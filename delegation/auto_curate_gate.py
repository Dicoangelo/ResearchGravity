"""
Coherence-Gated Auto-Curate Loop (E7).

Closes the loop from coherence-arc detection → NotebookLM auto-curation by
applying a deterministic, testable policy gate:

    candidates (coherence arcs) → [gate] → approved curations → lineage nodes

Why a separate module:

* `notebooklm_mcp/api/cognitive.py::check_curation_triggers` emits suggestions
  from Postgres but has no decision policy — whoever calls it has to invent
  thresholds on the fly.
* `auto_curate_notebook` creates the notebook but does not wire the result into
  lineage, so the compound-intelligence loop is broken.
* This gate is infrastructure-free: it takes plain dict candidates and a
  `LineageStore`, so it can be unit-tested without Postgres and reused from
  any daemon, MCP tool, or CLI.

Policy (all tunable via `CurateGatePolicy`):
  1. Significance floor — arcs below `min_significance` are skipped.
  2. Platform diversity — arcs below `min_platforms` are local patterns,
     not cross-platform convergence, so we skip them.
  3. Moment floor — arcs with fewer than `min_moments` are too thin.
  4. Cooldown — arcs curated within `cooldown_hours` are skipped (dedup).
  5. Budget — cap the number of curations emitted per cycle to avoid
     notebook spam from a burst of arcs.
  6. Ranking — ties broken by (significance desc, moments desc, platforms desc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from .lineage_store import LineageStore, LineageStoreError


AUTO_CURATE_ROOT_ID = "auto-curate-root"


@dataclass
class CurateGatePolicy:
    min_significance: float = 0.7
    min_platforms: int = 2
    min_moments: int = 3
    cooldown_hours: int = 24
    budget_per_cycle: int = 3


@dataclass
class CurateDecision:
    arc_id: str
    topic: str
    significance: float
    platforms: int
    moments: int
    approved: bool
    reason: str
    lineage_node_id: Optional[str] = None


@dataclass
class GateResult:
    approved: list[CurateDecision] = field(default_factory=list)
    rejected: list[CurateDecision] = field(default_factory=list)

    @property
    def approved_arc_ids(self) -> list[str]:
        return [d.arc_id for d in self.approved]


class CurateGate:
    """Policy gate that decides which coherence arcs get auto-curated.

    Side effect on approval: records a lineage node (node_type="concept") for
    each approved arc, wired under `AUTO_CURATE_ROOT_ID`. The returned
    `CurateDecision.lineage_node_id` is the created node id.
    """

    def __init__(
        self,
        store: LineageStore,
        policy: Optional[CurateGatePolicy] = None,
    ) -> None:
        self._store = store
        self._policy = policy or CurateGatePolicy()
        self._ensure_root()

    def _ensure_root(self) -> None:
        if self._store.get_node(AUTO_CURATE_ROOT_ID) is None:
            try:
                self._store.add_node(
                    AUTO_CURATE_ROOT_ID,
                    node_type="session",
                    metadata={"purpose": "auto-curate lineage anchor"},
                )
            except LineageStoreError:
                pass  # Race: another process created it first.

    @property
    def policy(self) -> CurateGatePolicy:
        return self._policy

    def evaluate(
        self,
        candidates: Iterable[dict[str, Any]],
        now: Optional[datetime] = None,
    ) -> GateResult:
        """Score + filter candidates; record lineage for approved ones.

        Each candidate dict must carry: arc_id, topic, significance,
        platforms (int), moments (int). Optional: last_curated_at (ISO str
        or datetime) — used for cooldown.
        """
        now = now or datetime.now(timezone.utc)
        cooldown_cutoff = now - timedelta(hours=self._policy.cooldown_hours)

        candidate_list = list(candidates)
        result = GateResult()

        scored: list[tuple[dict[str, Any], float]] = []
        for cand in candidate_list:
            decision = self._policy_check(cand, cooldown_cutoff)
            if not decision.approved:
                result.rejected.append(decision)
                continue
            # ranking key: significance desc, moments desc, platforms desc
            rank = (
                -float(cand.get("significance", 0.0)),
                -int(cand.get("moments", 0)),
                -int(cand.get("platforms", 0)),
            )
            scored.append((cand, rank))

        scored.sort(key=lambda x: x[1])

        for cand, _rank in scored[: self._policy.budget_per_cycle]:
            decision = self._approve_and_record(cand)
            result.approved.append(decision)

        for cand, _rank in scored[self._policy.budget_per_cycle :]:
            result.rejected.append(
                CurateDecision(
                    arc_id=str(cand["arc_id"]),
                    topic=str(cand.get("topic", "")),
                    significance=float(cand.get("significance", 0.0)),
                    platforms=int(cand.get("platforms", 0)),
                    moments=int(cand.get("moments", 0)),
                    approved=False,
                    reason="over_budget",
                )
            )

        return result

    def _policy_check(
        self, cand: dict[str, Any], cooldown_cutoff: datetime
    ) -> CurateDecision:
        arc_id = str(cand["arc_id"])
        topic = str(cand.get("topic", ""))
        sig = float(cand.get("significance", 0.0))
        platforms = int(cand.get("platforms", 0))
        moments = int(cand.get("moments", 0))

        if sig < self._policy.min_significance:
            return CurateDecision(
                arc_id, topic, sig, platforms, moments, False,
                f"below_significance_floor ({sig:.2f} < {self._policy.min_significance})",
            )
        if platforms < self._policy.min_platforms:
            return CurateDecision(
                arc_id, topic, sig, platforms, moments, False,
                f"below_platform_floor ({platforms} < {self._policy.min_platforms})",
            )
        if moments < self._policy.min_moments:
            return CurateDecision(
                arc_id, topic, sig, platforms, moments, False,
                f"below_moment_floor ({moments} < {self._policy.min_moments})",
            )

        last = cand.get("last_curated_at")
        if last is not None:
            last_dt = _coerce_dt(last)
            if last_dt is not None and last_dt > cooldown_cutoff:
                return CurateDecision(
                    arc_id, topic, sig, platforms, moments, False,
                    f"in_cooldown (last_curated_at={last_dt.isoformat()})",
                )

        lineage_id = self._lineage_id_for(arc_id)
        if self._store.get_node(lineage_id) is not None:
            return CurateDecision(
                arc_id, topic, sig, platforms, moments, False,
                "already_curated",
            )

        return CurateDecision(
            arc_id, topic, sig, platforms, moments, True, "approved"
        )

    def _approve_and_record(self, cand: dict[str, Any]) -> CurateDecision:
        arc_id = str(cand["arc_id"])
        topic = str(cand.get("topic", ""))
        sig = float(cand.get("significance", 0.0))
        platforms = int(cand.get("platforms", 0))
        moments = int(cand.get("moments", 0))
        lineage_id = self._lineage_id_for(arc_id)
        try:
            self._store.add_node(
                lineage_id,
                parent_id=AUTO_CURATE_ROOT_ID,
                node_type="concept",
                metadata={
                    "arc_id": arc_id,
                    "topic": topic,
                    "significance": sig,
                    "platforms": platforms,
                    "moments": moments,
                    "source": "auto_curate_gate",
                },
            )
            node_id: Optional[str] = lineage_id
        except LineageStoreError:
            node_id = None
        return CurateDecision(
            arc_id, topic, sig, platforms, moments, True, "approved",
            lineage_node_id=node_id,
        )

    @staticmethod
    def _lineage_id_for(arc_id: str) -> str:
        return f"arc:{arc_id}"


def _coerce_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None
