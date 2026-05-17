"""
Auto-Curate Runner (E7) — closes the coherence → NotebookLM loop.

The `CurateGate` decides *which* arcs get curated. This runner actually executes
the curation against whatever backend the caller provides (NotebookLM MCP,
ResearchGravity notebook, stub for tests) and records the result in the lineage
store so the compound-intelligence trail is unbroken:

    coherence_arc → gate-approved decision → lineage node (concept)
                                           → notebook created (executor)
                                           → lineage node (session) child of the concept
                                           → edge: session --derives_from--> concept

Design choices:

* The runner is infrastructure-free. The executor is any `async (arc_id) -> dict | None`,
  so we can unit-test it with a stub while production wires it to
  `CognitiveLayer.auto_curate_notebook`.
* Executor failures don't crash the cycle; each approved decision gets a
  `RunOutcome` with either `notebook_id` (success), `error` (raised), or both
  `None` (executor returned `None`).
* Lineage edges are best-effort: if the lineage store raises, we still return
  the outcome with `lineage_node_id=None` rather than losing the work.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .auto_curate_gate import CurateDecision, CurateGate, GateResult
from .lineage_store import LineageStore, LineageStoreError


logger = logging.getLogger(__name__)

ExecutorFn = Callable[[str], Awaitable[Optional[dict[str, Any]]]]


@dataclass
class RunOutcome:
    arc_id: str
    notebook_id: Optional[str] = None
    lineage_node_id: Optional[str] = None
    error: Optional[str] = None
    executor_result: Optional[dict[str, Any]] = None

    @property
    def succeeded(self) -> bool:
        return self.notebook_id is not None and self.error is None


@dataclass
class RunReport:
    gate_result: GateResult
    outcomes: list[RunOutcome] = field(default_factory=list)

    @property
    def approved_count(self) -> int:
        return len(self.gate_result.approved)

    @property
    def succeeded_count(self) -> int:
        return sum(1 for o in self.outcomes if o.succeeded)

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.outcomes if not o.succeeded)


class AutoCurateRunner:
    """Executes approved curations and records their lineage.

    Typical production wiring::

        from delegation import CurateGate, AutoCurateRunner, LineageStore

        store = LineageStore()
        gate = CurateGate(store)
        runner = AutoCurateRunner(gate, store, cognitive.auto_curate_notebook)

        candidates = await cognitive.check_curation_triggers()
        report = await runner.run_cycle(candidates)
    """

    def __init__(
        self,
        gate: CurateGate,
        store: LineageStore,
        executor: ExecutorFn,
    ) -> None:
        self._gate = gate
        self._store = store
        self._executor = executor

    async def run_cycle(
        self, candidates: list[dict[str, Any]]
    ) -> RunReport:
        gate_result = self._gate.evaluate(candidates)
        report = RunReport(gate_result=gate_result)

        for decision in gate_result.approved:
            outcome = await self._execute_one(decision)
            report.outcomes.append(outcome)

        return report

    async def _execute_one(self, decision: CurateDecision) -> RunOutcome:
        outcome = RunOutcome(arc_id=decision.arc_id)

        try:
            result = await self._executor(decision.arc_id)
        except Exception as exc:  # executor contract is best-effort
            logger.exception("auto-curate executor raised for arc %s", decision.arc_id)
            outcome.error = f"{type(exc).__name__}: {exc}"
            return outcome

        outcome.executor_result = result
        if not result:
            outcome.error = "executor_returned_none"
            return outcome

        notebook_id = str(result.get("notebook_id")) if result.get("notebook_id") else None
        outcome.notebook_id = notebook_id
        if not notebook_id:
            outcome.error = "missing_notebook_id"
            return outcome

        outcome.lineage_node_id = self._record_notebook_node(
            decision, notebook_id, result
        )
        return outcome

    def _record_notebook_node(
        self,
        decision: CurateDecision,
        notebook_id: str,
        executor_result: dict[str, Any],
    ) -> Optional[str]:
        parent = decision.lineage_node_id
        if parent is None:
            return None

        node_id = f"notebook:{notebook_id}"
        if self._store.get_node(node_id) is not None:
            return node_id  # idempotent

        metadata = {
            "notebook_id": notebook_id,
            "arc_id": decision.arc_id,
            "topic": decision.topic,
            "title": executor_result.get("title"),
            "sources_added": executor_result.get("sources_added"),
            "moments_included": executor_result.get("moments_included"),
            "source": "auto_curate_runner",
        }

        try:
            self._store.add_node(
                node_id,
                parent_id=parent,
                node_type="session",
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
            self._store.add_edge(
                node_id,
                parent,
                edge_type="derives_from",
                metadata={"source": "auto_curate_runner"},
            )
        except LineageStoreError as exc:
            logger.warning(
                "lineage write failed for notebook %s under %s: %s",
                notebook_id,
                parent,
                exc,
            )
            return None

        return node_id


async def run_auto_curate_cycle(
    gate: CurateGate,
    store: LineageStore,
    executor: ExecutorFn,
    candidates: list[dict[str, Any]],
) -> RunReport:
    """Convenience wrapper for one-shot callers (daemons, CLI, cron)."""
    runner = AutoCurateRunner(gate, store, executor)
    return await runner.run_cycle(candidates)
