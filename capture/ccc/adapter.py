"""
CCC Adapter — Ingest from Claude Command Center's claude.db

Reads operational data from ~/.claude/data/claude.db (SQLite):
  - sessions:       412 sessions with model, tokens, quality, cost
  - tool_events:    77K tool calls with success/failure/duration
  - routing_decisions: 1,150 DQ routing decisions
  - session_outcome_events: 1,371 session outcomes
  - git_events:     346 commits/pushes/PRs
  - recovery_events: 151 error recoveries

This is operational/infrastructure data — complements the transcript
data from claude-code adapter. Together they give full coverage:
  claude-code = WHAT was said (conversations)
  ccc         = WHAT happened (operations, tools, routing, git)

Stores watermark in ~/.ucw/capture_state/ccc.json
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..base import CapturedEvent, PlatformAdapter, AdapterStatus
from .. import config as cfg
from .normalizer import CCCNormalizer

log = logging.getLogger("capture.ccc")

CCC_DB_PATH = Path.home() / ".claude" / "data" / "claude.db"

# Tables to ingest, with their timestamp column and content builder
_TABLES = {
    "sessions": {
        "ts_col": "started_at",
        "ts_type": "datetime",  # ISO format
        "role": "system",
        "content_fn": "_session_content",
    },
    "tool_events": {
        "ts_col": "timestamp",
        "ts_type": "epoch_ms",
        "role": "system",
        "content_fn": "_tool_content",
    },
    "routing_decisions": {
        "ts_col": "timestamp",
        "ts_type": "datetime",
        "role": "system",
        "content_fn": "_routing_content",
    },
    "session_outcome_events": {
        "ts_col": "timestamp",
        "ts_type": "epoch_ms",
        "role": "system",
        "content_fn": "_outcome_content",
    },
    "git_events": {
        "ts_col": "timestamp",
        "ts_type": "epoch_ms",
        "role": "system",
        "content_fn": "_git_content",
    },
    "recovery_events": {
        "ts_col": "timestamp",
        "ts_type": "epoch_ms",
        "role": "system",
        "content_fn": "_recovery_content",
    },
    "coordinator_events": {
        "ts_col": "timestamp",
        "ts_type": "epoch_ms",
        "role": "system",
        "content_fn": "_coordinator_content",
    },
}


class CCCAdapter(PlatformAdapter):
    """Ingest operational data from Claude Command Center's claude.db."""

    def __init__(self):
        self._pool = None
        self._db_path = CCC_DB_PATH
        self._normalizer = CCCNormalizer()
        self._watermark = _Watermark()
        self._last_poll: float = 0
        self._events_captured: int = 0
        self._healthy = False
        self._error: Optional[str] = None

    @property
    def name(self) -> str:
        return "Claude Command Center"

    @property
    def platform(self) -> str:
        return "ccc"

    async def initialize(self, pool) -> bool:
        self._pool = pool
        self._watermark.load()

        if not self._db_path.exists():
            log.warning(f"CCC database not found: {self._db_path}")
            self._error = "claude.db not found"
            self._healthy = False
            return False

        # Verify DB is readable
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            conn.close()
            log.info(f"CCC database found: {self._db_path} ({count} sessions)")
            self._healthy = True
            return True
        except Exception as exc:
            log.error(f"CCC database error: {exc}")
            self._error = str(exc)
            return False

    async def poll(self) -> List[CapturedEvent]:
        """Read new rows from all tracked tables in claude.db."""
        events: List[CapturedEvent] = []

        if not self._db_path.exists():
            return events

        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row

            for table_name, table_cfg in _TABLES.items():
                try:
                    table_events = self._poll_table(conn, table_name, table_cfg)
                    events.extend(table_events)
                except Exception as exc:
                    log.error(f"Error polling {table_name}: {exc}")

            conn.close()

        except Exception as exc:
            log.error(f"CCC poll error: {exc}")
            self._error = str(exc)

        self._last_poll = time.time()
        self._events_captured += len(events)
        if events:
            self._watermark.save()
            self._error = None

        return events

    async def normalize(self, events: List[CapturedEvent]) -> List[dict]:
        return [self._normalizer.to_cognitive_event(e) for e in events]

    async def health_check(self) -> AdapterStatus:
        return AdapterStatus(
            healthy=self._healthy,
            last_poll=self._last_poll,
            events_captured=self._events_captured,
            error=self._error,
        )

    # ── table polling ────────────────────────────────────

    def _poll_table(
        self, conn: sqlite3.Connection, table_name: str, table_cfg: dict,
    ) -> List[CapturedEvent]:
        """Poll a single table for new rows since last watermark."""
        events: List[CapturedEvent] = []
        ts_col = table_cfg["ts_col"]
        ts_type = table_cfg["ts_type"]
        content_fn = getattr(self, table_cfg["content_fn"])

        # Get watermark for this table (always use rowid for reliable ordering)
        last_rowid = self._watermark.table_watermarks.get(table_name, 0)

        rows = conn.execute(
            f"SELECT rowid as _rowid, * FROM {table_name} WHERE rowid > ? ORDER BY rowid",
            (last_rowid,),
        ).fetchall()

        max_rowid = last_rowid
        for row in rows:
            row_dict = dict(row)
            row_id = row_dict.pop("_rowid", 0)

            # Parse timestamp
            ts = self._parse_timestamp(row_dict.get(ts_col, 0), ts_type)
            if ts <= 0:
                ts = time.time()

            # Build content text
            content = content_fn(row_dict)
            if not content or len(content) < 10:
                continue

            events.append(CapturedEvent(
                platform="ccc",
                session_id=f"ccc-{table_name}",
                content=content,
                role="system",
                timestamp=ts,
                metadata={
                    "table": table_name,
                    "row_id": row_id,
                    **{k: v for k, v in row_dict.items()
                       if k not in ("metadata",) and v is not None},
                },
            ))

            if isinstance(row_id, int) and row_id > max_rowid:
                max_rowid = row_id

        if max_rowid > last_rowid:
            self._watermark.table_watermarks[table_name] = max_rowid

        return events

    def _parse_timestamp(self, value, ts_type: str) -> float:
        """Convert various timestamp formats to Unix seconds."""
        if not value:
            return 0
        if ts_type == "epoch_ms":
            try:
                v = int(value)
                # Could be seconds or milliseconds
                if v > 1_000_000_000_000:
                    return v / 1000.0
                return float(v)
            except (ValueError, TypeError):
                return 0
        elif ts_type == "datetime":
            try:
                from datetime import datetime
                if isinstance(value, str):
                    return datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    ).timestamp()
                return float(value)
            except Exception:
                return 0
        return 0

    # ── content builders ─────────────────────────────────

    def _session_content(self, row: dict) -> str:
        model = row.get("model", "unknown")
        msgs = row.get("message_count", 0)
        tools = row.get("tool_count", 0)
        quality = row.get("quality_score", 0) or 0
        cost = row.get("cost_estimate", 0) or 0
        outcome = row.get("outcome", "unknown")
        project = row.get("project_path", "")
        complexity = row.get("complexity", 0) or 0
        inp = row.get("input_tokens", 0)
        out = row.get("output_tokens", 0)
        cache = row.get("cache_read_tokens", 0)

        return (
            f"Session on {model}: {msgs} messages, {tools} tool calls, "
            f"quality={quality:.2f}, complexity={complexity:.2f}, "
            f"cost=${cost:.4f}, outcome={outcome}. "
            f"Tokens: {inp} in / {out} out / {cache} cached. "
            f"Project: {project}"
        )

    def _tool_content(self, row: dict) -> str:
        tool = row.get("tool_name", "unknown")
        success = "success" if row.get("success") else "failure"
        duration = row.get("duration_ms", 0) or 0
        error = row.get("error_message", "")
        ctx = row.get("context", "")

        text = f"Tool call: {tool} → {success}"
        if duration:
            text += f" ({duration}ms)"
        if error:
            text += f" error: {error}"
        if ctx:
            text += f" context: {ctx[:200]}"
        return text

    def _routing_content(self, row: dict) -> str:
        model = row.get("selected_model", "unknown")
        dq = row.get("dq_score", 0) or 0
        complexity = row.get("complexity", 0) or 0
        preview = row.get("query_preview", "")
        validity = row.get("dq_validity", 0) or 0
        specificity = row.get("dq_specificity", 0) or 0
        correctness = row.get("dq_correctness", 0) or 0
        cost = row.get("cost_estimate", 0) or 0

        return (
            f"Routing decision: {preview} → {model} "
            f"(DQ={dq:.3f} V={validity:.2f} S={specificity:.2f} C={correctness:.2f}) "
            f"complexity={complexity:.2f} cost=${cost:.4f}"
        )

    def _outcome_content(self, row: dict) -> str:
        outcome = row.get("outcome", "unknown")
        quality = row.get("quality_score", 0) or 0
        complexity = row.get("complexity", 0) or 0
        model = row.get("model_used", "unknown")
        cost = row.get("cost", 0) or 0
        msgs = row.get("message_count", 0)

        return (
            f"Session outcome: {outcome} on {model}, "
            f"quality={quality:.2f}, complexity={complexity:.2f}, "
            f"cost=${cost:.4f}, {msgs} messages"
        )

    def _git_content(self, row: dict) -> str:
        event_type = row.get("event_type", "unknown")
        repo = row.get("repo", "unknown")
        branch = row.get("branch", "")
        message = row.get("message", "")
        files = row.get("files_changed", 0) or 0
        additions = row.get("additions", 0) or 0
        deletions = row.get("deletions", 0) or 0

        text = f"Git {event_type}: {repo}"
        if branch:
            text += f" ({branch})"
        if message:
            text += f" — {message[:200]}"
        if files:
            text += f" [{files} files, +{additions}/-{deletions}]"
        return text

    def _recovery_content(self, row: dict) -> str:
        error_type = row.get("error_type") or "unknown"
        strategy = row.get("recovery_strategy") or "unknown"
        success = "recovered" if row.get("success") else "failed"
        attempts = row.get("attempts") or 1
        method = row.get("recovery_method") or "unknown"
        details = row.get("error_details") or ""

        return (
            f"Error recovery: {error_type} → {strategy} ({method}) "
            f"→ {success} in {attempts} attempts. {details[:200]}"
        )

    def _coordinator_content(self, row: dict) -> str:
        action = row.get("action", "unknown")
        strategy = row.get("strategy", "")
        agent_id = row.get("agent_id", "")
        result = row.get("result", "")
        duration = row.get("duration_ms", 0) or 0
        file_path = row.get("file_path", "")

        text = f"Coordinator {action}"
        if strategy:
            text += f" ({strategy})"
        if agent_id:
            text += f" agent={agent_id}"
        if file_path:
            text += f" file={file_path}"
        if result:
            text += f" → {result[:200]}"
        if duration:
            text += f" ({duration}ms)"
        return text


# ── watermark ────────────────────────────────────────────────

class _Watermark:
    """Track last-polled row IDs per table."""

    def __init__(self):
        self.path = cfg.STATE_DIR / "ccc.json"
        self.table_watermarks: Dict[str, int] = {}

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.table_watermarks = data.get("table_watermarks", {})
                log.info(f"Watermark loaded: {self.table_watermarks}")
            except Exception as exc:
                log.warning(f"Watermark load failed: {exc}")

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps({
                "table_watermarks": self.table_watermarks,
                "updated_at": time.time(),
            }, indent=2))
        except Exception as exc:
            log.error(f"Watermark save failed: {exc}")
