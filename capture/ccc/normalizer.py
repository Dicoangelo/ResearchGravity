"""
CCC Normalizer — Convert Command Center operational data to cognitive_events.

Maps:
  - Session summaries, tool calls, routing decisions → cognitive events
  - Tags with platform='ccc', protocol='sqlite-local'
  - Operational events are infrastructure-level (system role)
  - Cognitive mode derived from event type:
      sessions/outcomes → deep_work (meta-cognitive)
      routing → exploration (decision-making)
      tools → deep_work (execution)
      git → deep_work (creation)
      recovery → exploration (problem-solving)
"""

from ..base import CapturedEvent
from ..normalizer import BaseNormalizer
from ..quality import score_event


# Event type → cognitive mode mapping
_TABLE_MODES = {
    "sessions": "deep_work",
    "session_outcome_events": "deep_work",
    "tool_events": "deep_work",
    "routing_decisions": "exploration",
    "git_events": "deep_work",
    "recovery_events": "exploration",
    "coordinator_events": "deep_work",
}


class CCCNormalizer(BaseNormalizer):
    """Normalize CCC operational data into cognitive_events rows."""

    def to_cognitive_event(self, captured: CapturedEvent, session_topic: str = "") -> dict:
        """Override to inject CCC-specific metadata."""
        table = captured.metadata.get("table", "")
        topic = f"ccc-{table}" if table else session_topic or "infrastructure"

        # Score quality
        qs, mode = score_event(captured.content, captured.role, captured.platform)
        captured.quality_score = max(qs, 0.5)  # CCC data is always at least moderate quality
        captured.cognitive_mode = _TABLE_MODES.get(table, mode)

        row = super().to_cognitive_event(captured, session_topic=topic)

        # Method reflects the table/event type
        row["method"] = f"ccc.{table}"
        row["protocol"] = "sqlite-local"

        return row

    def _protocol_for(self, platform: str) -> str:
        return "sqlite-local"
