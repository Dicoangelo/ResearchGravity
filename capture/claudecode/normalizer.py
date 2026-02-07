"""
Claude Code Normalizer — Convert CLI transcript messages to cognitive_events.

Maps:
  - User/assistant messages with full context
  - Tool use tracking (which tools were invoked)
  - Working directory and git branch as session context
  - Model + token usage metadata
  - Default cognitive_mode='deep_work' (Claude Code = always coding)
"""

from ..base import CapturedEvent
from ..normalizer import BaseNormalizer
from ..quality import score_event


class ClaudeCodeNormalizer(BaseNormalizer):
    """Normalize Claude Code transcript messages into cognitive_events rows."""

    def to_cognitive_event(self, captured: CapturedEvent, session_topic: str = "") -> dict:
        """Override to inject Claude Code-specific metadata."""
        # Use cwd as session topic hint
        cwd = captured.metadata.get("cwd", "")
        topic = _project_from_cwd(cwd) or session_topic or "coding"

        # Score quality
        qs, mode = score_event(captured.content, captured.role, captured.platform)
        captured.quality_score = qs
        # Claude Code is always deep work context (coding/architecture)
        captured.cognitive_mode = "deep_work" if qs >= 0.35 else mode

        row = super().to_cognitive_event(captured, session_topic=topic)

        # Enrich method with tool context
        tool_uses = captured.metadata.get("tool_uses", [])
        if tool_uses:
            row["method"] = f"claude-code.{captured.role}.tools({','.join(tool_uses[:3])})"
        else:
            row["method"] = f"claude-code.{captured.role}"

        # Protocol
        row["protocol"] = "cli-transcript"

        return row

    def _protocol_for(self, platform: str) -> str:
        return "cli-transcript"


def _project_from_cwd(cwd: str) -> str:
    """Extract project name from working directory."""
    if not cwd:
        return ""
    # ~/OS-App -> OS-App, ~/researchgravity -> researchgravity
    parts = cwd.rstrip("/").split("/")
    if parts:
        last = parts[-1]
        if last and last != "~" and not last.startswith("."):
            return last
    return ""
