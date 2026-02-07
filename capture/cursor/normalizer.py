"""
Cursor Normalizer — Convert Cursor AI interactions to cognitive_events.

Maps:
  - Cursor AI assistant messages to cognitive events
  - File context (which files were being edited)
  - Tags with platform='cursor', cognitive_mode='deep_work' (Cursor is always coding)
"""

from ..base import CapturedEvent
from ..normalizer import BaseNormalizer
from ..quality import score_event


class CursorNormalizer(BaseNormalizer):
    """Normalize Cursor AI interactions into cognitive_events rows."""

    def to_cognitive_event(self, captured: CapturedEvent, session_topic: str = "") -> dict:
        """Override to inject Cursor-specific metadata."""
        # Use workspace name as session topic
        topic = captured.metadata.get("workspace", session_topic or "coding")

        # Score quality — Cursor interactions tend to be deep_work
        qs, mode = score_event(captured.content, captured.role, captured.platform)
        captured.quality_score = qs
        # Override mode: Cursor is always coding context
        captured.cognitive_mode = "deep_work" if qs >= 0.4 else mode

        row = super().to_cognitive_event(captured, session_topic=topic)

        # Enrich with file context
        file_ctx = captured.metadata.get("file_context", "")
        if file_ctx:
            row["method"] = f"cursor.{captured.role}.{_ext(file_ctx)}"
        else:
            row["method"] = f"cursor.{captured.role}"

        return row


def _ext(filepath: str) -> str:
    """Extract file extension for method tagging."""
    if "." in filepath:
        return filepath.rsplit(".", 1)[-1][:10]
    return "unknown"
