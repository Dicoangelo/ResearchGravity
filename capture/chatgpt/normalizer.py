"""
ChatGPT Normalizer — Convert ChatGPT messages to cognitive_events.

Maps:
  - ChatGPT roles (user/assistant) to UCW direction (in/out)
  - Conversation title as session topic
  - Model info to metadata
  - Multi-part content handling
"""

from ..base import CapturedEvent
from ..normalizer import BaseNormalizer
from ..quality import score_event


class ChatGPTNormalizer(BaseNormalizer):
    """Normalize ChatGPT messages into cognitive_events rows."""

    def to_cognitive_event(self, captured: CapturedEvent, session_topic: str = "") -> dict:
        """Override to inject ChatGPT-specific metadata."""
        # Use conversation title as session topic if available
        topic = captured.metadata.get("conversation_title", session_topic or "")

        # Score quality
        qs, mode = score_event(captured.content, captured.role, captured.platform)
        captured.quality_score = qs
        captured.cognitive_mode = mode

        # Build base event
        row = super().to_cognitive_event(captured, session_topic=topic)

        # Enrich metadata with ChatGPT-specific fields
        row["method"] = f"chatgpt.{captured.role}"

        return row
