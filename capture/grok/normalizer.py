"""
Grok Normalizer — Convert Grok/X conversations to cognitive_events.

Maps:
  - Grok conversation format to cognitive events
  - Tags with platform='grok'
  - Cognitive mode based on content analysis
  - X/Grok conversations tend toward strategic/world-level thinking
"""

from ..base import CapturedEvent
from ..normalizer import BaseNormalizer
from ..quality import score_event


# Grok/X conversations tend toward strategic topics
_STRATEGIC_KEYWORDS = {
    "strategy", "market", "trend", "prediction", "future",
    "geopolit", "econom", "policy", "innovation", "disruption",
    "compete", "advantage", "landscape", "industry", "global",
}


class GrokNormalizer(BaseNormalizer):
    """Normalize Grok/X conversations into cognitive_events rows."""

    def to_cognitive_event(self, captured: CapturedEvent, session_topic: str = "") -> dict:
        """Override to inject Grok-specific metadata."""
        # Score quality
        qs, mode = score_event(captured.content, captured.role, captured.platform)
        captured.quality_score = qs

        # Check for strategic content (Grok's strength)
        cl = captured.content.lower()
        strategic_hits = sum(1 for kw in _STRATEGIC_KEYWORDS if kw in cl)
        if strategic_hits >= 2 and mode != "garbage":
            captured.cognitive_mode = "deep_work"
        else:
            captured.cognitive_mode = mode

        row = super().to_cognitive_event(captured, session_topic=session_topic)
        row["method"] = f"grok.{captured.role}"

        return row
