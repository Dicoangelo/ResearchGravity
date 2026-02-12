"""
Webhook Normalizer — Convert WebhookEvent to CapturedEvent.

Bridges webhook-specific events into the existing capture pipeline's
CapturedEvent format, which then flows through BaseNormalizer.to_cognitive_event().
"""

import time
import uuid
from typing import List

from capture.base import CapturedEvent
from .handlers.base import WebhookEvent


class WebhookNormalizer:
    """Convert webhook events to the shared CapturedEvent format."""

    def normalize_batch(
        self, provider: str, events: List[WebhookEvent]
    ) -> List[CapturedEvent]:
        return [self._normalize_one(provider, e) for e in events]

    def _normalize_one(
        self, provider: str, event: WebhookEvent
    ) -> CapturedEvent:
        platform = f"{provider}-webhook"
        ts = event.timestamp or time.time()

        return CapturedEvent(
            platform=platform,
            session_id=event.session_id or f"{provider}-{uuid.uuid4().hex[:8]}",
            content=event.content,
            role=event.role,
            timestamp=ts,
            metadata={
                "webhook_provider": provider,
                "webhook_event_type": event.event_type,
                **event.metadata,
            },
            quality_score=0.0,      # Scored later in server.py
            cognitive_mode="exploration",  # Classified later in server.py
        )
