"""
Generic Webhook Handler — Accepts any JSON payload.

Useful for custom integrations, testing, and services without
a dedicated handler. Stores the full payload as content.
"""

import json
import time
from typing import List, Mapping

from .base import WebhookHandler, WebhookEvent


class GenericHandler(WebhookHandler):

    @property
    def provider(self) -> str:
        return "generic"

    @property
    def platform(self) -> str:
        return "generic-webhook"

    async def handle(
        self, headers: Mapping[str, str], body: bytes
    ) -> List[WebhookEvent]:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body.decode("utf-8", errors="replace")[:2000]}

        # Try to extract meaningful fields
        content = (
            payload.get("text")
            or payload.get("message")
            or payload.get("content")
            or payload.get("body")
            or json.dumps(payload, default=str)[:1000]
        )

        event_type = (
            payload.get("event_type")
            or payload.get("type")
            or payload.get("action")
            or "generic"
        )

        session_id = (
            payload.get("session_id")
            or payload.get("channel")
            or payload.get("project")
            or f"generic-{int(time.time())}"
        )

        return [WebhookEvent(
            event_type=event_type,
            content=str(content),
            metadata={"source": "generic-webhook", "keys": list(payload.keys())[:20]},
            timestamp=time.time(),
            session_id=session_id,
            role=payload.get("role", "system"),
        )]
