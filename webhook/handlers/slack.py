"""
Slack Webhook Handler — Events API and URL verification.

Slack Events API flow:
1. URL verification challenge (responds with challenge token)
2. Event dispatching (message, app_mention, reaction_added, etc.)
"""

import json
import time
from typing import List, Mapping

from .base import WebhookHandler, WebhookEvent


class SlackHandler(WebhookHandler):

    @property
    def provider(self) -> str:
        return "slack"

    @property
    def platform(self) -> str:
        return "slack-webhook"

    def supported_events(self) -> List[str]:
        return [
            "message", "app_mention", "reaction_added",
            "channel_created", "member_joined_channel",
        ]

    async def handle(
        self, headers: Mapping[str, str], body: bytes
    ) -> List[WebhookEvent]:
        payload = json.loads(body)

        # URL verification challenge — return empty events,
        # server.py handles the challenge response directly
        if payload.get("type") == "url_verification":
            return []

        event_type = payload.get("type", "")

        if event_type == "event_callback":
            return self._handle_event(payload)

        return [self._handle_generic(payload)]

    def _handle_event(self, payload: dict) -> List[WebhookEvent]:
        event = payload.get("event", {})
        event_type = event.get("type", "unknown")
        user = event.get("user", "unknown")
        text = event.get("text", "")
        channel = event.get("channel", "")
        ts = float(event.get("ts", time.time()))

        # Skip bot messages to avoid loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return []

        content = f"Slack {event_type} from {user} in {channel}: {text}"

        return [WebhookEvent(
            event_type=event_type,
            content=content,
            metadata={
                "channel": channel,
                "user": user,
                "slack_event_type": event_type,
                "team_id": payload.get("team_id", ""),
                "event_id": payload.get("event_id", ""),
            },
            timestamp=ts,
            session_id=f"slack-{channel}",
            role="user",
        )]

    def _handle_generic(self, payload: dict) -> List[WebhookEvent]:
        event_type = payload.get("type", "unknown")
        content = f"Slack event: {event_type}"

        return [WebhookEvent(
            event_type=event_type,
            content=content,
            metadata={"raw_type": event_type},
            timestamp=time.time(),
            session_id="slack-generic",
            role="system",
        )]
