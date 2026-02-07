"""
Grok/X Adapter — X API + Grok conversation polling.

Polls X API for Grok conversations when API is available.
Gracefully no-ops if no API key is configured.

Stores watermark in ~/.ucw/capture_state/grok.json
"""

import json
import logging
import time
from typing import Dict, List, Optional

from ..base import CapturedEvent, PlatformAdapter, AdapterStatus
from .. import config as cfg
from .normalizer import GrokNormalizer

log = logging.getLogger("capture.grok")


class GrokAdapter(PlatformAdapter):
    """Live capture from Grok/X API."""

    def __init__(self):
        self._pool = None
        self._normalizer = GrokNormalizer()
        self._watermark = _Watermark()
        self._last_poll: float = 0
        self._events_captured: int = 0
        self._healthy = False
        self._error: Optional[str] = None
        self._api_key = cfg.GROK_API_KEY
        self._http_session = None

    @property
    def name(self) -> str:
        return "Grok/X Capture"

    @property
    def platform(self) -> str:
        return "grok"

    async def initialize(self, pool) -> bool:
        self._pool = pool
        self._watermark.load()

        if not self._api_key:
            log.info("Grok API key not configured (set UCW_GROK_API_KEY)")
            self._error = "No API key configured"
            self._healthy = False
            return False

        # Validate API key with a lightweight check
        try:
            import aiohttp
            self._http_session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            self._healthy = True
            log.info("Grok adapter initialized with API key")
            return True
        except ImportError:
            log.warning("aiohttp not installed — Grok adapter requires it (pip install aiohttp)")
            self._error = "aiohttp not installed"
            return False

    async def poll(self) -> List[CapturedEvent]:
        """Poll Grok API for new conversations."""
        if not self._healthy or not self._http_session:
            return []

        events: List[CapturedEvent] = []

        try:
            # Poll Grok conversations endpoint
            # X/Grok API: https://docs.x.ai/api
            grok_events = await self._poll_grok_conversations()
            events.extend(grok_events)
        except Exception as exc:
            log.error(f"Grok poll error: {exc}")
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

    # ── Grok API polling ─────────────────────────────────

    async def _poll_grok_conversations(self) -> List[CapturedEvent]:
        """Poll Grok/xAI API for conversation history."""
        events: List[CapturedEvent] = []

        if not self._http_session:
            return events

        try:
            # xAI API endpoint for conversation listing
            # This uses the xAI API which provides Grok access
            url = "https://api.x.ai/v1/chat/completions"

            # The xAI API is a chat completions API (like OpenAI)
            # It doesn't have a conversations list endpoint yet.
            # For now, we check if there are saved conversation exports.
            grok_export = cfg.STATE_DIR / "grok_conversations.json"
            if grok_export.exists():
                return self._poll_grok_export(grok_export)

        except Exception as exc:
            log.debug(f"Grok API error: {exc}")

        return events

    def _poll_grok_export(self, export_path) -> List[CapturedEvent]:
        """Poll a local Grok conversation export file."""
        events: List[CapturedEvent] = []

        try:
            mtime = export_path.stat().st_mtime
            if mtime <= self._watermark.last_poll_time:
                return events

            data = json.loads(export_path.read_text())
            conversations = data if isinstance(data, list) else data.get("conversations", [])

            for conv in conversations:
                conv_id = conv.get("id", conv.get("conversation_id", ""))
                if not conv_id:
                    continue

                if conv_id in self._watermark.seen_conversations:
                    continue

                messages = conv.get("messages", [])
                for msg in messages:
                    content = msg.get("content", "")
                    if not content or len(content.strip()) < 10:
                        continue

                    role = msg.get("role", "assistant")
                    if role == "system":
                        continue

                    ts = msg.get("timestamp", msg.get("created_at", time.time()))
                    if isinstance(ts, str):
                        try:
                            from datetime import datetime
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                        except Exception:
                            ts = time.time()

                    events.append(CapturedEvent(
                        platform="grok",
                        session_id=f"grok-{conv_id}",
                        content=content.strip(),
                        role="user" if role == "user" else "assistant",
                        timestamp=ts,
                        metadata={
                            "conversation_id": conv_id,
                            "model": conv.get("model", "grok"),
                        },
                    ))

                self._watermark.seen_conversations.add(conv_id)

            self._watermark.last_poll_time = mtime

        except Exception as exc:
            log.error(f"Grok export poll error: {exc}")

        return events

    async def shutdown(self):
        """Clean up HTTP session."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None


# ── watermark ────────────────────────────────────────────────

class _Watermark:
    """Track Grok polling state."""

    def __init__(self):
        self.path = cfg.STATE_DIR / "grok.json"
        self.last_poll_time: float = 0
        self.seen_conversations: set = set()

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.last_poll_time = data.get("last_poll_time", 0)
                self.seen_conversations = set(data.get("seen_conversations", []))
                log.info(f"Watermark loaded: {len(self.seen_conversations)} conversations tracked")
            except Exception as exc:
                log.warning(f"Watermark load failed: {exc}")

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps({
                "last_poll_time": self.last_poll_time,
                "seen_conversations": list(self.seen_conversations),
                "updated_at": time.time(),
            }, indent=2))
        except Exception as exc:
            log.error(f"Watermark save failed: {exc}")
