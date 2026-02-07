"""
ChatGPT Live Adapter — Export-diff polling + optional OpenAI API.

Primary strategy: Watch CHATGPT_EXPORT_PATH for new/updated conversations.json,
diff against previously seen conversation IDs, extract new messages.

Secondary (if OPENAI_API_KEY set): Poll OpenAI /v1/conversations endpoint.

Stores watermark in ~/.ucw/capture_state/chatgpt.json
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..base import CapturedEvent, PlatformAdapter, AdapterStatus
from .. import config as cfg
from .normalizer import ChatGPTNormalizer

log = logging.getLogger("capture.chatgpt")


class ChatGPTAdapter(PlatformAdapter):
    """Live capture from ChatGPT exports and optional OpenAI API."""

    def __init__(self):
        self._pool = None
        self._export_path = Path(cfg.CHATGPT_EXPORT_PATH)
        self._normalizer = ChatGPTNormalizer()
        self._watermark = _Watermark()
        self._last_poll: float = 0
        self._events_captured: int = 0
        self._healthy = False
        self._error: Optional[str] = None

    @property
    def name(self) -> str:
        return "ChatGPT Live Capture"

    @property
    def platform(self) -> str:
        return "chatgpt"

    async def initialize(self, pool) -> bool:
        self._pool = pool
        self._watermark.load()

        # Check if export path exists
        if not self._export_path.exists():
            log.info(f"ChatGPT export path not found: {self._export_path}")
            log.info("Will create when first export is placed there")
            # Still mark as healthy — path might appear later
            self._healthy = True
            return True

        conv_file = self._export_path / "conversations.json"
        if conv_file.exists():
            log.info(f"ChatGPT export found: {conv_file}")
            self._healthy = True
        else:
            log.info(f"No conversations.json in {self._export_path} yet")
            self._healthy = True

        return True

    async def poll(self) -> List[CapturedEvent]:
        """Fetch new events since last poll via export-diff."""
        events: List[CapturedEvent] = []

        # Primary: export-diff polling
        try:
            export_events = self._poll_export()
            events.extend(export_events)
        except Exception as exc:
            log.error(f"Export poll error: {exc}")
            self._error = str(exc)

        # Secondary: OpenAI API (if configured)
        if cfg.OPENAI_API_KEY:
            try:
                api_events = await self._poll_openai_api()
                events.extend(api_events)
            except Exception as exc:
                log.debug(f"OpenAI API poll error: {exc}")

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

    # ── export-diff polling ──────────────────────────────

    def _poll_export(self) -> List[CapturedEvent]:
        """Diff conversations.json against last-seen state."""
        conv_file = self._export_path / "conversations.json"
        if not conv_file.exists():
            return []

        # Check if file was modified since last poll
        mtime = conv_file.stat().st_mtime
        if mtime <= self._watermark.last_export_mtime:
            return []

        log.info(f"Export file changed (mtime={mtime:.0f}), scanning for new conversations...")

        conversations = json.loads(conv_file.read_text())
        events: List[CapturedEvent] = []

        for conv in conversations:
            conv_id = conv.get("id", conv.get("conversation_id", ""))
            if not conv_id:
                continue

            update_time = conv.get("update_time", 0) or 0

            # Skip already-processed conversations (by ID + update_time)
            if conv_id in self._watermark.seen_conversations:
                if update_time <= self._watermark.seen_conversations[conv_id]:
                    continue

            # Extract new messages
            messages = _extract_messages(conv)
            title = conv.get("title", "Untitled")

            # Filter to messages after our watermark for this conversation
            last_seen_time = self._watermark.seen_conversations.get(conv_id, 0)

            for msg in messages:
                msg_time = msg.get("create_time", 0)
                if msg_time <= last_seen_time:
                    continue

                events.append(CapturedEvent(
                    platform="chatgpt",
                    session_id=f"chatgpt-{conv_id}",
                    content=msg["content"],
                    role=msg["role"],
                    timestamp=msg_time if msg_time > 0 else time.time(),
                    metadata={
                        "conversation_id": conv_id,
                        "conversation_title": title,
                        "model": conv.get("default_model_slug", ""),
                    },
                ))

            # Update watermark for this conversation
            self._watermark.seen_conversations[conv_id] = update_time

        self._watermark.last_export_mtime = mtime
        log.info(f"Export diff: {len(events)} new messages from {len(conversations)} conversations")
        return events

    async def _poll_openai_api(self) -> List[CapturedEvent]:
        """Poll OpenAI conversations API (secondary source)."""
        # OpenAI doesn't currently expose a public conversations API
        # for consumer ChatGPT. This is a placeholder for when/if they do,
        # or for ChatGPT Team/Enterprise API access.
        return []


# ── message extraction (reused from chatgpt_importer.py) ────

def _extract_messages(conversation: Dict) -> List[Dict]:
    """Extract messages from a ChatGPT conversation mapping."""
    messages = []
    mapping = conversation.get("mapping", {})

    for msg_id, msg_data in mapping.items():
        message = msg_data.get("message")
        if not message:
            continue

        author = message.get("author", {})
        role = author.get("role", "")
        content = message.get("content", {})
        parts = content.get("parts", [])

        if role == "system":
            continue

        text = "\n".join(
            p if isinstance(p, str)
            else p.get("text", "") if isinstance(p, dict)
            else str(p)
            for p in parts
        ) if parts else ""

        if not text or not text.strip():
            continue

        messages.append({
            "role": role,
            "content": text.strip(),
            "create_time": message.get("create_time", 0),
        })

    messages.sort(key=lambda m: m["create_time"])
    return messages


# ── watermark persistence ────────────────────────────────────

class _Watermark:
    """Tracks last-processed state for incremental polling."""

    def __init__(self):
        self.path = cfg.STATE_DIR / "chatgpt.json"
        self.last_export_mtime: float = 0
        self.seen_conversations: Dict[str, float] = {}

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.last_export_mtime = data.get("last_export_mtime", 0)
                self.seen_conversations = data.get("seen_conversations", {})
                log.info(
                    f"Watermark loaded: {len(self.seen_conversations)} conversations tracked"
                )
            except Exception as exc:
                log.warning(f"Watermark load failed: {exc}")

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps({
                "last_export_mtime": self.last_export_mtime,
                "seen_conversations": self.seen_conversations,
                "updated_at": time.time(),
            }, indent=2))
        except Exception as exc:
            log.error(f"Watermark save failed: {exc}")
