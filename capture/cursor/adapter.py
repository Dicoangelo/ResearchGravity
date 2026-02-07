"""
Cursor Adapter — File watcher on Cursor workspace data.

Watches Cursor's workspace storage for AI chat interactions:
  macOS: ~/Library/Application Support/Cursor/User/workspaceStorage/
  Linux: ~/.config/Cursor/User/workspaceStorage/

Falls back to watching .cursor/ directories in active projects.

Stores watermark in ~/.ucw/capture_state/cursor.json
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..base import CapturedEvent, PlatformAdapter, AdapterStatus
from .. import config as cfg
from .normalizer import CursorNormalizer

log = logging.getLogger("capture.cursor")


class CursorAdapter(PlatformAdapter):
    """Live capture from Cursor IDE AI interactions."""

    def __init__(self):
        self._pool = None
        self._data_dir = Path(cfg.CURSOR_DATA_DIR)
        self._normalizer = CursorNormalizer()
        self._watermark = _Watermark()
        self._last_poll: float = 0
        self._events_captured: int = 0
        self._healthy = False
        self._error: Optional[str] = None

    @property
    def name(self) -> str:
        return "Cursor IDE Capture"

    @property
    def platform(self) -> str:
        return "cursor"

    async def initialize(self, pool) -> bool:
        self._pool = pool
        self._watermark.load()

        if not self._data_dir.exists():
            log.info(f"Cursor data dir not found: {self._data_dir}")
            self._error = "Cursor data directory not found"
            self._healthy = False
            return False

        log.info(f"Cursor data dir found: {self._data_dir}")
        self._healthy = True
        return True

    async def poll(self) -> List[CapturedEvent]:
        """Scan Cursor workspace storage for new AI chat data."""
        events: List[CapturedEvent] = []

        if not self._data_dir.exists():
            return events

        try:
            # Scan workspace storage directories for AI chat state
            for workspace_dir in self._data_dir.iterdir():
                if not workspace_dir.is_dir():
                    continue

                # Look for Cursor AI chat state files
                chat_events = self._scan_workspace(workspace_dir)
                events.extend(chat_events)

            # Also scan .cursor directories in common project locations
            home = Path.home()
            for project_dir in [home / "OS-App", home / "researchgravity"]:
                cursor_dir = project_dir / ".cursor"
                if cursor_dir.exists():
                    project_events = self._scan_cursor_dir(cursor_dir, project_dir.name)
                    events.extend(project_events)

        except Exception as exc:
            log.error(f"Cursor poll error: {exc}")
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

    # ── workspace scanning ───────────────────────────────

    def _scan_workspace(self, workspace_dir: Path) -> List[CapturedEvent]:
        """Scan a single workspace storage directory for AI chat data."""
        events: List[CapturedEvent] = []

        # Cursor stores AI chat history in various state.vscdb files
        # and chat-related JSON files
        for state_file in workspace_dir.rglob("*.json"):
            try:
                mtime = state_file.stat().st_mtime
                file_key = str(state_file)

                if file_key in self._watermark.seen_files:
                    if mtime <= self._watermark.seen_files[file_key]:
                        continue

                content = state_file.read_text(errors="replace")
                if len(content) < 50:
                    continue

                # Try to parse as chat data
                data = json.loads(content)
                file_events = self._extract_chat_events(data, workspace_dir.name)
                events.extend(file_events)

                self._watermark.seen_files[file_key] = mtime

            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception as exc:
                log.debug(f"Skip {state_file}: {exc}")
                continue

        return events

    def _scan_cursor_dir(self, cursor_dir: Path, project_name: str) -> List[CapturedEvent]:
        """Scan a .cursor directory in a project for AI interactions."""
        events: List[CapturedEvent] = []

        for chat_file in cursor_dir.rglob("*.json"):
            try:
                mtime = chat_file.stat().st_mtime
                file_key = str(chat_file)

                if file_key in self._watermark.seen_files:
                    if mtime <= self._watermark.seen_files[file_key]:
                        continue

                content = chat_file.read_text(errors="replace")
                if len(content) < 50:
                    continue

                data = json.loads(content)
                file_events = self._extract_chat_events(data, project_name)
                events.extend(file_events)

                self._watermark.seen_files[file_key] = mtime

            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception:
                continue

        return events

    def _extract_chat_events(
        self, data: dict, workspace_name: str,
    ) -> List[CapturedEvent]:
        """Extract AI chat events from Cursor state data."""
        events: List[CapturedEvent] = []

        # Cursor stores chat in various formats. Handle the common ones:
        # 1. Direct messages array
        messages = data.get("messages", data.get("chatMessages", []))
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                content = msg.get("content", msg.get("text", msg.get("message", "")))
                if not content or not isinstance(content, str) or len(content.strip()) < 10:
                    continue

                role = msg.get("role", msg.get("type", "assistant"))
                if role in ("system",):
                    continue

                ts = msg.get("timestamp", msg.get("createdAt", time.time()))
                if isinstance(ts, str):
                    try:
                        from datetime import datetime
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        ts = time.time()

                session_id = f"cursor-{workspace_name}"

                events.append(CapturedEvent(
                    platform="cursor",
                    session_id=session_id,
                    content=content.strip(),
                    role="user" if role in ("user", "human") else "assistant",
                    timestamp=ts,
                    metadata={
                        "workspace": workspace_name,
                        "file_context": msg.get("file", msg.get("filePath", "")),
                    },
                ))

        return events


# ── watermark ────────────────────────────────────────────────

class _Watermark:
    """Track scanned files and their modification times."""

    def __init__(self):
        self.path = cfg.STATE_DIR / "cursor.json"
        self.seen_files: Dict[str, float] = {}

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.seen_files = data.get("seen_files", {})
                log.info(f"Watermark loaded: {len(self.seen_files)} files tracked")
            except Exception as exc:
                log.warning(f"Watermark load failed: {exc}")

    def save(self) -> None:
        try:
            # Prune old entries (keep last 1000)
            if len(self.seen_files) > 1000:
                sorted_files = sorted(self.seen_files.items(), key=lambda x: x[1], reverse=True)
                self.seen_files = dict(sorted_files[:1000])

            self.path.write_text(json.dumps({
                "seen_files": self.seen_files,
                "updated_at": time.time(),
            }, indent=2))
        except Exception as exc:
            log.error(f"Watermark save failed: {exc}")
