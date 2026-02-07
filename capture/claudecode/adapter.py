"""
Claude Code Adapter — Session transcript watcher.

Watches ~/.claude/projects/-Users-dicoangelo/*.jsonl for new/modified
session transcripts. Extracts user + assistant text messages with
rich context (tool use, working directory, git branch, model info).

This is the richest data source — full conversation transcripts with
metadata that no other platform provides.

Stores watermark in ~/.ucw/capture_state/claudecode.json
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..base import CapturedEvent, PlatformAdapter, AdapterStatus
from .. import config as cfg
from .normalizer import ClaudeCodeNormalizer

log = logging.getLogger("capture.claudecode")

# Session transcript directory
TRANSCRIPT_DIR = Path.home() / ".claude" / "projects" / "-Users-dicoangelo"

# Message types to capture (skip progress, file-history-snapshot, queue-operation)
CAPTURE_TYPES = {"user", "assistant"}

# Minimum text length to capture
MIN_TEXT_LENGTH = 20


class ClaudeCodeAdapter(PlatformAdapter):
    """Live capture from Claude Code CLI session transcripts."""

    def __init__(self):
        self._pool = None
        self._transcript_dir = TRANSCRIPT_DIR
        self._normalizer = ClaudeCodeNormalizer()
        self._watermark = _Watermark()
        self._last_poll: float = 0
        self._events_captured: int = 0
        self._healthy = False
        self._error: Optional[str] = None

    @property
    def name(self) -> str:
        return "Claude Code CLI Capture"

    @property
    def platform(self) -> str:
        return "claude-code"

    async def initialize(self, pool) -> bool:
        self._pool = pool
        self._watermark.load()

        if not self._transcript_dir.exists():
            log.warning(f"Claude Code transcript dir not found: {self._transcript_dir}")
            self._error = "Transcript directory not found"
            self._healthy = False
            return False

        session_count = len(list(self._transcript_dir.glob("*.jsonl")))
        log.info(f"Claude Code transcripts found: {session_count} sessions in {self._transcript_dir}")
        self._healthy = True
        return True

    async def poll(self) -> List[CapturedEvent]:
        """Scan for new/modified session transcripts and extract messages."""
        events: List[CapturedEvent] = []

        if not self._transcript_dir.exists():
            return events

        try:
            for jsonl_file in self._transcript_dir.glob("*.jsonl"):
                session_id = jsonl_file.stem
                mtime = jsonl_file.stat().st_mtime

                # Skip if file hasn't changed since last poll
                last_seen_mtime = self._watermark.seen_sessions.get(session_id, {}).get("mtime", 0)
                if mtime <= last_seen_mtime:
                    continue

                # Extract new messages from this session
                last_seen_line = self._watermark.seen_sessions.get(session_id, {}).get("lines", 0)
                session_events, lines_read = self._extract_session(
                    jsonl_file, session_id, skip_lines=last_seen_line,
                )
                events.extend(session_events)

                # Update watermark
                self._watermark.seen_sessions[session_id] = {
                    "mtime": mtime,
                    "lines": last_seen_line + lines_read,
                }

        except Exception as exc:
            log.error(f"Claude Code poll error: {exc}")
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

    # ── transcript extraction ────────────────────────────

    def _extract_session(
        self, jsonl_path: Path, session_id: str, skip_lines: int = 0,
    ) -> tuple:
        """
        Extract user + assistant messages from a JSONL transcript.

        Returns (events, lines_read).
        """
        events: List[CapturedEvent] = []
        lines_read = 0

        try:
            with open(jsonl_path, "r", errors="replace") as f:
                for i, line in enumerate(f):
                    if i < skip_lines:
                        continue

                    lines_read += 1

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_type = record.get("type", "")
                    if msg_type not in CAPTURE_TYPES:
                        continue

                    # Extract text content
                    text, role, metadata = self._parse_record(record, msg_type)
                    if not text or len(text) < MIN_TEXT_LENGTH:
                        continue

                    # Timestamp
                    ts_str = record.get("timestamp", "")
                    if ts_str:
                        try:
                            from datetime import datetime
                            ts = datetime.fromisoformat(
                                ts_str.replace("Z", "+00:00")
                            ).timestamp()
                        except Exception:
                            ts = time.time()
                    else:
                        ts = time.time()

                    metadata.update({
                        "session_id": session_id,
                        "cwd": record.get("cwd", ""),
                        "git_branch": record.get("gitBranch", ""),
                        "version": record.get("version", ""),
                    })

                    events.append(CapturedEvent(
                        platform="claude-code",
                        session_id=f"cc-{session_id[:12]}",
                        content=text,
                        role=role,
                        timestamp=ts,
                        metadata=metadata,
                    ))

        except Exception as exc:
            log.error(f"Error reading {jsonl_path.name}: {exc}")

        return events, lines_read

    def _parse_record(self, record: dict, msg_type: str) -> tuple:
        """
        Parse a JSONL record into (text, role, metadata).

        User messages: message.content is a string
        Assistant messages: message.content is a list of content blocks
        """
        message = record.get("message", {})
        if not message:
            return "", "", {}

        role = message.get("role", msg_type)
        content = message.get("content", "")
        metadata = {}

        if msg_type == "user":
            # User content is a string
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                # Sometimes content is a list of blocks
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                text = "\n".join(parts).strip()
            else:
                text = str(content).strip()

            return text, "user", metadata

        elif msg_type == "assistant":
            # Assistant content is a list of content blocks
            text_parts = []
            tool_names = []

            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "tool_use":
                        tool_names.append(block.get("name", "unknown"))
            elif isinstance(content, str):
                text_parts.append(content)

            text = "\n".join(text_parts).strip()

            if tool_names:
                metadata["tool_uses"] = tool_names

            # Model info
            model = message.get("model", "")
            if model:
                metadata["model"] = model

            usage = message.get("usage", {})
            if usage:
                metadata["input_tokens"] = usage.get("input_tokens", 0)
                metadata["output_tokens"] = usage.get("output_tokens", 0)

            return text, "assistant", metadata

        return "", "", {}


# ── watermark ────────────────────────────────────────────────

class _Watermark:
    """Track scanned sessions and line positions."""

    def __init__(self):
        self.path = cfg.STATE_DIR / "claudecode.json"
        self.seen_sessions: Dict[str, Dict] = {}

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.seen_sessions = data.get("seen_sessions", {})
                log.info(f"Watermark loaded: {len(self.seen_sessions)} sessions tracked")
            except Exception as exc:
                log.warning(f"Watermark load failed: {exc}")

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps({
                "seen_sessions": self.seen_sessions,
                "updated_at": time.time(),
            }, indent=2))
        except Exception as exc:
            log.error(f"Watermark save failed: {exc}")
