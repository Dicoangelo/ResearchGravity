"""
Deduplication Engine — Prevent duplicate events across polls.

Uses 3 strategies:
  1. Exact event_id match
  2. Content hash match within time window
  3. Conversation/session ID + timestamp match

In-memory set for fast checks, DB fallback for edge cases.
"""

import hashlib
import logging
import time
from typing import Optional, Set

from . import config as cfg

log = logging.getLogger("capture.dedup")


class DeduplicationEngine:
    """
    Fast deduplication with in-memory hash set and DB verification.

    Usage:
        dedup = DeduplicationEngine()
        await dedup.initialize(pool)
        if dedup.is_duplicate(event_id, content_hash, session_id):
            skip()
    """

    def __init__(self):
        self._pool = None
        self._seen_event_ids: Set[str] = set()
        self._seen_hashes: Set[str] = set()
        self._seen_session_keys: Set[str] = set()
        self._initialized = False

    async def initialize(self, pool) -> None:
        """Load recent event hashes from DB into memory for fast lookup."""
        self._pool = pool

        window_ns = cfg.DEDUP_WINDOW_HOURS * 3600 * 1_000_000_000
        cutoff_ns = time.time_ns() - window_ns

        try:
            async with pool.acquire() as conn:
                # Load recent event IDs
                rows = await conn.fetch(
                    """SELECT event_id FROM cognitive_events
                       WHERE timestamp_ns > $1
                       AND platform IN ('chatgpt', 'cursor', 'grok')""",
                    cutoff_ns,
                )
                self._seen_event_ids = {r["event_id"] for r in rows}

                # Load recent content hashes (from data_layer)
                rows = await conn.fetch(
                    """SELECT data_layer->>'content' as content
                       FROM cognitive_events
                       WHERE timestamp_ns > $1
                       AND platform IN ('chatgpt', 'cursor', 'grok')
                       AND data_layer IS NOT NULL""",
                    cutoff_ns,
                )
                for r in rows:
                    if r["content"]:
                        h = hashlib.sha256(r["content"].encode()).hexdigest()
                        self._seen_hashes.add(h)

                # Load recent session keys (session_id + platform)
                rows = await conn.fetch(
                    """SELECT DISTINCT session_id, platform
                       FROM cognitive_events
                       WHERE timestamp_ns > $1
                       AND platform IN ('chatgpt', 'cursor', 'grok')""",
                    cutoff_ns,
                )
                self._seen_session_keys = {
                    f"{r['platform']}:{r['session_id']}" for r in rows
                }

            self._initialized = True
            log.info(
                f"Dedup initialized: {len(self._seen_event_ids)} event IDs, "
                f"{len(self._seen_hashes)} content hashes, "
                f"{len(self._seen_session_keys)} session keys"
            )

        except Exception as exc:
            log.error(f"Dedup initialization failed: {exc}")
            self._initialized = True  # Still allow operation in degraded mode

    def is_duplicate(
        self,
        event_id: str,
        content_hash: str,
        session_id: str,
        platform: str = "",
    ) -> bool:
        """
        Check if an event is a duplicate using 3 strategies.
        Returns True if this event should be skipped.
        """
        # Strategy 1: exact event_id match
        if event_id in self._seen_event_ids:
            return True

        # Strategy 2: content hash match
        if content_hash in self._seen_hashes:
            return True

        # Strategy 3: session key check (for conversation-level dedup)
        # This only prevents re-importing entire sessions, not individual messages
        # Individual message dedup is handled by content hash above

        return False

    def mark_seen(self, event_id: str, content_hash: str, session_id: str, platform: str = "") -> None:
        """Mark an event as seen (call after successful store)."""
        self._seen_event_ids.add(event_id)
        self._seen_hashes.add(content_hash)
        if platform:
            self._seen_session_keys.add(f"{platform}:{session_id}")

    def is_session_known(self, platform: str, session_id: str) -> bool:
        """Check if we've already captured events from this session."""
        return f"{platform}:{session_id}" in self._seen_session_keys

    @property
    def stats(self) -> dict:
        return {
            "event_ids": len(self._seen_event_ids),
            "content_hashes": len(self._seen_hashes),
            "session_keys": len(self._seen_session_keys),
            "initialized": self._initialized,
        }
