"""
Coherence Engine — Background Monitoring Daemon

Polls the cognitive events table for new events,
embeds them, searches for cross-platform matches,
scores coherence, and fires alerts.

Modes:
  - poll:   Check DB every N seconds for new events
  - oneshot: Process all unscored events and exit
"""

import asyncio
import json
import os
import signal
import time
from typing import Optional

import asyncpg

from . import config as cfg
from .embeddings import embed_event_row, event_to_text
from .similarity import SimilarityIndex
from .scorer import CoherenceScorer
from .alerts import AlertSystem
from mcp_raw.embeddings import embed_single

import logging

log = logging.getLogger("coherence.daemon")


class CoherenceDaemon:
    """
    Background daemon that monitors cognitive events for coherence.

    Lifecycle:
      Start -> Connect to DB -> Poll for new events -> Process -> Loop
      On new event: embed -> search similar -> score -> alert
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._similarity: Optional[SimilarityIndex] = None
        self._scorer: Optional[CoherenceScorer] = None
        self._alerts = AlertSystem()
        self._last_event_ns: int = 0
        self._running = False
        self._events_processed = 0
        self._moments_detected = 0

    async def start(self):
        """Initialize database connection and components."""
        log.info("Coherence daemon starting...")

        self._pool = await asyncpg.create_pool(
            cfg.PG_DSN,
            min_size=cfg.PG_MIN_POOL,
            max_size=cfg.PG_MAX_POOL,
            command_timeout=30,
        )

        self._similarity = SimilarityIndex(self._pool)
        self._scorer = CoherenceScorer(self._pool)

        # Find the latest event timestamp we've already processed
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(timestamp_ns) as max_ts FROM cognitive_events"
            )
            if row and row["max_ts"]:
                self._last_event_ns = row["max_ts"]

        log.info(
            f"Daemon ready — last event at {self._last_event_ns}, "
            f"polling every {cfg.POLL_INTERVAL_S}s"
        )

    async def run(self):
        """Main daemon loop: poll -> process -> sleep -> repeat."""
        await self.start()
        self._running = True

        # Handle signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass

        log.info("Daemon running (poll mode)")

        try:
            while self._running:
                processed = await self._poll_and_process()
                if processed > 0:
                    log.info(
                        f"Processed {processed} events | "
                        f"Total: {self._events_processed} events, "
                        f"{self._moments_detected} moments, "
                        f"{self._alerts.alert_count} alerts"
                    )
                await asyncio.sleep(cfg.POLL_INTERVAL_S)
        except asyncio.CancelledError:
            log.info("Daemon cancelled")
        finally:
            await self.stop()

    async def oneshot(self):
        """Process all unscored events and exit."""
        await self.start()
        log.info("Running one-shot coherence scan...")

        processed = await self._poll_and_process(all_unscored=True)

        log.info(
            f"One-shot complete: {processed} events processed, "
            f"{self._moments_detected} moments detected"
        )

        await self.stop()
        return self._moments_detected

    async def _poll_and_process(self, all_unscored: bool = False) -> int:
        """Poll for new events and process them."""
        if not self._pool:
            return 0

        # Get new events since last check
        async with self._pool.acquire() as conn:
            if all_unscored:
                # Get events that have embeddings but haven't been coherence-scored
                rows = await conn.fetch(
                    """SELECT ce.event_id, ce.session_id, ce.timestamp_ns,
                              ce.platform, ce.data_layer, ce.light_layer,
                              ce.instinct_layer, ce.coherence_sig
                       FROM cognitive_events ce
                       JOIN embedding_cache ec ON ec.source_event_id = ce.event_id
                       ORDER BY ce.timestamp_ns ASC
                       LIMIT 10000""",
                )
            else:
                rows = await conn.fetch(
                    """SELECT ce.event_id, ce.session_id, ce.timestamp_ns,
                              ce.platform, ce.data_layer, ce.light_layer,
                              ce.instinct_layer, ce.coherence_sig
                       FROM cognitive_events ce
                       WHERE ce.timestamp_ns > $1
                       ORDER BY ce.timestamp_ns ASC
                       LIMIT 100""",
                    self._last_event_ns,
                )

        if not rows:
            return 0

        processed = 0
        for row in rows:
            event = dict(row)
            # Parse JSON fields if needed
            for field in ("data_layer", "light_layer", "instinct_layer"):
                if isinstance(event.get(field), str):
                    event[field] = json.loads(event[field])

            await self._process_event(event)
            processed += 1

            ts = event.get("timestamp_ns", 0)
            if ts > self._last_event_ns:
                self._last_event_ns = ts

        return processed

    async def _process_event(self, event: dict):
        """Process a single event through the full coherence pipeline."""
        event_id = event["event_id"]

        # 1. Get or compute embedding
        text = event_to_text(event)
        if not text or len(text) < 10:
            return

        embedding = embed_single(text)

        # 2. Embed and store if not already cached
        await embed_event_row(self._pool, event)

        # 3. Find cross-platform similar events
        similar = await self._similarity.cross_platform_similar(
            event, embedding, threshold=cfg.SEMANTIC_MEDIUM_THRESHOLD
        )

        if not similar:
            self._events_processed += 1
            return

        # 4. Score coherence
        moments = await self._scorer.score(event, embedding, similar)

        # 5. Store and alert on significant moments
        for moment in moments:
            if moment.confidence >= cfg.MIN_ALERT_CONFIDENCE:
                await self._scorer.store_moment(moment)
                await self._alerts.notify(moment)
                self._moments_detected += 1

        self._events_processed += 1

    async def stop(self):
        """Graceful shutdown."""
        if not self._running and not self._pool:
            return

        self._running = False
        log.info(
            f"Daemon stopping — {self._events_processed} events processed, "
            f"{self._moments_detected} moments detected, "
            f"{self._alerts.alert_count} alerts fired"
        )

        if self._pool:
            await self._pool.close()
            self._pool = None

    @property
    def stats(self) -> dict:
        return {
            "events_processed": self._events_processed,
            "moments_detected": self._moments_detected,
            "alerts_fired": self._alerts.alert_count,
            "last_event_ns": self._last_event_ns,
            "running": self._running,
        }
