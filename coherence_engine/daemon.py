"""
Coherence Engine — Background Monitoring Daemon

Monitors the cognitive events table for new events,
embeds them, searches for cross-platform matches,
scores coherence, and fires alerts.

Modes:
  - realtime: LISTEN/NOTIFY with 500ms micro-batch + 60s fallback poll (default)
  - poll:     Check DB every N seconds for new events (legacy)
  - oneshot:  Process all unscored events and exit
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
from .temporal import MultiScaleDetector
from mcp_raw.embeddings import embed_single, embed_texts

import logging

log = logging.getLogger("coherence.daemon")


class CoherenceDaemon:
    """
    Background daemon that monitors cognitive events for coherence.

    Lifecycle:
      Start -> Connect to DB -> Poll for new events -> Process -> Loop
      On new event: embed -> search similar -> score -> alert
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self._pool: Optional[asyncpg.Pool] = pool
        self._owns_pool = pool is None  # only close pool if we created it
        self._similarity: Optional[SimilarityIndex] = None
        self._scorer: Optional[CoherenceScorer] = None
        self._alerts = AlertSystem()
        self._last_event_ns: int = 0
        self._running = False
        self._events_processed = 0
        self._moments_detected = 0
        self._multi_scale = None
        self._realtime_listener = None
        self._mode = "poll"  # "poll" or "realtime"

    async def initialize(self):
        """Initialize components using an existing pool (injected via __init__)."""
        if not self._pool:
            raise RuntimeError("No pool available — use start() or pass pool to __init__")
        self._similarity = SimilarityIndex(self._pool)
        self._scorer = CoherenceScorer(self._pool)
        if cfg.MULTI_SCALE_ENABLED:
            self._multi_scale = MultiScaleDetector(
                self._pool, self._similarity, self._scorer
            )
            log.info("Multi-scale temporal detection enabled (6 windows)")

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

    async def start(self):
        """Initialize database connection and components (creates own pool)."""
        log.info("Coherence daemon starting...")

        if not self._pool:
            self._pool = await asyncpg.create_pool(
                cfg.PG_DSN,
                min_size=cfg.PG_MIN_POOL,
                max_size=cfg.PG_MAX_POOL,
                command_timeout=30,
            )
            self._owns_pool = True

        await self.initialize()

    async def start_realtime(self):
        """Set up the LISTEN/NOTIFY realtime listener."""
        from .realtime import RealtimeListener

        self._realtime_listener = RealtimeListener(self)
        await self._realtime_listener.start()
        self._mode = "realtime"
        log.info("Realtime LISTEN/NOTIFY listener initialized")

    async def run(self, realtime: bool = True):
        """Main daemon loop.

        Args:
            realtime: If True (default), use LISTEN/NOTIFY with 500ms
                      micro-batch and 60s fallback poll.  If False,
                      use legacy polling every POLL_INTERVAL_S seconds.
        """
        await self.start()
        self._running = True

        # Handle signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass

        if realtime:
            try:
                await self.start_realtime()
            except Exception as exc:
                log.warning(
                    f"Failed to start realtime listener: {exc}. "
                    f"Falling back to poll mode."
                )
                realtime = False

        if realtime:
            log.info("Daemon running (realtime mode: LISTEN/NOTIFY + 60s fallback)")
            try:
                await self._realtime_listener.run()
            except asyncio.CancelledError:
                log.info("Daemon cancelled")
            finally:
                await self.stop()
        else:
            self._mode = "poll"
            log.info(f"Daemon running (poll mode: every {cfg.POLL_INTERVAL_S}s)")
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
        if not self._similarity:
            # Not yet initialized — full start
            await self.start()

        log.info("Running one-shot coherence scan...")

        processed = await self._poll_and_process(all_unscored=True)

        log.info(
            f"One-shot complete: {processed} events processed, "
            f"{self._moments_detected} moments detected"
        )

        # Only stop (close pool) if we created it ourselves
        if self._owns_pool:
            await self.stop()
        return processed

    async def _poll_and_process(self, all_unscored: bool = False) -> int:
        """Poll for new events and process them."""
        if not self._pool:
            return 0

        # Get events that haven't been coherence-scanned yet
        limit = 10000 if all_unscored else 100
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT ce.event_id, ce.session_id, ce.timestamp_ns,
                          ce.platform, ce.data_layer, ce.light_layer,
                          ce.instinct_layer, ce.coherence_sig
                   FROM cognitive_events ce
                   JOIN embedding_cache ec ON ec.source_event_id = ce.event_id
                   WHERE ce.coherence_scanned_at IS NULL
                   ORDER BY ce.timestamp_ns ASC
                   LIMIT $1""",
                limit,
            )

        if not rows:
            return 0

        # Parse all events
        events = []
        for row in rows:
            event = dict(row)
            for fld in ("data_layer", "light_layer", "instinct_layer"):
                if isinstance(event.get(fld), str):
                    event[fld] = json.loads(event[fld])
            events.append(event)

        # Filter out noise content before embedding
        noise_prefixes = [p.lower() for p in cfg.NOISE_PREFIXES]
        min_len = cfg.MIN_CONTENT_LENGTH

        # Batch embed all event texts at once (instead of 1-by-1 GPU calls)
        texts = [event_to_text(e) for e in events]
        valid_indices = []
        for i, t in enumerate(texts):
            if not t or len(t) < min_len:
                continue
            t_lower = t[:120].lower()
            if any(t_lower.startswith(p) for p in noise_prefixes):
                continue
            valid_indices.append(i)

        if valid_indices:
            valid_texts = [texts[i] for i in valid_indices]
            batch_embeddings = embed_texts(valid_texts, batch_size=256)
            log.info(f"Batch-embedded {len(valid_texts)} texts")
        else:
            batch_embeddings = []

        # Build embedding lookup
        embedding_map = {}
        for idx, emb in zip(valid_indices, batch_embeddings):
            embedding_map[events[idx]["event_id"]] = emb

        # Process each event through similarity + scoring
        processed = 0
        scanned_ids = []
        for event in events:
            eid = event["event_id"]
            embedding = embedding_map.get(eid)
            if embedding:
                await self._process_event_with_embedding(event, embedding)

            scanned_ids.append(eid)
            processed += 1

            ts = event.get("timestamp_ns", 0)
            if ts > self._last_event_ns:
                self._last_event_ns = ts

        # Mark events as scanned in batch
        if scanned_ids:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """UPDATE cognitive_events
                       SET coherence_scanned_at = NOW()
                       WHERE event_id = ANY($1::text[])""",
                    scanned_ids,
                )

        return processed

    async def _process_event_with_embedding(self, event: dict, embedding: list):
        """Process a single event with a pre-computed embedding."""
        # Multi-scale detection: run similarity + scoring at 6 time scales
        if self._multi_scale:
            detections = await self._multi_scale.detect_multi_scale(event, embedding)

            for moment, window_scale in detections:
                if moment.confidence >= cfg.MIN_ALERT_CONFIDENCE:
                    await self._scorer.store_moment(moment)
                    await self._alerts.notify(moment)
                    self._moments_detected += 1

            self._events_processed += 1
            return

        # Fallback: single-window detection (MULTI_SCALE_ENABLED = False)
        similar = await self._similarity.cross_platform_similar(
            event, embedding, threshold=cfg.SEMANTIC_MEDIUM_THRESHOLD
        )

        if not similar:
            self._events_processed += 1
            return

        moments = await self._scorer.score(event, embedding, similar)

        for moment in moments:
            if moment.confidence >= cfg.MIN_ALERT_CONFIDENCE:
                await self._scorer.store_moment(moment)
                await self._alerts.notify(moment)
                self._moments_detected += 1

        self._events_processed += 1

    async def _process_event(self, event: dict):
        """Process a single event through the full coherence pipeline (legacy)."""
        text = event_to_text(event)
        if not text or len(text) < 10:
            return

        embedding = embed_single(text)
        await embed_event_row(self._pool, event)
        await self._process_event_with_embedding(event, embedding)

    async def stop(self):
        """Graceful shutdown."""
        if not self._running and not self._pool:
            return

        self._running = False

        # Stop the realtime listener if active
        if self._realtime_listener:
            await self._realtime_listener.stop()
            self._realtime_listener = None

        log.info(
            f"Daemon stopping ({self._mode} mode) — "
            f"{self._events_processed} events processed, "
            f"{self._moments_detected} moments detected, "
            f"{self._alerts.alert_count} alerts fired"
        )

        if self._pool and self._owns_pool:
            await self._pool.close()
            self._pool = None

    @property
    def stats(self) -> dict:
        base = {
            "events_processed": self._events_processed,
            "moments_detected": self._moments_detected,
            "alerts_fired": self._alerts.alert_count,
            "last_event_ns": self._last_event_ns,
            "running": self._running,
            "mode": self._mode,
        }
        if self._realtime_listener:
            base["realtime"] = self._realtime_listener.stats
        return base
