"""
Capture Manager — Orchestrates all platform adapters.

Handles:
  - Adapter registration and lifecycle
  - Per-adapter poll scheduling
  - Event dedup, quality scoring, embedding, storage
  - Health monitoring
"""

import asyncio
import json
import logging
import signal
import time
from typing import Dict, List, Optional

import asyncpg

from . import config as cfg
from .base import CapturedEvent, PlatformAdapter, AdapterStatus
from .dedup import DeduplicationEngine
from .normalizer import BaseNormalizer
from .quality import score_event

log = logging.getLogger("capture.manager")


class CaptureManager:
    """
    Orchestrates cross-platform capture across all registered adapters.

    Usage:
        manager = CaptureManager()
        manager.register(ChatGPTAdapter())
        manager.register(CursorAdapter())
        await manager.start()     # continuous polling
        # or
        await manager.poll_all()  # one-shot
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self._pool: Optional[asyncpg.Pool] = pool
        self._owns_pool = pool is None
        self._adapters: Dict[str, PlatformAdapter] = {}
        self._dedup = DeduplicationEngine()
        self._normalizer = BaseNormalizer()
        self._running = False
        self._stats = {
            "events_captured": 0,
            "events_stored": 0,
            "events_skipped_dedup": 0,
            "events_skipped_quality": 0,
            "errors": 0,
        }

    def register(self, adapter: PlatformAdapter) -> None:
        """Register a platform adapter."""
        self._adapters[adapter.platform] = adapter
        log.info(f"Registered adapter: {adapter.name} ({adapter.platform})")

    async def start(self) -> None:
        """Initialize and run continuous poll loops for all adapters."""
        await self._init_pool()
        await self._dedup.initialize(self._pool)

        # Initialize all adapters
        for name, adapter in list(self._adapters.items()):
            try:
                ok = await adapter.initialize(self._pool)
                if not ok:
                    log.warning(f"Adapter {name} failed to initialize, removing")
                    del self._adapters[name]
            except Exception as exc:
                log.error(f"Adapter {name} init error: {exc}")
                del self._adapters[name]

        if not self._adapters:
            log.error("No adapters initialized, exiting")
            return

        self._running = True
        log.info(f"Capture manager started with {len(self._adapters)} adapters")

        # Handle signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                pass

        # Run per-adapter poll loops concurrently
        tasks = []
        for platform, adapter in self._adapters.items():
            interval = self._poll_interval(platform)
            tasks.append(asyncio.create_task(
                self._poll_loop(adapter, interval)
            ))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def poll_all(self) -> Dict[str, int]:
        """One-shot poll of all adapters. Returns events captured per platform."""
        await self._init_pool()
        await self._dedup.initialize(self._pool)

        results = {}
        for name, adapter in self._adapters.items():
            try:
                ok = await adapter.initialize(self._pool)
                if not ok:
                    results[name] = 0
                    continue

                events = await adapter.poll()
                stored = await self._process_events(adapter, events)
                results[name] = stored
            except Exception as exc:
                log.error(f"Poll error for {name}: {exc}")
                results[name] = 0

        if self._owns_pool and self._pool:
            await self._pool.close()
            self._pool = None

        return results

    async def status(self) -> dict:
        """Health check all adapters and return status."""
        statuses = {}
        for name, adapter in self._adapters.items():
            try:
                statuses[name] = await adapter.health_check()
            except Exception as exc:
                statuses[name] = AdapterStatus(
                    healthy=False, last_poll=0, events_captured=0, error=str(exc),
                )
        return {
            "adapters": {
                name: {
                    "healthy": s.healthy,
                    "last_poll": s.last_poll,
                    "events_captured": s.events_captured,
                    "error": s.error,
                }
                for name, s in statuses.items()
            },
            "stats": self._stats,
            "dedup": self._dedup.stats,
        }

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        log.info(f"Capture manager shutting down — {self._stats}")
        if self._pool and self._owns_pool:
            await self._pool.close()
            self._pool = None

    # ── internal ──────────────────────────────────────────

    async def _init_pool(self) -> None:
        if self._pool:
            return
        self._pool = await asyncpg.create_pool(
            cfg.PG_DSN,
            min_size=cfg.PG_MIN_POOL,
            max_size=cfg.PG_MAX_POOL,
            command_timeout=30,
        )
        self._owns_pool = True

    async def _poll_loop(self, adapter: PlatformAdapter, interval: int) -> None:
        """Continuous poll loop for a single adapter."""
        while self._running:
            try:
                events = await adapter.poll()
                if events:
                    stored = await self._process_events(adapter, events)
                    if stored > 0:
                        log.info(
                            f"[{adapter.platform}] Captured {stored} new events "
                            f"({len(events) - stored} skipped)"
                        )
            except Exception as exc:
                log.error(f"[{adapter.platform}] Poll error: {exc}")
                self._stats["errors"] += 1

            await asyncio.sleep(interval)

    async def _process_events(
        self, adapter: PlatformAdapter, events: List[CapturedEvent],
    ) -> int:
        """Process and store captured events. Returns count stored."""
        if not events or not self._pool:
            return 0

        # Normalize via adapter
        rows = await adapter.normalize(events)

        stored = 0
        for event, row in zip(events, rows):
            content_hash = self._normalizer.content_hash(event.content)

            # Dedup check
            if self._dedup.is_duplicate(event.event_id, content_hash, event.session_id, event.platform):
                self._stats["events_skipped_dedup"] += 1
                continue

            # Quality gate (skip garbage)
            qs, mode = score_event(event.content, event.role, event.platform)
            if mode == "garbage":
                self._stats["events_skipped_quality"] += 1
                continue

            row["quality_score"] = qs
            row["cognitive_mode"] = mode

            # Store to database
            try:
                await self._store_event(row)
                self._dedup.mark_seen(event.event_id, content_hash, event.session_id, event.platform)
                stored += 1
                self._stats["events_stored"] += 1
            except Exception as exc:
                log.error(f"Store error for {event.event_id}: {exc}")
                self._stats["errors"] += 1

        self._stats["events_captured"] += len(events)
        return stored

    async def _store_event(self, row: dict) -> None:
        """INSERT a cognitive_event row into PostgreSQL."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO cognitive_events (
                    event_id, session_id, timestamp_ns, direction, stage,
                    method, request_id, parent_event_id, turn,
                    raw_bytes, parsed_json, content_length, error,
                    data_layer, light_layer, instinct_layer,
                    coherence_sig, platform, protocol,
                    quality_score, cognitive_mode
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18, $19,
                    $20, $21
                )
                ON CONFLICT (event_id) DO NOTHING""",
                row["event_id"],
                row["session_id"],
                row["timestamp_ns"],
                row["direction"],
                row["stage"],
                row["method"],
                row.get("request_id"),
                row.get("parent_event_id"),
                row.get("turn", 0),
                row.get("raw_bytes"),
                row.get("parsed_json"),
                row.get("content_length", 0),
                row.get("error"),
                row.get("data_layer"),
                row.get("light_layer"),
                row.get("instinct_layer"),
                row.get("coherence_sig"),
                row["platform"],
                row.get("protocol", "external"),
                row.get("quality_score"),
                row.get("cognitive_mode"),
            )

    def _poll_interval(self, platform: str) -> int:
        return {
            "chatgpt": cfg.CHATGPT_POLL_INTERVAL_S,
            "cursor": cfg.CURSOR_POLL_INTERVAL_S,
            "grok": cfg.GROK_POLL_INTERVAL_S,
        }.get(platform, 300)
