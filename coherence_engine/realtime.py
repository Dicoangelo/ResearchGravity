"""
Coherence Engine — Real-time PostgreSQL LISTEN/NOTIFY Listener

Replaces polling-based event detection with push-based notifications.
When a new cognitive_event is inserted, a PostgreSQL trigger fires
pg_notify('new_cognitive_event', ...) and this listener reacts within
milliseconds instead of waiting for a 10-second poll cycle.

Features:
  - asyncpg-based LISTEN on 'new_cognitive_event' channel
  - 500ms micro-batch window to group GPU embedding calls
  - 60-second fallback poll as safety net (catches missed notifications)
  - Automatic reconnection on connection loss

Usage:
    listener = RealtimeListener(daemon)
    await listener.start()
    await listener.run()  # blocks, calls daemon._poll_and_process()
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, TYPE_CHECKING

import asyncpg

from . import config as cfg

if TYPE_CHECKING:
    from .daemon import CoherenceDaemon

log = logging.getLogger("coherence.realtime")

# Channel name must match the PostgreSQL trigger
NOTIFY_CHANNEL = "new_cognitive_event"

# Micro-batch window: collect notifications for this long before processing
MICRO_BATCH_WINDOW_S = float(os.environ.get("UCW_MICROBATCH_WINDOW", "0.5"))

# Fallback poll interval when using realtime mode (safety net)
FALLBACK_POLL_S = int(os.environ.get("UCW_FALLBACK_POLL", "60"))

# Reconnection delay on connection loss
RECONNECT_DELAY_S = 5
MAX_RECONNECT_DELAY_S = 60


class RealtimeListener:
    """
    Async listener for PostgreSQL LISTEN/NOTIFY on cognitive_events inserts.

    Uses an asyncio.Queue to buffer incoming notifications, then drains the
    queue after a 500ms micro-batch window to trigger a single
    daemon._poll_and_process() call that batches GPU embeddings.
    """

    def __init__(self, daemon: "CoherenceDaemon"):
        self._daemon = daemon
        self._conn: Optional[asyncpg.Connection] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._notifications_received = 0
        self._batches_dispatched = 0
        self._last_poll_time: float = 0.0
        self._reconnect_delay = RECONNECT_DELAY_S

    async def start(self):
        """Create a dedicated connection for LISTEN and subscribe."""
        await self._connect()
        self._running = True
        log.info(
            f"Realtime listener started on channel '{NOTIFY_CHANNEL}' "
            f"(micro-batch: {MICRO_BATCH_WINDOW_S}s, "
            f"fallback poll: {FALLBACK_POLL_S}s)"
        )

    async def _connect(self):
        """Establish the LISTEN connection."""
        if self._conn and not self._conn.is_closed():
            try:
                await self._conn.close()
            except Exception:
                pass

        self._conn = await asyncpg.connect(cfg.PG_DSN)
        await self._conn.add_listener(NOTIFY_CHANNEL, self._on_notification)
        self._reconnect_delay = RECONNECT_DELAY_S  # reset on success
        log.info(f"LISTEN connection established on '{NOTIFY_CHANNEL}'")

    def _on_notification(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ):
        """
        Callback fired by asyncpg when a NOTIFY arrives.
        Parses the JSON payload and enqueues the event_id.
        This runs on the event loop, so it must be non-blocking.
        """
        try:
            data = json.loads(payload)
            event_id = data.get("event_id", "")
            platform = data.get("platform", "unknown")
            self._queue.put_nowait(
                {"event_id": event_id, "platform": platform, "received_at": time.time()}
            )
            self._notifications_received += 1
            log.debug(
                f"Notification #{self._notifications_received}: "
                f"event={event_id[:12]}... platform={platform}"
            )
        except (json.JSONDecodeError, Exception) as exc:
            log.warning(f"Failed to parse notification payload: {exc}")

    async def run(self):
        """
        Main loop: wait for notifications or fallback timeout, then process.

        The loop has two triggers:
        1. A notification arrives -> wait 500ms for more -> batch process
        2. No notification for 60s -> fallback poll (safety net)
        """
        self._last_poll_time = time.time()

        try:
            while self._running:
                try:
                    await self._run_loop_iteration()
                except asyncpg.PostgresConnectionError:
                    await self._handle_disconnect()
                except OSError as exc:
                    if "closed" in str(exc).lower() or "connection" in str(exc).lower():
                        await self._handle_disconnect()
                    else:
                        raise
        except asyncio.CancelledError:
            log.info("Realtime listener cancelled")
        finally:
            await self.stop()

    async def _run_loop_iteration(self):
        """Single iteration of the main loop."""
        time_since_last_poll = time.time() - self._last_poll_time
        timeout = max(0.1, FALLBACK_POLL_S - time_since_last_poll)

        try:
            # Wait for first notification (or timeout for fallback poll)
            notification = await asyncio.wait_for(
                self._queue.get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Fallback poll: no notifications received within the window
            log.debug(
                f"Fallback poll triggered ({FALLBACK_POLL_S}s without notifications)"
            )
            await self._dispatch_poll(reason="fallback")
            return

        # Got a notification! Wait for the micro-batch window to collect more
        batch = [notification]
        batch_deadline = time.time() + MICRO_BATCH_WINDOW_S

        while True:
            remaining = batch_deadline - time.time()
            if remaining <= 0:
                break
            try:
                more = await asyncio.wait_for(
                    self._queue.get(), timeout=remaining
                )
                batch.append(more)
            except asyncio.TimeoutError:
                break

        # Drain any remaining items that arrived during processing
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        event_ids = [n["event_id"] for n in batch]
        platforms = set(n["platform"] for n in batch)
        log.info(
            f"Micro-batch ready: {len(batch)} notifications "
            f"from {platforms} -> dispatching poll"
        )

        await self._dispatch_poll(
            reason="realtime",
            event_count=len(batch),
            event_ids=event_ids,
        )

    async def _dispatch_poll(
        self,
        reason: str = "unknown",
        event_count: int = 0,
        event_ids: Optional[list] = None,
    ):
        """Call the daemon's _poll_and_process and update tracking."""
        start = time.time()
        try:
            processed = await self._daemon._poll_and_process()
            elapsed = time.time() - start
            self._last_poll_time = time.time()
            self._batches_dispatched += 1

            if processed > 0:
                log.info(
                    f"[{reason}] Processed {processed} events "
                    f"in {elapsed:.2f}s (batch #{self._batches_dispatched})"
                )
        except Exception as exc:
            log.error(f"Error during poll_and_process: {exc}", exc_info=True)

    async def _handle_disconnect(self):
        """Reconnect after a connection loss with exponential backoff."""
        log.warning(
            f"LISTEN connection lost. "
            f"Reconnecting in {self._reconnect_delay}s..."
        )
        await asyncio.sleep(self._reconnect_delay)

        # Exponential backoff (capped)
        self._reconnect_delay = min(
            self._reconnect_delay * 2, MAX_RECONNECT_DELAY_S
        )

        try:
            await self._connect()
            log.info("LISTEN connection re-established")
        except Exception as exc:
            log.error(f"Reconnection failed: {exc}")
            # Will retry on next loop iteration

    async def stop(self):
        """Clean up the LISTEN connection."""
        self._running = False
        if self._conn and not self._conn.is_closed():
            try:
                await self._conn.remove_listener(
                    NOTIFY_CHANNEL, self._on_notification
                )
            except Exception:
                pass
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None

        log.info(
            f"Realtime listener stopped — "
            f"{self._notifications_received} notifications received, "
            f"{self._batches_dispatched} batches dispatched"
        )

    @property
    def stats(self) -> dict:
        return {
            "notifications_received": self._notifications_received,
            "batches_dispatched": self._batches_dispatched,
            "queue_size": self._queue.qsize(),
            "connected": self._conn is not None and not self._conn.is_closed(),
            "running": self._running,
            "micro_batch_window_s": MICRO_BATCH_WINDOW_S,
            "fallback_poll_s": FALLBACK_POLL_S,
        }
