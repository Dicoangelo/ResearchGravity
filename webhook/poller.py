"""
Webhook Queue Poller — Pulls pending webhooks from Supabase and forwards to local server.

Architecture:
    GitHub/Slack → Supabase Edge Function → webhook_queue table
                                                ↓ (this poller)
                                          localhost:3848/webhook/{provider}
                                                ↓
                                          cognitive_events (full UCW pipeline)

The poller connects to the Supabase PostgreSQL database (same PG_DSN used by the
capture pipeline) and polls for pending rows every POLL_INTERVAL seconds.
"""

import asyncio
import base64
import json
import logging
import os
import signal
import time
from typing import Optional

import asyncpg
import httpx

from . import config as cfg

log = logging.getLogger("webhook.poller")

# ── Configuration ─────────────────────────────────────────
POLL_INTERVAL = int(os.environ.get("UCW_POLLER_INTERVAL", "30"))
BATCH_SIZE = int(os.environ.get("UCW_POLLER_BATCH", "50"))
MAX_RETRIES = int(os.environ.get("UCW_POLLER_MAX_RETRIES", "5"))
LOCAL_WEBHOOK_URL = f"http://{cfg.WEBHOOK_HOST}:{cfg.WEBHOOK_PORT}"
CLEANUP_DAYS = 7

# Supabase remote DB (for the queue table)
# Falls back to local PG_DSN if not set — allows same DB for dev
SUPABASE_PG_DSN = os.environ.get("UCW_SUPABASE_DATABASE_URL", "")

_running = True


def _signal_handler(sig, frame):
    global _running
    log.info(f"Received signal {sig}, shutting down gracefully...")
    _running = False


async def _get_pool() -> asyncpg.Pool:
    """Connect to the Supabase PostgreSQL database where webhook_queue lives."""
    dsn = SUPABASE_PG_DSN
    if not dsn:
        log.error(
            "UCW_SUPABASE_DATABASE_URL not set. "
            "Set it to your Supabase database connection string."
        )
        raise RuntimeError("UCW_SUPABASE_DATABASE_URL required for poller")

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    log.info("Connected to Supabase database")
    return pool


async def _poll_once(pool: asyncpg.Pool, client: httpx.AsyncClient) -> int:
    """Fetch pending webhooks and forward to local server. Returns count processed."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, provider, headers_json, body, retry_count
               FROM webhook_queue
               WHERE status = 'pending' AND retry_count < $1
               ORDER BY created_at ASC
               LIMIT $2""",
            MAX_RETRIES,
            BATCH_SIZE,
        )

    if not rows:
        return 0

    processed = 0
    for row in rows:
        row_id = row["id"]
        provider = row["provider"]
        headers_json = row["headers_json"]
        body_b64 = row["body"]

        try:
            # Decode the base64 body back to raw bytes
            body_bytes = base64.b64decode(body_b64)
        except Exception as exc:
            log.error(f"Row {row_id}: Failed to decode body: {exc}")
            await _mark_failed(pool, row_id, f"base64 decode error: {exc}")
            continue

        # Reconstruct headers — include original provider headers for HMAC
        if isinstance(headers_json, str):
            forward_headers = json.loads(headers_json)
        elif isinstance(headers_json, dict):
            forward_headers = dict(headers_json)
        else:
            forward_headers = {}
        # Ensure content-type is set
        if "content-type" not in forward_headers:
            forward_headers["content-type"] = "application/json"

        # Forward to local webhook server
        url = f"{LOCAL_WEBHOOK_URL}/webhook/{provider}"
        try:
            resp = await client.post(url, content=body_bytes, headers=forward_headers)

            if resp.status_code == 200:
                await _mark_processed(pool, row_id)
                data = resp.json()
                stored = data.get("stored", data.get("events_stored", 0))
                log.info(
                    f"Row {row_id}: {provider} → stored={stored} "
                    f"(HTTP {resp.status_code})"
                )
                processed += 1
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                await _mark_retry(pool, row_id, error_msg)
                log.warning(f"Row {row_id}: {provider} failed — {error_msg}")

        except httpx.ConnectError:
            await _mark_retry(pool, row_id, "Local webhook server unreachable")
            log.warning(f"Row {row_id}: Local server unreachable at {url}")
        except Exception as exc:
            await _mark_retry(pool, row_id, str(exc)[:500])
            log.error(f"Row {row_id}: Unexpected error: {exc}")

    return processed


async def _mark_processed(pool: asyncpg.Pool, row_id: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE webhook_queue SET status = 'processed', processed_at = NOW() WHERE id = $1",
            row_id,
        )


async def _mark_retry(pool: asyncpg.Pool, row_id: int, error: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE webhook_queue
               SET retry_count = retry_count + 1,
                   error_message = $2,
                   status = CASE
                       WHEN retry_count + 1 >= $3 THEN 'failed'
                       ELSE 'pending'
                   END
               WHERE id = $1""",
            row_id,
            error,
            MAX_RETRIES,
        )


async def _mark_failed(pool: asyncpg.Pool, row_id: int, error: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE webhook_queue SET status = 'failed', error_message = $1 WHERE id = $2",
            error,
            row_id,
        )


async def _cleanup_old(pool: asyncpg.Pool) -> int:
    """Delete processed rows older than CLEANUP_DAYS."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            """DELETE FROM webhook_queue
               WHERE status = 'processed'
               AND processed_at < NOW() - INTERVAL '$1 days'""".replace(
                "$1", str(CLEANUP_DAYS)
            )
        )
    # Result is like "DELETE 42"
    count = int(result.split()[-1]) if result else 0
    if count > 0:
        log.info(f"Cleaned up {count} processed rows older than {CLEANUP_DAYS} days")
    return count


async def poll_loop() -> None:
    """Main polling loop — runs until signaled to stop."""
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    pool = await _get_pool()
    cleanup_counter = 0

    log.info(
        f"Poller started — interval={POLL_INTERVAL}s, batch={BATCH_SIZE}, "
        f"target={LOCAL_WEBHOOK_URL}"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        while _running:
            try:
                count = await _poll_once(pool, client)
                if count > 0:
                    log.info(f"Processed {count} queued webhooks")

                # Cleanup every ~100 polls (~50 min at 30s interval)
                cleanup_counter += 1
                if cleanup_counter >= 100:
                    await _cleanup_old(pool)
                    cleanup_counter = 0

            except asyncpg.PostgresConnectionError as exc:
                log.error(f"Database connection error: {exc}")
                # Reconnect
                try:
                    await pool.close()
                except Exception:
                    pass
                await asyncio.sleep(5)
                pool = await _get_pool()
            except Exception as exc:
                log.error(f"Poll error: {exc}", exc_info=True)

            await asyncio.sleep(POLL_INTERVAL)

    await pool.close()
    log.info("Poller stopped")


async def poll_once_cli() -> None:
    """Single poll for CLI usage."""
    pool = await _get_pool()

    async with httpx.AsyncClient(timeout=30) as client:
        count = await _poll_once(pool, client)
        print(f"Processed {count} queued webhooks")

    await pool.close()


async def queue_status() -> None:
    """Show queue status."""
    pool = await _get_pool()

    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM webhook_queue")
        pending = await conn.fetchval(
            "SELECT COUNT(*) FROM webhook_queue WHERE status = 'pending'"
        )
        processed = await conn.fetchval(
            "SELECT COUNT(*) FROM webhook_queue WHERE status = 'processed'"
        )
        failed = await conn.fetchval(
            "SELECT COUNT(*) FROM webhook_queue WHERE status = 'failed'"
        )
        recent = await conn.fetch(
            """SELECT id, provider, status, retry_count, created_at, processed_at
               FROM webhook_queue ORDER BY created_at DESC LIMIT 10"""
        )

    await pool.close()

    print("=" * 55)
    print("  UCW WEBHOOK QUEUE STATUS")
    print("=" * 55)
    print(f"  Total:      {total}")
    print(f"  Pending:    {pending}")
    print(f"  Processed:  {processed}")
    print(f"  Failed:     {failed}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print(f"  Target:     {LOCAL_WEBHOOK_URL}")
    print()

    if recent:
        print("  Recent queue entries:")
        for r in recent:
            ts = r["created_at"].strftime("%H:%M:%S") if r["created_at"] else "—"
            pt = r["processed_at"].strftime("%H:%M:%S") if r["processed_at"] else "—"
            retries = f"r={r['retry_count']}" if r["retry_count"] > 0 else ""
            print(
                f"    [{ts}] {r['provider']:10s} {r['status']:10s} "
                f"processed={pt} {retries}"
            )

    print("=" * 55)
