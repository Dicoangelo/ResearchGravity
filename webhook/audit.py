"""
Audit Trail — Track all incoming webhook deliveries for debugging and replay.

Stores in webhook_events table (created on first use).
"""

import logging
import time
from typing import Optional

log = logging.getLogger("webhook.audit")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS webhook_events (
    id                  BIGSERIAL PRIMARY KEY,
    provider            TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    delivery_id         TEXT,
    signature_valid     BOOLEAN NOT NULL,
    status              TEXT NOT NULL,
    events_parsed       INTEGER DEFAULT 0,
    events_stored       INTEGER DEFAULT 0,
    error_message       TEXT,
    processing_time_ms  INTEGER,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_we_provider ON webhook_events (provider);
CREATE INDEX IF NOT EXISTS idx_we_status   ON webhook_events (status);
CREATE INDEX IF NOT EXISTS idx_we_created  ON webhook_events (created_at);
"""


class AuditTrail:
    """Webhook audit trail backed by PostgreSQL."""

    def __init__(self, pool):
        self._pool = pool

    async def ensure_table(self) -> None:
        """Create webhook_events table if it doesn't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.execute(_CREATE_INDEXES_SQL)
        log.info("Audit table ready")

    async def log_received(
        self,
        provider: str,
        event_type: str,
        delivery_id: Optional[str],
        signature_valid: bool,
        events_parsed: int,
        events_stored: int,
        processing_time_ms: int,
        error_message: Optional[str] = None,
    ) -> None:
        """Log a webhook delivery to the audit trail."""
        status = "error" if error_message else ("stored" if events_stored > 0 else "received")
        if not signature_valid:
            status = "rejected"

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO webhook_events (
                        provider, event_type, delivery_id, signature_valid,
                        status, events_parsed, events_stored,
                        error_message, processing_time_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                    provider, event_type, delivery_id, signature_valid,
                    status, events_parsed, events_stored,
                    error_message, processing_time_ms,
                )
        except Exception as exc:
            log.error(f"Audit log failed: {exc}")

    async def recent(self, limit: int = 20, provider: Optional[str] = None) -> list:
        """Get recent audit entries."""
        async with self._pool.acquire() as conn:
            if provider:
                rows = await conn.fetch(
                    """SELECT * FROM webhook_events
                       WHERE provider = $1
                       ORDER BY created_at DESC LIMIT $2""",
                    provider, limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM webhook_events
                       ORDER BY created_at DESC LIMIT $1""",
                    limit,
                )
        return [dict(r) for r in rows]

    async def stats(self) -> dict:
        """Get aggregate stats from the audit trail."""
        async with self._pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM webhook_events")
            by_provider = await conn.fetch(
                """SELECT provider, status, COUNT(*) as cnt
                   FROM webhook_events
                   GROUP BY provider, status
                   ORDER BY provider"""
            )
        return {
            "total_deliveries": total,
            "by_provider": [dict(r) for r in by_provider],
        }
