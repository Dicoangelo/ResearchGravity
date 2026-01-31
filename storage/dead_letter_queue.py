"""
Dead-Letter Queue for Failed Storage Operations

Provides reliable storage for failed write operations with:
1. Persistent storage in SQLite
2. Automatic retry with exponential backoff
3. Metrics tracking for monitoring
4. Configurable retention policies

Usage:
    dlq = DeadLetterQueue(db_path)
    await dlq.initialize()

    # Add failed operation
    await dlq.add_failed_write(
        operation="upsert_finding",
        target="qdrant",
        payload={"finding_id": "...", "content": "..."},
        error="Connection refused"
    )

    # Retry failed operations
    results = await dlq.retry_failed_writes(max_retries=3)

    # Get DLQ stats
    stats = await dlq.get_stats()
"""

import asyncio
import aiosqlite
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Awaitable
from enum import Enum

logger = logging.getLogger("researchgravity.dlq")


class DLQStatus(Enum):
    """Status of a dead-letter queue entry."""
    PENDING = "pending"      # Awaiting retry
    RETRYING = "retrying"    # Currently being retried
    SUCCEEDED = "succeeded"  # Retry succeeded
    FAILED = "failed"        # All retries exhausted
    EXPIRED = "expired"      # Retention period passed


@dataclass
class DLQEntry:
    """A dead-letter queue entry."""
    id: int
    operation: str           # e.g., "upsert_finding", "upsert_session"
    target: str              # e.g., "qdrant", "sqlite_vec"
    payload: Dict[str, Any]  # The data that failed to write
    error: str               # Error message
    status: str              # DLQStatus value
    retry_count: int
    max_retries: int
    created_at: str
    last_retry_at: Optional[str]
    next_retry_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeadLetterQueue:
    """
    Dead-letter queue for failed storage operations.

    Features:
    - Persistent SQLite storage
    - Exponential backoff retry
    - Metrics and monitoring
    - Automatic cleanup of old entries
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_retries: int = 5,
        base_delay_seconds: int = 60,
        retention_days: int = 7
    ):
        """
        Initialize the dead-letter queue.

        Args:
            db_path: Path to SQLite database (default: ~/.agent-core/storage/dlq.db)
            max_retries: Maximum retry attempts per entry
            base_delay_seconds: Base delay for exponential backoff
            retention_days: How long to keep entries
        """
        self.db_path = db_path or Path.home() / ".agent-core" / "storage" / "dlq.db"
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.retention_days = retention_days
        self._initialized = False
        self._retry_handlers: Dict[str, Callable[..., Awaitable[bool]]] = {}

    async def initialize(self):
        """Initialize the DLQ database."""
        if self._initialized:
            return

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dead_letter_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT NOT NULL,
                    target TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    error TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_retry_at TEXT,
                    next_retry_at TEXT
                )
            """)

            # Create indexes
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_dlq_status
                ON dead_letter_queue(status)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_dlq_next_retry
                ON dead_letter_queue(next_retry_at)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_dlq_target
                ON dead_letter_queue(target, operation)
            """)

            await db.commit()

        self._initialized = True
        logger.info(f"DLQ initialized at {self.db_path}")

    def register_retry_handler(
        self,
        operation: str,
        target: str,
        handler: Callable[..., Awaitable[bool]]
    ):
        """
        Register a handler function for retrying a specific operation.

        Args:
            operation: Operation name (e.g., "upsert_finding")
            target: Target backend (e.g., "qdrant")
            handler: Async function that takes payload and returns True on success
        """
        key = f"{target}:{operation}"
        self._retry_handlers[key] = handler
        logger.debug(f"Registered retry handler for {key}")

    async def add_failed_write(
        self,
        operation: str,
        target: str,
        payload: Dict[str, Any],
        error: str,
        max_retries: Optional[int] = None
    ) -> int:
        """
        Add a failed write operation to the dead-letter queue.

        Args:
            operation: Operation name (e.g., "upsert_finding")
            target: Target backend (e.g., "qdrant", "sqlite_vec")
            payload: The data that failed to write
            error: Error message from the failure
            max_retries: Override default max retries

        Returns:
            Entry ID
        """
        if not self._initialized:
            await self.initialize()

        now = datetime.utcnow()
        next_retry = now + timedelta(seconds=self.base_delay_seconds)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO dead_letter_queue
                (operation, target, payload, error, status, retry_count,
                 max_retries, created_at, next_retry_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operation,
                target,
                json.dumps(payload),
                error,
                DLQStatus.PENDING.value,
                0,
                max_retries or self.max_retries,
                now.isoformat(),
                next_retry.isoformat()
            ))
            await db.commit()
            entry_id = cursor.lastrowid

        logger.warning(
            f"Added to DLQ: {operation}@{target} (id={entry_id}): {error[:100]}"
        )
        return entry_id

    async def get_pending_entries(
        self,
        limit: int = 100,
        target: Optional[str] = None
    ) -> List[DLQEntry]:
        """Get entries ready for retry."""
        if not self._initialized:
            await self.initialize()

        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if target:
                rows = await db.execute_fetchall("""
                    SELECT * FROM dead_letter_queue
                    WHERE status = ? AND target = ? AND next_retry_at <= ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (DLQStatus.PENDING.value, target, now, limit))
            else:
                rows = await db.execute_fetchall("""
                    SELECT * FROM dead_letter_queue
                    WHERE status = ? AND next_retry_at <= ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (DLQStatus.PENDING.value, now, limit))

        return [
            DLQEntry(
                id=row["id"],
                operation=row["operation"],
                target=row["target"],
                payload=json.loads(row["payload"]),
                error=row["error"],
                status=row["status"],
                retry_count=row["retry_count"],
                max_retries=row["max_retries"],
                created_at=row["created_at"],
                last_retry_at=row["last_retry_at"],
                next_retry_at=row["next_retry_at"]
            )
            for row in rows
        ]

    async def retry_entry(self, entry: DLQEntry) -> bool:
        """
        Retry a single DLQ entry.

        Returns True if retry succeeded.
        """
        handler_key = f"{entry.target}:{entry.operation}"
        handler = self._retry_handlers.get(handler_key)

        if not handler:
            logger.error(f"No retry handler for {handler_key}")
            return False

        now = datetime.utcnow()

        # Mark as retrying
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE dead_letter_queue
                SET status = ?, last_retry_at = ?, retry_count = retry_count + 1
                WHERE id = ?
            """, (DLQStatus.RETRYING.value, now.isoformat(), entry.id))
            await db.commit()

        try:
            success = await handler(entry.payload)

            if success:
                # Mark as succeeded
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE dead_letter_queue
                        SET status = ?
                        WHERE id = ?
                    """, (DLQStatus.SUCCEEDED.value, entry.id))
                    await db.commit()
                logger.info(f"DLQ entry {entry.id} succeeded on retry")
                return True
            else:
                raise Exception("Handler returned False")

        except Exception as e:
            new_retry_count = entry.retry_count + 1
            error_msg = str(e)

            if new_retry_count >= entry.max_retries:
                # Mark as permanently failed
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE dead_letter_queue
                        SET status = ?, error = ?
                        WHERE id = ?
                    """, (DLQStatus.FAILED.value, error_msg, entry.id))
                    await db.commit()
                logger.error(f"DLQ entry {entry.id} permanently failed: {error_msg}")
            else:
                # Calculate next retry with exponential backoff
                delay = self.base_delay_seconds * (2 ** new_retry_count)
                next_retry = now + timedelta(seconds=delay)

                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE dead_letter_queue
                        SET status = ?, error = ?, next_retry_at = ?
                        WHERE id = ?
                    """, (
                        DLQStatus.PENDING.value,
                        error_msg,
                        next_retry.isoformat(),
                        entry.id
                    ))
                    await db.commit()
                logger.warning(
                    f"DLQ entry {entry.id} retry failed, next attempt at {next_retry}"
                )

            return False

    async def retry_failed_writes(
        self,
        target: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, int]:
        """
        Retry pending entries.

        Args:
            target: Only retry entries for this target
            limit: Maximum entries to retry

        Returns:
            Dict with counts: {"attempted": N, "succeeded": M, "failed": K}
        """
        entries = await self.get_pending_entries(limit=limit, target=target)

        results = {"attempted": 0, "succeeded": 0, "failed": 0}

        for entry in entries:
            results["attempted"] += 1
            if await self.retry_entry(entry):
                results["succeeded"] += 1
            else:
                results["failed"] += 1

        logger.info(
            f"DLQ retry complete: {results['succeeded']}/{results['attempted']} succeeded"
        )
        return results

    async def cleanup_old_entries(self) -> int:
        """Remove entries older than retention period."""
        if not self._initialized:
            await self.initialize()

        cutoff = (datetime.utcnow() - timedelta(days=self.retention_days)).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Mark old entries as expired
            cursor = await db.execute("""
                UPDATE dead_letter_queue
                SET status = ?
                WHERE created_at < ? AND status IN (?, ?)
            """, (
                DLQStatus.EXPIRED.value,
                cutoff,
                DLQStatus.SUCCEEDED.value,
                DLQStatus.FAILED.value
            ))
            await db.commit()
            expired_count = cursor.rowcount

            # Delete very old expired entries (2x retention)
            very_old = (
                datetime.utcnow() - timedelta(days=self.retention_days * 2)
            ).isoformat()
            cursor = await db.execute("""
                DELETE FROM dead_letter_queue
                WHERE created_at < ? AND status = ?
            """, (very_old, DLQStatus.EXPIRED.value))
            await db.commit()
            deleted_count = cursor.rowcount

        logger.info(f"DLQ cleanup: {expired_count} expired, {deleted_count} deleted")
        return expired_count + deleted_count

    async def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Count by status
            rows = await db.execute_fetchall("""
                SELECT status, COUNT(*) as count
                FROM dead_letter_queue
                GROUP BY status
            """)
            status_counts = {row[0]: row[1] for row in rows}

            # Count by target
            rows = await db.execute_fetchall("""
                SELECT target, COUNT(*) as count
                FROM dead_letter_queue
                WHERE status = 'pending'
                GROUP BY target
            """)
            pending_by_target = {row[0]: row[1] for row in rows}

            # Recent failures (last 24h)
            cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            rows = await db.execute_fetchall("""
                SELECT operation, target, COUNT(*) as count
                FROM dead_letter_queue
                WHERE created_at > ?
                GROUP BY operation, target
                ORDER BY count DESC
                LIMIT 10
            """, (cutoff,))
            recent_failures = [
                {"operation": r[0], "target": r[1], "count": r[2]}
                for r in rows
            ]

        return {
            "status_counts": status_counts,
            "pending_by_target": pending_by_target,
            "recent_failures": recent_failures,
            "total_pending": status_counts.get(DLQStatus.PENDING.value, 0),
            "total_failed": status_counts.get(DLQStatus.FAILED.value, 0),
        }


# Singleton instance
_dlq_instance: Optional[DeadLetterQueue] = None


async def get_dlq() -> DeadLetterQueue:
    """Get the singleton DLQ instance."""
    global _dlq_instance
    if _dlq_instance is None:
        _dlq_instance = DeadLetterQueue()
        await _dlq_instance.initialize()
    return _dlq_instance
