"""
PostgreSQL Cognitive Database — Production-grade persistent storage

Uses asyncpg for async PostgreSQL access with pgvector for embeddings.
Falls back gracefully to SQLite (db.py) if PostgreSQL is unavailable.

Tables (7 from PRD):
  cognitive_events      — Every MCP message with UCW layers
  cognitive_sessions    — Work sessions across platforms
  coherence_moments     — Detected cross-platform alignment
  coherence_links       — Event-to-event coherence relationships
  embedding_cache       — Semantic embeddings for similarity
  cognitive_signatures  — Unique patterns for coherence detection
  supermemory_entries   — Long-term memory with spaced repetition
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from .config import Config
from .logger import get_logger

log = get_logger("database")

# Try to import asyncpg (optional dependency)
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    log.warning("asyncpg not installed — PostgreSQL disabled, using SQLite fallback")


# Connection config from environment
PG_DSN = os.environ.get(
    "UCW_DATABASE_URL",
    "postgresql://localhost:5432/ucw_cognitive"
)
PG_MIN_POOL = int(os.environ.get("UCW_PG_MIN_POOL", "2"))
PG_MAX_POOL = int(os.environ.get("UCW_PG_MAX_POOL", "10"))


SCHEMA_VERSION = 1


class CognitiveDatabase:
    """
    PostgreSQL database with connection pooling and pgvector.

    Usage:
        db = CognitiveDatabase()
        connected = await db.initialize()
        if not connected:
            # fall back to SQLite (db.py)
            ...
    """

    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or PG_DSN
        self._pool: Optional[asyncpg.Pool] = None
        self._session_id: Optional[str] = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    async def initialize(self) -> bool:
        """
        Connect to PostgreSQL and ensure schema exists.
        Returns True if connected, False if unavailable.
        """
        if not HAS_ASYNCPG:
            return False

        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=PG_MIN_POOL,
                max_size=PG_MAX_POOL,
                command_timeout=30,
            )

            # Ensure pgvector extension and schema
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await self._ensure_schema(conn)

            # Start session (nanosecond precision to avoid collisions)
            self._session_id = f"mcp-pg-{time.time_ns()}"
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO cognitive_sessions
                       (session_id, started_ns, platform, status)
                       VALUES ($1, $2, $3, 'active')
                       ON CONFLICT (session_id) DO UPDATE
                       SET started_ns = EXCLUDED.started_ns,
                           status = 'active'""",
                    self._session_id, time.time_ns(), Config.PLATFORM,
                )

            self._available = True
            log.info(f"PostgreSQL connected: {self._dsn} session={self._session_id}")
            return True

        except Exception as exc:
            log.warning(f"PostgreSQL unavailable: {exc} — use SQLite fallback")
            self._pool = None
            return False

    async def _ensure_schema(self, conn):
        """Create tables if they don't exist."""
        # Read and execute the schema file
        schema_path = Config.AGENT_CORE.parent / "researchgravity" / "unified_cognitive_schema.sql"
        if schema_path.exists():
            sql = schema_path.read_text()
            await conn.execute(sql)
            log.info("Schema applied from unified_cognitive_schema.sql")
        else:
            # Inline minimal schema if file not found
            await conn.execute(self._inline_schema())
            log.info("Schema applied (inline)")

    def _inline_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS cognitive_events (
            event_id        TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            timestamp_ns    BIGINT NOT NULL,
            direction       TEXT NOT NULL,
            stage           TEXT NOT NULL,
            method          TEXT,
            request_id      TEXT,
            parent_event_id TEXT,
            turn            INTEGER DEFAULT 0,
            raw_bytes       BYTEA,
            parsed_json     JSONB,
            content_length  INTEGER DEFAULT 0,
            error           TEXT,
            data_layer      JSONB,
            light_layer     JSONB,
            instinct_layer  JSONB,
            coherence_sig   TEXT,
            platform        TEXT DEFAULT 'claude-desktop',
            protocol        TEXT DEFAULT 'mcp',
            quality_score   REAL,
            cognitive_mode  TEXT,
            semantic_embedding vector(384),
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS cognitive_sessions (
            session_id      TEXT PRIMARY KEY,
            started_ns      BIGINT NOT NULL,
            ended_ns        BIGINT,
            platform        TEXT DEFAULT 'claude-desktop',
            status          TEXT DEFAULT 'active',
            event_count     INTEGER DEFAULT 0,
            turn_count      INTEGER DEFAULT 0,
            topics          JSONB,
            summary         TEXT,
            cognitive_mode  TEXT,
            quality_score   REAL,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS coherence_moments (
            moment_id       TEXT PRIMARY KEY,
            detected_ns     BIGINT NOT NULL,
            event_ids       TEXT[] NOT NULL,
            platforms       TEXT[] NOT NULL,
            coherence_type  TEXT NOT NULL,
            confidence      REAL NOT NULL,
            description     TEXT,
            time_window_s   INTEGER,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS coherence_links (
            link_id         TEXT PRIMARY KEY,
            source_event_id TEXT NOT NULL,
            target_event_id TEXT NOT NULL,
            link_type       TEXT NOT NULL,
            confidence      REAL NOT NULL,
            metadata        JSONB,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS embedding_cache (
            content_hash    TEXT PRIMARY KEY,
            embedding       vector(384),
            model           TEXT NOT NULL,
            dimensions      INTEGER NOT NULL,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS cognitive_signatures (
            signature_id    TEXT PRIMARY KEY,
            signature_hash  TEXT NOT NULL UNIQUE,
            intent          TEXT,
            topic           TEXT,
            concepts        JSONB,
            occurrence_count INTEGER DEFAULT 1,
            first_seen_ns   BIGINT,
            last_seen_ns    BIGINT,
            platforms       TEXT[],
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS supermemory_entries (
            entry_id        TEXT PRIMARY KEY,
            content         TEXT NOT NULL,
            entry_type      TEXT NOT NULL,
            source_session  TEXT,
            source_platform TEXT,
            importance      REAL DEFAULT 0.5,
            review_count    INTEGER DEFAULT 0,
            next_review_at  TIMESTAMPTZ,
            embedding       vector(1024),
            metadata        JSONB,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );
        """

    # ── event storage ────────────────────────────────────────────

    async def store_event(self, event) -> None:
        """Store a CaptureEvent to PostgreSQL."""
        if not self._pool:
            return

        data = event.data_layer or {}
        light = event.light_layer or {}
        instinct = event.instinct_layer or {}

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO cognitive_events (
                        event_id, session_id, timestamp_ns, direction, stage,
                        method, request_id, parent_event_id, turn,
                        raw_bytes, parsed_json, content_length, error,
                        data_layer, light_layer, instinct_layer,
                        coherence_sig, platform, protocol
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9,
                        $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                    )""",
                    event.event_id,
                    self._session_id,
                    event.timestamp_ns,
                    event.direction,
                    event.stage,
                    event.method,
                    str(event.request_id) if event.request_id is not None else None,
                    event.parent_protocol_id,
                    event.turn,
                    event.raw_bytes,
                    json.dumps(event.parsed, default=str),
                    event.content_length,
                    event.error,
                    json.dumps(data),
                    json.dumps(light),
                    json.dumps(instinct),
                    event.coherence_signature,
                    Config.PLATFORM,
                    Config.PROTOCOL,
                )
        except Exception as exc:
            log.error(f"Failed to store event {event.event_id}: {exc}")

    # ── queries ──────────────────────────────────────────────────

    async def find_coherent_events(
        self,
        signature: str,
        time_window_ns: int = 30 * 60 * 1_000_000_000,
        limit: int = 20,
    ) -> List[Dict]:
        """Find events with matching coherence signature within time window."""
        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT event_id, session_id, timestamp_ns, direction,
                          method, platform, data_layer, light_layer, instinct_layer
                   FROM cognitive_events
                   WHERE coherence_sig = $1
                   ORDER BY timestamp_ns DESC
                   LIMIT $2""",
                signature, limit,
            )
            return [dict(r) for r in rows]

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get stats for the current capture session."""
        if not self._pool or not self._session_id:
            return {}

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COUNT(*) as event_count, MAX(turn) as turn_count
                   FROM cognitive_events WHERE session_id = $1""",
                self._session_id,
            )
            topics = await conn.fetch(
                """SELECT light_layer->>'topic' as topic, COUNT(*) as cnt
                   FROM cognitive_events WHERE session_id = $1
                   GROUP BY topic ORDER BY cnt DESC LIMIT 10""",
                self._session_id,
            )
            return {
                "session_id": self._session_id,
                "event_count": row["event_count"] if row else 0,
                "turn_count": row["turn_count"] if row else 0,
                "topics": {r["topic"]: r["cnt"] for r in topics},
                "backend": "postgresql",
            }

    async def get_all_stats(self) -> Dict[str, Any]:
        """Get stats across all sessions."""
        if not self._pool:
            return {}

        async with self._pool.acquire() as conn:
            events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
            sessions = await conn.fetchval("SELECT COUNT(*) FROM cognitive_sessions")
            total_bytes = await conn.fetchval(
                "SELECT COALESCE(SUM(content_length), 0) FROM cognitive_events"
            )
            coherence = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")

            return {
                "total_events": events,
                "total_sessions": sessions,
                "total_bytes_captured": total_bytes,
                "coherence_moments": coherence,
                "current_session": self._session_id,
                "backend": "postgresql",
            }

    # ── lifecycle ────────────────────────────────────────────────

    async def close(self):
        """Close pool and finalize session."""
        if self._pool and self._session_id:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """UPDATE cognitive_sessions SET
                              ended_ns = $1,
                              event_count = (SELECT COUNT(*) FROM cognitive_events WHERE session_id = $2),
                              turn_count = (SELECT MAX(turn) FROM cognitive_events WHERE session_id = $2),
                              status = 'completed'
                           WHERE session_id = $2""",
                        time.time_ns(), self._session_id,
                    )
            except Exception as exc:
                log.error(f"Error finalizing session: {exc}")

            await self._pool.close()
            log.info(f"PostgreSQL closed, session {self._session_id} finalized")
