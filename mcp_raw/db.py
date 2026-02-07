"""
SQLite Capture Database — Perfect fidelity storage

Stores every captured event with full UCW semantic layers.
Uses WAL mode for concurrent reads during writes.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .logger import get_logger

log = get_logger("db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cognitive_events (
    event_id        TEXT PRIMARY KEY,
    session_id      TEXT,
    timestamp_ns    INTEGER NOT NULL,
    direction       TEXT NOT NULL,          -- 'in' or 'out'
    stage           TEXT NOT NULL,          -- 'received','parsed','routed','executed','sent'
    method          TEXT,
    request_id      TEXT,
    parent_event_id TEXT,
    turn            INTEGER DEFAULT 0,
    raw_bytes       BLOB,
    parsed_json     TEXT,
    content_length  INTEGER DEFAULT 0,
    error           TEXT,

    -- UCW Data layer
    data_content    TEXT,
    data_tokens_est INTEGER,

    -- UCW Light layer
    light_intent    TEXT,
    light_topic     TEXT,
    light_concepts  TEXT,                   -- JSON array
    light_summary   TEXT,

    -- UCW Instinct layer
    instinct_coherence   REAL,
    instinct_indicators  TEXT,              -- JSON array
    instinct_gut_signal  TEXT,

    -- Coherence
    coherence_sig   TEXT,

    -- Platform
    platform        TEXT DEFAULT 'claude-desktop',
    protocol        TEXT DEFAULT 'mcp',

    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON cognitive_events(timestamp_ns);
CREATE INDEX IF NOT EXISTS idx_events_session ON cognitive_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_method ON cognitive_events(method);
CREATE INDEX IF NOT EXISTS idx_events_direction ON cognitive_events(direction);
CREATE INDEX IF NOT EXISTS idx_events_turn ON cognitive_events(turn);
CREATE INDEX IF NOT EXISTS idx_events_coherence ON cognitive_events(coherence_sig);
CREATE INDEX IF NOT EXISTS idx_events_topic ON cognitive_events(light_topic);
CREATE INDEX IF NOT EXISTS idx_events_intent ON cognitive_events(light_intent);
CREATE INDEX IF NOT EXISTS idx_events_gut ON cognitive_events(instinct_gut_signal);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    started_ns      INTEGER NOT NULL,
    ended_ns        INTEGER,
    platform        TEXT DEFAULT 'claude-desktop',
    event_count     INTEGER DEFAULT 0,
    turn_count      INTEGER DEFAULT 0,
    topics          TEXT,                   -- JSON array of detected topics
    summary         TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS capture_stats (
    stat_key        TEXT PRIMARY KEY,
    stat_value      TEXT,
    updated_at      TEXT DEFAULT (datetime('now'))
);
"""


class CaptureDB:
    """SQLite database for perfect-fidelity cognitive capture."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or Config.DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._session_id: Optional[str] = None

    async def initialize(self):
        """Open database and create schema."""
        Config.ensure_dirs()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

        # Generate session ID
        self._session_id = f"mcp-{int(time.time())}"
        self._conn.execute(
            "INSERT INTO sessions (session_id, started_ns, platform) VALUES (?, ?, ?)",
            (self._session_id, time.time_ns(), Config.PLATFORM),
        )
        self._conn.commit()
        log.info(f"Database initialized: {self._db_path} session={self._session_id}")

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    async def store_event(self, event) -> None:
        """Store a CaptureEvent to the database."""
        if not self._conn:
            return

        data = event.data_layer or {}
        light = event.light_layer or {}
        instinct = event.instinct_layer or {}

        try:
            self._conn.execute(
                """INSERT INTO cognitive_events (
                    event_id, session_id, timestamp_ns, direction, stage,
                    method, request_id, parent_event_id, turn,
                    raw_bytes, parsed_json, content_length, error,
                    data_content, data_tokens_est,
                    light_intent, light_topic, light_concepts, light_summary,
                    instinct_coherence, instinct_indicators, instinct_gut_signal,
                    coherence_sig, platform, protocol
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
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
                    data.get("content"),
                    data.get("tokens_est"),
                    light.get("intent"),
                    light.get("topic"),
                    json.dumps(light.get("concepts", [])),
                    light.get("summary"),
                    instinct.get("coherence_potential"),
                    json.dumps(instinct.get("emergence_indicators", [])),
                    instinct.get("gut_signal"),
                    event.coherence_signature,
                    Config.PLATFORM,
                    Config.PROTOCOL,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            log.error(f"Failed to store event {event.event_id}: {exc}")

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get stats for the current capture session."""
        if not self._conn or not self._session_id:
            return {}

        cur = self._conn.execute(
            "SELECT COUNT(*), MAX(turn) FROM cognitive_events WHERE session_id = ?",
            (self._session_id,),
        )
        row = cur.fetchone()
        event_count = row[0] if row else 0
        turn_count = row[1] if row else 0

        # Topic distribution
        cur = self._conn.execute(
            """SELECT light_topic, COUNT(*) as cnt
               FROM cognitive_events WHERE session_id = ?
               GROUP BY light_topic ORDER BY cnt DESC LIMIT 10""",
            (self._session_id,),
        )
        topics = {r[0]: r[1] for r in cur.fetchall()}

        # Gut signal distribution
        cur = self._conn.execute(
            """SELECT instinct_gut_signal, COUNT(*) as cnt
               FROM cognitive_events WHERE session_id = ?
               GROUP BY instinct_gut_signal ORDER BY cnt DESC""",
            (self._session_id,),
        )
        signals = {r[0]: r[1] for r in cur.fetchall() if r[0]}

        return {
            "session_id": self._session_id,
            "event_count": event_count,
            "turn_count": turn_count or 0,
            "topics": topics,
            "gut_signals": signals,
        }

    async def get_all_stats(self) -> Dict[str, Any]:
        """Get stats across ALL capture sessions."""
        if not self._conn:
            return {}

        cur = self._conn.execute("SELECT COUNT(*) FROM cognitive_events")
        total_events = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cur.fetchone()[0]

        cur = self._conn.execute(
            "SELECT SUM(content_length) FROM cognitive_events"
        )
        total_bytes = cur.fetchone()[0] or 0

        cur = self._conn.execute(
            """SELECT instinct_gut_signal, COUNT(*) FROM cognitive_events
               WHERE instinct_gut_signal IS NOT NULL
               GROUP BY instinct_gut_signal"""
        )
        signals = {r[0]: r[1] for r in cur.fetchall()}

        return {
            "total_events": total_events,
            "total_sessions": total_sessions,
            "total_bytes_captured": total_bytes,
            "gut_signals": signals,
            "current_session": self._session_id,
        }

    async def close(self):
        """Close database and finalize session."""
        if self._conn and self._session_id:
            self._conn.execute(
                "UPDATE sessions SET ended_ns = ?, event_count = ("
                "  SELECT COUNT(*) FROM cognitive_events WHERE session_id = ?"
                "), turn_count = ("
                "  SELECT MAX(turn) FROM cognitive_events WHERE session_id = ?"
                ") WHERE session_id = ?",
                (time.time_ns(), self._session_id, self._session_id, self._session_id),
            )
            self._conn.commit()
            self._conn.close()
            log.info(f"Database closed, session {self._session_id} finalized")
