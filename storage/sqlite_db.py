"""
SQLite Database Module

Provides concurrent-safe relational storage with WAL mode for:
- Sessions
- Findings (with evidence)
- URLs
- Context Packs
- Provenance tracking (for UCW imports)

Uses aiosqlite for async operations and connection pooling.
"""

import aiosqlite
import json
import asyncio
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Database location
DB_PATH = Path.home() / ".agent-core" / "storage" / "antigravity.db"

# Schema version for migrations
SCHEMA_VERSION = 3  # Phase 4: Added prediction_tracking

SCHEMA = """
-- Enable WAL mode for concurrent reads/writes
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    topic TEXT,
    status TEXT DEFAULT 'active',
    project TEXT,
    started_at TEXT,
    archived_at TEXT,
    transcript_tokens INTEGER,
    finding_count INTEGER DEFAULT 0,
    url_count INTEGER DEFAULT 0,
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- URLs with tier classification
CREATE TABLE IF NOT EXISTS urls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    tier INTEGER CHECK (tier BETWEEN 1 AND 3),
    category TEXT,
    source TEXT,
    context TEXT,
    relevance INTEGER,
    captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, url)
);

-- Findings with evidence
CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    evidence TEXT,  -- JSON: {sources: [], confidence: float, reasoning_chain: []}
    confidence REAL,
    derived_from TEXT,  -- JSON array of parent finding IDs
    enables TEXT,  -- JSON array of child finding IDs
    project TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Context packs for retrieval
CREATE TABLE IF NOT EXISTS context_packs (
    id TEXT PRIMARY KEY,
    name TEXT,
    type TEXT NOT NULL,  -- 'domain', 'project', 'pattern', 'ucw'
    content TEXT,  -- JSON
    tokens INTEGER,
    dq_metadata TEXT,  -- JSON
    source TEXT,  -- 'local', 'ucw_trade', 'agent_produced'
    source_id TEXT,  -- Original pack ID if imported
    validated INTEGER DEFAULT 0,
    validation_result TEXT,  -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Provenance tracking for UCW imports and agent production
CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,  -- 'session', 'finding', 'pack', 'url'
    entity_id TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- 'local', 'ucw_trade', 'agent', 'import'
    source_id TEXT,  -- UCW wallet ID, agent ID, etc.
    source_metadata TEXT,  -- JSON
    imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_id)
);

-- Papers referenced in research
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,  -- arxiv_id or DOI
    title TEXT,
    authors TEXT,  -- JSON array
    abstract TEXT,
    url TEXT,
    relevance INTEGER,
    applied INTEGER DEFAULT 0,
    session_ids TEXT,  -- JSON array of sessions that reference this
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Lineage/relationships (lightweight graph in relational)
CREATE TABLE IF NOT EXISTS lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,  -- 'session', 'paper', 'finding', 'pack'
    source_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,  -- 'enables', 'informs', 'derives_from', 'cites', 'contains'
    weight REAL DEFAULT 1.0,
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, source_id, target_type, target_id, relation)
);

-- Session outcomes for meta-learning
CREATE TABLE IF NOT EXISTS session_outcomes (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    intent TEXT NOT NULL,
    outcome TEXT,  -- 'success', 'partial', 'failed'
    quality REAL,  -- 1-5
    model_efficiency REAL,
    models_used TEXT,  -- JSON
    date TEXT,
    messages INTEGER,
    tools INTEGER,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Cognitive states for temporal prediction
CREATE TABLE IF NOT EXISTS cognitive_states (
    id TEXT PRIMARY KEY,
    mode TEXT,  -- 'morning', 'peak', 'dip', 'evening', 'deep_night'
    energy_level REAL,
    flow_score REAL,
    hour INTEGER,
    day TEXT,
    predictions TEXT,  -- JSON
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Error patterns for preventive action
CREATE TABLE IF NOT EXISTS error_patterns (
    id TEXT PRIMARY KEY,
    error_type TEXT NOT NULL,
    context TEXT,
    solution TEXT,
    success_rate REAL DEFAULT 0.0,
    occurrences INTEGER DEFAULT 1,
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Prediction tracking for calibration loop (Phase 4)
CREATE TABLE IF NOT EXISTS prediction_tracking (
    id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    predicted_quality REAL,
    predicted_success_probability REAL,
    predicted_optimal_hour INTEGER,
    actual_quality REAL,
    actual_outcome TEXT,
    actual_session_id TEXT,
    prediction_timestamp TEXT NOT NULL,
    outcome_timestamp TEXT,
    cognitive_state TEXT,  -- JSON snapshot of state at prediction time
    error_magnitude REAL,  -- |predicted - actual| for quality
    success_match INTEGER,  -- 1 if prediction matched outcome, 0 otherwise
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_archived ON sessions(archived_at);

CREATE INDEX IF NOT EXISTS idx_urls_session ON urls(session_id);
CREATE INDEX IF NOT EXISTS idx_urls_tier ON urls(tier);
CREATE INDEX IF NOT EXISTS idx_urls_category ON urls(category);

CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(type);
CREATE INDEX IF NOT EXISTS idx_findings_project ON findings(project);
CREATE INDEX IF NOT EXISTS idx_findings_confidence ON findings(confidence);

CREATE INDEX IF NOT EXISTS idx_packs_type ON context_packs(type);
CREATE INDEX IF NOT EXISTS idx_packs_source ON context_packs(source);

CREATE INDEX IF NOT EXISTS idx_provenance_entity ON provenance(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_provenance_source ON provenance(source_type, source_id);

CREATE INDEX IF NOT EXISTS idx_papers_relevance ON papers(relevance);
CREATE INDEX IF NOT EXISTS idx_papers_applied ON papers(applied);

CREATE INDEX IF NOT EXISTS idx_lineage_source ON lineage(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_lineage_target ON lineage(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_lineage_relation ON lineage(relation);

CREATE INDEX IF NOT EXISTS idx_outcomes_session ON session_outcomes(session_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_outcome ON session_outcomes(outcome);
CREATE INDEX IF NOT EXISTS idx_outcomes_quality ON session_outcomes(quality);
CREATE INDEX IF NOT EXISTS idx_outcomes_date ON session_outcomes(date);

CREATE INDEX IF NOT EXISTS idx_cognitive_mode ON cognitive_states(mode);
CREATE INDEX IF NOT EXISTS idx_cognitive_hour ON cognitive_states(hour);
CREATE INDEX IF NOT EXISTS idx_cognitive_day ON cognitive_states(day);

CREATE INDEX IF NOT EXISTS idx_errors_type ON error_patterns(error_type);
CREATE INDEX IF NOT EXISTS idx_errors_success ON error_patterns(success_rate);

CREATE INDEX IF NOT EXISTS idx_predictions_intent ON prediction_tracking(intent);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON prediction_tracking(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON prediction_tracking(actual_session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON prediction_tracking(success_match);

-- Full-text search for content
CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(
    id,
    content,
    type,
    content='findings',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    id,
    topic,
    content='sessions',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
    INSERT INTO findings_fts(id, content, type) VALUES (new.id, new.content, new.type);
END;

CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, id, content, type) VALUES('delete', old.id, old.content, old.type);
END;

CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, id, content, type) VALUES('delete', old.id, old.content, old.type);
    INSERT INTO findings_fts(id, content, type) VALUES (new.id, new.content, new.type);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO sessions_fts(id, topic) VALUES (new.id, new.topic);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, id, topic) VALUES('delete', old.id, old.topic);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, id, topic) VALUES('delete', old.id, old.topic);
    INSERT INTO sessions_fts(id, topic) VALUES (new.id, new.topic);
END;
"""


class SQLiteDB:
    """Async SQLite database with connection pooling and WAL mode.

    Features:
    - Semaphore-guarded connection pool to prevent unbounded growth
    - Pre-warmed pool for faster initial requests
    - Proper resource tracking and cleanup
    """

    def __init__(self, db_path: Path = DB_PATH, pool_size: int = 5):
        self.db_path = db_path
        self._pool: List[aiosqlite.Connection] = []
        self._pool_size = pool_size
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(pool_size)  # Limit concurrent connections
        self._initialized = False
        self._borrowed_count = 0  # Track active connections for diagnostics

    async def initialize(self):
        """Initialize database, create schema, and pre-warm connection pool."""
        if self._initialized:
            return

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create schema with initial connection
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)

            # Check/update schema version
            cursor = await db.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            row = await cursor.fetchone()

            if not row:
                await db.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now().isoformat())
                )

            await db.commit()

        # Pre-warm the connection pool
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            self._pool.append(conn)

        self._initialized = True

    @asynccontextmanager
    async def connection(self):
        """Get a connection from the pool with semaphore guard.

        Uses semaphore to prevent unbounded connection growth under load.
        Pre-warmed pool ensures fast initial access.
        """
        # Semaphore limits total concurrent connections
        await self._semaphore.acquire()

        conn = None
        try:
            async with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                    self._borrowed_count += 1
                else:
                    # Fallback: create new connection (shouldn't happen with pre-warming)
                    conn = await aiosqlite.connect(self.db_path)
                    conn.row_factory = aiosqlite.Row
                    self._borrowed_count += 1

            yield conn

        finally:
            if conn is not None:
                async with self._lock:
                    self._borrowed_count -= 1
                    if len(self._pool) < self._pool_size:
                        self._pool.append(conn)
                    else:
                        await conn.close()

            self._semaphore.release()

    async def close(self):
        """Close all connections in the pool."""
        async with self._lock:
            # Warn if connections are still borrowed
            if self._borrowed_count > 0:
                import logging
                logging.warning(
                    f"SQLiteDB.close(): {self._borrowed_count} connections still borrowed"
                )

            for conn in self._pool:
                try:
                    await conn.close()
                except Exception:
                    pass  # Ignore close errors during shutdown
            self._pool.clear()

    @property
    def pool_stats(self) -> Dict[str, int]:
        """Get current pool statistics for monitoring."""
        return {
            "pool_size": self._pool_size,
            "available": len(self._pool),
            "borrowed": self._borrowed_count,
        }

    # --- Session Operations ---

    async def store_session(self, session: Dict[str, Any]) -> str:
        """Store or update a session."""
        async with self.connection() as db:
            await db.execute("""
                INSERT INTO sessions (id, topic, status, project, started_at, archived_at,
                                     transcript_tokens, finding_count, url_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    topic = excluded.topic,
                    status = excluded.status,
                    project = excluded.project,
                    archived_at = excluded.archived_at,
                    transcript_tokens = excluded.transcript_tokens,
                    finding_count = excluded.finding_count,
                    url_count = excluded.url_count,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                session['id'],
                session.get('topic'),
                session.get('status', 'active'),
                session.get('project'),
                session.get('started_at'),
                session.get('archived_at'),
                session.get('transcript_tokens'),
                session.get('finding_count', 0),
                session.get('url_count', 0),
                json.dumps(session.get('metadata', {}))
            ))
            await db.commit()
            return session['id']

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        async with self.connection() as db:
            cursor = await db.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row:
                return dict(row)
            return None

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        project: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List sessions with optional filtering."""
        query = "SELECT * FROM sessions WHERE 1=1"
        params = []

        if project:
            query += " AND project = ?"
            params.append(project)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self.connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # --- Finding Operations ---

    async def store_finding(self, finding: Dict[str, Any]) -> str:
        """Store or update a finding."""
        async with self.connection() as db:
            await db.execute("""
                INSERT INTO findings (id, session_id, content, type, evidence,
                                     confidence, derived_from, enables, project)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    type = excluded.type,
                    evidence = excluded.evidence,
                    confidence = excluded.confidence,
                    derived_from = excluded.derived_from,
                    enables = excluded.enables,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                finding['id'],
                finding.get('session_id'),
                finding['content'],
                finding['type'],
                json.dumps(finding.get('evidence', {})),
                finding.get('confidence'),
                json.dumps(finding.get('derived_from', [])),
                json.dumps(finding.get('enables', [])),
                finding.get('project')
            ))
            await db.commit()
            return finding['id']

    async def store_findings_batch(self, findings: List[Dict[str, Any]]) -> int:
        """Store multiple findings in a single transaction."""
        async with self.connection() as db:
            await db.executemany("""
                INSERT INTO findings (id, session_id, content, type, evidence,
                                     confidence, derived_from, enables, project)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    type = excluded.type,
                    evidence = excluded.evidence,
                    confidence = excluded.confidence,
                    updated_at = CURRENT_TIMESTAMP
            """, [
                (
                    f['id'],
                    f.get('session_id'),
                    f['content'],
                    f['type'],
                    json.dumps(f.get('evidence', {})),
                    f.get('confidence'),
                    json.dumps(f.get('derived_from', [])),
                    json.dumps(f.get('enables', [])),
                    f.get('project')
                )
                for f in findings
            ])
            await db.commit()
            return len(findings)

    async def get_findings(
        self,
        session_id: Optional[str] = None,
        finding_type: Optional[str] = None,
        project: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get findings with filtering."""
        query = "SELECT * FROM findings WHERE 1=1"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if finding_type:
            query += " AND type = ?"
            params.append(finding_type)
        if project:
            query += " AND project = ?"
            params.append(project)
        if min_confidence:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with self.connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d['evidence'] = json.loads(d['evidence']) if d['evidence'] else {}
                d['derived_from'] = json.loads(d['derived_from']) if d['derived_from'] else []
                d['enables'] = json.loads(d['enables']) if d['enables'] else []
                results.append(d)
            return results

    async def search_findings_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search on findings."""
        async with self.connection() as db:
            cursor = await db.execute("""
                SELECT f.* FROM findings f
                JOIN findings_fts fts ON f.id = fts.id
                WHERE findings_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # --- URL Operations ---

    async def store_url(self, url_data: Dict[str, Any]) -> int:
        """Store a URL."""
        async with self.connection() as db:
            cursor = await db.execute("""
                INSERT INTO urls (session_id, url, tier, category, source, context, relevance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, url) DO UPDATE SET
                    tier = excluded.tier,
                    category = excluded.category,
                    context = excluded.context,
                    relevance = excluded.relevance
            """, (
                url_data.get('session_id'),
                url_data['url'],
                url_data.get('tier'),
                url_data.get('category'),
                url_data.get('source'),
                url_data.get('context'),
                url_data.get('relevance')
            ))
            await db.commit()
            return cursor.lastrowid

    async def store_urls_batch(self, urls: List[Dict[str, Any]]) -> int:
        """Store multiple URLs in a single transaction."""
        async with self.connection() as db:
            await db.executemany("""
                INSERT INTO urls (session_id, url, tier, category, source, context, relevance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, url) DO NOTHING
            """, [
                (
                    u.get('session_id'),
                    u['url'],
                    u.get('tier'),
                    u.get('category'),
                    u.get('source'),
                    u.get('context'),
                    u.get('relevance')
                )
                for u in urls
            ])
            await db.commit()
            return len(urls)

    # --- Context Pack Operations ---

    async def store_pack(self, pack: Dict[str, Any], source: str = 'local') -> str:
        """Store a context pack."""
        async with self.connection() as db:
            await db.execute("""
                INSERT INTO context_packs (id, name, type, content, tokens, dq_metadata,
                                          source, source_id, validated, validation_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    content = excluded.content,
                    tokens = excluded.tokens,
                    dq_metadata = excluded.dq_metadata,
                    validated = excluded.validated,
                    validation_result = excluded.validation_result,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                pack['id'],
                pack.get('name'),
                pack.get('type', 'pattern'),
                json.dumps(pack.get('content', {})),
                pack.get('tokens'),
                json.dumps(pack.get('dq_metadata', {})),
                source,
                pack.get('source_id'),
                pack.get('validated', 0),
                json.dumps(pack.get('validation_result', {}))
            ))
            await db.commit()
            return pack['id']

    async def store_packs_batch(
        self,
        packs: List[Dict[str, Any]],
        source: str = 'local',
        source_id: Optional[str] = None
    ) -> int:
        """Store multiple packs in a single transaction (for UCW imports)."""
        async with self.connection() as db:
            await db.executemany("""
                INSERT INTO context_packs (id, name, type, content, tokens, dq_metadata,
                                          source, source_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    updated_at = CURRENT_TIMESTAMP
            """, [
                (
                    p['id'],
                    p.get('name'),
                    p.get('type', 'pattern'),
                    json.dumps(p.get('content', {})),
                    p.get('tokens'),
                    json.dumps(p.get('dq_metadata', {})),
                    source,
                    source_id or p.get('source_id')
                )
                for p in packs
            ])
            await db.commit()
            return len(packs)

    async def get_packs(
        self,
        pack_type: Optional[str] = None,
        source: Optional[str] = None,
        validated_only: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get context packs with filtering."""
        query = "SELECT * FROM context_packs WHERE 1=1"
        params = []

        if pack_type:
            query += " AND type = ?"
            params.append(pack_type)
        if source:
            query += " AND source = ?"
            params.append(source)
        if validated_only:
            query += " AND validated = 1"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with self.connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d['content'] = json.loads(d['content']) if d['content'] else {}
                d['dq_metadata'] = json.loads(d['dq_metadata']) if d['dq_metadata'] else {}
                d['validation_result'] = json.loads(d['validation_result']) if d['validation_result'] else {}
                results.append(d)
            return results

    # --- Provenance Tracking ---

    async def track_provenance(
        self,
        entity_type: str,
        entity_id: str,
        source_type: str,
        source_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Track provenance of an entity."""
        async with self.connection() as db:
            await db.execute("""
                INSERT INTO provenance (entity_type, entity_id, source_type, source_id, source_metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    source_id = excluded.source_id,
                    source_metadata = excluded.source_metadata
            """, (
                entity_type,
                entity_id,
                source_type,
                source_id,
                json.dumps(metadata or {})
            ))
            await db.commit()

    # --- Lineage Operations ---

    async def add_lineage(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        metadata: Optional[Dict] = None
    ):
        """Add a lineage relationship."""
        async with self.connection() as db:
            await db.execute("""
                INSERT INTO lineage (source_type, source_id, target_type, target_id, relation, weight, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_type, source_id, target_type, target_id, relation) DO UPDATE SET
                    weight = excluded.weight,
                    metadata = excluded.metadata
            """, (
                source_type, source_id, target_type, target_id, relation, weight,
                json.dumps(metadata or {})
            ))
            await db.commit()

    async def get_lineage(
        self,
        entity_type: str,
        entity_id: str,
        direction: str = 'both'  # 'outgoing', 'incoming', 'both'
    ) -> List[Dict[str, Any]]:
        """Get lineage connections for an entity."""
        results = []

        async with self.connection() as db:
            if direction in ('outgoing', 'both'):
                cursor = await db.execute("""
                    SELECT * FROM lineage
                    WHERE source_type = ? AND source_id = ?
                """, (entity_type, entity_id))
                rows = await cursor.fetchall()
                results.extend([dict(row) for row in rows])

            if direction in ('incoming', 'both'):
                cursor = await db.execute("""
                    SELECT * FROM lineage
                    WHERE target_type = ? AND target_id = ?
                """, (entity_type, entity_id))
                rows = await cursor.fetchall()
                results.extend([dict(row) for row in rows])

        return results

    # --- Session Outcome Operations ---

    async def store_outcome(self, outcome: Dict[str, Any]) -> str:
        """Store a session outcome."""
        async with self.connection() as db:
            outcome_id = outcome.get("id", outcome.get("session_id"))
            await db.execute("""
                INSERT INTO session_outcomes (id, session_id, intent, outcome, quality,
                                              model_efficiency, models_used, date, messages, tools)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    intent = excluded.intent,
                    outcome = excluded.outcome,
                    quality = excluded.quality,
                    model_efficiency = excluded.model_efficiency,
                    models_used = excluded.models_used,
                    messages = excluded.messages,
                    tools = excluded.tools
            """, (
                outcome_id,
                outcome.get("session_id"),
                outcome.get("intent", outcome.get("title", "")),
                outcome.get("outcome"),
                outcome.get("quality"),
                outcome.get("model_efficiency"),
                json.dumps(outcome.get("models_used", {})),
                outcome.get("date"),
                outcome.get("messages"),
                outcome.get("tools")
            ))
            await db.commit()
            return outcome_id

    async def store_outcomes_batch(self, outcomes: List[Dict[str, Any]]) -> int:
        """Store multiple outcomes in a single transaction."""
        async with self.connection() as db:
            await db.executemany("""
                INSERT INTO session_outcomes (id, session_id, intent, outcome, quality,
                                              model_efficiency, models_used, date, messages, tools)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    intent = excluded.intent,
                    outcome = excluded.outcome,
                    quality = excluded.quality
            """, [
                (
                    o.get("id", o.get("session_id")),
                    o.get("session_id"),
                    o.get("intent", o.get("title", "")),
                    o.get("outcome"),
                    o.get("quality"),
                    o.get("model_efficiency"),
                    json.dumps(o.get("models_used", {})),
                    o.get("date"),
                    o.get("messages"),
                    o.get("tools")
                )
                for o in outcomes
            ])
            await db.commit()
            return len(outcomes)

    async def get_outcomes(
        self,
        limit: int = 100,
        min_quality: Optional[float] = None,
        outcome_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get session outcomes with filtering."""
        query = "SELECT * FROM session_outcomes WHERE 1=1"
        params = []

        if min_quality:
            query += " AND quality >= ?"
            params.append(min_quality)
        if outcome_filter:
            query += " AND outcome = ?"
            params.append(outcome_filter)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self.connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d['models_used'] = json.loads(d['models_used']) if d['models_used'] else {}
                results.append(d)
            return results

    # --- Cognitive State Operations ---

    async def store_cognitive_state(self, state: Dict[str, Any]) -> str:
        """Store a cognitive state."""
        async with self.connection() as db:
            state_id = state.get("id", f"state-{state.get('timestamp', datetime.now().isoformat())}")
            await db.execute("""
                INSERT INTO cognitive_states (id, mode, energy_level, flow_score, hour, day, predictions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    mode = excluded.mode,
                    energy_level = excluded.energy_level,
                    flow_score = excluded.flow_score
            """, (
                state_id,
                state.get("mode"),
                state.get("energy_level"),
                state.get("flow_score"),
                state.get("hour"),
                state.get("day"),
                json.dumps(state.get("predictions", {}))
            ))
            await db.commit()
            return state_id

    async def store_cognitive_states_batch(self, states: List[Dict[str, Any]]) -> int:
        """Store multiple cognitive states."""
        async with self.connection() as db:
            await db.executemany("""
                INSERT INTO cognitive_states (id, mode, energy_level, flow_score, hour, day, predictions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
            """, [
                (
                    s.get("id", f"state-{s.get('timestamp', i)}"),
                    s.get("mode"),
                    s.get("energy_level"),
                    s.get("flow_score"),
                    s.get("hour"),
                    s.get("day"),
                    json.dumps(s.get("predictions", {}))
                )
                for i, s in enumerate(states)
            ])
            await db.commit()
            return len(states)

    async def get_cognitive_states(
        self,
        limit: int = 100,
        mode: Optional[str] = None,
        hour: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get cognitive states with filtering."""
        query = "SELECT * FROM cognitive_states WHERE 1=1"
        params = []

        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if hour is not None:
            query += " AND hour = ?"
            params.append(hour)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self.connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d['predictions'] = json.loads(d['predictions']) if d['predictions'] else {}
                results.append(d)
            return results

    # --- Error Pattern Operations ---

    async def store_error_pattern(self, error: Dict[str, Any]) -> str:
        """Store an error pattern."""
        async with self.connection() as db:
            error_id = error.get("id", f"error-{uuid.uuid4().hex[:8]}")
            await db.execute("""
                INSERT INTO error_patterns (id, error_type, context, solution, success_rate, occurrences)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    occurrences = occurrences + 1,
                    success_rate = excluded.success_rate,
                    last_seen = CURRENT_TIMESTAMP
            """, (
                error_id,
                error.get("error_type"),
                error.get("context"),
                error.get("solution"),
                error.get("success_rate", 0.0),
                error.get("occurrences", 1)
            ))
            await db.commit()
            return error_id

    async def store_error_patterns_batch(self, errors: List[Dict[str, Any]]) -> int:
        """Store multiple error patterns."""
        async with self.connection() as db:
            await db.executemany("""
                INSERT INTO error_patterns (id, error_type, context, solution, success_rate)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
            """, [
                (
                    e.get("id", f"error-{i}"),
                    e.get("error_type"),
                    e.get("context"),
                    e.get("solution"),
                    e.get("success_rate", 0.0)
                )
                for i, e in enumerate(errors)
            ])
            await db.commit()
            return len(errors)

    async def get_error_patterns(
        self,
        limit: int = 100,
        min_success_rate: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get error patterns with filtering."""
        query = "SELECT * FROM error_patterns WHERE 1=1"
        params = []

        if min_success_rate:
            query += " AND success_rate >= ?"
            params.append(min_success_rate)

        query += " ORDER BY occurrences DESC, success_rate DESC LIMIT ?"
        params.append(limit)

        async with self.connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # --- Prediction Tracking (Phase 4: Calibration Loop) ---

    async def store_prediction(self, prediction: Dict[str, Any]) -> str:
        """Store a prediction for later calibration."""
        prediction_id = prediction.get("id", str(uuid.uuid4()))

        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO prediction_tracking
                (id, intent, predicted_quality, predicted_success_probability,
                 predicted_optimal_hour, cognitive_state, prediction_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    prediction.get("intent", ""),
                    prediction.get("predicted_quality"),
                    prediction.get("success_probability"),
                    prediction.get("optimal_time"),
                    json.dumps(prediction.get("cognitive_state", {})),
                    prediction.get("timestamp", datetime.now().isoformat())
                )
            )
            await db.commit()

        return prediction_id

    async def update_prediction_outcome(
        self,
        prediction_id: str,
        actual_quality: float,
        actual_outcome: str,
        session_id: str
    ):
        """Update a prediction with actual outcome for calibration."""
        async with self.connection() as db:
            # Get the prediction
            cursor = await db.execute(
                "SELECT predicted_quality, predicted_success_probability FROM prediction_tracking WHERE id = ?",
                (prediction_id,)
            )
            row = await cursor.fetchone()

            if row:
                predicted_quality = row[0] or 3.0
                predicted_success = row[1] or 0.5

                # Calculate error and match
                error_magnitude = abs(predicted_quality - actual_quality)
                success_match = 1 if (
                    (predicted_success >= 0.7 and actual_outcome == "success") or
                    (predicted_success < 0.7 and actual_outcome != "success")
                ) else 0

                # Update with actual outcome
                await db.execute(
                    """
                    UPDATE prediction_tracking
                    SET actual_quality = ?,
                        actual_outcome = ?,
                        actual_session_id = ?,
                        outcome_timestamp = ?,
                        error_magnitude = ?,
                        success_match = ?
                    WHERE id = ?
                    """,
                    (
                        actual_quality,
                        actual_outcome,
                        session_id,
                        datetime.now().isoformat(),
                        error_magnitude,
                        success_match,
                        prediction_id
                    )
                )
                await db.commit()

    async def get_prediction_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """Calculate prediction accuracy metrics."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        async with self.connection() as db:
            # Total predictions with outcomes
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM prediction_tracking
                WHERE outcome_timestamp IS NOT NULL
                AND prediction_timestamp >= ?
                """,
                (cutoff,)
            )
            total = (await cursor.fetchone())[0]

            if total == 0:
                return {
                    "total_predictions": 0,
                    "accurate_predictions": 0,
                    "accuracy": 0.0,
                    "avg_quality_error": 0.0,
                    "success_prediction_rate": 0.0
                }

            # Success matches
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM prediction_tracking
                WHERE success_match = 1
                AND prediction_timestamp >= ?
                """,
                (cutoff,)
            )
            successes = (await cursor.fetchone())[0]

            # Average quality error
            cursor = await db.execute(
                """
                SELECT AVG(error_magnitude) FROM prediction_tracking
                WHERE error_magnitude IS NOT NULL
                AND prediction_timestamp >= ?
                """,
                (cutoff,)
            )
            avg_error = (await cursor.fetchone())[0] or 0.0

            return {
                "total_predictions": total,
                "accurate_predictions": successes,
                "accuracy": successes / total if total > 0 else 0.0,
                "avg_quality_error": round(avg_error, 2),
                "success_prediction_rate": round(successes / total, 2) if total > 0 else 0.0,
                "period_days": days
            }

    # --- Statistics ---

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with self.connection() as db:
            stats = {}

            cursor = await db.execute("SELECT COUNT(*) FROM sessions")
            stats['sessions'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM findings")
            stats['findings'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM urls")
            stats['urls'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM context_packs")
            stats['packs'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM papers")
            stats['papers'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM lineage")
            stats['lineage_edges'] = (await cursor.fetchone())[0]

            cursor = await db.execute(
                "SELECT COUNT(*) FROM provenance WHERE source_type = 'ucw_trade'"
            )
            stats['ucw_imports'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM session_outcomes")
            stats['session_outcomes'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM cognitive_states")
            stats['cognitive_states'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM error_patterns")
            stats['error_patterns'] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM prediction_tracking")
            stats['predictions_tracked'] = (await cursor.fetchone())[0]

            return stats


# Global instance
_db: Optional[SQLiteDB] = None


async def get_db() -> SQLiteDB:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = SQLiteDB()
        await _db.initialize()
    return _db
