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
import sqlite3
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Database location
DB_PATH = Path.home() / ".agent-core" / "storage" / "antigravity.db"

# Schema version for migrations
SCHEMA_VERSION = 1

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
    """Async SQLite database with connection pooling and WAL mode."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._pool: List[aiosqlite.Connection] = []
        self._pool_size = 5
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize database and create schema."""
        if self._initialized:
            return

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create schema
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

        self._initialized = True

    @asynccontextmanager
    async def connection(self):
        """Get a connection from the pool."""
        async with self._lock:
            if self._pool:
                conn = self._pool.pop()
            else:
                conn = await aiosqlite.connect(self.db_path)
                conn.row_factory = aiosqlite.Row

        try:
            yield conn
        finally:
            async with self._lock:
                if len(self._pool) < self._pool_size:
                    self._pool.append(conn)
                else:
                    await conn.close()

    async def close(self):
        """Close all connections in the pool."""
        async with self._lock:
            for conn in self._pool:
                await conn.close()
            self._pool.clear()

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
