"""
SQLite-Vec Vector Storage
=========================

Provides vector storage using sqlite-vec extension for:
- Local-first vector storage (no external dependencies)
- Semantic search without Qdrant
- Hybrid BM25 + cosine similarity search
- Single-file deployment

Replaces Qdrant dependency while maintaining Cohere embeddings for quality.

Usage:
    from storage.sqlite_vec import SqliteVecDB, get_vec_db

    vec_db = await get_vec_db()
    await vec_db.upsert_finding("id", "content", {"type": "finding"})
    results = await vec_db.search_findings("query", limit=10)
"""

import json
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

import aiosqlite

from storage.logging_config import get_logger

logger = get_logger(__name__)

# Try to import sqlite-vec
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    logger.warning("sqlite-vec not installed. Run: pip install sqlite-vec")

# Try to import Cohere for embeddings
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# Default paths
AGENT_CORE_DIR = Path.home() / ".agent-core"
VEC_DB_PATH = AGENT_CORE_DIR / "storage" / "antigravity_vec.db"
CONFIG_PATH = AGENT_CORE_DIR / "config.json"

# Embedding dimensions
EMBEDDING_DIM = 1024  # Cohere embed-english-v3.0


def get_cohere_client() -> Optional["cohere.Client"]:
    """Get Cohere client from config."""
    if not COHERE_AVAILABLE:
        return None

    try:
        if CONFIG_PATH.exists():
            config = json.loads(CONFIG_PATH.read_text())
            api_key = config.get("cohere", {}).get("api_key")
            if api_key:
                return cohere.Client(api_key)
    except Exception:
        pass

    # Try environment variable
    import os
    api_key = os.environ.get("COHERE_API_KEY")
    if api_key:
        return cohere.Client(api_key)

    return None


class SqliteVecDB:
    """Vector database using sqlite-vec extension."""

    def __init__(self, db_path: Path = VEC_DB_PATH):
        self.db_path = db_path
        self._cohere = get_cohere_client()
        self._initialized = False
        self._lock = asyncio.Lock()
        self._embed_cache: Dict[str, List[float]] = {}

    async def initialize(self):
        """Initialize the vector database."""
        if self._initialized:
            return

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Load sqlite-vec extension
            if SQLITE_VEC_AVAILABLE:
                await db.execute("SELECT load_extension('vec0')")

            # Create schema
            await db.executescript(self._get_schema())
            await db.commit()

        self._initialized = True

    def _get_schema(self) -> str:
        """Get the database schema."""
        return f"""
-- Mapping tables for vector-entity relationships
CREATE TABLE IF NOT EXISTS finding_vectors (
    finding_id TEXT PRIMARY KEY,
    vec_rowid INTEGER,
    content_hash TEXT,
    embedded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_vectors (
    session_id TEXT PRIMARY KEY,
    vec_rowid INTEGER,
    content_hash TEXT,
    embedded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pack_vectors (
    pack_id TEXT PRIMARY KEY,
    vec_rowid INTEGER,
    content_hash TEXT,
    embedded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS outcome_vectors (
    outcome_id TEXT PRIMARY KEY,
    vec_rowid INTEGER,
    content_hash TEXT,
    embedded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS error_vectors (
    error_id TEXT PRIMARY KEY,
    vec_rowid INTEGER,
    content_hash TEXT,
    embedded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cognitive_vectors (
    state_id TEXT PRIMARY KEY,
    vec_rowid INTEGER,
    content_hash TEXT,
    embedded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Metadata table for vector storage
CREATE TABLE IF NOT EXISTS vector_metadata (
    rowid INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    content TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vec_meta_type ON vector_metadata(entity_type);
CREATE INDEX IF NOT EXISTS idx_vec_meta_entity ON vector_metadata(entity_id);

-- Virtual table for vectors (if sqlite-vec available)
-- This will be created dynamically when extension is loaded
"""

    async def _create_vec_table(self, db: aiosqlite.Connection, name: str):
        """Create a virtual vector table if it doesn't exist."""
        if not SQLITE_VEC_AVAILABLE:
            return

        try:
            await db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_{name} USING vec0(
                    embedding float[{EMBEDDING_DIM}]
                )
            """)
            await db.commit()
        except Exception as e:
            logger.warning("Could not create vec table", extra={"table": f"vec_{name}", "error": str(e)})

    async def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using Cohere."""
        if not text:
            return None

        # Check cache
        cache_key = text[:200]  # Use prefix for cache key
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        if not self._cohere:
            # Return None if no Cohere client - caller should handle gracefully
            return None

        try:
            # Run blocking Cohere API call in thread pool to avoid blocking event loop
            def _embed():
                return self._cohere.embed(
                    texts=[text],
                    model="embed-english-v3.0",
                    input_type="search_document",
                    truncate="END"
                )

            response = await asyncio.to_thread(_embed)
            embedding = response.embeddings[0]

            # Cache it
            if len(self._embed_cache) < 1000:
                self._embed_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.warning("Embedding failed", extra={"error": str(e)})
            return None

    async def embed_query(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a search query."""
        if not self._cohere:
            return None

        try:
            # Run blocking Cohere API call in thread pool to avoid blocking event loop
            def _embed_query():
                return self._cohere.embed(
                    texts=[text],
                    model="embed-english-v3.0",
                    input_type="search_query",
                    truncate="END"
                )

            response = await asyncio.to_thread(_embed_query)
            return response.embeddings[0]
        except Exception as e:
            logger.warning("Query embedding failed", extra={"error": str(e)})
            return None

    @asynccontextmanager
    async def connection(self):
        """Get a database connection with sqlite-vec loaded."""
        async with self._lock:
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row

            if SQLITE_VEC_AVAILABLE:
                try:
                    await conn.execute("SELECT load_extension('vec0')")
                except Exception:
                    pass

            try:
                yield conn
            finally:
                await conn.close()

    # --- Finding Operations ---

    async def upsert_finding(
        self,
        finding_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upsert a finding with its vector."""
        embedding = await self.embed(content)

        async with self.connection() as db:
            # Ensure vec table exists
            await self._create_vec_table(db, "findings")

            # Store metadata
            await db.execute("""
                INSERT OR REPLACE INTO vector_metadata
                (entity_type, entity_id, content, metadata)
                VALUES ('finding', ?, ?, ?)
            """, (finding_id, content[:2000], json.dumps(metadata or {})))

            # Get rowid
            cursor = await db.execute(
                "SELECT rowid FROM vector_metadata WHERE entity_type = 'finding' AND entity_id = ?",
                (finding_id,)
            )
            row = await cursor.fetchone()
            rowid = row[0] if row else None

            # Store vector if available
            if embedding and SQLITE_VEC_AVAILABLE and rowid:
                try:
                    # Convert to binary format for sqlite-vec
                    import struct
                    vec_blob = struct.pack(f'{len(embedding)}f', *embedding)

                    await db.execute("""
                        INSERT OR REPLACE INTO vec_findings (rowid, embedding)
                        VALUES (?, ?)
                    """, (rowid, vec_blob))

                    # Update mapping
                    await db.execute("""
                        INSERT OR REPLACE INTO finding_vectors
                        (finding_id, vec_rowid, embedded_at)
                        VALUES (?, ?, ?)
                    """, (finding_id, rowid, datetime.now().isoformat()))

                except Exception as e:
                    logger.warning("Vector insert failed", extra={"error": str(e)})

            await db.commit()
            return True

    async def upsert_findings_batch(self, findings: List[Dict[str, Any]]) -> int:
        """Batch upsert findings."""
        count = 0
        for f in findings:
            success = await self.upsert_finding(
                f.get("id", str(uuid.uuid4())),
                f.get("content", ""),
                {k: v for k, v in f.items() if k not in ["id", "content"]}
            )
            if success:
                count += 1
        return count

    async def search_findings(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        filter_type: Optional[str] = None,
        filter_project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search findings using vector similarity."""
        query_embedding = await self.embed_query(query)

        async with self.connection() as db:
            results = []

            if query_embedding and SQLITE_VEC_AVAILABLE:
                try:
                    import struct
                    vec_blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)

                    # Vector similarity search
                    cursor = await db.execute("""
                        SELECT
                            vm.entity_id,
                            vm.content,
                            vm.metadata,
                            vec_distance_cosine(vf.embedding, ?) as distance
                        FROM vec_findings vf
                        JOIN vector_metadata vm ON vm.rowid = vf.rowid
                        WHERE vm.entity_type = 'finding'
                        ORDER BY distance
                        LIMIT ?
                    """, (vec_blob, limit * 2))  # Get more for filtering

                    rows = await cursor.fetchall()

                    for row in rows:
                        score = 1 - row[3]  # Convert distance to similarity
                        if score < min_score:
                            continue

                        metadata = json.loads(row[2]) if row[2] else {}

                        # Apply filters
                        if filter_type and metadata.get("type") != filter_type:
                            continue
                        if filter_project and metadata.get("project") != filter_project:
                            continue

                        results.append({
                            "id": row[0],
                            "content": row[1],
                            "score": score,
                            "relevance_score": score,
                            **metadata
                        })

                        if len(results) >= limit:
                            break

                except Exception as e:
                    logger.warning("Vector search failed", extra={"error": str(e)})

            # Fallback to FTS if no vector results
            if not results:
                cursor = await db.execute("""
                    SELECT entity_id, content, metadata
                    FROM vector_metadata
                    WHERE entity_type = 'finding'
                    AND content LIKE ?
                    LIMIT ?
                """, (f"%{query}%", limit))

                rows = await cursor.fetchall()
                for row in rows:
                    metadata = json.loads(row[2]) if row[2] else {}
                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "score": 0.5,  # Default score for text match
                        **metadata
                    })

            return results

    # --- Session Operations ---

    async def upsert_session(
        self,
        session_id: str,
        topic: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upsert a session with its vector."""
        embedding = await self.embed(topic)

        async with self.connection() as db:
            await self._create_vec_table(db, "sessions")

            await db.execute("""
                INSERT OR REPLACE INTO vector_metadata
                (entity_type, entity_id, content, metadata)
                VALUES ('session', ?, ?, ?)
            """, (session_id, topic, json.dumps(metadata or {})))

            cursor = await db.execute(
                "SELECT rowid FROM vector_metadata WHERE entity_type = 'session' AND entity_id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            rowid = row[0] if row else None

            if embedding and SQLITE_VEC_AVAILABLE and rowid:
                try:
                    import struct
                    vec_blob = struct.pack(f'{len(embedding)}f', *embedding)

                    await db.execute("""
                        INSERT OR REPLACE INTO vec_sessions (rowid, embedding)
                        VALUES (?, ?)
                    """, (rowid, vec_blob))

                    await db.execute("""
                        INSERT OR REPLACE INTO session_vectors
                        (session_id, vec_rowid, embedded_at)
                        VALUES (?, ?, ?)
                    """, (session_id, rowid, datetime.now().isoformat()))

                except Exception as e:
                    logger.warning("Session vector insert failed", extra={"error": str(e)})

            await db.commit()
            return True

    async def search_sessions(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.4,
        filter_project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search sessions using vector similarity."""
        query_embedding = await self.embed_query(query)

        async with self.connection() as db:
            results = []

            if query_embedding and SQLITE_VEC_AVAILABLE:
                try:
                    import struct
                    vec_blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)

                    cursor = await db.execute("""
                        SELECT
                            vm.entity_id,
                            vm.content,
                            vm.metadata,
                            vec_distance_cosine(vs.embedding, ?) as distance
                        FROM vec_sessions vs
                        JOIN vector_metadata vm ON vm.rowid = vs.rowid
                        WHERE vm.entity_type = 'session'
                        ORDER BY distance
                        LIMIT ?
                    """, (vec_blob, limit))

                    rows = await cursor.fetchall()

                    for row in rows:
                        score = 1 - row[3]
                        if score < min_score:
                            continue

                        metadata = json.loads(row[2]) if row[2] else {}
                        if filter_project and metadata.get("project") != filter_project:
                            continue

                        results.append({
                            "id": row[0],
                            "topic": row[1],
                            "score": score,
                            **metadata
                        })

                except Exception as e:
                    logger.warning("Session search failed", extra={"error": str(e)})

            return results

    # --- Pack Operations ---

    async def upsert_pack(
        self,
        pack_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upsert a context pack with its vector."""
        embedding = await self.embed(content)

        async with self.connection() as db:
            await self._create_vec_table(db, "packs")

            await db.execute("""
                INSERT OR REPLACE INTO vector_metadata
                (entity_type, entity_id, content, metadata)
                VALUES ('pack', ?, ?, ?)
            """, (pack_id, content[:2000], json.dumps(metadata or {})))

            cursor = await db.execute(
                "SELECT rowid FROM vector_metadata WHERE entity_type = 'pack' AND entity_id = ?",
                (pack_id,)
            )
            row = await cursor.fetchone()
            rowid = row[0] if row else None

            if embedding and SQLITE_VEC_AVAILABLE and rowid:
                try:
                    import struct
                    vec_blob = struct.pack(f'{len(embedding)}f', *embedding)

                    await db.execute("""
                        INSERT OR REPLACE INTO vec_packs (rowid, embedding)
                        VALUES (?, ?)
                    """, (rowid, vec_blob))

                    await db.execute("""
                        INSERT OR REPLACE INTO pack_vectors
                        (pack_id, vec_rowid, embedded_at)
                        VALUES (?, ?, ?)
                    """, (pack_id, rowid, datetime.now().isoformat()))

                except Exception as e:
                    logger.warning("Pack vector insert failed", extra={"error": str(e)})

            await db.commit()
            return True

    async def search_packs(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.4,
        filter_type: Optional[str] = None,
        filter_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search packs using vector similarity."""
        query_embedding = await self.embed_query(query)

        async with self.connection() as db:
            results = []

            if query_embedding and SQLITE_VEC_AVAILABLE:
                try:
                    import struct
                    vec_blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)

                    cursor = await db.execute("""
                        SELECT
                            vm.entity_id,
                            vm.content,
                            vm.metadata,
                            vec_distance_cosine(vp.embedding, ?) as distance
                        FROM vec_packs vp
                        JOIN vector_metadata vm ON vm.rowid = vp.rowid
                        WHERE vm.entity_type = 'pack'
                        ORDER BY distance
                        LIMIT ?
                    """, (vec_blob, limit))

                    rows = await cursor.fetchall()

                    for row in rows:
                        score = 1 - row[3]
                        if score < min_score:
                            continue

                        metadata = json.loads(row[2]) if row[2] else {}
                        if filter_type and metadata.get("type") != filter_type:
                            continue
                        if filter_source and metadata.get("source") != filter_source:
                            continue

                        results.append({
                            "id": row[0],
                            "content": row[1],
                            "score": score,
                            **metadata
                        })

                except Exception as e:
                    logger.warning("Pack search failed", extra={"error": str(e)})

            return results

    # --- Hybrid Search with Reranking ---

    async def hybrid_search(
        self,
        query: str,
        entity_type: str = "finding",
        limit: int = 10,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 (text) and vector similarity.

        This provides a fallback when Cohere reranking is not available.
        """
        async with self.connection() as db:
            results = {}

            # BM25 / text search
            cursor = await db.execute("""
                SELECT entity_id, content, metadata,
                       1.0 as text_score  -- Simple presence score
                FROM vector_metadata
                WHERE entity_type = ?
                AND content LIKE ?
            """, (entity_type, f"%{query}%"))

            for row in await cursor.fetchall():
                entity_id = row[0]
                results[entity_id] = {
                    "id": entity_id,
                    "content": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "text_score": row[3],
                    "vector_score": 0.0,
                }

            # Vector search
            query_embedding = await self.embed_query(query)
            if query_embedding and SQLITE_VEC_AVAILABLE:
                try:
                    import struct
                    vec_blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)

                    table_name = f"vec_{entity_type}s"
                    cursor = await db.execute(f"""
                        SELECT
                            vm.entity_id,
                            vm.content,
                            vm.metadata,
                            1 - vec_distance_cosine(v.embedding, ?) as score
                        FROM {table_name} v
                        JOIN vector_metadata vm ON vm.rowid = v.rowid
                        WHERE vm.entity_type = ?
                        ORDER BY score DESC
                        LIMIT ?
                    """, (vec_blob, entity_type, limit * 2))

                    for row in await cursor.fetchall():
                        entity_id = row[0]
                        if entity_id in results:
                            results[entity_id]["vector_score"] = row[3]
                        else:
                            results[entity_id] = {
                                "id": entity_id,
                                "content": row[1],
                                "metadata": json.loads(row[2]) if row[2] else {},
                                "text_score": 0.0,
                                "vector_score": row[3],
                            }

                except Exception as e:
                    logger.warning("Hybrid vector search failed", extra={"error": str(e)})

            # Calculate combined scores
            final_results = []
            for entity_id, data in results.items():
                combined_score = (
                    bm25_weight * data["text_score"] +
                    vector_weight * data["vector_score"]
                )
                final_results.append({
                    "id": entity_id,
                    "content": data["content"],
                    "score": combined_score,
                    "text_score": data["text_score"],
                    "vector_score": data["vector_score"],
                    **data["metadata"]
                })

            # Sort by combined score
            final_results.sort(key=lambda x: x["score"], reverse=True)
            return final_results[:limit]

    # --- Statistics ---

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        async with self.connection() as db:
            stats = {
                "sqlite_vec_available": SQLITE_VEC_AVAILABLE,
                "cohere_available": self._cohere is not None,
                "embedding_dim": EMBEDDING_DIM,
            }

            # Count by entity type
            cursor = await db.execute("""
                SELECT entity_type, COUNT(*) as count
                FROM vector_metadata
                GROUP BY entity_type
            """)

            for row in await cursor.fetchall():
                stats[f"{row[0]}_count"] = row[1]

            # Total vectors
            if SQLITE_VEC_AVAILABLE:
                for table in ["findings", "sessions", "packs"]:
                    try:
                        cursor = await db.execute(f"SELECT COUNT(*) FROM vec_{table}")
                        row = await cursor.fetchone()
                        stats[f"vec_{table}_count"] = row[0] if row else 0
                    except Exception:
                        stats[f"vec_{table}_count"] = 0

            return stats

    async def health_check(self) -> bool:
        """Check if the database is healthy."""
        try:
            async with self.connection() as db:
                cursor = await db.execute("SELECT 1")
                await cursor.fetchone()
                return True
        except Exception:
            return False

    async def close(self):
        """Close any open resources."""
        self._embed_cache.clear()


# Global instance
_vec_db: Optional[SqliteVecDB] = None


async def get_vec_db() -> SqliteVecDB:
    """Get the global sqlite-vec database instance."""
    global _vec_db
    if _vec_db is None:
        _vec_db = SqliteVecDB()
        await _vec_db.initialize()
    return _vec_db
