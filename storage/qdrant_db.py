"""
Qdrant Vector Database Module

Provides semantic search capabilities using local embeddings.
Uses sentence-transformers for embedding generation (no API costs).

Collections:
- findings: Finding content embeddings
- sessions: Session topic embeddings
- packs: Context pack embeddings
"""

import asyncio
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json

try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchRequest,
        UpdateStatus,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384 dimensions
EMBEDDING_DIM = 384

# Collection definitions
COLLECTIONS = {
    "findings": {
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE,
    },
    "sessions": {
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE,
    },
    "packs": {
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE,
    },
}


class QdrantDB:
    """Vector database for semantic search."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        embedding_model: str = EMBEDDING_MODEL
    ):
        self.host = host
        self.port = port
        self.embedding_model_name = embedding_model
        self._client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._initialized = False

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            )
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    @property
    def embedder(self) -> 'SentenceTransformer':
        """Lazy-load the embedding model."""
        if self._embedder is None:
            self._check_dependencies()
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def client(self) -> 'QdrantClient':
        """Get sync client."""
        if self._client is None:
            self._check_dependencies()
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    @property
    def async_client(self) -> 'AsyncQdrantClient':
        """Get async client."""
        if self._async_client is None:
            self._check_dependencies()
            self._async_client = AsyncQdrantClient(host=self.host, port=self.port)
        return self._async_client

    async def initialize(self):
        """Initialize collections."""
        if self._initialized:
            return

        self._check_dependencies()

        for name, config in COLLECTIONS.items():
            try:
                # Check if collection exists
                collections = await self.async_client.get_collections()
                exists = any(c.name == name for c in collections.collections)

                if not exists:
                    await self.async_client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=config["vector_size"],
                            distance=config["distance"]
                        )
                    )
                    print(f"Created Qdrant collection: {name}")
            except Exception as e:
                print(f"Warning: Could not create collection {name}: {e}")

        self._initialized = True

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.embedder.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.embedder.encode(texts).tolist()

    def _generate_id(self, text: str, prefix: str = "") -> str:
        """Generate deterministic ID from text."""
        content = f"{prefix}:{text}" if prefix else text
        return hashlib.md5(content.encode()).hexdigest()

    # --- Finding Operations ---

    async def upsert_finding(
        self,
        finding_id: str,
        content: str,
        metadata: Dict[str, Any]
    ):
        """Store a finding with its embedding."""
        embedding = self.embed(content)

        await self.async_client.upsert(
            collection_name="findings",
            points=[
                PointStruct(
                    id=self._generate_id(finding_id, "finding"),
                    vector=embedding,
                    payload={
                        "finding_id": finding_id,
                        "content": content[:1000],  # Truncate for storage
                        **metadata
                    }
                )
            ]
        )

    async def upsert_findings_batch(
        self,
        findings: List[Dict[str, Any]]
    ) -> int:
        """Store multiple findings with embeddings."""
        if not findings:
            return 0

        # Generate embeddings in batch
        texts = [f["content"] for f in findings]
        embeddings = self.embed_batch(texts)

        points = [
            PointStruct(
                id=self._generate_id(f["id"], "finding"),
                vector=emb,
                payload={
                    "finding_id": f["id"],
                    "content": f["content"][:1000],
                    "type": f.get("type"),
                    "session_id": f.get("session_id"),
                    "project": f.get("project"),
                    "confidence": f.get("confidence"),
                }
            )
            for f, emb in zip(findings, embeddings)
        ]

        await self.async_client.upsert(
            collection_name="findings",
            points=points
        )

        return len(points)

    async def search_findings(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        filter_type: Optional[str] = None,
        filter_project: Optional[str] = None,
        filter_session: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for findings."""
        embedding = self.embed(query)

        # Build filter
        conditions = []
        if filter_type:
            conditions.append(
                FieldCondition(key="type", match=MatchValue(value=filter_type))
            )
        if filter_project:
            conditions.append(
                FieldCondition(key="project", match=MatchValue(value=filter_project))
            )
        if filter_session:
            conditions.append(
                FieldCondition(key="session_id", match=MatchValue(value=filter_session))
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = await self.async_client.query_points(
            collection_name="findings",
            query=embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=min_score
        )

        return [
            {
                "finding_id": r.payload.get("finding_id"),
                "content": r.payload.get("content"),
                "type": r.payload.get("type"),
                "session_id": r.payload.get("session_id"),
                "project": r.payload.get("project"),
                "confidence": r.payload.get("confidence"),
                "score": r.score,
            }
            for r in results.points
        ]

    # --- Session Operations ---

    async def upsert_session(
        self,
        session_id: str,
        topic: str,
        metadata: Dict[str, Any]
    ):
        """Store a session with its embedding."""
        embedding = self.embed(topic)

        await self.async_client.upsert(
            collection_name="sessions",
            points=[
                PointStruct(
                    id=self._generate_id(session_id, "session"),
                    vector=embedding,
                    payload={
                        "session_id": session_id,
                        "topic": topic,
                        **metadata
                    }
                )
            ]
        )

    async def upsert_sessions_batch(
        self,
        sessions: List[Dict[str, Any]]
    ) -> int:
        """Store multiple sessions with embeddings."""
        if not sessions:
            return 0

        # Filter out sessions without topics
        valid_sessions = [s for s in sessions if s.get("topic")]
        if not valid_sessions:
            return 0

        texts = [s["topic"] for s in valid_sessions]
        embeddings = self.embed_batch(texts)

        points = [
            PointStruct(
                id=self._generate_id(s["id"], "session"),
                vector=emb,
                payload={
                    "session_id": s["id"],
                    "topic": s["topic"],
                    "project": s.get("project"),
                    "status": s.get("status"),
                    "finding_count": s.get("finding_count", 0),
                    "url_count": s.get("url_count", 0),
                }
            )
            for s, emb in zip(valid_sessions, embeddings)
        ]

        await self.async_client.upsert(
            collection_name="sessions",
            points=points
        )

        return len(points)

    async def search_sessions(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.4,
        filter_project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for sessions."""
        embedding = self.embed(query)

        conditions = []
        if filter_project:
            conditions.append(
                FieldCondition(key="project", match=MatchValue(value=filter_project))
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = await self.async_client.query_points(
            collection_name="sessions",
            query=embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=min_score
        )

        return [
            {
                "session_id": r.payload.get("session_id"),
                "topic": r.payload.get("topic"),
                "project": r.payload.get("project"),
                "status": r.payload.get("status"),
                "finding_count": r.payload.get("finding_count"),
                "score": r.score,
            }
            for r in results.points
        ]

    # --- Pack Operations ---

    async def upsert_pack(
        self,
        pack_id: str,
        content: str,
        metadata: Dict[str, Any]
    ):
        """Store a context pack with its embedding."""
        embedding = self.embed(content)

        await self.async_client.upsert(
            collection_name="packs",
            points=[
                PointStruct(
                    id=self._generate_id(pack_id, "pack"),
                    vector=embedding,
                    payload={
                        "pack_id": pack_id,
                        "content_preview": content[:500],
                        **metadata
                    }
                )
            ]
        )

    async def upsert_packs_batch(
        self,
        packs: List[Dict[str, Any]]
    ) -> int:
        """Store multiple packs with embeddings."""
        if not packs:
            return 0

        # Extract searchable content from packs
        texts = []
        for p in packs:
            content = p.get("content", {})
            if isinstance(content, dict):
                # Combine relevant fields for embedding
                text_parts = [
                    p.get("name", ""),
                    content.get("description", ""),
                    " ".join(content.get("keywords", [])),
                ]
                texts.append(" ".join(filter(None, text_parts)))
            else:
                texts.append(str(content)[:1000])

        embeddings = self.embed_batch(texts)

        points = [
            PointStruct(
                id=self._generate_id(p["id"], "pack"),
                vector=emb,
                payload={
                    "pack_id": p["id"],
                    "name": p.get("name"),
                    "type": p.get("type"),
                    "source": p.get("source"),
                    "tokens": p.get("tokens"),
                }
            )
            for p, emb in zip(packs, embeddings)
        ]

        await self.async_client.upsert(
            collection_name="packs",
            points=points
        )

        return len(points)

    async def search_packs(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.4,
        filter_type: Optional[str] = None,
        filter_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for context packs."""
        embedding = self.embed(query)

        conditions = []
        if filter_type:
            conditions.append(
                FieldCondition(key="type", match=MatchValue(value=filter_type))
            )
        if filter_source:
            conditions.append(
                FieldCondition(key="source", match=MatchValue(value=filter_source))
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = await self.async_client.query_points(
            collection_name="packs",
            query=embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=min_score
        )

        return [
            {
                "pack_id": r.payload.get("pack_id"),
                "name": r.payload.get("name"),
                "type": r.payload.get("type"),
                "source": r.payload.get("source"),
                "tokens": r.payload.get("tokens"),
                "score": r.score,
            }
            for r in results.points
        ]

    # --- Unified Search ---

    async def semantic_search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.4
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple collections."""
        if collections is None:
            collections = ["findings", "sessions", "packs"]

        embedding = self.embed(query)
        results = {}

        for collection in collections:
            if collection not in COLLECTIONS:
                continue

            search_results = await self.async_client.query_points(
                collection_name=collection,
                query=embedding,
                limit=limit,
                score_threshold=min_score
            )

            results[collection] = [
                {**r.payload, "score": r.score}
                for r in search_results.points
            ]

        return results

    # --- Statistics ---

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = {}

        for name in COLLECTIONS:
            try:
                info = await self.async_client.get_collection(name)
                stats[name] = {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status.value if info.status else "unknown",
                }
            except Exception as e:
                stats[name] = {"error": str(e)}

        return stats

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            await self.async_client.get_collections()
            return True
        except Exception:
            return False

    async def close(self):
        """Close connections."""
        if self._async_client:
            await self._async_client.close()
        if self._client:
            self._client.close()


# Global instance
_qdrant: Optional[QdrantDB] = None


async def get_qdrant() -> QdrantDB:
    """Get the global Qdrant instance."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantDB()
        await _qdrant.initialize()
    return _qdrant
