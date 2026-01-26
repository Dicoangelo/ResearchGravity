"""
Qdrant Vector Database Module

Provides semantic search capabilities using Cohere embeddings and reranking.
Uses Cohere embed-english-v3.0 (1024 dims) for embeddings.
Uses Cohere rerank-v3.5 for result reranking.

Collections:
- findings: Finding content embeddings
- sessions: Session topic embeddings
- packs: Context pack embeddings

Environment:
- COHERE_API_KEY: Required for embeddings and reranking
"""

import hashlib
import os
import uuid
from typing import Optional, List, Dict, Any
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
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "embed-english-v3.0"  # Cohere v3, 1024 dimensions
RERANK_MODEL = "rerank-v3.5"  # Cohere rerank v3.5
EMBEDDING_DIM = 1024

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
    "session_outcomes": {
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE,
    },
    "cognitive_states": {
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE,
    },
    "error_patterns": {
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE,
    },
}


def get_cohere_api_key() -> str:
    """Get Cohere API key from environment."""
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        # Check config file as fallback
        config_path = Path.home() / ".agent-core" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    api_key = config.get("cohere", {}).get("api_key")
            except Exception:
                pass
    if not api_key:
        raise ValueError(
            "COHERE_API_KEY not set. Run: export COHERE_API_KEY='your-key-here'\n"
            "Get your key at: https://dashboard.cohere.com/api-keys"
        )
    return api_key


class QdrantDB:
    """Vector database for semantic search with Cohere embeddings."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
    ):
        self.host = host
        self.port = port
        self._client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None
        self._cohere_client: Optional[cohere.Client] = None
        self._initialized = False

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            )
        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere not installed. Run: pip install cohere"
            )

    @property
    def cohere_client(self) -> 'cohere.Client':
        """Lazy-load the Cohere client."""
        if self._cohere_client is None:
            self._check_dependencies()
            api_key = get_cohere_api_key()
            self._cohere_client = cohere.Client(api_key)
        return self._cohere_client

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
                else:
                    # Check if dimension matches (migration needed if not)
                    info = await self.async_client.get_collection(name)
                    if hasattr(info.config.params, 'vectors'):
                        current_size = info.config.params.vectors.size
                    else:
                        current_size = info.config.params.size

                    if current_size != config["vector_size"]:
                        print(f"⚠️  Collection '{name}' has dimension {current_size}, expected {config['vector_size']}")
                        print("   Run migration: python -m storage.migrate --recreate")
            except Exception as e:
                print(f"Warning: Could not create collection {name}: {e}")

        self._initialized = True

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text using Cohere."""
        response = self.cohere_client.embed(
            texts=[text],
            model=EMBEDDING_MODEL,
            input_type="search_document",
            truncate="END"
        )
        return response.embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using Cohere."""
        if not texts:
            return []

        # Cohere has a limit of 96 texts per request
        all_embeddings = []
        batch_size = 96

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.cohere_client.embed(
                texts=batch,
                model=EMBEDDING_MODEL,
                input_type="search_document",
                truncate="END"
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query using Cohere."""
        response = self.cohere_client.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="search_query",
            truncate="END"
        )
        return response.embeddings[0]

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 10,
        content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cohere rerank model.

        Args:
            query: The search query
            documents: List of documents with content to rerank
            top_n: Number of top results to return
            content_key: Key in document dict containing text to rerank

        Returns:
            Reranked documents with relevance_score added
        """
        if not documents:
            return []

        # Extract text content for reranking
        texts = [doc.get(content_key, str(doc)) for doc in documents]

        response = self.cohere_client.rerank(
            query=query,
            documents=texts,
            model=RERANK_MODEL,
            top_n=min(top_n, len(documents))
        )

        # Build reranked results
        reranked = []
        for result in response.results:
            doc = documents[result.index].copy()
            doc["relevance_score"] = result.relevance_score
            doc["rerank_index"] = result.index
            reranked.append(doc)

        return reranked

    def _generate_id(self, text: str, prefix: str = "") -> str:
        """Generate deterministic ID from text (for deduplication)."""
        content = f"{prefix}:{text}" if prefix else text
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_unique_id(self, prefix: str = "") -> str:
        """Generate unique ID using UUID (for temporal records)."""
        unique_id = uuid.uuid4().hex[:12]
        return f"{prefix}-{unique_id}" if prefix else unique_id

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
        filter_session: Optional[str] = None,
        rerank: bool = True,
        rerank_top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for findings with optional reranking.

        Args:
            query: Search query
            limit: Number of results to return
            min_score: Minimum similarity score
            filter_type: Filter by finding type
            filter_project: Filter by project
            filter_session: Filter by session
            rerank: Whether to rerank results using Cohere
            rerank_top_n: Number of results after reranking (defaults to limit)
        """
        # Use query-specific embedding
        embedding = self.embed_query(query)

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

        # Fetch more results for reranking
        fetch_limit = limit * 3 if rerank else limit

        results = await self.async_client.query_points(
            collection_name="findings",
            query=embedding,
            query_filter=search_filter,
            limit=fetch_limit,
            score_threshold=min_score
        )

        findings = [
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

        # Rerank if enabled and we have results
        if rerank and findings:
            rerank_n = rerank_top_n or limit
            findings = self.rerank(query, findings, top_n=rerank_n)

        return findings[:limit]

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
        filter_project: Optional[str] = None,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """Semantic search for sessions with optional reranking."""
        embedding = self.embed_query(query)

        conditions = []
        if filter_project:
            conditions.append(
                FieldCondition(key="project", match=MatchValue(value=filter_project))
            )

        search_filter = Filter(must=conditions) if conditions else None

        fetch_limit = limit * 3 if rerank else limit

        results = await self.async_client.query_points(
            collection_name="sessions",
            query=embedding,
            query_filter=search_filter,
            limit=fetch_limit,
            score_threshold=min_score
        )

        sessions = [
            {
                "session_id": r.payload.get("session_id"),
                "topic": r.payload.get("topic"),
                "project": r.payload.get("project"),
                "status": r.payload.get("status"),
                "finding_count": r.payload.get("finding_count"),
                "content": r.payload.get("topic"),  # For reranking
                "score": r.score,
            }
            for r in results.points
        ]

        if rerank and sessions:
            sessions = self.rerank(query, sessions, top_n=limit, content_key="topic")

        return sessions[:limit]

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
        filter_source: Optional[str] = None,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """Semantic search for context packs with optional reranking."""
        embedding = self.embed_query(query)

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

        fetch_limit = limit * 3 if rerank else limit

        results = await self.async_client.query_points(
            collection_name="packs",
            query=embedding,
            query_filter=search_filter,
            limit=fetch_limit,
            score_threshold=min_score
        )

        packs = [
            {
                "pack_id": r.payload.get("pack_id"),
                "name": r.payload.get("name"),
                "type": r.payload.get("type"),
                "source": r.payload.get("source"),
                "tokens": r.payload.get("tokens"),
                "content": r.payload.get("name", ""),  # For reranking
                "score": r.score,
            }
            for r in results.points
        ]

        if rerank and packs:
            packs = self.rerank(query, packs, top_n=limit, content_key="name")

        return packs[:limit]

    # --- Session Outcome Operations ---

    async def upsert_outcome(
        self,
        outcome_id: str,
        intent: str,
        metadata: Dict[str, Any]
    ):
        """Store a session outcome with its embedding."""
        # Embed the intent (what the session was about)
        embedding = self.embed(intent)

        await self.async_client.upsert(
            collection_name="session_outcomes",
            points=[
                PointStruct(
                    id=self._generate_id(outcome_id, "outcome"),
                    vector=embedding,
                    payload={
                        "outcome_id": outcome_id,
                        "session_id": metadata.get("session_id"),
                        "intent": intent[:500],  # Truncate for storage
                        "outcome": metadata.get("outcome"),
                        "quality": metadata.get("quality"),
                        "model_efficiency": metadata.get("model_efficiency"),
                        "models_used": metadata.get("models_used"),
                        "date": metadata.get("date"),
                        "messages": metadata.get("messages"),
                        "tools": metadata.get("tools"),
                    }
                )
            ]
        )

    async def upsert_outcomes_batch(
        self,
        outcomes: List[Dict[str, Any]]
    ) -> int:
        """Store multiple session outcomes with embeddings."""
        if not outcomes:
            return 0

        # Generate embeddings for intents
        texts = [o.get("intent", o.get("title", "")) for o in outcomes]
        embeddings = self.embed_batch(texts)

        points = [
            PointStruct(
                id=self._generate_id(o.get("session_id", f"outcome-{i}"), "outcome"),
                vector=emb,
                payload={
                    "outcome_id": o.get("session_id"),
                    "session_id": o.get("session_id"),
                    "intent": o.get("intent", "")[:500],
                    "outcome": o.get("outcome"),
                    "quality": o.get("quality"),
                    "model_efficiency": o.get("model_efficiency"),
                    "models_used": o.get("models_used"),
                    "date": o.get("date"),
                    "messages": o.get("messages"),
                    "tools": o.get("tools"),
                }
            )
            for i, (o, emb) in enumerate(zip(outcomes, embeddings))
        ]

        await self.async_client.upsert(
            collection_name="session_outcomes",
            points=points
        )

        return len(points)

    async def search_outcomes(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        filter_outcome: Optional[str] = None,
        min_quality: Optional[float] = None,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for session outcomes with optional reranking.

        Args:
            query: Search query (e.g., "implement authentication system")
            limit: Number of results to return
            min_score: Minimum similarity score
            filter_outcome: Filter by outcome type ('success', 'partial', 'failed')
            min_quality: Minimum quality score (1-5)
            rerank: Whether to rerank results using Cohere
        """
        embedding = self.embed_query(query)

        # Build filter
        conditions = []
        if filter_outcome:
            conditions.append(
                FieldCondition(key="outcome", match=MatchValue(value=filter_outcome))
            )
        if min_quality:
            conditions.append(
                FieldCondition(key="quality", range=models.Range(gte=min_quality))
            )

        search_filter = Filter(must=conditions) if conditions else None

        # Fetch more results for reranking
        fetch_limit = limit * 3 if rerank else limit

        results = await self.async_client.query_points(
            collection_name="session_outcomes",
            query=embedding,
            query_filter=search_filter,
            limit=fetch_limit,
            score_threshold=min_score
        )

        outcomes = [
            {
                "outcome_id": r.payload.get("outcome_id"),
                "session_id": r.payload.get("session_id"),
                "intent": r.payload.get("intent"),
                "outcome": r.payload.get("outcome"),
                "quality": r.payload.get("quality"),
                "model_efficiency": r.payload.get("model_efficiency"),
                "models_used": r.payload.get("models_used"),
                "date": r.payload.get("date"),
                "messages": r.payload.get("messages"),
                "tools": r.payload.get("tools"),
                "content": r.payload.get("intent"),  # For reranking
                "score": r.score,
            }
            for r in results.points
        ]

        # Rerank if enabled
        if rerank and outcomes:
            outcomes = self.rerank(query, outcomes, top_n=limit, content_key="intent")

        return outcomes[:limit]

    # --- Cognitive State Operations ---

    async def upsert_cognitive_state(
        self,
        state_id: str,
        context: str,
        metadata: Dict[str, Any]
    ):
        """Store a cognitive state with its embedding."""
        embedding = self.embed(context)

        await self.async_client.upsert(
            collection_name="cognitive_states",
            points=[
                PointStruct(
                    id=self._generate_id(state_id, "cognitive"),
                    vector=embedding,
                    payload={
                        "state_id": state_id,
                        "mode": metadata.get("mode"),
                        "energy_level": metadata.get("energy_level"),
                        "flow_score": metadata.get("flow_score"),
                        "hour": metadata.get("hour"),
                        "day": metadata.get("day"),
                        "timestamp": metadata.get("timestamp"),
                        "predictions": metadata.get("predictions"),
                    }
                )
            ]
        )

    async def upsert_cognitive_states_batch(
        self,
        states: List[Dict[str, Any]]
    ) -> int:
        """Store multiple cognitive states with embeddings."""
        if not states:
            return 0

        # Create context strings for embedding
        texts = [
            f"{s.get('mode', '')} energy_{s.get('energy_level', 0):.2f} flow_{s.get('flow_score', 0):.2f} hour_{s.get('hour', 0)}"
            for s in states
        ]
        embeddings = self.embed_batch(texts)

        points = [
            PointStruct(
                id=self._generate_id(s.get("id", f"state-{i}"), "cognitive"),
                vector=emb,
                payload={
                    "state_id": s.get("id"),
                    "mode": s.get("mode"),
                    "energy_level": s.get("energy_level"),
                    "flow_score": s.get("flow_score"),
                    "hour": s.get("hour"),
                    "day": s.get("day"),
                    "timestamp": s.get("timestamp"),
                    "predictions": s.get("predictions"),
                }
            )
            for i, (s, emb) in enumerate(zip(states, embeddings))
        ]

        await self.async_client.upsert(
            collection_name="cognitive_states",
            points=points
        )

        return len(points)

    async def search_cognitive_states(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Semantic search for cognitive states."""
        embedding = self.embed_query(query)

        results = await self.async_client.query_points(
            collection_name="cognitive_states",
            query=embedding,
            limit=limit,
            score_threshold=min_score
        )

        return [
            {
                "state_id": r.payload.get("state_id"),
                "mode": r.payload.get("mode"),
                "energy_level": r.payload.get("energy_level"),
                "flow_score": r.payload.get("flow_score"),
                "hour": r.payload.get("hour"),
                "day": r.payload.get("day"),
                "timestamp": r.payload.get("timestamp"),
                "predictions": r.payload.get("predictions"),
                "score": r.score,
            }
            for r in results.points
        ]

    # --- Error Pattern Operations ---

    async def upsert_error_pattern(
        self,
        error_id: str,
        context: str,
        metadata: Dict[str, Any]
    ):
        """Store an error pattern with its embedding."""
        embedding = self.embed(context)

        await self.async_client.upsert(
            collection_name="error_patterns",
            points=[
                PointStruct(
                    id=self._generate_id(error_id, "error"),
                    vector=embedding,
                    payload={
                        "error_id": error_id,
                        "error_type": metadata.get("error_type"),
                        "context": context[:1000],
                        "solution": metadata.get("solution"),
                        "success_rate": metadata.get("success_rate"),
                    }
                )
            ]
        )

    async def upsert_error_patterns_batch(
        self,
        errors: List[Dict[str, Any]]
    ) -> int:
        """Store multiple error patterns with embeddings."""
        if not errors:
            return 0

        # Create context strings for embedding
        texts = [
            f"{e.get('error_type', '')} in {e.get('context', '')} solved_by {e.get('solution', '')}"
            for e in errors
        ]
        embeddings = self.embed_batch(texts)

        points = [
            PointStruct(
                id=self._generate_id(e.get("id", f"error-{i}"), "error"),
                vector=emb,
                payload={
                    "error_id": e.get("id"),
                    "error_type": e.get("error_type"),
                    "context": e.get("context", "")[:1000],
                    "solution": e.get("solution"),
                    "success_rate": e.get("success_rate", 0.0),
                }
            )
            for i, (e, emb) in enumerate(zip(errors, embeddings))
        ]

        await self.async_client.upsert(
            collection_name="error_patterns",
            points=points
        )

        return len(points)

    async def search_error_patterns(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        min_success_rate: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for error patterns."""
        embedding = self.embed_query(query)

        conditions = []
        if min_success_rate:
            conditions.append(
                FieldCondition(key="success_rate", range=models.Range(gte=min_success_rate))
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = await self.async_client.query_points(
            collection_name="error_patterns",
            query=embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=min_score
        )

        return [
            {
                "error_id": r.payload.get("error_id"),
                "error_type": r.payload.get("error_type"),
                "context": r.payload.get("context"),
                "solution": r.payload.get("solution"),
                "success_rate": r.payload.get("success_rate"),
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
        min_score: float = 0.4,
        rerank: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple collections with optional reranking."""
        if collections is None:
            collections = ["findings", "sessions", "packs"]

        embedding = self.embed_query(query)
        results = {}

        for collection in collections:
            if collection not in COLLECTIONS:
                continue

            fetch_limit = limit * 3 if rerank else limit

            search_results = await self.async_client.query_points(
                collection_name=collection,
                query=embedding,
                limit=fetch_limit,
                score_threshold=min_score
            )

            items = [
                {**r.payload, "score": r.score}
                for r in search_results.points
            ]

            # Rerank each collection
            if rerank and items:
                content_key = "content" if collection == "findings" else "topic" if collection == "sessions" else "name"
                items = self.rerank(query, items, top_n=limit, content_key=content_key)

            results[collection] = items[:limit]

        return results

    # --- Statistics ---

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = {
            "embedding_model": EMBEDDING_MODEL,
            "rerank_model": RERANK_MODEL,
            "embedding_dim": EMBEDDING_DIM,
        }

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
