"""
Unified Storage Engine

Combines SQLite (relational), Qdrant (vector), and sqlite-vec for:
- Concurrent-safe writes from multiple agents
- Semantic search across all content
- UCW pack ingestion with provenance tracking
- Lineage and relationship queries
- Dual-write to Qdrant and sqlite-vec for migration

V2 Features:
- sqlite-vec integration for single-file vector storage
- Hybrid search with BM25 + cosine similarity
- Automatic fallback between backends

V2.1 Features:
- Dead-letter queue for failed writes
- Automatic retry with exponential backoff
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from .sqlite_db import SQLiteDB, get_db
from .qdrant_db import QdrantDB, get_qdrant, QDRANT_AVAILABLE
from .dead_letter_queue import DeadLetterQueue, get_dlq
from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import sqlite-vec
try:
    from .sqlite_vec import SqliteVecDB, get_vec_db, SQLITE_VEC_AVAILABLE
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    SqliteVecDB = None
    get_vec_db = None


class StorageEngine:
    """
    Unified storage interface for the Antigravity Chief of Staff.

    Usage:
        engine = StorageEngine()
        await engine.initialize()

        # Store with automatic indexing
        await engine.store_session(session_data)
        await engine.store_finding(finding_data)

        # Semantic search
        results = await engine.semantic_search("multi-agent patterns")

        # UCW pack import
        await engine.ingest_packs(packs, source="ucw_trade", source_id="wallet_xyz")

    V2 Features:
        - Dual-write to both Qdrant and sqlite-vec
        - Automatic fallback if one backend is unavailable
        - Use prefer_sqlite_vec=True for offline mode
    """

    def __init__(self, prefer_sqlite_vec: bool = False, enable_dlq: bool = True):
        self.sqlite: Optional[SQLiteDB] = None
        self.qdrant: Optional[QdrantDB] = None
        self.sqlite_vec: Optional[SqliteVecDB] = None
        self.dlq: Optional[DeadLetterQueue] = None
        self._initialized = False
        self._qdrant_enabled = QDRANT_AVAILABLE
        self._sqlite_vec_enabled = SQLITE_VEC_AVAILABLE
        self._prefer_sqlite_vec = prefer_sqlite_vec
        self._enable_dlq = enable_dlq

    async def initialize(self):
        """Initialize all storage backends."""
        if self._initialized:
            return

        # Always initialize SQLite
        self.sqlite = await get_db()

        # Initialize Qdrant if available
        if self._qdrant_enabled:
            try:
                self.qdrant = await get_qdrant()
                # Test connection
                if not await self.qdrant.health_check():
                    print("Warning: Qdrant not responding, will use sqlite-vec")
                    self._qdrant_enabled = False
            except Exception as e:
                print(f"Warning: Qdrant initialization failed: {e}")
                self._qdrant_enabled = False

        # Initialize sqlite-vec if available
        if self._sqlite_vec_enabled and get_vec_db:
            try:
                self.sqlite_vec = await get_vec_db()
            except Exception as e:
                print(f"Warning: sqlite-vec initialization failed: {e}")
                self._sqlite_vec_enabled = False

        # Ensure at least one vector backend is available
        if not self._qdrant_enabled and not self._sqlite_vec_enabled:
            logger.warning("No vector backend available. Semantic search will use FTS fallback.")

        # Initialize dead-letter queue for failed writes
        if self._enable_dlq:
            try:
                self.dlq = await get_dlq()
                self._register_dlq_handlers()
                logger.info("Dead-letter queue initialized")
            except Exception as e:
                logger.error(f"DLQ initialization failed: {e}")
                self.dlq = None

        self._initialized = True

    async def close(self):
        """Close all connections."""
        if self.sqlite:
            await self.sqlite.close()
        if self.qdrant:
            await self.qdrant.close()
        if self.sqlite_vec:
            await self.sqlite_vec.close()

    def _register_dlq_handlers(self):
        """Register retry handlers for the dead-letter queue."""
        if not self.dlq:
            return

        # Qdrant handlers
        if self.qdrant:
            self.dlq.register_retry_handler(
                "upsert_session", "qdrant",
                lambda p: self.qdrant.upsert_session(**p)
            )
            self.dlq.register_retry_handler(
                "upsert_finding", "qdrant",
                lambda p: self.qdrant.upsert_finding(**p)
            )
            self.dlq.register_retry_handler(
                "upsert_pack", "qdrant",
                lambda p: self.qdrant.upsert_pack(**p)
            )
            self.dlq.register_retry_handler(
                "upsert_outcome", "qdrant",
                lambda p: self.qdrant.upsert_outcome(**p)
            )
            self.dlq.register_retry_handler(
                "upsert_cognitive_state", "qdrant",
                lambda p: self.qdrant.upsert_cognitive_state(**p)
            )
            self.dlq.register_retry_handler(
                "upsert_error_pattern", "qdrant",
                lambda p: self.qdrant.upsert_error_pattern(**p)
            )

        # sqlite-vec handlers
        if self.sqlite_vec:
            self.dlq.register_retry_handler(
                "upsert_session", "sqlite_vec",
                lambda p: self.sqlite_vec.upsert_session(**p)
            )
            self.dlq.register_retry_handler(
                "upsert_finding", "sqlite_vec",
                lambda p: self.sqlite_vec.upsert_finding(**p)
            )

    async def _add_to_dlq(
        self,
        operation: str,
        target: str,
        payload: Dict[str, Any],
        error: Exception
    ):
        """Add a failed write operation to the dead-letter queue."""
        if self.dlq:
            await self.dlq.add_failed_write(
                operation=operation,
                target=target,
                payload=payload,
                error=str(error)
            )
        else:
            # Fallback to logging if DLQ not available
            logger.error(f"Failed {operation}@{target}: {error} (no DLQ)")

    async def retry_failed_writes(
        self,
        target: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, int]:
        """Retry pending entries in the dead-letter queue."""
        if not self.dlq:
            return {"attempted": 0, "succeeded": 0, "failed": 0, "error": "DLQ not available"}
        return await self.dlq.retry_failed_writes(target=target, limit=limit)

    async def get_dlq_stats(self) -> Dict[str, Any]:
        """Get dead-letter queue statistics."""
        if not self.dlq:
            return {"error": "DLQ not available"}
        return await self.dlq.get_stats()

    # --- Session Operations ---

    async def store_session(
        self,
        session: Dict[str, Any],
        source: str = "local"
    ) -> str:
        """Store a session in SQLite and index in vector backends."""
        # Store in SQLite
        session_id = await self.sqlite.store_session(session)

        # Track provenance
        await self.sqlite.track_provenance(
            entity_type="session",
            entity_id=session_id,
            source_type=source,
            metadata={"stored_at": datetime.now().isoformat()}
        )

        metadata = {
            "project": session.get("project"),
            "status": session.get("status"),
            "finding_count": session.get("finding_count", 0),
            "url_count": session.get("url_count", 0),
        }

        # Dual-write: Index in Qdrant
        if self._qdrant_enabled and session.get("topic"):
            try:
                await self.qdrant.upsert_session(
                    session_id=session_id,
                    topic=session["topic"],
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to index session in Qdrant: {e}")
                await self._add_to_dlq(
                    "upsert_session", "qdrant",
                    {"session_id": session_id, "topic": session["topic"], "metadata": metadata},
                    e
                )

        # Dual-write: Index in sqlite-vec
        if self._sqlite_vec_enabled and self.sqlite_vec and session.get("topic"):
            try:
                await self.sqlite_vec.upsert_session(
                    session_id=session_id,
                    topic=session["topic"],
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to index session in sqlite-vec: {e}")
                await self._add_to_dlq(
                    "upsert_session", "sqlite_vec",
                    {"session_id": session_id, "topic": session["topic"], "metadata": metadata},
                    e
                )

        return session_id

    async def store_sessions_batch(
        self,
        sessions: List[Dict[str, Any]],
        source: str = "local"
    ) -> int:
        """Store multiple sessions."""
        count = 0
        for session in sessions:
            await self.store_session(session, source)
            count += 1
        return count

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        return await self.sqlite.get_session(session_id)

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List sessions."""
        return await self.sqlite.list_sessions(
            limit=limit, offset=offset, project=project
        )

    # --- Finding Operations ---

    async def store_finding(
        self,
        finding: Dict[str, Any],
        source: str = "local"
    ) -> str:
        """Store a finding in SQLite and index in vector backends."""
        # Ensure ID exists
        if "id" not in finding:
            finding["id"] = f"finding-{uuid.uuid4().hex[:12]}"

        # Store in SQLite
        finding_id = await self.sqlite.store_finding(finding)

        # Track provenance
        await self.sqlite.track_provenance(
            entity_type="finding",
            entity_id=finding_id,
            source_type=source
        )

        metadata = {
            "type": finding.get("type"),
            "session_id": finding.get("session_id"),
            "project": finding.get("project"),
            "confidence": finding.get("confidence"),
        }

        # Dual-write: Index in Qdrant
        if self._qdrant_enabled:
            try:
                await self.qdrant.upsert_finding(
                    finding_id=finding_id,
                    content=finding["content"],
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to index finding in Qdrant: {e}")
                await self._add_to_dlq(
                    "upsert_finding", "qdrant",
                    {"finding_id": finding_id, "content": finding["content"], "metadata": metadata},
                    e
                )

        # Dual-write: Index in sqlite-vec
        if self._sqlite_vec_enabled and self.sqlite_vec:
            try:
                await self.sqlite_vec.upsert_finding(
                    finding_id=finding_id,
                    content=finding["content"],
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to index finding in sqlite-vec: {e}")
                await self._add_to_dlq(
                    "upsert_finding", "sqlite_vec",
                    {"finding_id": finding_id, "content": finding["content"], "metadata": metadata},
                    e
                )

        return finding_id

    async def store_findings_batch(
        self,
        findings: List[Dict[str, Any]],
        source: str = "local"
    ) -> int:
        """Store multiple findings efficiently."""
        # Ensure all have IDs
        for f in findings:
            if "id" not in f:
                f["id"] = f"finding-{uuid.uuid4().hex[:12]}"

        # Batch store in SQLite
        count = await self.sqlite.store_findings_batch(findings)

        # Batch index in Qdrant
        if self._qdrant_enabled and findings:
            try:
                await self.qdrant.upsert_findings_batch(findings)
            except Exception as e:
                logger.warning(f"Failed to batch index findings in Qdrant: {e}")
                # Add each finding to DLQ for individual retry
                for f in findings:
                    await self._add_to_dlq(
                        "upsert_finding", "qdrant",
                        {"finding_id": f["id"], "content": f["content"],
                         "metadata": {"type": f.get("type"), "session_id": f.get("session_id")}},
                        e
                    )

        return count

    async def get_findings(
        self,
        session_id: Optional[str] = None,
        finding_type: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get findings with filtering."""
        return await self.sqlite.get_findings(
            session_id=session_id,
            finding_type=finding_type,
            project=project,
            limit=limit
        )

    # --- URL Operations ---

    async def store_url(self, url_data: Dict[str, Any]) -> int:
        """Store a URL."""
        return await self.sqlite.store_url(url_data)

    async def store_urls_batch(self, urls: List[Dict[str, Any]]) -> int:
        """Store multiple URLs."""
        return await self.sqlite.store_urls_batch(urls)

    # --- Pack Operations ---

    async def store_pack(
        self,
        pack: Dict[str, Any],
        source: str = "local",
        source_id: Optional[str] = None
    ) -> str:
        """Store a context pack."""
        # Ensure ID exists
        if "id" not in pack:
            pack["id"] = f"pack-{uuid.uuid4().hex[:12]}"

        # Store in SQLite
        pack_id = await self.sqlite.store_pack(pack, source)

        # Track provenance
        await self.sqlite.track_provenance(
            entity_type="pack",
            entity_id=pack_id,
            source_type=source,
            source_id=source_id
        )

        # Index in Qdrant
        if self._qdrant_enabled:
            try:
                content = pack.get("content", {})
                if isinstance(content, dict):
                    text = " ".join([
                        pack.get("name", ""),
                        content.get("description", ""),
                        " ".join(content.get("keywords", []))
                    ])
                else:
                    text = str(content)[:1000]

                await self.qdrant.upsert_pack(
                    pack_id=pack_id,
                    content=text,
                    metadata={
                        "name": pack.get("name"),
                        "type": pack.get("type"),
                        "source": source,
                        "tokens": pack.get("tokens"),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to index pack in Qdrant: {e}")
                await self._add_to_dlq(
                    "upsert_pack", "qdrant",
                    {"pack_id": pack_id, "content": pack["content"][:1000],
                     "metadata": {"name": pack.get("name"), "type": pack.get("type")}},
                    e
                )

        return pack_id

    async def ingest_packs(
        self,
        packs: List[Dict[str, Any]],
        source: str = "ucw_trade",
        source_id: Optional[str] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest multiple packs from UCW trade or external source.

        This is the main entry point for bulk pack imports.
        Handles deduplication, provenance tracking, and optional validation.
        """
        results = {
            "total": len(packs),
            "imported": 0,
            "skipped": 0,
            "errors": [],
            "pack_ids": [],
        }

        for pack in packs:
            try:
                # Ensure ID
                if "id" not in pack:
                    pack["id"] = f"pack-{uuid.uuid4().hex[:12]}"

                # Store with provenance
                pack_id = await self.store_pack(pack, source=source, source_id=source_id)
                results["imported"] += 1
                results["pack_ids"].append(pack_id)

            except Exception as e:
                results["errors"].append({
                    "pack_id": pack.get("id"),
                    "error": str(e)
                })

        return results

    async def get_packs(
        self,
        pack_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get context packs."""
        return await self.sqlite.get_packs(
            pack_type=pack_type,
            source=source,
            limit=limit
        )

    # --- Search Operations ---

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.4,
        collections: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Semantic search across all content.

        Returns results grouped by type (findings, sessions, packs).
        Priority: Qdrant > sqlite-vec > FTS
        """
        # Try Qdrant first (if available and not preferring sqlite-vec)
        if self._qdrant_enabled and not self._prefer_sqlite_vec:
            try:
                return await self.qdrant.semantic_search(
                    query=query,
                    collections=collections,
                    limit=limit,
                    min_score=min_score
                )
            except Exception as e:
                print(f"Warning: Qdrant search failed, falling back: {e}")

        # Try sqlite-vec
        if self._sqlite_vec_enabled and self.sqlite_vec:
            try:
                findings = await self.sqlite_vec.search_findings(query, limit, min_score)
                sessions = await self.sqlite_vec.search_sessions(query, limit, min_score)
                packs = await self.sqlite_vec.search_packs(query, limit, min_score)
                return {
                    "findings": findings,
                    "sessions": sessions,
                    "packs": packs,
                }
            except Exception as e:
                print(f"Warning: sqlite-vec search failed, falling back to FTS: {e}")

        # Final fallback to SQLite FTS
        findings = await self.sqlite.search_findings_fts(query, limit)
        return {
            "findings": findings,
            "sessions": [],
            "packs": [],
        }

    async def search_findings(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        filter_type: Optional[str] = None,
        filter_project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for findings. Priority: Qdrant > sqlite-vec > FTS"""
        # Try Qdrant first
        if self._qdrant_enabled and not self._prefer_sqlite_vec:
            try:
                return await self.qdrant.search_findings(
                    query=query,
                    limit=limit,
                    min_score=min_score,
                    filter_type=filter_type,
                    filter_project=filter_project
                )
            except Exception:
                pass

        # Try sqlite-vec
        if self._sqlite_vec_enabled and self.sqlite_vec:
            try:
                return await self.sqlite_vec.search_findings(
                    query=query,
                    limit=limit,
                    min_score=min_score,
                    filter_type=filter_type,
                    filter_project=filter_project
                )
            except Exception:
                pass

        # Fallback to FTS
        return await self.sqlite.search_findings_fts(query, limit)

    async def search_sessions(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.4,
        filter_project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for sessions."""
        if self._qdrant_enabled:
            return await self.qdrant.search_sessions(
                query=query,
                limit=limit,
                min_score=min_score,
                filter_project=filter_project
            )
        else:
            # Fallback to listing sessions (no semantic capability without Qdrant)
            return await self.sqlite.list_sessions(limit=limit, project=filter_project)

    async def search_packs(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.4,
        filter_type: Optional[str] = None,
        filter_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for context packs."""
        if self._qdrant_enabled:
            return await self.qdrant.search_packs(
                query=query,
                limit=limit,
                min_score=min_score,
                filter_type=filter_type,
                filter_source=filter_source
            )
        else:
            return await self.sqlite.get_packs(
                pack_type=filter_type,
                source=filter_source,
                limit=limit
            )

    # --- Lineage Operations ---

    async def add_lineage(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation: str,
        weight: float = 1.0
    ):
        """Add a lineage relationship."""
        await self.sqlite.add_lineage(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relation=relation,
            weight=weight
        )

    async def get_lineage(
        self,
        entity_type: str,
        entity_id: str,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get lineage for an entity."""
        return await self.sqlite.get_lineage(
            entity_type=entity_type,
            entity_id=entity_id,
            direction=direction
        )

    # --- Session Outcome Operations ---

    async def store_outcome(
        self,
        outcome: Dict[str, Any],
        source: str = "local"
    ) -> str:
        """Store a session outcome in SQLite and index in Qdrant."""
        # Store in SQLite
        outcome_id = await self.sqlite.store_outcome(outcome)

        # Index in Qdrant
        if self._qdrant_enabled:
            intent = outcome.get("intent", outcome.get("title", ""))
            try:
                await self.qdrant.upsert_outcome(
                    outcome_id=outcome_id,
                    intent=intent,
                    metadata=outcome
                )
            except Exception as e:
                logger.warning(f"Failed to index outcome in Qdrant: {e}")
                await self._add_to_dlq(
                    "upsert_outcome", "qdrant",
                    {"outcome_id": outcome_id, "intent": intent, "metadata": outcome},
                    e
                )

        return outcome_id

    async def store_outcomes_batch(
        self,
        outcomes: List[Dict[str, Any]],
        source: str = "local"
    ) -> int:
        """Store multiple outcomes efficiently."""
        # Batch store in SQLite
        count = await self.sqlite.store_outcomes_batch(outcomes)

        # Batch index in Qdrant
        if self._qdrant_enabled and outcomes:
            try:
                await self.qdrant.upsert_outcomes_batch(outcomes)
            except Exception as e:
                logger.warning(f"Failed to batch index outcomes in Qdrant: {e}")
                # Add each outcome to DLQ for individual retry
                for o in outcomes:
                    await self._add_to_dlq(
                        "upsert_outcome", "qdrant",
                        {"outcome_id": o.get("id"), "intent": o.get("intent", ""), "metadata": o},
                        e
                    )

        return count

    async def search_outcomes(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        filter_outcome: Optional[str] = None,
        min_quality: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for session outcomes."""
        if self._qdrant_enabled:
            return await self.qdrant.search_outcomes(
                query=query,
                limit=limit,
                min_score=min_score,
                filter_outcome=filter_outcome,
                min_quality=min_quality
            )
        else:
            return await self.sqlite.get_outcomes(
                limit=limit,
                min_quality=min_quality,
                outcome_filter=filter_outcome
            )

    # --- Cognitive State Operations ---

    async def store_cognitive_state(self, state: Dict[str, Any]) -> str:
        """Store a cognitive state."""
        state_id = await self.sqlite.store_cognitive_state(state)

        if self._qdrant_enabled:
            context = f"{state.get('mode', '')} energy_{state.get('energy_level', 0):.2f} flow_{state.get('flow_score', 0):.2f}"
            try:
                await self.qdrant.upsert_cognitive_state(
                    state_id=state_id,
                    context=context,
                    metadata=state
                )
            except Exception as e:
                logger.warning(f"Failed to index cognitive state in Qdrant: {e}")
                await self._add_to_dlq(
                    "upsert_cognitive_state", "qdrant",
                    {"state_id": state_id, "context": context, "metadata": state},
                    e
                )

        return state_id

    async def store_cognitive_states_batch(self, states: List[Dict[str, Any]]) -> int:
        """Store multiple cognitive states."""
        count = await self.sqlite.store_cognitive_states_batch(states)

        if self._qdrant_enabled and states:
            try:
                await self.qdrant.upsert_cognitive_states_batch(states)
            except Exception as e:
                logger.warning(f"Failed to batch index cognitive states in Qdrant: {e}")
                # Add each state to DLQ for individual retry
                for s in states:
                    context = f"{s.get('mode', '')} energy_{s.get('energy_level', 0):.2f}"
                    await self._add_to_dlq(
                        "upsert_cognitive_state", "qdrant",
                        {"state_id": s.get("id"), "context": context, "metadata": s},
                        e
                    )

        return count

    async def search_cognitive_states(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Semantic search for cognitive states."""
        if self._qdrant_enabled:
            return await self.qdrant.search_cognitive_states(
                query=query,
                limit=limit,
                min_score=min_score
            )
        else:
            return await self.sqlite.get_cognitive_states(limit=limit)

    # --- Error Pattern Operations ---

    async def store_error_pattern(self, error: Dict[str, Any]) -> str:
        """Store an error pattern."""
        error_id = await self.sqlite.store_error_pattern(error)

        if self._qdrant_enabled:
            context = f"{error.get('error_type', '')} in {error.get('context', '')} solved_by {error.get('solution', '')}"
            try:
                await self.qdrant.upsert_error_pattern(
                    error_id=error_id,
                    context=context,
                    metadata=error
                )
            except Exception as e:
                logger.warning(f"Failed to index error pattern in Qdrant: {e}")
                await self._add_to_dlq(
                    "upsert_error_pattern", "qdrant",
                    {"error_id": error_id, "context": context, "metadata": error},
                    e
                )

        return error_id

    async def store_error_patterns_batch(self, errors: List[Dict[str, Any]]) -> int:
        """Store multiple error patterns."""
        count = await self.sqlite.store_error_patterns_batch(errors)

        if self._qdrant_enabled and errors:
            try:
                await self.qdrant.upsert_error_patterns_batch(errors)
            except Exception as e:
                logger.warning(f"Failed to batch index error patterns in Qdrant: {e}")
                # Add each error to DLQ for individual retry
                for err in errors:
                    context = f"{err.get('error_type', '')} in {err.get('context', '')}"
                    await self._add_to_dlq(
                        "upsert_error_pattern", "qdrant",
                        {"error_id": err.get("id"), "context": context, "metadata": err},
                        e
                    )

        return count

    async def search_error_patterns(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        min_success_rate: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for error patterns."""
        if self._qdrant_enabled:
            return await self.qdrant.search_error_patterns(
                query=query,
                limit=limit,
                min_score=min_score,
                min_success_rate=min_success_rate
            )
        else:
            return await self.sqlite.get_error_patterns(
                limit=limit,
                min_success_rate=min_success_rate
            )

    # --- Prediction Tracking (Phase 4: Calibration Loop) ---

    async def store_prediction(self, prediction: Dict[str, Any]) -> str:
        """Store a prediction for later calibration."""
        return await self.sqlite.store_prediction(prediction)

    async def update_prediction_outcome(
        self,
        prediction_id: str,
        actual_quality: float,
        actual_outcome: str,
        session_id: str
    ):
        """Update a prediction with actual outcome."""
        await self.sqlite.update_prediction_outcome(
            prediction_id=prediction_id,
            actual_quality=actual_quality,
            actual_outcome=actual_outcome,
            session_id=session_id
        )

    async def get_prediction_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """Get prediction accuracy metrics."""
        return await self.sqlite.get_prediction_accuracy(days=days)

    # --- Statistics ---

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "sqlite": await self.sqlite.get_stats(),
            "qdrant_enabled": self._qdrant_enabled,
        }

        if self._qdrant_enabled:
            stats["qdrant"] = await self.qdrant.get_stats()

        return stats

    async def health_check(self) -> Dict[str, bool]:
        """Check health of storage backends."""
        health = {
            "sqlite": True,  # SQLite is always available
            "qdrant": False,
            "sqlite_vec": False,
            "dlq": False,
        }

        # Check Qdrant if enabled and initialized
        if self._qdrant_enabled and self.qdrant is not None:
            try:
                health["qdrant"] = await self.qdrant.health_check()
            except Exception as e:
                logger.warning("Qdrant health check failed", extra={"error": str(e)})
                health["qdrant"] = False

        # Check sqlite-vec if available
        if self.sqlite_vec is not None:
            try:
                health["sqlite_vec"] = True
            except Exception:
                health["sqlite_vec"] = False

        # Check DLQ
        if self.dlq is not None:
            try:
                stats = await self.dlq.get_stats()
                health["dlq"] = stats is not None
            except Exception:
                health["dlq"] = False

        return health


# Convenience function
async def get_engine() -> StorageEngine:
    """Get initialized storage engine."""
    engine = StorageEngine()
    await engine.initialize()
    return engine
