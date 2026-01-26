"""
Unified Storage Engine

Combines SQLite (relational) and Qdrant (vector) for:
- Concurrent-safe writes from multiple agents
- Semantic search across all content
- UCW pack ingestion with provenance tracking
- Lineage and relationship queries
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from .sqlite_db import SQLiteDB, get_db
from .qdrant_db import QdrantDB, get_qdrant, QDRANT_AVAILABLE


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
    """

    def __init__(self):
        self.sqlite: Optional[SQLiteDB] = None
        self.qdrant: Optional[QdrantDB] = None
        self._initialized = False
        self._qdrant_enabled = QDRANT_AVAILABLE

    async def initialize(self):
        """Initialize both storage backends."""
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
                    print("Warning: Qdrant not responding, semantic search disabled")
                    self._qdrant_enabled = False
            except Exception as e:
                print(f"Warning: Qdrant initialization failed: {e}")
                print("Semantic search disabled. Run Qdrant to enable.")
                self._qdrant_enabled = False

        self._initialized = True

    async def close(self):
        """Close all connections."""
        if self.sqlite:
            await self.sqlite.close()
        if self.qdrant:
            await self.qdrant.close()

    # --- Session Operations ---

    async def store_session(
        self,
        session: Dict[str, Any],
        source: str = "local"
    ) -> str:
        """Store a session in SQLite and index in Qdrant."""
        # Store in SQLite
        session_id = await self.sqlite.store_session(session)

        # Track provenance
        await self.sqlite.track_provenance(
            entity_type="session",
            entity_id=session_id,
            source_type=source,
            metadata={"stored_at": datetime.now().isoformat()}
        )

        # Index in Qdrant for semantic search
        if self._qdrant_enabled and session.get("topic"):
            try:
                await self.qdrant.upsert_session(
                    session_id=session_id,
                    topic=session["topic"],
                    metadata={
                        "project": session.get("project"),
                        "status": session.get("status"),
                        "finding_count": session.get("finding_count", 0),
                        "url_count": session.get("url_count", 0),
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to index session in Qdrant: {e}")

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
        """Store a finding in SQLite and index in Qdrant."""
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

        # Index in Qdrant
        if self._qdrant_enabled:
            try:
                await self.qdrant.upsert_finding(
                    finding_id=finding_id,
                    content=finding["content"],
                    metadata={
                        "type": finding.get("type"),
                        "session_id": finding.get("session_id"),
                        "project": finding.get("project"),
                        "confidence": finding.get("confidence"),
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to index finding in Qdrant: {e}")

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
                print(f"Warning: Failed to batch index findings in Qdrant: {e}")

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
                print(f"Warning: Failed to index pack in Qdrant: {e}")

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
        Falls back to FTS if Qdrant is unavailable.
        """
        if self._qdrant_enabled:
            return await self.qdrant.semantic_search(
                query=query,
                collections=collections,
                limit=limit,
                min_score=min_score
            )
        else:
            # Fallback to SQLite FTS
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
        """Semantic search for findings."""
        if self._qdrant_enabled:
            return await self.qdrant.search_findings(
                query=query,
                limit=limit,
                min_score=min_score,
                filter_type=filter_type,
                filter_project=filter_project
            )
        else:
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
            try:
                await self.qdrant.upsert_outcome(
                    outcome_id=outcome_id,
                    intent=outcome.get("intent", outcome.get("title", "")),
                    metadata=outcome
                )
            except Exception as e:
                print(f"Warning: Failed to index outcome in Qdrant: {e}")

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
                print(f"Warning: Failed to batch index outcomes in Qdrant: {e}")

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
            try:
                context = f"{state.get('mode', '')} energy_{state.get('energy_level', 0):.2f} flow_{state.get('flow_score', 0):.2f}"
                await self.qdrant.upsert_cognitive_state(
                    state_id=state_id,
                    context=context,
                    metadata=state
                )
            except Exception as e:
                print(f"Warning: Failed to index cognitive state in Qdrant: {e}")

        return state_id

    async def store_cognitive_states_batch(self, states: List[Dict[str, Any]]) -> int:
        """Store multiple cognitive states."""
        count = await self.sqlite.store_cognitive_states_batch(states)

        if self._qdrant_enabled and states:
            try:
                await self.qdrant.upsert_cognitive_states_batch(states)
            except Exception as e:
                print(f"Warning: Failed to batch index cognitive states in Qdrant: {e}")

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
            try:
                context = f"{error.get('error_type', '')} in {error.get('context', '')} solved_by {error.get('solution', '')}"
                await self.qdrant.upsert_error_pattern(
                    error_id=error_id,
                    context=context,
                    metadata=error
                )
            except Exception as e:
                print(f"Warning: Failed to index error pattern in Qdrant: {e}")

        return error_id

    async def store_error_patterns_batch(self, errors: List[Dict[str, Any]]) -> int:
        """Store multiple error patterns."""
        count = await self.sqlite.store_error_patterns_batch(errors)

        if self._qdrant_enabled and errors:
            try:
                await self.qdrant.upsert_error_patterns_batch(errors)
            except Exception as e:
                print(f"Warning: Failed to batch index error patterns in Qdrant: {e}")

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
        }

        if self._qdrant_enabled:
            health["qdrant"] = await self.qdrant.health_check()

        return health


# Convenience function
async def get_engine() -> StorageEngine:
    """Get initialized storage engine."""
    engine = StorageEngine()
    await engine.initialize()
    return engine
