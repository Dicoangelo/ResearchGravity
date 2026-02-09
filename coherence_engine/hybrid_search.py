"""
Coherence Engine — Hybrid Search (Semantic + BM25 with RRF)

Combines pgvector cosine similarity (embedding_768) with PostgreSQL
full-text search (tsvector/GIN) using Reciprocal Rank Fusion to produce
results that are both semantically relevant and keyword-precise.

RRF formula: score = sum(1 / (k + rank))  where k = 60

Usage:
    search = HybridSearch(pool)
    results = await search.search("sovereignty infrastructure")
    results = await search.search_cross_platform("UCW", exclude_platform="chatgpt")
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp_raw.embeddings import embed_single, _embedding_column

log = logging.getLogger("coherence.hybrid_search")

# RRF constant — standard value from Cormack et al. (2009)
RRF_K = 60


@dataclass
class HybridResult:
    """A result from hybrid search with combined RRF score."""
    event_id: str
    platform: str
    session_id: str
    preview: str
    rrf_score: float
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    semantic_similarity: Optional[float] = None
    bm25_score: Optional[float] = None
    coherence_sig: Optional[str] = None
    cognitive_mode: Optional[str] = None


class HybridSearch:
    """
    Hybrid search combining semantic vectors and BM25 full-text search.

    Both retrieval paths run independently, then results are merged
    using Reciprocal Rank Fusion (RRF) for robust ranking that
    benefits from the strengths of each method.

    Semantic search captures meaning and paraphrases.
    BM25 captures exact keyword matches and rare terms.
    """

    def __init__(self, pool):
        self._pool = pool

    async def search(
        self,
        query: str,
        limit: int = 20,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> List[HybridResult]:
        """
        Run hybrid search: semantic + BM25 with RRF fusion.

        Args:
            query: Natural language search query.
            limit: Maximum results to return.
            semantic_weight: Weight for semantic RRF scores (default 0.6).
            bm25_weight: Weight for BM25 RRF scores (default 0.4).

        Returns:
            List of HybridResult sorted by combined RRF score.
        """
        return await self._hybrid_search(
            query=query,
            limit=limit,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            exclude_platform=None,
        )

    async def search_cross_platform(
        self,
        query: str,
        exclude_platform: str,
        limit: int = 20,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> List[HybridResult]:
        """
        Hybrid search excluding a specific platform.

        Useful for finding cross-platform coherence: search for what
        was discussed on one platform and find matches on others.

        Args:
            query: Natural language search query.
            exclude_platform: Platform to exclude (e.g. "chatgpt").
            limit: Maximum results to return.
            semantic_weight: Weight for semantic RRF scores.
            bm25_weight: Weight for BM25 RRF scores.

        Returns:
            List of HybridResult sorted by combined RRF score.
        """
        return await self._hybrid_search(
            query=query,
            limit=limit,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            exclude_platform=exclude_platform,
        )

    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        semantic_weight: float,
        bm25_weight: float,
        exclude_platform: Optional[str],
    ) -> List[HybridResult]:
        """Internal: run both search paths and fuse with RRF."""
        if not self._pool:
            log.warning("No database pool available for hybrid search")
            return []

        # Fetch more candidates than final limit so RRF has enough to fuse
        candidate_limit = limit * 3

        # Run both searches in parallel
        semantic_results, bm25_results = await self._run_dual_search(
            query, candidate_limit, exclude_platform
        )

        log.info(
            f"Hybrid search '{query[:50]}': "
            f"{len(semantic_results)} semantic, {len(bm25_results)} BM25"
        )

        # Fuse results using Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

        # Sort by RRF score and trim to limit
        fused.sort(key=lambda r: r.rrf_score, reverse=True)
        return fused[:limit]

    async def _run_dual_search(
        self,
        query: str,
        candidate_limit: int,
        exclude_platform: Optional[str],
    ) -> tuple:
        """Run semantic and BM25 searches against the database."""
        col = _embedding_column

        # Embed query for semantic search (use "search_query" prefix per Nomic)
        query_vec = embed_single(query, prefix="search_query")
        vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"

        async with self._pool.acquire() as conn:
            # ── Semantic search (pgvector cosine distance) ──
            if exclude_platform:
                semantic_rows = await conn.fetch(
                    f"""SELECT ec.source_event_id,
                               ec.content_preview,
                               ce.platform,
                               ce.session_id,
                               ce.coherence_sig,
                               ce.cognitive_mode,
                               1 - (ec.{col} <=> $1::vector) AS similarity
                        FROM embedding_cache ec
                        JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                        WHERE ec.{col} IS NOT NULL
                          AND ce.platform != $2
                        ORDER BY ec.{col} <=> $1::vector
                        LIMIT $3""",
                    vec_str, exclude_platform, candidate_limit,
                )
            else:
                semantic_rows = await conn.fetch(
                    f"""SELECT ec.source_event_id,
                               ec.content_preview,
                               ce.platform,
                               ce.session_id,
                               ce.coherence_sig,
                               ce.cognitive_mode,
                               1 - (ec.{col} <=> $1::vector) AS similarity
                        FROM embedding_cache ec
                        JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                        WHERE ec.{col} IS NOT NULL
                        ORDER BY ec.{col} <=> $1::vector
                        LIMIT $2""",
                    vec_str, candidate_limit,
                )

            # ── BM25 search (ts_rank_cd + plainto_tsquery) ──
            if exclude_platform:
                bm25_rows = await conn.fetch(
                    """SELECT ec.source_event_id,
                              ec.content_preview,
                              ce.platform,
                              ce.session_id,
                              ce.coherence_sig,
                              ce.cognitive_mode,
                              ts_rank_cd(ec.content_tsv, plainto_tsquery('english', $1)) AS rank
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ec.content_tsv @@ plainto_tsquery('english', $1)
                         AND ce.platform != $2
                       ORDER BY rank DESC
                       LIMIT $3""",
                    query, exclude_platform, candidate_limit,
                )
            else:
                bm25_rows = await conn.fetch(
                    """SELECT ec.source_event_id,
                              ec.content_preview,
                              ce.platform,
                              ce.session_id,
                              ce.coherence_sig,
                              ce.cognitive_mode,
                              ts_rank_cd(ec.content_tsv, plainto_tsquery('english', $1)) AS rank
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ec.content_tsv @@ plainto_tsquery('english', $1)
                       ORDER BY rank DESC
                       LIMIT $2""",
                    query, candidate_limit,
                )

        return semantic_rows, bm25_rows

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list,
        bm25_results: list,
        semantic_weight: float,
        bm25_weight: float,
    ) -> List[HybridResult]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        For each document d appearing in a ranked list at position r:
            RRF_score(d) += weight * (1 / (k + r))

        where k = 60 (smoothing constant that prevents top-ranked documents
        from dominating the fused score).
        """
        # Build a map: event_id -> metadata + scores
        candidates: Dict[str, Dict[str, Any]] = {}

        # Process semantic results
        for rank, row in enumerate(semantic_results, start=1):
            eid = row["source_event_id"]
            if eid not in candidates:
                candidates[eid] = {
                    "event_id": eid,
                    "platform": row["platform"],
                    "session_id": row["session_id"],
                    "preview": row["content_preview"] or "",
                    "coherence_sig": row["coherence_sig"],
                    "cognitive_mode": row["cognitive_mode"],
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "semantic_similarity": None,
                    "bm25_score": None,
                }
            candidates[eid]["semantic_rank"] = rank
            candidates[eid]["semantic_similarity"] = round(float(row["similarity"]), 4)
            candidates[eid]["rrf_score"] += semantic_weight * (1.0 / (RRF_K + rank))

        # Process BM25 results
        for rank, row in enumerate(bm25_results, start=1):
            eid = row["source_event_id"]
            if eid not in candidates:
                candidates[eid] = {
                    "event_id": eid,
                    "platform": row["platform"],
                    "session_id": row["session_id"],
                    "preview": row["content_preview"] or "",
                    "coherence_sig": row["coherence_sig"],
                    "cognitive_mode": row["cognitive_mode"],
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "semantic_similarity": None,
                    "bm25_score": None,
                }
            candidates[eid]["bm25_rank"] = rank
            candidates[eid]["bm25_score"] = round(float(row["rank"]), 6)
            candidates[eid]["rrf_score"] += bm25_weight * (1.0 / (RRF_K + rank))

        # Convert to HybridResult objects
        return [
            HybridResult(
                event_id=c["event_id"],
                platform=c["platform"],
                session_id=c["session_id"],
                preview=c["preview"],
                rrf_score=round(c["rrf_score"], 6),
                semantic_rank=c["semantic_rank"],
                bm25_rank=c["bm25_rank"],
                semantic_similarity=c["semantic_similarity"],
                bm25_score=c["bm25_score"],
                coherence_sig=c["coherence_sig"],
                cognitive_mode=c["cognitive_mode"],
            )
            for c in candidates.values()
        ]
