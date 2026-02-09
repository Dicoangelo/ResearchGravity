"""
Coherence Engine — Similarity Search

Fast cross-platform similarity search using pgvector HNSW index.
Falls back to brute-force cosine similarity if pgvector unavailable.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp_raw.embeddings import embed_single, cosine_similarity, _embedding_column
from . import config as cfg

import logging

log = logging.getLogger("coherence.similarity")


@dataclass
class SimilarityResult:
    """A similar event found by similarity search."""
    event_id: str
    platform: str
    session_id: str
    similarity: float
    preview: str
    coherence_sig: Optional[str] = None
    light_layer: Optional[Dict] = None
    instinct_layer: Optional[Dict] = None
    timestamp_ns: int = 0


class SimilarityIndex:
    """
    Similarity search across embedded cognitive events.

    Uses pgvector's <=> operator for cosine distance when available.
    The HNSW index on embedding_cache makes this O(log n).
    """

    def __init__(self, pool):
        self._pool = pool

    async def find_similar(
        self,
        query_embedding: List[float],
        threshold: float = 0.85,
        limit: int = 20,
        exclude_platform: Optional[str] = None,
    ) -> List[SimilarityResult]:
        """
        Find events similar to query embedding using pgvector.

        The <=> operator returns cosine distance (0 = identical, 2 = opposite).
        We convert to similarity: 1 - distance.
        """
        if not self._pool:
            return []

        vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        # cosine distance threshold: 1 - similarity
        dist_threshold = 1.0 - threshold
        col = _embedding_column

        try:
            async with self._pool.acquire() as conn:
                if exclude_platform:
                    rows = await conn.fetch(
                        f"""SELECT ec.content_hash, ec.content_preview, ec.source_event_id,
                                  ce.platform, ce.session_id, ce.light_layer,
                                  ce.instinct_layer, ce.coherence_sig, ce.timestamp_ns,
                                  (ec.{col} <=> $1::vector) AS distance
                           FROM embedding_cache ec
                           JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                           WHERE ce.platform != $2
                             AND ec.{col} IS NOT NULL
                             AND (ec.{col} <=> $1::vector) < $3
                           ORDER BY distance
                           LIMIT $4""",
                        vec_str, exclude_platform, dist_threshold, limit,
                    )
                else:
                    rows = await conn.fetch(
                        f"""SELECT ec.content_hash, ec.content_preview, ec.source_event_id,
                                  ce.platform, ce.session_id, ce.light_layer,
                                  ce.instinct_layer, ce.coherence_sig, ce.timestamp_ns,
                                  (ec.{col} <=> $1::vector) AS distance
                           FROM embedding_cache ec
                           JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                           WHERE ec.{col} IS NOT NULL
                             AND (ec.{col} <=> $1::vector) < $2
                           ORDER BY distance
                           LIMIT $3""",
                        vec_str, dist_threshold, limit,
                    )
        except Exception as e:
            log.warning(f"pgvector search failed, falling back to brute-force: {e}")
            return await self._brute_force_search(
                query_embedding, threshold, limit, exclude_platform
            )

        results = []
        for row in rows:
            light = row["light_layer"]
            if isinstance(light, str):
                light = json.loads(light)
            instinct = row.get("instinct_layer")
            if isinstance(instinct, str):
                instinct = json.loads(instinct)

            results.append(SimilarityResult(
                event_id=row["source_event_id"],
                platform=row["platform"],
                session_id=row["session_id"],
                similarity=round(1.0 - row["distance"], 4),
                preview=row["content_preview"],
                coherence_sig=row["coherence_sig"],
                light_layer=light,
                instinct_layer=instinct,
                timestamp_ns=row["timestamp_ns"],
            ))

        return results

    async def cross_platform_similar(
        self,
        event_row: Dict[str, Any],
        embedding: List[float],
        threshold: float = 0.80,
        limit: int = None,
    ) -> List[SimilarityResult]:
        """Find events from OTHER platforms similar to this event."""
        platform = event_row.get("platform", "")
        return await self.find_similar(
            embedding,
            threshold=threshold,
            limit=limit or cfg.MAX_CANDIDATES_PER_EVENT,
            exclude_platform=platform,
        )

    async def _brute_force_search(
        self,
        query_embedding: List[float],
        threshold: float,
        limit: int,
        exclude_platform: Optional[str],
    ) -> List[SimilarityResult]:
        """Fallback: load all embeddings and compute cosine similarity in Python."""
        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            if exclude_platform:
                rows = await conn.fetch(
                    """SELECT ec.content_hash, ec.content_preview, ec.embedding,
                              ec.source_event_id, ce.platform, ce.session_id,
                              ce.light_layer, ce.instinct_layer, ce.coherence_sig,
                              ce.timestamp_ns
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ce.platform != $1""",
                    exclude_platform,
                )
            else:
                rows = await conn.fetch(
                    """SELECT ec.content_hash, ec.content_preview, ec.embedding,
                              ec.source_event_id, ce.platform, ce.session_id,
                              ce.light_layer, ce.instinct_layer, ce.coherence_sig,
                              ce.timestamp_ns
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id""",
                )

        results = []
        for row in rows:
            try:
                emb_raw = row["embedding"]
                emb = json.loads(emb_raw) if isinstance(emb_raw, str) else list(emb_raw)
                sim = cosine_similarity(query_embedding, emb)
                if sim >= threshold:
                    light = row["light_layer"]
                    if isinstance(light, str):
                        light = json.loads(light)
                    instinct = row.get("instinct_layer")
                    if isinstance(instinct, str):
                        instinct = json.loads(instinct)

                    results.append(SimilarityResult(
                        event_id=row["source_event_id"],
                        platform=row["platform"],
                        session_id=row["session_id"],
                        similarity=round(sim, 4),
                        preview=row["content_preview"],
                        coherence_sig=row["coherence_sig"],
                        light_layer=light,
                        instinct_layer=instinct,
                        timestamp_ns=row["timestamp_ns"],
                    ))
            except Exception:
                continue

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]
