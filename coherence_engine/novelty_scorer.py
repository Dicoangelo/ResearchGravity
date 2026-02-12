"""
Coherence Engine — Novelty Scorer for Entity Pairs

Scores how novel an entity pair is based on:
  - First-time co-occurrence (never seen together before)
  - Cross-domain pairing (entities from different domains)
  - Rarity of each entity (less common = more novel)
  - Recency (newer pairings are more interesting)

Used by the emergence listener to detect genuinely new connections
in the knowledge graph.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import asyncpg

from . import config as cfg

log = logging.getLogger("coherence.novelty_scorer")


@dataclass
class NoveltyScore:
    """Novelty assessment for an entity pair."""
    entity_a: str
    entity_b: str
    overall_score: float  # 0-1 composite
    first_occurrence: bool  # True if never seen together before
    entity_a_rarity: float  # 0-1 (1 = very rare)
    entity_b_rarity: float
    cross_domain: bool  # True if different entity types
    edge_weight: float  # Current edge weight (0 if new)
    components: Dict[str, float]  # Score breakdown


class NoveltyScorer:
    """
    Scores novelty of entity pairs in the knowledge graph.

    Novelty is a composite of:
    - **first_time** (0.4): Is this the first time these entities appear together?
    - **rarity** (0.3): How rare are the individual entities?
    - **cross_domain** (0.2): Are they from different entity types/domains?
    - **recency** (0.1): How recently were they first seen together?
    """

    WEIGHTS = {
        "first_time": 0.4,
        "rarity": 0.3,
        "cross_domain": 0.2,
        "recency": 0.1,
    }

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._total_entities: Optional[int] = None
        self._max_mentions: Optional[int] = None

    async def _ensure_stats(self):
        """Cache aggregate stats for scoring."""
        if self._total_entities is not None:
            return
        async with self._pool.acquire() as conn:
            self._total_entities = await conn.fetchval(
                "SELECT COUNT(*) FROM cognitive_entities"
            ) or 1
            self._max_mentions = await conn.fetchval(
                "SELECT MAX(mention_count) FROM cognitive_entities"
            ) or 1

    async def score_pair(
        self, entity_a_id: str, entity_b_id: str
    ) -> NoveltyScore:
        """
        Score the novelty of an entity pair.

        Args:
            entity_a_id: Entity ID (e.g., "ent-concept-abc123")
            entity_b_id: Entity ID

        Returns:
            NoveltyScore with composite score and component breakdown
        """
        await self._ensure_stats()

        src, tgt = sorted([entity_a_id, entity_b_id])

        async with self._pool.acquire() as conn:
            # Check if edge exists
            edge = await conn.fetchrow(
                """SELECT weight, evidence_count, first_seen_ns
                   FROM cognitive_edges
                   WHERE source_entity = $1 AND target_entity = $2""",
                src, tgt,
            )

            # Get entity details
            ent_a = await conn.fetchrow(
                "SELECT entity_type, mention_count FROM cognitive_entities WHERE entity_id = $1",
                entity_a_id,
            )
            ent_b = await conn.fetchrow(
                "SELECT entity_type, mention_count FROM cognitive_entities WHERE entity_id = $1",
                entity_b_id,
            )

        if not ent_a or not ent_b:
            return NoveltyScore(
                entity_a=entity_a_id, entity_b=entity_b_id,
                overall_score=0.0, first_occurrence=False,
                entity_a_rarity=0.0, entity_b_rarity=0.0,
                cross_domain=False, edge_weight=0.0, components={},
            )

        # Component 1: First-time occurrence
        is_first = edge is None
        first_time_score = 1.0 if is_first else max(0, 1.0 - (edge["evidence_count"] / 10.0))

        # Component 2: Rarity (inverse of mention frequency)
        rarity_a = 1.0 - (ent_a["mention_count"] / self._max_mentions)
        rarity_b = 1.0 - (ent_b["mention_count"] / self._max_mentions)
        rarity_score = (rarity_a + rarity_b) / 2.0

        # Component 3: Cross-domain
        is_cross = ent_a["entity_type"] != ent_b["entity_type"]
        cross_score = 1.0 if is_cross else 0.3

        # Component 4: Recency (how recently first seen, if exists)
        if edge and edge["first_seen_ns"]:
            age_hours = (time.time_ns() - edge["first_seen_ns"]) / 3.6e12
            recency_score = max(0, 1.0 - (age_hours / (24 * 30)))  # Decays over 30 days
        else:
            recency_score = 1.0  # New = max recency

        # Composite score
        components = {
            "first_time": first_time_score,
            "rarity": rarity_score,
            "cross_domain": cross_score,
            "recency": recency_score,
        }
        overall = sum(
            self.WEIGHTS[k] * v for k, v in components.items()
        )

        return NoveltyScore(
            entity_a=entity_a_id,
            entity_b=entity_b_id,
            overall_score=min(1.0, overall),
            first_occurrence=is_first,
            entity_a_rarity=rarity_a,
            entity_b_rarity=rarity_b,
            cross_domain=is_cross,
            edge_weight=float(edge["weight"]) if edge else 0.0,
            components=components,
        )

    async def find_novel_pairs(
        self,
        min_novelty: float = 0.6,
        limit: int = 20,
        since_hours: int = 168,
    ) -> List[NoveltyScore]:
        """
        Find the most novel entity pairs in recent events.

        Scans recent edges and scores them for novelty.
        """
        await self._ensure_stats()
        cutoff_ns = int((time.time() - since_hours * 3600) * 1e9)

        async with self._pool.acquire() as conn:
            # Get recent edges (newest first)
            edges = await conn.fetch(
                """SELECT source_entity, target_entity, weight,
                          evidence_count, first_seen_ns
                   FROM cognitive_edges
                   WHERE first_seen_ns > $1
                   ORDER BY first_seen_ns DESC
                   LIMIT 200""",
                cutoff_ns,
            )

        results = []
        for edge in edges:
            score = await self.score_pair(
                edge["source_entity"], edge["target_entity"]
            )
            if score.overall_score >= min_novelty:
                results.append(score)

        # Sort by novelty descending
        results.sort(key=lambda s: s.overall_score, reverse=True)
        return results[:limit]

    async def score_event_entities(
        self, entity_ids: List[str]
    ) -> List[NoveltyScore]:
        """Score all entity pairs from a single event for novelty."""
        scores = []
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                score = await self.score_pair(entity_ids[i], entity_ids[j])
                if score.overall_score > 0.3:
                    scores.append(score)

        scores.sort(key=lambda s: s.overall_score, reverse=True)
        return scores
