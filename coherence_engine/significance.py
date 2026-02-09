"""
Coherence Engine — Significance Testing & Coherence Arcs

Provides:
  - Permutation-based significance testing for coherence moments
  - Coherence arc detection (grouping moments into narrative arcs)

Significance testing validates whether a coherence score is statistically
significant vs random chance. Arcs group related moments over time into
narrative threads.
"""

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg

log = logging.getLogger("coherence.significance")


# ── Significance Testing ────────────────────────────────────


@dataclass
class SignificanceResult:
    """Result of a permutation significance test."""
    moment_id: str
    real_score: float
    p_value: float
    n_permutations: int
    null_mean: float
    null_std: float
    is_significant: bool  # p_value < alpha

    @property
    def z_score(self) -> float:
        if self.null_std == 0:
            return 0.0
        return (self.real_score - self.null_mean) / self.null_std


class SignificanceTester:
    """
    Permutation-based significance testing for coherence moments.

    Tests whether an observed coherence score is significantly higher
    than what would be expected by chance (random event pairing).

    Used as a second-pass filter: fast heuristic identifies candidates,
    permutation test validates the top-N.
    """

    def __init__(self, pool=None, n_permutations: int = 100, alpha: float = 0.05):
        self._pool = pool
        self._n_permutations = n_permutations
        self._alpha = alpha
        self._platform_events_cache: Dict[str, List[Dict]] = {}

    async def _get_random_events(self, platform: str, n: int = 50) -> List[Dict]:
        """Get a sample of random events from a platform for null distribution."""
        if platform in self._platform_events_cache:
            return self._platform_events_cache[platform]

        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT event_id, platform, timestamp_ns,
                          light_layer, instinct_layer, coherence_sig
                   FROM cognitive_events
                   WHERE platform = $1
                   ORDER BY RANDOM()
                   LIMIT $2""",
                platform, max(n, self._n_permutations * 2),
            )

        import json
        events = []
        for row in rows:
            event = dict(row)
            for fld in ("light_layer", "instinct_layer"):
                if isinstance(event.get(fld), str):
                    event[fld] = json.loads(event[fld])
            events.append(event)

        self._platform_events_cache[platform] = events
        return events

    def _compute_similarity_score(self, event_a: Dict, event_b: Dict) -> float:
        """
        Compute a lightweight coherence score between two events.

        Uses light_layer topic/intent matching + instinct signals.
        This is NOT the full coherence pipeline — it's a fast approximation
        for the null distribution.
        """
        light_a = event_a.get("light_layer") or {}
        light_b = event_b.get("light_layer") or {}

        score = 0.0

        # Topic match
        topic_a = (light_a.get("topic") or "").lower()
        topic_b = (light_b.get("topic") or "").lower()
        if topic_a and topic_b:
            words_a = set(topic_a.split())
            words_b = set(topic_b.split())
            if words_a and words_b:
                overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
                score += overlap * 0.4

        # Intent match
        if light_a.get("intent") == light_b.get("intent") and light_a.get("intent"):
            score += 0.15

        # Concept overlap
        concepts_a = set(light_a.get("concepts") or [])
        concepts_b = set(light_b.get("concepts") or [])
        if concepts_a and concepts_b:
            concept_overlap = len(concepts_a & concepts_b) / max(len(concepts_a | concepts_b), 1)
            score += concept_overlap * 0.3

        # Coherence signature match
        sig_a = event_a.get("coherence_sig") or ""
        sig_b = event_b.get("coherence_sig") or ""
        if sig_a and sig_b and sig_a == sig_b:
            score += 0.15

        return min(score, 1.0)

    async def test(
        self,
        event_a: Dict,
        event_b: Dict,
        real_score: float,
        moment_id: str = "",
    ) -> SignificanceResult:
        """
        Test whether the coherence between event_a and event_b is significant.

        Builds a null distribution by pairing event_a with random events
        from event_b's platform, then computes the p-value.
        """
        platform_b = event_b.get("platform", "")
        random_events = await self._get_random_events(platform_b)

        if len(random_events) < 10:
            # Not enough data for permutation test — assume significant
            return SignificanceResult(
                moment_id=moment_id,
                real_score=real_score,
                p_value=0.01,
                n_permutations=0,
                null_mean=0.0,
                null_std=0.0,
                is_significant=True,
            )

        # Build null distribution
        null_scores = []
        sample = random.sample(random_events, min(self._n_permutations, len(random_events)))
        for rand_event in sample:
            null_score = self._compute_similarity_score(event_a, rand_event)
            null_scores.append(null_score)

        # Compute p-value
        n_extreme = sum(1 for s in null_scores if s >= real_score)
        p_value = (n_extreme + 1) / (len(null_scores) + 1)  # Laplace smoothing

        null_mean = sum(null_scores) / len(null_scores) if null_scores else 0
        null_std = (
            (sum((s - null_mean) ** 2 for s in null_scores) / len(null_scores)) ** 0.5
            if null_scores else 0
        )

        return SignificanceResult(
            moment_id=moment_id,
            real_score=real_score,
            p_value=p_value,
            n_permutations=len(null_scores),
            null_mean=null_mean,
            null_std=null_std,
            is_significant=p_value < self._alpha,
        )

    def clear_cache(self):
        """Clear the random events cache."""
        self._platform_events_cache.clear()


# ── Coherence Arcs ──────────────────────────────────────────


@dataclass
class CoherenceArc:
    """A narrative arc of related coherence moments over time."""
    arc_id: str
    title: str
    started_ns: int
    last_activity_ns: int
    status: str  # active, dormant, resolved
    moment_ids: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    arc_strength: float = 0.0
    moment_count: int = 0

    def add_moment(self, moment_id: str, entities: List[str],
                   platforms: List[str], confidence: float, timestamp_ns: int):
        """Add a moment to this arc, updating all derived fields."""
        self.moment_ids.append(moment_id)
        self.moment_count += 1

        # Update platforms (deduplicated)
        for p in platforms:
            if p and p not in self.platforms:
                self.platforms.append(p)

        # Update entities (keep top entities by frequency)
        for e in entities:
            if e not in self.key_entities:
                self.key_entities.append(e)

        # Update timing
        if timestamp_ns > self.last_activity_ns:
            self.last_activity_ns = timestamp_ns

        # Cumulative strength (weighted by recency)
        self.arc_strength += confidence


def _extract_entities_from_description(description: str) -> List[str]:
    """Extract entity-like tokens from a moment description."""
    import re
    entities = set()

    # Extract platform names
    for match in re.finditer(r'\b(chatgpt|claude-code|claude-cli|claude-desktop|grok)\b', description.lower()):
        entities.add(match.group(1))

    # Extract topic-like words (after | or : delimiters)
    for match in re.finditer(r'[|:]\s*(\w[\w\s]{2,30}?)(?:\s*[|]|$)', description):
        token = match.group(1).strip().lower()
        if len(token) > 3 and token not in ("the", "and", "for", "with", "from", "about"):
            entities.add(token)

    return list(entities)


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class ArcDetector:
    """
    Detects coherence arcs — narrative threads that span multiple moments.

    Groups moments by entity overlap into arcs. An arc is a sequence of
    coherence moments that share concepts/entities, forming a coherent
    intellectual narrative over time.
    """

    def __init__(self, pool=None, overlap_threshold: float = 0.2,
                 dormant_hours: float = 72):
        self._pool = pool
        self._overlap_threshold = overlap_threshold
        self._dormant_ns = int(dormant_hours * 3600 * 1e9)

    async def detect_arcs(self, limit: int = 500) -> List[CoherenceArc]:
        """
        Detect arcs from stored coherence moments.

        Reads moments from the database, groups them by entity overlap,
        and returns the detected arcs.
        """
        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT moment_id, detected_ns, event_ids, platforms,
                          coherence_type, confidence, description
                   FROM coherence_moments
                   ORDER BY detected_ns ASC
                   LIMIT $1""",
                limit,
            )

        if not rows:
            return []

        arcs: List[CoherenceArc] = []
        now_ns = time.time_ns()

        for row in rows:
            entities = _extract_entities_from_description(row["description"] or "")
            entity_set = set(entities)
            platforms = row["platforms"] or []
            timestamp_ns = row["detected_ns"] or 0

            # Find best matching arc
            best_arc = None
            best_overlap = 0.0
            for arc in arcs:
                if arc.status == "resolved":
                    continue
                overlap = _jaccard(entity_set, set(arc.key_entities))
                if overlap > self._overlap_threshold and overlap > best_overlap:
                    best_arc = arc
                    best_overlap = overlap

            if best_arc:
                best_arc.add_moment(
                    row["moment_id"], entities, platforms,
                    row["confidence"], timestamp_ns,
                )
            else:
                # Create new arc
                arc_id = f"arc-{hashlib.sha256(row['moment_id'].encode()).hexdigest()[:12]}"
                # Title from first moment's description
                desc = (row["description"] or "")[:80]
                arc = CoherenceArc(
                    arc_id=arc_id,
                    title=desc,
                    started_ns=timestamp_ns,
                    last_activity_ns=timestamp_ns,
                    status="active",
                    moment_ids=[row["moment_id"]],
                    platforms=list(platforms),
                    key_entities=entities,
                    arc_strength=row["confidence"],
                    moment_count=1,
                )
                arcs.append(arc)

        # Mark dormant arcs
        for arc in arcs:
            if now_ns - arc.last_activity_ns > self._dormant_ns:
                arc.status = "dormant"

        # Sort by strength
        arcs.sort(key=lambda a: a.arc_strength, reverse=True)
        return arcs

    async def store_arcs(self, arcs: List[CoherenceArc]):
        """Persist detected arcs to the database."""
        if not self._pool or not arcs:
            return

        async with self._pool.acquire() as conn:
            # Ensure table exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS coherence_arcs (
                    arc_id TEXT PRIMARY KEY,
                    title TEXT,
                    started_ns BIGINT,
                    last_activity_ns BIGINT,
                    status TEXT DEFAULT 'active',
                    moment_ids TEXT[] DEFAULT '{}',
                    platforms TEXT[] DEFAULT '{}',
                    key_entities TEXT[] DEFAULT '{}',
                    arc_strength REAL DEFAULT 0.0,
                    moment_count INT DEFAULT 0,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            for arc in arcs:
                if arc.moment_count < 2:
                    continue  # Only store arcs with 2+ moments

                await conn.execute(
                    """INSERT INTO coherence_arcs
                       (arc_id, title, started_ns, last_activity_ns, status,
                        moment_ids, platforms, key_entities, arc_strength, moment_count)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                       ON CONFLICT (arc_id) DO UPDATE SET
                           last_activity_ns = GREATEST(coherence_arcs.last_activity_ns, EXCLUDED.last_activity_ns),
                           status = EXCLUDED.status,
                           moment_ids = EXCLUDED.moment_ids,
                           platforms = EXCLUDED.platforms,
                           key_entities = EXCLUDED.key_entities,
                           arc_strength = EXCLUDED.arc_strength,
                           moment_count = EXCLUDED.moment_count""",
                    arc.arc_id, arc.title, arc.started_ns, arc.last_activity_ns,
                    arc.status, arc.moment_ids, arc.platforms, arc.key_entities,
                    arc.arc_strength, arc.moment_count,
                )

        stored = sum(1 for a in arcs if a.moment_count >= 2)
        log.info(f"Stored {stored} arcs ({sum(a.moment_count for a in arcs if a.moment_count >= 2)} moments)")
