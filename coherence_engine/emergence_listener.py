"""
Coherence Engine — Emergence Listener

A persistent background agent that watches the event stream for REAL emergence:
- Concept crystallization: A vague idea becomes concrete across 3+ sessions
- Cross-domain synthesis: Ideas from different domains merge into something new
- Temporal acceleration: Suddenly working faster on a topic = insight happened
- Convergence cascade: Multiple platforms converge on same solution within hours

Tracks rolling windows of concept velocity, cross-domain bridges, and flow state
to detect genuine cognitive breakthroughs.
"""

import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg

from . import config as cfg

log = logging.getLogger("coherence.emergence")


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class ConceptVelocity:
    """Track how fast a concept is evolving across sessions."""
    concept: str
    mentions_1h: int = 0
    mentions_24h: int = 0
    mentions_7d: int = 0
    platforms: Set[str] = field(default_factory=set)
    sessions: Set[str] = field(default_factory=set)
    velocity: float = 0.0  # mentions_1h / max(mentions_24h, 1) — acceleration


@dataclass
class CrossDomainBridge:
    """A new connection between concepts from different domains."""
    concept_a: str
    concept_b: str
    domain_a: str
    domain_b: str
    first_seen_ns: int
    co_occurrence_count: int
    platforms: Set[str] = field(default_factory=set)


@dataclass
class FlowIndicator:
    """Flow state signals from event patterns."""
    platform: str
    events_per_hour: float
    avg_content_length: float
    topic_depth: float  # How focused on a single topic (0-1)
    quality_score_trend: float  # Positive = improving


@dataclass
class CognitiveBreakthrough:
    """A detected breakthrough — the culmination of emergence signals."""
    breakthrough_id: str
    breakthrough_type: str  # crystallization, synthesis, convergence, acceleration
    title: str
    narrative: str
    evidence_moment_ids: List[str]
    evidence_session_ids: List[str]
    platforms: List[str]
    concepts: List[str]
    novelty_score: float
    impact_score: float


# ── Concept Velocity Tracker (US-014) ────────────────────────────────────────


class ConceptVelocityTracker:
    """
    Track concept evolution speed across a rolling window.

    Acceleration = recent mentions / historical mentions.
    A concept accelerating means it's becoming more important.
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def compute(self, limit: int = 500) -> List[ConceptVelocity]:
        """Compute concept velocities from recent events."""
        async with self._pool.acquire() as conn:
            # Get concept mentions from the last hour
            hour_ago = time.time_ns() - 3_600_000_000_000
            day_ago = time.time_ns() - 86_400_000_000_000
            week_ago = time.time_ns() - 604_800_000_000_000

            # Fetch events with concepts from last 7 days
            rows = await conn.fetch(
                """SELECT event_id, session_id, timestamp_ns, platform,
                          light_layer->'concepts' AS concepts,
                          light_layer->>'topic' AS topic
                   FROM cognitive_events
                   WHERE timestamp_ns > $1
                     AND light_layer IS NOT NULL
                   ORDER BY timestamp_ns DESC
                   LIMIT $2""",
                week_ago, limit,
            )

        # Count concept mentions across time windows
        concepts_1h: Counter = Counter()
        concepts_24h: Counter = Counter()
        concepts_7d: Counter = Counter()
        concept_platforms: Dict[str, Set[str]] = defaultdict(set)
        concept_sessions: Dict[str, Set[str]] = defaultdict(set)

        for row in rows:
            ts = row["timestamp_ns"]
            platform = row["platform"]
            session_id = row["session_id"]

            # Extract concepts
            concepts_raw = row["concepts"]
            if isinstance(concepts_raw, str):
                try:
                    concepts_raw = json.loads(concepts_raw)
                except (json.JSONDecodeError, TypeError):
                    concepts_raw = []
            if not isinstance(concepts_raw, list):
                concepts_raw = []

            topic = row.get("topic")
            if topic and topic != "general":
                concepts_raw.append(topic)

            for concept in concepts_raw:
                if not concept or len(concept) < 3:
                    continue
                concept = concept.lower().strip()

                concepts_7d[concept] += 1
                concept_platforms[concept].add(platform or "")
                concept_sessions[concept].add(session_id or "")

                if ts > day_ago:
                    concepts_24h[concept] += 1
                if ts > hour_ago:
                    concepts_1h[concept] += 1

        # Build velocity objects
        velocities = []
        for concept, count_7d in concepts_7d.most_common(100):
            count_1h = concepts_1h.get(concept, 0)
            count_24h = concepts_24h.get(concept, 0)

            # Velocity = recent activity relative to baseline
            baseline = max(count_7d / 7, 1)
            velocity = count_1h / baseline if count_1h > 0 else 0

            velocities.append(ConceptVelocity(
                concept=concept,
                mentions_1h=count_1h,
                mentions_24h=count_24h,
                mentions_7d=count_7d,
                platforms=concept_platforms[concept],
                sessions=concept_sessions[concept],
                velocity=velocity,
            ))

        velocities.sort(key=lambda v: v.velocity, reverse=True)
        return velocities


# ── Cross-Domain Bridge Detector (US-015) ────────────────────────────────────


class CrossDomainBridgeDetector:
    """
    Detect new connections between concepts from different domains.

    A "bridge" is when two concepts that have never co-occurred before
    suddenly appear together — potentially signaling a novel synthesis.
    """

    # Domain classification based on topic patterns
    DOMAIN_MAP = {
        "ai_agents": "ai", "machine_learning": "ai", "llm": "ai",
        "neural_networks": "ai", "nlp": "ai", "embeddings": "ai",
        "web_development": "engineering", "typescript": "engineering",
        "react": "engineering", "python": "engineering", "database": "engineering",
        "devops": "engineering", "infrastructure": "engineering",
        "ucw": "philosophy", "sovereignty": "philosophy", "cognitive": "philosophy",
        "emergence": "philosophy", "consciousness": "philosophy",
        "blockchain": "crypto", "token": "crypto", "defi": "crypto",
        "career": "personal", "portfolio": "personal", "identity": "personal",
        "research": "research", "papers": "research", "arxiv": "research",
    }

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def detect(self, limit: int = 200) -> List[CrossDomainBridge]:
        """Find new cross-domain concept bridges in recent events."""
        async with self._pool.acquire() as conn:
            day_ago = time.time_ns() - 86_400_000_000_000

            rows = await conn.fetch(
                """SELECT event_id, timestamp_ns, platform,
                          light_layer->'concepts' AS concepts,
                          light_layer->>'topic' AS topic
                   FROM cognitive_events
                   WHERE timestamp_ns > $1
                     AND light_layer IS NOT NULL
                   ORDER BY timestamp_ns DESC
                   LIMIT $2""",
                day_ago, limit,
            )

            # Get existing co-occurrences from KG
            existing_pairs = set()
            edges = await conn.fetch(
                """SELECT source_entity, target_entity
                   FROM cognitive_edges
                   WHERE weight > 2"""
            )
            for e in edges:
                existing_pairs.add(
                    tuple(sorted([e["source_entity"], e["target_entity"]]))
                )

        # Find new concept co-occurrences
        new_bridges = []
        co_occurrences: Dict[Tuple[str, str], CrossDomainBridge] = {}

        for row in rows:
            concepts_raw = row["concepts"]
            if isinstance(concepts_raw, str):
                try:
                    concepts_raw = json.loads(concepts_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(concepts_raw, list):
                continue

            topic = row.get("topic", "")
            concepts = [c.lower().strip() for c in concepts_raw if c and len(c) > 2]
            if topic and topic != "general":
                concepts.append(topic.lower())

            # Check all pairs
            for i, a in enumerate(concepts):
                domain_a = self._classify_domain(a)
                for b in concepts[i + 1:]:
                    domain_b = self._classify_domain(b)
                    if domain_a == domain_b:
                        continue  # Same domain, not a bridge

                    pair = tuple(sorted([a, b]))
                    # Check if this is a NEW pair (not in existing KG)
                    if pair in existing_pairs:
                        continue

                    if pair not in co_occurrences:
                        co_occurrences[pair] = CrossDomainBridge(
                            concept_a=pair[0],
                            concept_b=pair[1],
                            domain_a=domain_a,
                            domain_b=domain_b,
                            first_seen_ns=row["timestamp_ns"],
                            co_occurrence_count=0,
                            platforms=set(),
                        )
                    bridge = co_occurrences[pair]
                    bridge.co_occurrence_count += 1
                    bridge.platforms.add(row["platform"] or "")

        # Filter to significant bridges (appeared 2+ times)
        bridges = [
            b for b in co_occurrences.values()
            if b.co_occurrence_count >= 2
        ]
        bridges.sort(key=lambda b: b.co_occurrence_count, reverse=True)
        return bridges[:20]

    def _classify_domain(self, concept: str) -> str:
        """Classify a concept into a domain."""
        concept_lower = concept.lower()
        for keyword, domain in self.DOMAIN_MAP.items():
            if keyword in concept_lower:
                return domain
        return "general"


# ── Flow State Indicator Tracker (US-016) ────────────────────────────────────


class FlowStateTracker:
    """
    Track flow state indicators from event patterns.

    Flow signals:
    - High event frequency (messages per hour)
    - Increasing content length (deeper engagement)
    - Topic consistency (focused, not scattered)
    - Quality score trends (improving over session)
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def compute(self) -> List[FlowIndicator]:
        """Compute flow state indicators for each active platform."""
        async with self._pool.acquire() as conn:
            two_hours_ago = time.time_ns() - 7_200_000_000_000

            # Get recent events grouped by platform
            rows = await conn.fetch(
                """SELECT platform,
                          COUNT(*) AS event_count,
                          AVG(content_length) AS avg_length,
                          AVG(quality_score) AS avg_quality,
                          array_agg(DISTINCT light_layer->>'topic') AS topics
                   FROM cognitive_events
                   WHERE timestamp_ns > $1
                     AND platform IS NOT NULL
                   GROUP BY platform""",
                two_hours_ago,
            )

            # Get quality trend (compare last hour vs hour before)
            hour_ago = time.time_ns() - 3_600_000_000_000
            trends = await conn.fetch(
                """SELECT platform,
                          AVG(quality_score) FILTER (WHERE timestamp_ns > $1) AS recent_avg,
                          AVG(quality_score) FILTER (WHERE timestamp_ns <= $1 AND timestamp_ns > $2) AS older_avg
                   FROM cognitive_events
                   WHERE timestamp_ns > $2
                     AND quality_score IS NOT NULL
                   GROUP BY platform""",
                hour_ago, two_hours_ago,
            )
            trend_map = {
                r["platform"]: (float(r["recent_avg"] or 0) - float(r["older_avg"] or 0))
                for r in trends
                if r["recent_avg"] and r["older_avg"]
            }

        indicators = []
        for row in rows:
            event_count = row["event_count"]
            topics = [t for t in (row["topics"] or []) if t and t != "general"]
            unique_topics = len(set(topics))
            topic_depth = 1 - (unique_topics / max(event_count, 1))  # 1 = focused, 0 = scattered

            indicators.append(FlowIndicator(
                platform=row["platform"],
                events_per_hour=event_count / 2.0,  # 2-hour window
                avg_content_length=float(row["avg_length"] or 0),
                topic_depth=max(0, min(1, topic_depth)),
                quality_score_trend=trend_map.get(row["platform"], 0),
            ))

        return indicators


# ── Breakthrough Generation (US-017) ─────────────────────────────────────────


class BreakthroughDetector:
    """
    Combines all emergence signals to detect genuine cognitive breakthroughs.

    Breakthrough types:
    - crystallization: Concept velocity spike + topic depth increase
    - synthesis: New cross-domain bridge + multi-platform convergence
    - convergence: 3+ platforms discussing the same thing within hours
    - acceleration: Sudden flow state on a topic that was stalled
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._velocity_tracker = ConceptVelocityTracker(pool)
        self._bridge_detector = CrossDomainBridgeDetector(pool)
        self._flow_tracker = FlowStateTracker(pool)

    async def scan(self) -> List[CognitiveBreakthrough]:
        """Run all emergence detectors and synthesize breakthroughs."""
        velocities = await self._velocity_tracker.compute()
        bridges = await self._bridge_detector.detect()
        flow = await self._flow_tracker.compute()

        breakthroughs = []

        # Detect CRYSTALLIZATION: top accelerating concepts across 3+ sessions
        for v in velocities[:10]:
            if v.velocity > 3.0 and len(v.sessions) >= 3 and len(v.platforms) >= 2:
                bt = CognitiveBreakthrough(
                    breakthrough_id=self._gen_id("crystallization", v.concept),
                    breakthrough_type="crystallization",
                    title=f"Concept crystallizing: {v.concept}",
                    narrative=(
                        f"The concept '{v.concept}' is rapidly crystallizing across "
                        f"{len(v.platforms)} platforms and {len(v.sessions)} sessions. "
                        f"Velocity: {v.velocity:.1f}x baseline. "
                        f"This suggests a vague idea is becoming concrete through "
                        f"cross-platform exploration."
                    ),
                    evidence_moment_ids=[],
                    evidence_session_ids=list(v.sessions)[:10],
                    platforms=list(v.platforms),
                    concepts=[v.concept],
                    novelty_score=min(1.0, v.velocity / 10.0),
                    impact_score=min(1.0, len(v.sessions) / 10.0),
                )
                breakthroughs.append(bt)

        # Detect SYNTHESIS: new cross-domain bridges on multiple platforms
        for bridge in bridges[:5]:
            if bridge.co_occurrence_count >= 3 and len(bridge.platforms) >= 2:
                bt = CognitiveBreakthrough(
                    breakthrough_id=self._gen_id(
                        "synthesis", f"{bridge.concept_a}+{bridge.concept_b}"
                    ),
                    breakthrough_type="synthesis",
                    title=f"Cross-domain synthesis: {bridge.concept_a} + {bridge.concept_b}",
                    narrative=(
                        f"New bridge between '{bridge.concept_a}' ({bridge.domain_a}) and "
                        f"'{bridge.concept_b}' ({bridge.domain_b}) detected — "
                        f"{bridge.co_occurrence_count} co-occurrences across "
                        f"{len(bridge.platforms)} platforms. This is a novel connection "
                        f"not seen in the existing knowledge graph."
                    ),
                    evidence_moment_ids=[],
                    evidence_session_ids=[],
                    platforms=list(bridge.platforms),
                    concepts=[bridge.concept_a, bridge.concept_b],
                    novelty_score=0.8,
                    impact_score=min(1.0, bridge.co_occurrence_count / 5.0),
                )
                breakthroughs.append(bt)

        # Detect CONVERGENCE: multiple platforms in flow on the same topic
        active_platforms = {f.platform for f in flow if f.events_per_hour > 5}
        if len(active_platforms) >= 3:
            # Find shared topics across active platforms
            top_concepts = velocities[:5] if velocities else []
            for v in top_concepts:
                if len(v.platforms & active_platforms) >= 3:
                    bt = CognitiveBreakthrough(
                        breakthrough_id=self._gen_id("convergence", v.concept),
                        breakthrough_type="convergence",
                        title=f"Multi-platform convergence: {v.concept}",
                        narrative=(
                            f"{len(v.platforms & active_platforms)} active platforms are "
                            f"converging on '{v.concept}'. Platforms: "
                            f"{', '.join(v.platforms & active_platforms)}. "
                            f"This simultaneous focus suggests a breakthrough moment."
                        ),
                        evidence_moment_ids=[],
                        evidence_session_ids=list(v.sessions)[:10],
                        platforms=list(v.platforms & active_platforms),
                        concepts=[v.concept],
                        novelty_score=0.7,
                        impact_score=0.9,
                    )
                    breakthroughs.append(bt)

        return breakthroughs

    async def scan_and_store(self) -> List[CognitiveBreakthrough]:
        """Scan for breakthroughs and store them in the database."""
        breakthroughs = await self.scan()

        for bt in breakthroughs:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """INSERT INTO cognitive_breakthroughs
                           (breakthrough_id, breakthrough_type, title, narrative,
                            evidence_moment_ids, evidence_session_ids, platforms,
                            concepts, novelty_score, impact_score)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                           ON CONFLICT (breakthrough_id) DO UPDATE SET
                               narrative = EXCLUDED.narrative,
                               novelty_score = GREATEST(cognitive_breakthroughs.novelty_score, EXCLUDED.novelty_score),
                               impact_score = GREATEST(cognitive_breakthroughs.impact_score, EXCLUDED.impact_score),
                               detected_at = NOW()""",
                        bt.breakthrough_id,
                        bt.breakthrough_type,
                        bt.title,
                        bt.narrative,
                        bt.evidence_moment_ids,
                        bt.evidence_session_ids,
                        bt.platforms,
                        bt.concepts,
                        bt.novelty_score,
                        bt.impact_score,
                    )
                log.info(f"Breakthrough stored: {bt.breakthrough_type} — {bt.title}")
            except Exception as e:
                log.error(f"Failed to store breakthrough {bt.breakthrough_id}: {e}")

        return breakthroughs

    @staticmethod
    def _gen_id(btype: str, key: str) -> str:
        h = hashlib.sha256(f"{btype}|{key}".encode()).hexdigest()[:16]
        return f"bt-{btype[:4]}-{h}"
