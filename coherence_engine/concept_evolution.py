"""
Coherence Engine — Concept Evolution Tracker

Tracks how concepts evolve over time across sessions and platforms.
Uses the `concept_versions` table to record concept definitions,
evolution chains, and temporal progression.

Detects:
  - Concept refinement: same concept gets more specific over time
  - Concept merging: two concepts merge into one
  - Concept branching: one concept splits into specialized forms
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from . import config as cfg

log = logging.getLogger("coherence.concept_evolution")


@dataclass
class ConceptVersion:
    """A versioned snapshot of a concept."""
    concept: str
    version: int
    definition: str
    first_seen_ns: int
    last_seen_ns: int
    platform: str
    session_id: str
    evolved_from: Optional[str] = None


@dataclass
class EvolutionChain:
    """A chain of concept versions showing evolution over time."""
    concept: str
    versions: List[ConceptVersion]
    total_versions: int
    platforms_involved: List[str]
    time_span_hours: float


class ConceptEvolutionTracker:
    """
    Tracks concept evolution across sessions and platforms.

    Monitors the knowledge graph for concepts that appear repeatedly
    with changing definitions or expanding context.
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def track_concept(
        self,
        concept: str,
        definition: str,
        platform: str,
        session_id: str,
        timestamp_ns: int = None,
        evolved_from: Optional[str] = None,
    ) -> ConceptVersion:
        """
        Record a new version of a concept.

        If the concept already exists with the same definition, updates last_seen.
        If the definition has changed, creates a new version.
        """
        ts = timestamp_ns or time.time_ns()

        async with self._pool.acquire() as conn:
            # Get current latest version
            latest = await conn.fetchrow(
                """SELECT version, definition FROM concept_versions
                   WHERE concept = $1
                   ORDER BY version DESC LIMIT 1""",
                concept.lower(),
            )

            if latest and latest["definition"] == definition:
                # Same definition — update last_seen
                await conn.execute(
                    """UPDATE concept_versions
                       SET last_seen_ns = $2
                       WHERE concept = $1 AND version = $3""",
                    concept.lower(), ts, latest["version"],
                )
                return ConceptVersion(
                    concept=concept.lower(),
                    version=latest["version"],
                    definition=definition,
                    first_seen_ns=ts,
                    last_seen_ns=ts,
                    platform=platform,
                    session_id=session_id,
                )

            # New version
            new_version = (latest["version"] + 1) if latest else 1
            evolved = evolved_from or (concept.lower() if latest else None)

            await conn.execute(
                """INSERT INTO concept_versions
                       (concept, version, definition, first_seen_ns, last_seen_ns,
                        platform, session_id, evolved_from)
                   VALUES ($1, $2, $3, $4, $4, $5, $6, $7)
                   ON CONFLICT (concept, version) DO UPDATE SET
                       last_seen_ns = GREATEST(concept_versions.last_seen_ns, EXCLUDED.last_seen_ns)""",
                concept.lower(), new_version, definition,
                ts, platform, session_id, evolved,
            )

            log.info(f"Concept '{concept}' evolved to v{new_version} on {platform}")
            return ConceptVersion(
                concept=concept.lower(),
                version=new_version,
                definition=definition,
                first_seen_ns=ts,
                last_seen_ns=ts,
                platform=platform,
                session_id=session_id,
                evolved_from=evolved,
            )

    async def get_evolution_chain(self, concept: str) -> Optional[EvolutionChain]:
        """Get the full evolution chain for a concept."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT concept, version, definition, first_seen_ns,
                          last_seen_ns, platform, session_id, evolved_from
                   FROM concept_versions
                   WHERE concept = $1
                   ORDER BY version ASC""",
                concept.lower(),
            )

        if not rows:
            return None

        versions = [
            ConceptVersion(
                concept=r["concept"],
                version=r["version"],
                definition=r["definition"] or "",
                first_seen_ns=r["first_seen_ns"] or 0,
                last_seen_ns=r["last_seen_ns"] or 0,
                platform=r["platform"] or "",
                session_id=r["session_id"] or "",
                evolved_from=r["evolved_from"],
            )
            for r in rows
        ]

        platforms = list(set(v.platform for v in versions if v.platform))
        first_ns = min(v.first_seen_ns for v in versions if v.first_seen_ns)
        last_ns = max(v.last_seen_ns for v in versions if v.last_seen_ns)
        span_hours = (last_ns - first_ns) / 3.6e12 if first_ns and last_ns else 0

        return EvolutionChain(
            concept=concept.lower(),
            versions=versions,
            total_versions=len(versions),
            platforms_involved=platforms,
            time_span_hours=span_hours,
        )

    async def detect_evolutions(
        self,
        since_hours: int = 168,
        min_versions: int = 2,
    ) -> List[EvolutionChain]:
        """
        Find concepts that have evolved recently.

        Returns concepts with 2+ versions in the given time window.
        """
        cutoff_ns = int((time.time() - since_hours * 3600) * 1e9)

        async with self._pool.acquire() as conn:
            concepts = await conn.fetch(
                """SELECT concept, COUNT(*) AS version_count
                   FROM concept_versions
                   WHERE last_seen_ns > $1
                   GROUP BY concept
                   HAVING COUNT(*) >= $2
                   ORDER BY version_count DESC
                   LIMIT 50""",
                cutoff_ns, min_versions,
            )

        chains = []
        for row in concepts:
            chain = await self.get_evolution_chain(row["concept"])
            if chain:
                chains.append(chain)

        return chains

    async def auto_track_from_events(
        self,
        limit: int = 500,
    ) -> int:
        """
        Auto-track concept evolution from recent cognitive events.

        Extracts concepts from light_layer and tracks their definitions
        over time using topic + summary as the definition proxy.
        """
        tracked = 0

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT event_id, session_id, timestamp_ns, platform,
                          light_layer->>'topic' AS topic,
                          light_layer->>'summary' AS summary,
                          light_layer->'concepts' AS concepts
                   FROM cognitive_events
                   WHERE light_layer IS NOT NULL
                     AND light_layer->'concepts' IS NOT NULL
                   ORDER BY timestamp_ns DESC
                   LIMIT $1""",
                limit,
            )

        for row in rows:
            concepts_raw = row["concepts"]
            if isinstance(concepts_raw, str):
                try:
                    concepts_list = json.loads(concepts_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(concepts_raw, list):
                concepts_list = concepts_raw
            else:
                continue

            topic = row["topic"] or ""
            summary = row["summary"] or ""
            definition = f"{topic}: {summary}"[:500] if summary else topic

            for concept in concepts_list[:5]:
                if not isinstance(concept, str) or len(concept) < 3:
                    continue
                await self.track_concept(
                    concept=concept,
                    definition=definition,
                    platform=row["platform"] or "",
                    session_id=row["session_id"] or "",
                    timestamp_ns=row["timestamp_ns"],
                )
                tracked += 1

        log.info(f"Auto-tracked {tracked} concept versions from {len(rows)} events")
        return tracked
