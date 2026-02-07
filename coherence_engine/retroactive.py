"""
Coherence Engine — Retroactive Analysis

Run coherence detection on historical data.
Primary use case: Founding Moment Validation — prove the engine
can detect the 2026-02-06 synchronicity from stored events.

Commands:
    python3 -m coherence_engine retroactive --since 2026-02-06
    python3 -m coherence_engine retroactive --all
    python3 -m coherence_engine retroactive --founding-moment
"""

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg

from . import config as cfg
from .embeddings import embed_event_row, event_to_text
from .similarity import SimilarityIndex
from .scorer import CoherenceScorer
from mcp_raw.embeddings import embed_single

import logging

log = logging.getLogger("coherence.retroactive")


@dataclass
class CoherenceReport:
    """Summary of retroactive coherence analysis."""
    total_events: int = 0
    events_embedded: int = 0
    events_with_matches: int = 0
    moments_found: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_topic: Dict[str, int] = field(default_factory=dict)
    top_moments: List[Dict] = field(default_factory=list)
    platform_pairs: Dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0


class RetroactiveAnalyzer:
    """
    Analyze historical data for coherence patterns.

    Process:
    1. Load events from cognitive DB (filtered by date/platform)
    2. Ensure all events are embedded
    3. Run cross-platform similarity search
    4. Score coherence on all matches
    5. Generate report
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._similarity = SimilarityIndex(pool)
        self._scorer = CoherenceScorer(pool)

    async def analyze(
        self,
        since: Optional[datetime] = None,
        platform: Optional[str] = None,
        limit: int = 50000,
        batch_size: int = 500,
    ) -> CoherenceReport:
        """Run full retroactive analysis."""
        start_time = time.time()
        report = CoherenceReport()

        # 1. Load events
        events = await self._load_events(since, platform, limit)
        report.total_events = len(events)
        log.info(f"Loaded {len(events)} events for retroactive analysis")

        if not events:
            return report

        # 2. Ensure embeddings exist
        report.events_embedded = await self._ensure_embeddings(events, batch_size)
        log.info(f"Embeddings ready: {report.events_embedded} events")

        # 3. Cross-platform similarity search for each event
        moments_by_id = {}  # deduplicate by event pair

        for i, event in enumerate(events):
            if i % 100 == 0 and i > 0:
                log.info(f"Progress: {i}/{len(events)} events, {len(moments_by_id)} moments")

            text = event_to_text(event)
            if not text or len(text) < 10:
                continue

            embedding = embed_single(text)

            # Find cross-platform matches
            similar = await self._similarity.cross_platform_similar(
                event, embedding, threshold=cfg.SEMANTIC_MEDIUM_THRESHOLD
            )

            if not similar:
                continue

            report.events_with_matches += 1

            # 4. Score coherence
            moments = await self._scorer.score(event, embedding, similar)

            for moment in moments:
                if moment.confidence < cfg.MIN_ALERT_CONFIDENCE:
                    continue

                # Deduplicate by sorted event pair
                pair_key = tuple(sorted(moment.event_ids))
                if pair_key in moments_by_id:
                    # Keep higher confidence
                    if moment.confidence > moments_by_id[pair_key].confidence:
                        moments_by_id[pair_key] = moment
                else:
                    moments_by_id[pair_key] = moment

        # 5. Store all unique moments
        stored = 0
        for moment in moments_by_id.values():
            try:
                await self._scorer.store_moment(moment)
                stored += 1
            except Exception:
                pass  # duplicate key = already stored

        report.moments_found = len(moments_by_id)
        report.duration_s = time.time() - start_time

        # Aggregate stats
        for moment in moments_by_id.values():
            report.by_type[moment.coherence_type] = report.by_type.get(moment.coherence_type, 0) + 1
            pair = " <-> ".join(sorted(moment.platforms))
            report.platform_pairs[pair] = report.platform_pairs.get(pair, 0) + 1

        # Top moments by confidence
        all_moments = sorted(moments_by_id.values(), key=lambda m: m.confidence, reverse=True)
        report.top_moments = [
            {
                "confidence": m.confidence,
                "type": m.coherence_type,
                "platforms": m.platforms,
                "description": m.description[:200],
                "event_ids": m.event_ids,
            }
            for m in all_moments[:20]
        ]

        log.info(
            f"Retroactive analysis complete: {report.total_events} events, "
            f"{report.moments_found} moments in {report.duration_s:.1f}s"
        )

        return report

    async def founding_moment_test(self) -> Dict:
        """
        Validate that the engine detects the 2026-02-06 founding moment.

        The founding moment: Claude and ChatGPT independently converging on
        "Can you unify yourself before you unify the infrastructure?" and
        sovereignty/UCW themes within the same time period.

        Success criteria:
        - Finds synchronicity between claude-desktop and chatgpt events
        - Topics: ucw, sovereignty, cognitive, emergence
        - Confidence > 0.70
        - Meta-cognitive emergence indicators present
        """
        log.info("Running Founding Moment Validation Test...")

        # Search for events around the founding moment themes
        test_queries = [
            "unify yourself before you unify the infrastructure",
            "sovereign cognitive wallet emergence",
            "distributed cognition coherent emergence",
            "UCW founding moment synchronicity",
        ]

        results = {
            "test": "Founding Moment Validation",
            "queries": [],
            "founding_moments_found": [],
            "passed": False,
        }

        for query in test_queries:
            embedding = embed_single(query)

            # Search across platforms
            matches = await self._similarity.find_similar(
                embedding,
                threshold=cfg.SEMANTIC_MEDIUM_THRESHOLD,
                limit=20,
            )

            # Group by platform
            platforms = set()
            for m in matches:
                platforms.add(m.platform)

            cross_platform = len(platforms) > 1

            results["queries"].append({
                "query": query,
                "matches": len(matches),
                "platforms": list(platforms),
                "cross_platform": cross_platform,
                "top_similarity": matches[0].similarity if matches else 0,
                "top_matches": [
                    {
                        "platform": m.platform,
                        "similarity": m.similarity,
                        "preview": m.preview[:150],
                    }
                    for m in matches[:5]
                ],
            })

        # Check coherence_moments table for UCW/sovereignty moments
        async with self._pool.acquire() as conn:
            ucw_moments = await conn.fetch("""
                SELECT moment_id, confidence, coherence_type, platforms, description
                FROM coherence_moments
                WHERE description ILIKE '%ucw%'
                   OR description ILIKE '%sovereign%'
                   OR description ILIKE '%unify%'
                   OR description ILIKE '%coherence%'
                ORDER BY confidence DESC
                LIMIT 10
            """)

            for m in ucw_moments:
                results["founding_moments_found"].append({
                    "confidence": m["confidence"],
                    "type": m["coherence_type"],
                    "platforms": list(m["platforms"]),
                    "description": m["description"][:200] if m["description"] else "",
                })

        # Pass criteria: found cross-platform moments with UCW/sovereignty themes
        has_cross_platform = any(q["cross_platform"] for q in results["queries"])
        has_ucw_moments = len(results["founding_moments_found"]) > 0
        has_high_similarity = any(
            q["top_similarity"] > 0.60 for q in results["queries"]
        )

        results["passed"] = has_cross_platform and has_ucw_moments and has_high_similarity
        results["verdict"] = (
            "FOUNDING MOMENT VALIDATED" if results["passed"]
            else "Founding moment not fully detected — may need more data or lower thresholds"
        )

        return results

    async def _load_events(
        self,
        since: Optional[datetime],
        platform: Optional[str],
        limit: int,
    ) -> List[Dict]:
        """Load events from database."""
        async with self._pool.acquire() as conn:
            if since and platform:
                since_ns = int(since.timestamp() * 1_000_000_000)
                rows = await conn.fetch("""
                    SELECT event_id, session_id, timestamp_ns, platform,
                           data_layer, light_layer, instinct_layer, coherence_sig
                    FROM cognitive_events
                    WHERE timestamp_ns >= $1 AND platform = $2
                    ORDER BY timestamp_ns ASC
                    LIMIT $3
                """, since_ns, platform, limit)
            elif since:
                since_ns = int(since.timestamp() * 1_000_000_000)
                rows = await conn.fetch("""
                    SELECT event_id, session_id, timestamp_ns, platform,
                           data_layer, light_layer, instinct_layer, coherence_sig
                    FROM cognitive_events
                    WHERE timestamp_ns >= $1
                    ORDER BY timestamp_ns ASC
                    LIMIT $2
                """, since_ns, limit)
            else:
                rows = await conn.fetch("""
                    SELECT event_id, session_id, timestamp_ns, platform,
                           data_layer, light_layer, instinct_layer, coherence_sig
                    FROM cognitive_events
                    ORDER BY timestamp_ns ASC
                    LIMIT $1
                """, limit)

        return [dict(r) for r in rows]

    async def _ensure_embeddings(self, events: List[Dict], batch_size: int) -> int:
        """Ensure all events have embeddings. Returns count of embedded events."""
        # Check which are already embedded
        event_ids = [e["event_id"] for e in events]

        async with self._pool.acquire() as conn:
            existing = await conn.fetch(
                "SELECT source_event_id FROM embedding_cache WHERE source_event_id = ANY($1::text[])",
                event_ids,
            )
            existing_ids = {r["source_event_id"] for r in existing}

        need_embedding = [e for e in events if e["event_id"] not in existing_ids]
        log.info(f"Need to embed {len(need_embedding)} of {len(events)} events")

        if need_embedding:
            for i in range(0, len(need_embedding), batch_size):
                batch = need_embedding[i : i + batch_size]
                for event in batch:
                    try:
                        await embed_event_row(self._pool, event)
                    except Exception:
                        pass

        return len(existing_ids) + len(need_embedding)


def format_report(report: CoherenceReport) -> str:
    """Format a CoherenceReport as readable text."""
    out = "# Retroactive Coherence Analysis\n\n"
    out += f"**Events analyzed:** {report.total_events:,}\n"
    out += f"**Events embedded:** {report.events_embedded:,}\n"
    out += f"**Events with matches:** {report.events_with_matches:,}\n"
    out += f"**Moments detected:** {report.moments_found}\n"
    out += f"**Duration:** {report.duration_s:.1f}s\n\n"

    if report.by_type:
        out += "## By Detection Type\n\n"
        for t, count in sorted(report.by_type.items(), key=lambda x: -x[1]):
            out += f"- **{t}**: {count}\n"
        out += "\n"

    if report.platform_pairs:
        out += "## Platform Pairs\n\n"
        for pair, count in sorted(report.platform_pairs.items(), key=lambda x: -x[1]):
            out += f"- {pair}: {count} moments\n"
        out += "\n"

    if report.top_moments:
        out += "## Top Moments\n\n"
        for i, m in enumerate(report.top_moments[:10], 1):
            out += (
                f"### #{i} — {m['confidence']:.0%} — {m['type']}\n"
                f"Platforms: {m['platforms']}\n"
                f"> {m['description']}\n\n"
            )

    return out


def format_founding_test(results: Dict) -> str:
    """Format founding moment test results."""
    out = f"# {results['verdict']}\n\n"

    out += "## Semantic Search Probes\n\n"
    for q in results["queries"]:
        status = "CROSS-PLATFORM" if q["cross_platform"] else "single-platform"
        out += f"### \"{q['query']}\"\n"
        out += f"Matches: {q['matches']} | Platforms: {q['platforms']} | {status}\n"
        out += f"Top similarity: {q['top_similarity']:.0%}\n\n"

        for m in q["top_matches"]:
            out += f"- **{m['similarity']:.0%}** [{m['platform']}] {m['preview']}\n"
        out += "\n"

    if results["founding_moments_found"]:
        out += "## Founding Moments in Database\n\n"
        for m in results["founding_moments_found"]:
            out += (
                f"- **{m['confidence']:.0%}** {m['type']} | {m['platforms']}\n"
                f"  {m['description']}\n\n"
            )

    return out
