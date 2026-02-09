"""
Coherence Engine — Nightly Consolidation Daemon

Runs at 2 AM (configurable) to consolidate the day's cognitive activity:
  1. Detect and store coherence arcs from new moments
  2. Schedule new high-confidence insights for FSRS review
  3. Refresh materialized views for analytics
  4. Run significance tests on unvalidated moments
  5. Log consolidation summary

Like human sleep, this "replays" the day's cognitive events to
strengthen connections and prepare insights for resurfacing.

Usage:
    python3 -m coherence_engine consolidate     # Run once
    # Or via LaunchAgent (com.ucw.consolidation.plist) at 2 AM
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

import asyncpg

from . import config as cfg
from .significance import ArcDetector, SignificanceTester
from .fsrs import InsightScheduler
from .knowledge_graph import extract_and_ingest_batch

log = logging.getLogger("coherence.consolidation")


class ConsolidationDaemon:
    """Nightly consolidation of cognitive activity."""

    def __init__(self):
        self._pool = None

    async def start(self):
        """Run a single consolidation pass."""
        t0 = time.time()
        log.info("=== Consolidation starting ===")

        self._pool = await asyncpg.create_pool(
            cfg.PG_DSN, min_size=1, max_size=5,
        )

        results = {}

        try:
            # 1. Arc detection
            results["arcs"] = await self._detect_arcs()

            # 2. FSRS scheduling
            results["insights"] = await self._schedule_insights()

            # 3. Significance testing on recent moments
            results["significance"] = await self._test_significance()

            # 4. Knowledge graph extraction (new events since last run)
            results["graph"] = await self._extract_entities()

            # 5. Refresh materialized views
            results["views"] = await self._refresh_views()

            # 6. Summary stats
            results["stats"] = await self._gather_stats()

        except Exception as e:
            log.error(f"Consolidation error: {e}")
            results["error"] = str(e)
        finally:
            if self._pool:
                await self._pool.close()

        elapsed = time.time() - t0
        log.info(f"=== Consolidation complete in {elapsed:.1f}s ===")
        for key, val in results.items():
            log.info(f"  {key}: {val}")

        return results

    async def _detect_arcs(self) -> str:
        """Detect and store coherence arcs."""
        detector = ArcDetector(pool=self._pool, overlap_threshold=0.15)
        arcs = await detector.detect_arcs(limit=500)
        multi = [a for a in arcs if a.moment_count >= 2]
        await detector.store_arcs(arcs)
        return f"{len(multi)} arcs ({sum(a.moment_count for a in multi)} moments)"

    async def _schedule_insights(self) -> str:
        """Schedule new high-confidence moments for FSRS review."""
        scheduler = InsightScheduler(pool=self._pool)
        await scheduler.ensure_schema()
        count = await scheduler.schedule_new_moments(min_confidence=0.82)
        due = await scheduler.get_due_insights(limit=100)
        return f"{count} new scheduled, {len(due)} due for review"

    async def _test_significance(self) -> str:
        """Run significance tests on recent unvalidated moments."""
        import json

        async with self._pool.acquire() as conn:
            # Get moments without significance metadata
            rows = await conn.fetch(
                """SELECT moment_id, event_ids, confidence, metadata
                   FROM coherence_moments
                   WHERE confidence >= 0.82
                     AND (metadata IS NULL OR metadata::text NOT LIKE '%p_value%')
                   ORDER BY detected_ns DESC
                   LIMIT 50""",
            )

        if not rows:
            return "0 tested (all validated)"

        tester = SignificanceTester(pool=self._pool, n_permutations=50)
        tested = 0
        significant = 0

        for row in rows:
            event_ids = row["event_ids"]
            if not event_ids or len(event_ids) < 2:
                continue

            async with self._pool.acquire() as conn:
                ev_a = await conn.fetchrow(
                    "SELECT * FROM cognitive_events WHERE event_id = $1", event_ids[0]
                )
                ev_b = await conn.fetchrow(
                    "SELECT * FROM cognitive_events WHERE event_id = $1", event_ids[1]
                )

            if not ev_a or not ev_b:
                continue

            ea, eb = dict(ev_a), dict(ev_b)
            for fld in ("light_layer", "instinct_layer", "data_layer"):
                for e in (ea, eb):
                    if isinstance(e.get(fld), str):
                        e[fld] = json.loads(e[fld])

            result = await tester.test(ea, eb, row["confidence"], row["moment_id"])
            tested += 1
            if result.is_significant:
                significant += 1

            # Store significance result in metadata
            existing_meta = json.loads(row["metadata"] or "{}") if row["metadata"] else {}
            existing_meta["p_value"] = round(result.p_value, 4)
            existing_meta["z_score"] = round(result.z_score, 2)
            existing_meta["significant"] = result.is_significant

            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE coherence_moments SET metadata = $1::jsonb WHERE moment_id = $2",
                    json.dumps(existing_meta), row["moment_id"],
                )

        tester.clear_cache()
        return f"{tested} tested, {significant} significant"

    async def _extract_entities(self) -> str:
        """Extract entities from recent events into the knowledge graph."""
        # Get the count of events already processed (via entity mentions)
        async with self._pool.acquire() as conn:
            last_entity_ns = await conn.fetchval(
                "SELECT COALESCE(MAX(last_seen_ns), 0) FROM cognitive_entities"
            )
            new_events = await conn.fetchval(
                "SELECT COUNT(*) FROM cognitive_events WHERE timestamp_ns > $1 AND light_layer IS NOT NULL",
                last_entity_ns,
            )

        if new_events == 0:
            return "0 new events"

        # Process in batches of 2000
        total = {"events_processed": 0, "entities_created": 0, "edges_created": 0}
        offset = 0
        while True:
            result = await extract_and_ingest_batch(self._pool, limit=2000, offset=offset)
            if result["events_processed"] == 0:
                break
            for k in total:
                total[k] += result[k]
            offset += 2000
            if offset > new_events + 2000:  # Safety cap
                break

        return f"{total['entities_created']} entities, {total['edges_created']} edges from {total['events_processed']} events"

    async def _refresh_views(self) -> str:
        """Refresh materialized views."""
        views_refreshed = 0
        async with self._pool.acquire() as conn:
            for view in ("mv_platform_coherence",):
                try:
                    await conn.execute(f"REFRESH MATERIALIZED VIEW {view}")
                    views_refreshed += 1
                except Exception as e:
                    log.warning(f"Failed to refresh {view}: {e}")

        return f"{views_refreshed} views refreshed"

    async def _gather_stats(self) -> str:
        """Gather summary statistics."""
        async with self._pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
            moments = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")
            embeddings = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
            migrated = await conn.fetchval(
                "SELECT COUNT(*) FROM embedding_cache WHERE embedding_768 IS NOT NULL"
            )

        return (
            f"events={total:,} moments={moments} "
            f"embeddings={embeddings:,} migrated_768={migrated:,}"
        )
