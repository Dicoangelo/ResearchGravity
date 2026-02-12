"""
Coherence Engine — CLI Entry Point

Commands:
  start             Run daemon in foreground (realtime mode by default)
  start --realtime  Explicit realtime mode (LISTEN/NOTIFY, default)
  start --poll      Legacy polling mode (every 10s)
  oneshot           Process all events once and exit
  status            Show engine status
  dashboard         Live TUI dashboard
  retroactive       Run retroactive analysis on historical data
  founding-moment   Validate the 2026-02-06 founding moment detection
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Suppress tqdm progress bars before any imports that use it
os.environ["TQDM_DISABLE"] = "1"

from .daemon import CoherenceDaemon
from . import config as cfg


def setup_logging():
    from .logging_config import setup_logging as _setup
    _setup("coherence")
    _setup("coherence.daemon")
    _setup("coherence.scorer")
    _setup("coherence.alerts")
    _setup("coherence.embeddings")
    _setup("coherence.realtime")
    # Suppress noisy libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


async def cmd_start():
    """Run the coherence daemon in foreground.

    Flags:
        --realtime   Use LISTEN/NOTIFY with 500ms micro-batch (default)
        --poll       Use legacy polling every POLL_INTERVAL_S seconds
    """
    # Default: realtime=True unless --poll is passed
    use_realtime = "--poll" not in sys.argv
    # Explicit --realtime overrides --poll if both are somehow present
    if "--realtime" in sys.argv:
        use_realtime = True

    mode_label = "realtime (LISTEN/NOTIFY)" if use_realtime else "poll"
    print(f"Starting coherence daemon in {mode_label} mode...")

    daemon = CoherenceDaemon()
    await daemon.run(realtime=use_realtime)


async def cmd_oneshot():
    """Process all events once and exit."""
    daemon = CoherenceDaemon()
    processed = await daemon.oneshot()
    print(f"Coherence scan complete: {processed} events processed")


async def cmd_status():
    """Show current coherence engine status."""
    import asyncpg

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=1, max_size=2)

    async with pool.acquire() as conn:
        events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        embedded = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
        moments = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")
        platforms = await conn.fetch(
            "SELECT platform, COUNT(*) as cnt FROM cognitive_events GROUP BY platform"
        )
        recent_moments = await conn.fetch(
            """SELECT coherence_type, confidence, description, platforms
               FROM coherence_moments
               ORDER BY detected_ns DESC LIMIT 5"""
        )

    await pool.close()

    print("=" * 60)
    print("  UCW COHERENCE ENGINE STATUS")
    print("=" * 60)
    print(f"  Events:     {events:,}")
    print(f"  Embedded:   {embedded:,}")
    print(f"  Moments:    {moments:,}")
    print()
    print("  Platforms:")
    for p in platforms:
        print(f"    {p['platform']:20s}  {p['cnt']:,} events")
    print()

    if recent_moments:
        print("  Recent Coherence Moments:")
        for m in recent_moments:
            plats = " <-> ".join(m["platforms"])
            print(f"    [{m['confidence']:.0%}] {m['coherence_type']}")
            print(f"          {plats}")
            print(f"          {m['description'][:80]}")
            print()
    else:
        print("  No coherence moments detected yet.")
        print("  Run: python3 -m coherence_engine oneshot")

    print("=" * 60)


async def cmd_dashboard():
    """Run the live TUI dashboard."""
    from .dashboard import CoherenceDashboard

    dashboard = CoherenceDashboard()
    await dashboard.run_live()


async def cmd_retroactive():
    """Run retroactive analysis on historical data."""
    import asyncpg
    from .retroactive import RetroactiveAnalyzer, format_report

    since = None
    if len(sys.argv) > 2 and sys.argv[2] == "--since" and len(sys.argv) > 3:
        since = datetime.fromisoformat(sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[2] == "--all":
        since = None
    else:
        # Default: last 7 days
        since = datetime.now().replace(hour=0, minute=0, second=0) - __import__("datetime").timedelta(days=7)

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=2, max_size=5)
    analyzer = RetroactiveAnalyzer(pool)

    print(f"Running retroactive analysis{f' since {since}' if since else ' (all time)'}...")
    report = await analyzer.analyze(since=since)
    print(format_report(report))

    await pool.close()


async def cmd_consolidate():
    """Run nightly consolidation (arcs, FSRS, significance, views)."""
    from .consolidation import ConsolidationDaemon

    daemon = ConsolidationDaemon()
    results = await daemon.start()
    print("Consolidation complete:")
    for key, val in results.items():
        print(f"  {key}: {val}")


async def cmd_graph():
    """Run knowledge graph entity extraction on cognitive events."""
    import asyncpg
    from .knowledge_graph import extract_and_ingest_batch, GraphManager

    batch_size = 5000
    start_offset = 0
    for arg in sys.argv[2:]:
        if arg.startswith("--batch="):
            batch_size = int(arg.split("=")[1])
        elif arg.startswith("--offset="):
            start_offset = int(arg.split("=")[1])

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=2, max_size=5)

    # Run in batches until no more events
    offset = start_offset
    total = {"events_processed": 0, "entities_created": 0, "edges_created": 0}
    while True:
        result = await extract_and_ingest_batch(pool, limit=batch_size, offset=offset)
        if result["events_processed"] == 0:
            break
        for k in total:
            total[k] += result[k]
        offset += batch_size
        print(f"  Batch done: offset={offset}, entities={total['entities_created']}, edges={total['edges_created']}")

    # Print stats
    graph = GraphManager(pool)
    stats = await graph.graph_stats()
    await pool.close()

    print(f"\nKnowledge Graph Extraction Complete:")
    print(f"  Events processed: {total['events_processed']:,}")
    print(f"  Entities created: {total['entities_created']:,}")
    print(f"  Edges created:    {total['edges_created']:,}")
    print(f"\nGraph Stats:")
    print(f"  Total entities: {stats['entity_count']:,}")
    print(f"  Total edges:    {stats['edge_count']:,}")
    if stats.get("entities_by_type"):
        print(f"  By type: {stats['entities_by_type']}")
    if stats.get("top_entities"):
        print(f"  Top entities:")
        for e in stats["top_entities"][:5]:
            print(f"    {e['name']} ({e['entity_type']}) — {e['mention_count']} mentions, {e['platform_count']} platforms")


async def cmd_insights():
    """Extract real insights from coherence moments via LLM synthesis."""
    import asyncpg
    from .insight_extractor import extract_insight_for_moment, backfill_insights

    limit = 72
    moment_id = None
    for arg in sys.argv[2:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])
        elif arg.startswith("--moment="):
            moment_id = arg.split("=")[1]

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=2, max_size=5)

    if moment_id:
        print(f"Extracting insight for moment {moment_id}...")
        result = await extract_insight_for_moment(pool, moment_id)
        if result:
            print(f"\n{'='*60}")
            print(f"  Moment:   {result.moment_id}")
            print(f"  Category: {result.category}")
            print(f"  Novelty:  {result.novelty:.2f}")
            print(f"  Summary:  {result.summary}")
            print(f"{'='*60}")
        else:
            print("Failed to extract insight.")
    else:
        print(f"Backfilling insights on up to {limit} moments...")
        results = await backfill_insights(pool, limit=limit)
        print(f"\nDone: {len(results)} insights extracted")
        for r in results[:10]:
            print(f"  [{r.category}] (novelty={r.novelty:.2f}) {r.summary[:80]}...")

    await pool.close()


async def cmd_emergence():
    """Scan for cognitive breakthroughs (emergence signals)."""
    import asyncpg
    from .emergence_listener import BreakthroughDetector

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=2, max_size=5)
    detector = BreakthroughDetector(pool)

    print("Scanning for emergence signals...")
    breakthroughs = await detector.scan_and_store()

    if breakthroughs:
        print(f"\n{len(breakthroughs)} breakthroughs detected:\n")
        for bt in breakthroughs:
            print(f"  [{bt.breakthrough_type}] {bt.title}")
            print(f"    Novelty: {bt.novelty_score:.2f} | Impact: {bt.impact_score:.2f}")
            print(f"    Platforms: {', '.join(bt.platforms)}")
            print(f"    {bt.narrative[:200]}...")
            print()
    else:
        print("\nNo breakthroughs detected. Continue working — patterns emerge over time.")

    await pool.close()


async def cmd_sessions():
    """Generate session embeddings and find cross-platform session coherence."""
    import asyncpg
    from .session_coherence import (
        backfill_session_embeddings,
        find_session_coherence,
    )

    subcmd = sys.argv[2] if len(sys.argv) > 2 else "embed"

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=2, max_size=5)

    if subcmd == "embed":
        limit = 5000
        for arg in sys.argv[3:]:
            if arg.startswith("--limit="):
                limit = int(arg.split("=")[1])
        print(f"Generating session embeddings (limit={limit})...")
        count = await backfill_session_embeddings(pool, limit=limit)
        print(f"Done: {count} session embeddings generated")

    elif subcmd == "coherence":
        hours = 168
        min_sim = 0.70
        for arg in sys.argv[3:]:
            if arg.startswith("--hours="):
                hours = int(arg.split("=")[1])
            elif arg.startswith("--min="):
                min_sim = float(arg.split("=")[1])
        print(f"Finding cross-platform session coherence (last {hours}h, min={min_sim})...")
        results = await find_session_coherence(pool, since_hours=hours, min_similarity=min_sim)
        print(f"\n{len(results)} session coherence pairs found:\n")
        for sc in results[:20]:
            print(f"  [{sc.similarity:.2f}] {sc.platform_a} <-> {sc.platform_b}")
            print(f"    Session A: {sc.session_a_id}")
            print(f"    Session B: {sc.session_b_id}")
            if sc.shared_themes:
                print(f"    Shared: {', '.join(sc.shared_themes[:5])}")
            print()

    else:
        print("Usage: python3 -m coherence_engine sessions <embed|coherence>")
        print("  embed       Generate session embeddings (--limit=N)")
        print("  coherence   Find cross-platform session pairs (--hours=N --min=0.7)")

    await pool.close()


async def cmd_founding_moment():
    """Run the Founding Moment Validation test."""
    import asyncpg
    from .retroactive import RetroactiveAnalyzer, format_founding_test

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=2, max_size=5)
    analyzer = RetroactiveAnalyzer(pool)

    print("Running Founding Moment Validation Test...")
    results = await analyzer.founding_moment_test()
    print(format_founding_test(results))

    await pool.close()


def main():
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python3 -m coherence_engine <command>")
        print()
        print("Commands:")
        print("  start             Run daemon (foreground, realtime LISTEN/NOTIFY)")
        print("    --realtime      Use LISTEN/NOTIFY mode (default)")
        print("    --poll          Use legacy polling mode (every 10s)")
        print("  oneshot           One-shot scan of all embedded events")
        print("  status            Show engine status")
        print("  dashboard         Live TUI dashboard")
        print("  retroactive       Retroactive analysis (--since YYYY-MM-DD | --all)")
        print("  consolidate       Nightly consolidation (arcs, FSRS, significance)")
        print("  graph             Extract entities into knowledge graph (--batch=N)")
        print("  insights          Extract real insights from moments via LLM (--moment=ID | --limit=N)")
        print("  emergence         Scan for cognitive breakthroughs")
        print("  sessions          Session embeddings + cross-platform coherence (embed|coherence)")
        print("  founding-moment   Validate 2026-02-06 founding moment detection")
        sys.exit(1)

    cmd = sys.argv[1]

    commands = {
        "start": cmd_start,
        "oneshot": cmd_oneshot,
        "status": cmd_status,
        "dashboard": cmd_dashboard,
        "retroactive": cmd_retroactive,
        "founding-moment": cmd_founding_moment,
        "consolidate": cmd_consolidate,
        "graph": cmd_graph,
        "insights": cmd_insights,
        "emergence": cmd_emergence,
        "sessions": cmd_sessions,
    }

    handler = commands.get(cmd)
    if not handler:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands.keys())}")
        sys.exit(1)

    asyncio.run(handler())


main()
