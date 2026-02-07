#!/usr/bin/env python3
"""
Cross-Platform Coherence Pipeline

Runs the full pipeline:
  1. Batch-embed any unembedded events (new CLI imports, etc.)
  2. Run cross-platform coherence analysis (ChatGPT vs Claude CLI)
  3. Generate a detailed report

Usage:
    python3 run_coherence_pipeline.py                  # Full pipeline
    python3 run_coherence_pipeline.py --embed-only      # Just embedding
    python3 run_coherence_pipeline.py --analyze-only    # Just analysis
    python3 run_coherence_pipeline.py --top 50          # Top N matches
"""

import asyncio
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, "/Users/dicoangelo/researchgravity")


async def get_pool():
    """Get asyncpg connection pool."""
    import asyncpg
    return await asyncpg.create_pool(
        "postgresql://localhost:5432/ucw_cognitive",
        min_size=2, max_size=8,
    )


async def get_db_stats(pool):
    """Get current database statistics."""
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        platforms = await conn.fetch(
            """SELECT platform, COUNT(*) as cnt, COUNT(DISTINCT session_id) as sessions
               FROM cognitive_events GROUP BY platform ORDER BY cnt DESC"""
        )
        embedded = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
        unembedded = await conn.fetchval(
            """SELECT COUNT(*) FROM cognitive_events ce
               WHERE NOT EXISTS (
                   SELECT 1 FROM embedding_cache ec
                   WHERE ec.source_event_id = ce.event_id
               )"""
        )
    return {
        "total_events": total,
        "platforms": [(r["platform"], r["cnt"], r["sessions"]) for r in platforms],
        "embedded": embedded,
        "unembedded": unembedded,
    }


async def batch_embed(pool, batch_size=500):
    """Embed all unembedded events using SBERT."""
    from mcp_raw.embeddings import EmbeddingPipeline

    pipeline = EmbeddingPipeline(pool)
    stats = await get_db_stats(pool)
    unembedded = stats["unembedded"]

    if unembedded == 0:
        print("  All events already embedded!")
        return 0

    print(f"  Embedding {unembedded} events (batch_size={batch_size})...")
    t0 = time.time()

    # Process in chunks to show progress
    total_stored = 0
    stall_count = 0
    while True:
        stored = await pipeline.batch_embed(limit=batch_size, skip_existing=True)
        if stored == 0:
            stall_count += 1
            if stall_count >= 3:
                # Remaining events are content dupes — stop
                print(f"    No new embeddings for 3 rounds — remaining events are content duplicates")
                break
        else:
            stall_count = 0
            total_stored += stored
            elapsed = time.time() - t0
            rate = total_stored / elapsed if elapsed > 0 else 0
            remaining = (unembedded - total_stored) / rate if rate > 0 else 0
            print(f"    Embedded {total_stored}/{unembedded} ({rate:.0f}/sec, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Done: {total_stored} events embedded in {elapsed:.1f}s")
    return total_stored


async def run_cross_platform_analysis(pool, top_n=100):
    """
    Cross-platform coherence analysis using pgvector.
    Finds semantic matches between Claude CLI and ChatGPT.
    """
    from mcp_raw.embeddings import EmbeddingPipeline

    pipeline = EmbeddingPipeline(pool)
    results = {}

    # 1. ChatGPT → Claude matches
    print("\n  Finding ChatGPT → Claude CLI coherence...")
    t0 = time.time()
    chatgpt_to_claude = await pipeline.find_cross_platform_matches(
        platform="chatgpt",
        threshold=0.65,
        limit=top_n,
    )
    results["chatgpt_to_claude"] = chatgpt_to_claude
    print(f"    Found {len(chatgpt_to_claude)} matches in {time.time()-t0:.1f}s")

    # 2. Claude CLI → ChatGPT matches
    print("  Finding Claude CLI → ChatGPT coherence...")
    t0 = time.time()
    claude_to_chatgpt = await pipeline.find_cross_platform_matches(
        platform="claude-code",
        threshold=0.65,
        limit=top_n,
    )
    results["claude_to_chatgpt"] = claude_to_chatgpt
    print(f"    Found {len(claude_to_chatgpt)} matches in {time.time()-t0:.1f}s")

    # 3. Sample queries — search for specific topics across platforms
    key_topics = [
        "multi-agent orchestration architecture",
        "sovereign AI infrastructure",
        "cognitive wallet data capture",
        "research session management",
        "career coaching system",
        "agentic kernel design",
        "cross-platform coherence detection",
        "embedding pipeline vector search",
    ]

    print(f"\n  Running {len(key_topics)} topic coherence probes...")
    topic_results = []
    for topic in key_topics:
        matches = await pipeline.find_similar_pgvector(
            topic, limit=10,
        )
        platforms_found = set(m["platform"] for m in matches if m["similarity"] > 0.5)
        top_sim = matches[0]["similarity"] if matches else 0
        topic_results.append({
            "topic": topic,
            "matches": len(matches),
            "platforms": list(platforms_found),
            "top_similarity": top_sim,
            "cross_platform": len(platforms_found) > 1,
        })
        cross = "CROSS" if len(platforms_found) > 1 else "single"
        print(f"    [{cross}] {topic}: {len(matches)} matches, top={top_sim:.3f}, platforms={list(platforms_found)}")

    results["topic_probes"] = topic_results

    # 4. Coherence distribution analysis
    print("\n  Analyzing similarity distribution...")
    async with pool.acquire() as conn:
        # Sample 2000 cross-platform pairs for distribution analysis
        dist_rows = await conn.fetch(
            """WITH sample_pairs AS (
                   SELECT ec1.embedding AS emb1, ce1.platform AS p1,
                          ec2.embedding AS emb2, ce2.platform AS p2
                   FROM embedding_cache ec1
                   JOIN cognitive_events ce1 ON ec1.source_event_id = ce1.event_id
                   CROSS JOIN LATERAL (
                       SELECT ec2.embedding, ec2.source_event_id
                       FROM embedding_cache ec2
                       JOIN cognitive_events ce2 ON ec2.source_event_id = ce2.event_id
                       WHERE ce2.platform != ce1.platform
                       ORDER BY ec2.embedding <=> ec1.embedding
                       LIMIT 1
                   ) ec2
                   JOIN cognitive_events ce2 ON ec2.source_event_id = ce2.event_id
                   WHERE ce1.platform IN ('chatgpt', 'claude-code')
                   ORDER BY RANDOM()
                   LIMIT 2000
               )
               SELECT p1, p2, 1 - (emb1 <=> emb2) AS similarity
               FROM sample_pairs"""
        )

    if dist_rows:
        sims = [float(r["similarity"]) for r in dist_rows]
        import statistics
        results["distribution"] = {
            "sample_size": len(sims),
            "mean": round(statistics.mean(sims), 4),
            "median": round(statistics.median(sims), 4),
            "stdev": round(statistics.stdev(sims), 4) if len(sims) > 1 else 0,
            "min": round(min(sims), 4),
            "max": round(max(sims), 4),
            "above_70": sum(1 for s in sims if s >= 0.7),
            "above_80": sum(1 for s in sims if s >= 0.8),
            "above_90": sum(1 for s in sims if s >= 0.9),
        }
        print(f"    Sampled {len(sims)} cross-platform pairs")
        print(f"    Mean similarity: {results['distribution']['mean']}")
        print(f"    Median: {results['distribution']['median']}")
        print(f"    >0.7: {results['distribution']['above_70']}, >0.8: {results['distribution']['above_80']}, >0.9: {results['distribution']['above_90']}")

    return results


def generate_report(stats, analysis, output_path):
    """Generate a formatted coherence report."""
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("=" * 72)
    lines.append("  UNIVERSAL COGNITIVE WALLET — Cross-Platform Coherence Report")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 72)

    # Database overview
    lines.append("\n## Database Overview\n")
    lines.append(f"Total events: {stats['total_events']:,}")
    lines.append(f"Total embeddings: {stats['embedded']:,}")
    lines.append(f"Unembedded: {stats['unembedded']:,}")
    lines.append("")
    lines.append(f"{'Platform':<20} {'Events':>10} {'Sessions':>10}")
    lines.append("-" * 42)
    for plat, cnt, sess in stats["platforms"]:
        lines.append(f"{plat:<20} {cnt:>10,} {sess:>10,}")

    # Cross-platform matches
    for direction, key in [
        ("ChatGPT → Claude CLI", "chatgpt_to_claude"),
        ("Claude CLI → ChatGPT", "claude_to_chatgpt"),
    ]:
        matches = analysis.get(key, [])
        lines.append(f"\n## {direction} Coherence ({len(matches)} matches)\n")
        if matches:
            lines.append(f"{'#':>3} {'Similarity':>10} {'Src Mode':<12} {'Tgt Platform':<15} {'Tgt Mode':<12}")
            lines.append("-" * 55)
            for i, m in enumerate(matches[:30]):
                lines.append(
                    f"{i+1:>3} {m['similarity']:>10.4f} "
                    f"{m.get('source_mode','?'):<12} "
                    f"{m.get('target_platform','?'):<15} "
                    f"{m.get('target_mode','?'):<12}"
                )
            if matches:
                lines.append(f"\nTop match ({matches[0]['similarity']:.4f}):")
                lines.append(f"  Source: {matches[0].get('source_preview','')[:100]}")
                lines.append(f"  Target: {matches[0].get('target_preview','')[:100]}")

    # Topic probes
    topic_probes = analysis.get("topic_probes", [])
    lines.append(f"\n## Topic Coherence Probes ({len(topic_probes)} topics)\n")
    cross_count = sum(1 for t in topic_probes if t["cross_platform"])
    lines.append(f"Cross-platform coherence detected: {cross_count}/{len(topic_probes)} topics\n")
    for t in topic_probes:
        status = "CROSS-PLATFORM" if t["cross_platform"] else "single-platform"
        lines.append(f"  [{status}] {t['topic']}")
        lines.append(f"    Matches: {t['matches']}, Top sim: {t['top_similarity']:.3f}, Platforms: {t['platforms']}")

    # Distribution
    dist = analysis.get("distribution", {})
    if dist:
        lines.append(f"\n## Similarity Distribution (n={dist['sample_size']})\n")
        lines.append(f"  Mean:   {dist['mean']:.4f}")
        lines.append(f"  Median: {dist['median']:.4f}")
        lines.append(f"  StDev:  {dist['stdev']:.4f}")
        lines.append(f"  Min:    {dist['min']:.4f}")
        lines.append(f"  Max:    {dist['max']:.4f}")
        lines.append(f"  >0.7:   {dist['above_70']} pairs ({dist['above_70']/dist['sample_size']*100:.1f}%)")
        lines.append(f"  >0.8:   {dist['above_80']} pairs ({dist['above_80']/dist['sample_size']*100:.1f}%)")
        lines.append(f"  >0.9:   {dist['above_90']} pairs ({dist['above_90']/dist['sample_size']*100:.1f}%)")

    # Summary
    lines.append(f"\n{'=' * 72}")
    lines.append("  COHERENCE SUMMARY")
    lines.append(f"{'=' * 72}")
    total_matches = len(analysis.get("chatgpt_to_claude", [])) + len(analysis.get("claude_to_chatgpt", []))
    peak_sim = 0
    for key in ["chatgpt_to_claude", "claude_to_chatgpt"]:
        matches = analysis.get(key, [])
        if matches:
            peak_sim = max(peak_sim, matches[0]["similarity"])
    lines.append(f"  Total cross-platform matches: {total_matches}")
    lines.append(f"  Peak similarity: {peak_sim:.4f}")
    lines.append(f"  Cross-platform topics: {cross_count}/{len(topic_probes)}")
    if dist:
        lines.append(f"  Mean cross-platform similarity: {dist['mean']:.4f}")
    lines.append(f"{'=' * 72}\n")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    return report


async def main():
    parser = argparse.ArgumentParser(description="UCW Cross-Platform Coherence Pipeline")
    parser.add_argument("--embed-only", action="store_true", help="Only run batch embedding")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis")
    parser.add_argument("--top", type=int, default=100, help="Top N matches per direction")
    parser.add_argument("--output", default="/Users/dicoangelo/researchgravity/coherence_report_full.txt")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  UCW Cross-Platform Coherence Pipeline")
    print("=" * 60)

    pool = await get_pool()

    try:
        # Stats
        stats = await get_db_stats(pool)
        print(f"\n  Database: {stats['total_events']:,} events, {stats['embedded']:,} embedded, {stats['unembedded']:,} unembedded")
        for plat, cnt, sess in stats["platforms"]:
            print(f"    {plat}: {cnt:,} events ({sess:,} sessions)")

        # Step 1: Batch embed
        if not args.analyze_only:
            print(f"\n--- Step 1: Batch Embedding ---")
            newly_embedded = await batch_embed(pool)
            # Refresh stats
            stats = await get_db_stats(pool)
            print(f"  Post-embed: {stats['embedded']:,} total embeddings")

        # Step 2: Cross-platform analysis
        if not args.embed_only:
            print(f"\n--- Step 2: Cross-Platform Coherence Analysis ---")
            analysis = await run_cross_platform_analysis(pool, top_n=args.top)

            # Step 3: Report
            print(f"\n--- Step 3: Generating Report ---")
            report = generate_report(stats, analysis, args.output)
            print(f"  Report saved to: {args.output}")

            # Also save raw JSON
            json_path = args.output.replace(".txt", ".json")
            with open(json_path, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"  JSON data saved to: {json_path}")

    finally:
        await pool.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
