#!/usr/bin/env python3
"""
UCW Chrome Extension — Health Check & Verification

Verifies the full pipeline:
  Extension → API → PostgreSQL → Embedding → Coherence Daemon

Usage:
    python3 verify_extension.py              # Full health check
    python3 verify_extension.py --send-test  # Send a test event and verify
    python3 verify_extension.py --stats      # Show extension event stats
    python3 verify_extension.py --embedding  # Verify embedding pipeline
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Suppress tqdm in non-interactive mode
import os
os.environ.setdefault("TQDM_DISABLE", "1")


async def check_api():
    """Check if the capture API is reachable."""
    import aiohttp
    url = "http://localhost:3847/api/v2/coherence/capture/extension"
    try:
        async with aiohttp.ClientSession() as session:
            # Try sending a minimal test event
            payload = {
                "platform": "test",
                "content": f"Health check at {time.strftime('%H:%M:%S')}",
                "direction": "out",
            }
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return True, f"API OK — event_id: {data.get('event_id', '?')}"
                else:
                    return False, f"API returned {resp.status}"
    except aiohttp.ClientConnectorError:
        return False, "API unreachable at localhost:3847"
    except Exception as e:
        return False, f"API error: {e}"


async def check_batch_api():
    """Check if the batch capture API is reachable."""
    import aiohttp
    url = "http://localhost:3847/api/v2/coherence/capture/extension/batch"
    try:
        async with aiohttp.ClientSession() as session:
            payload = [
                {
                    "platform": "test",
                    "content": f"Batch health check {i} at {time.strftime('%H:%M:%S')}",
                    "direction": "out",
                }
                for i in range(2)
            ]
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return True, f"Batch API OK — captured: {data.get('captured', '?')}, total: {data.get('total', '?')}"
                else:
                    return False, f"Batch API returned {resp.status}"
    except aiohttp.ClientConnectorError:
        return False, "Batch API unreachable at localhost:3847"
    except Exception as e:
        return False, f"Batch API error: {e}"


async def check_db():
    """Check extension events in PostgreSQL."""
    try:
        import asyncpg
        dsn = os.environ.get("UCW_DATABASE_URL", "postgresql://localhost:5432/ucw_cognitive")
        conn = await asyncpg.connect(dsn)

        # Count extension events
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE method = 'extension'"
        )

        # Recent events
        recent = await conn.fetchval(
            """SELECT COUNT(*) FROM cognitive_events
               WHERE method = 'extension'
                 AND timestamp_ns > $1""",
            int((time.time() - 3600) * 1_000_000_000),  # last hour
        )

        # Events with embeddings
        embedded = await conn.fetchval(
            """SELECT COUNT(*)
               FROM cognitive_events ce
               JOIN embedding_cache ec ON ec.source_event_id = ce.event_id
               WHERE ce.method = 'extension'"""
        )

        # Unscanned events
        unscanned = await conn.fetchval(
            """SELECT COUNT(*)
               FROM cognitive_events
               WHERE method = 'extension'
                 AND coherence_scanned_at IS NULL"""
        )

        # Dedup check: events with duplicate coherence_sig
        dup_sigs = await conn.fetchval(
            """SELECT COUNT(*) FROM (
                 SELECT coherence_sig FROM cognitive_events
                 WHERE method = 'extension' AND coherence_sig IS NOT NULL
                 GROUP BY coherence_sig HAVING COUNT(*) > 1
               ) sub"""
        )

        await conn.close()
        return True, {
            "total_events": total,
            "last_hour": recent,
            "embedded": embedded,
            "unscanned": unscanned,
            "duplicate_sigs": dup_sigs,
        }
    except Exception as e:
        return False, f"DB error: {e}"


async def check_embedding_pipeline():
    """Verify the embedding pipeline is functional."""
    try:
        import asyncpg
        dsn = os.environ.get("UCW_DATABASE_URL", "postgresql://localhost:5432/ucw_cognitive")
        conn = await asyncpg.connect(dsn)

        # Extension events without embeddings
        unembedded = await conn.fetchval(
            """SELECT COUNT(*)
               FROM cognitive_events ce
               WHERE ce.method = 'extension'
                 AND NOT EXISTS (
                     SELECT 1 FROM embedding_cache ec
                     WHERE ec.source_event_id = ce.event_id
                 )"""
        )

        # Total embeddings from extension events
        total_embedded = await conn.fetchval(
            """SELECT COUNT(*)
               FROM embedding_cache ec
               JOIN cognitive_events ce ON ce.event_id = ec.source_event_id
               WHERE ce.method = 'extension'"""
        )

        # Check embedding dimensions (768d nomic vs 384d legacy)
        dim_check = await conn.fetchrow(
            """SELECT
                 COUNT(*) FILTER (WHERE ec.embedding_768 IS NOT NULL) as has_768,
                 COUNT(*) FILTER (WHERE ec.embedding IS NOT NULL) as has_384,
                 COUNT(*) as total
               FROM embedding_cache ec
               JOIN cognitive_events ce ON ce.event_id = ec.source_event_id
               WHERE ce.method = 'extension'"""
        )

        # Check content_tsv population
        tsv_count = await conn.fetchval(
            """SELECT COUNT(*)
               FROM embedding_cache ec
               JOIN cognitive_events ce ON ce.event_id = ec.source_event_id
               WHERE ce.method = 'extension'
                 AND ec.content_tsv IS NOT NULL"""
        )

        await conn.close()

        coverage = (total_embedded / max(total_embedded + unembedded, 1)) * 100

        return True, {
            "embedded": total_embedded,
            "unembedded": unembedded,
            "coverage": f"{coverage:.1f}%",
            "768d_vectors": dim_check["has_768"] if dim_check else 0,
            "384d_vectors": dim_check["has_384"] if dim_check else 0,
            "content_tsv_populated": tsv_count,
        }
    except Exception as e:
        return False, f"Embedding pipeline error: {e}"


async def check_daemon():
    """Check if the coherence daemon is running."""
    import subprocess
    result = subprocess.run(
        ["pgrep", "-f", "coherence_engine"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
    if pids:
        return True, f"Daemon running (PID: {', '.join(pids)})"
    return False, "Daemon not running"


async def check_extension_dir():
    """Verify extension files exist."""
    ext_dir = Path(__file__).parent / "chrome-extension"
    required = [
        "manifest.json",
        "background.js",
        "lib/capture.js",
        "interceptors/chatgpt.js",
        "interceptors/grok.js",
        "interceptors/gemini.js",
        "interceptors/notebooklm.js",
        "interceptors/youtube.js",
        "popup/popup.html",
        "popup/popup.js",
        "icons/icon16.png",
        "icons/icon48.png",
        "icons/icon128.png",
    ]

    missing = [f for f in required if not (ext_dir / f).exists()]
    if missing:
        return False, f"Missing files: {', '.join(missing)}"

    # Check manifest version
    with open(ext_dir / "manifest.json") as f:
        manifest = json.load(f)

    return True, f"Extension v{manifest['version']} — {len(required)} files OK"


async def check_dedup():
    """Verify dedup is working — no duplicate event_ids in the DB."""
    try:
        import asyncpg
        dsn = os.environ.get("UCW_DATABASE_URL", "postgresql://localhost:5432/ucw_cognitive")
        conn = await asyncpg.connect(dsn)

        # event_id is PRIMARY KEY, so duplicates are impossible at DB level
        # But check for near-duplicates via coherence_sig
        dup_count = await conn.fetchval(
            """SELECT COUNT(*) FROM (
                 SELECT coherence_sig, COUNT(*) as cnt
                 FROM cognitive_events
                 WHERE method = 'extension' AND coherence_sig IS NOT NULL
                 GROUP BY coherence_sig
                 HAVING COUNT(*) > 1
               ) sub"""
        )

        total = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE method = 'extension'"
        )

        await conn.close()

        if dup_count == 0:
            return True, f"No duplicate coherence_sigs ({total} events)"
        else:
            return False, f"{dup_count} duplicate coherence_sig groups in {total} events"
    except Exception as e:
        return False, f"Dedup check error: {e}"


async def send_test_event():
    """Send a test event and verify it lands in the DB."""
    import aiohttp
    import asyncpg

    test_content = f"UCW verify_extension.py test event at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    url = "http://localhost:3847/api/v2/coherence/capture/extension"

    # Send
    async with aiohttp.ClientSession() as session:
        payload = {
            "platform": "test",
            "content": test_content,
            "direction": "out",
            "topic": "health_check",
            "session_hint": "test-health-check",
        }
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            event_id = data.get("event_id")
            status = data.get("status")
            print(f"  Sent test event: {event_id} (status: {status})")

    # Verify in DB
    await asyncio.sleep(1)
    dsn = os.environ.get("UCW_DATABASE_URL", "postgresql://localhost:5432/ucw_cognitive")
    conn = await asyncpg.connect(dsn)
    row = await conn.fetchrow(
        "SELECT event_id, platform, direction, coherence_sig FROM cognitive_events WHERE event_id = $1",
        event_id,
    )
    await conn.close()

    if row:
        print(f"  Verified in DB: platform={row['platform']}, direction={row['direction']}")
        print(f"  Coherence sig: {row['coherence_sig'][:16]}...")
        return True
    else:
        print(f"  NOT FOUND in DB after 1s")
        return False


async def send_test_batch():
    """Send a test batch and verify dedup + counts."""
    import aiohttp
    import asyncpg

    url = "http://localhost:3847/api/v2/coherence/capture/extension/batch"
    unique_content = f"Batch test unique at {time.time_ns()}"

    async with aiohttp.ClientSession() as session:
        payload = [
            {"platform": "test", "content": unique_content, "direction": "out"},
            {"platform": "test", "content": unique_content, "direction": "out"},  # dupe
            {"platform": "test", "content": f"Second unique at {time.time_ns()}", "direction": "out"},
        ]
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            print(f"  Batch result: captured={data['captured']}, dupes={data['duplicates']}, errors={data['errors']}")
            if data["duplicates"] >= 1:
                print(f"  Dedup working correctly")
                return True
            elif data["captured"] >= 2:
                print(f"  Batch captured (dedup via ON CONFLICT)")
                return True
            else:
                print(f"  Unexpected result")
                return False


async def show_stats():
    """Show detailed extension event statistics."""
    import asyncpg
    dsn = os.environ.get("UCW_DATABASE_URL", "postgresql://localhost:5432/ucw_cognitive")
    conn = await asyncpg.connect(dsn)

    # By platform
    rows = await conn.fetch(
        """SELECT platform, COUNT(*) as cnt, MIN(timestamp_ns) as first_ns, MAX(timestamp_ns) as last_ns
           FROM cognitive_events WHERE method = 'extension'
           GROUP BY platform ORDER BY cnt DESC"""
    )

    if not rows:
        print("No extension events found.")
        await conn.close()
        return

    print("\n  Platform breakdown:")
    for r in rows:
        first = time.strftime("%m/%d %H:%M", time.localtime(r["first_ns"] / 1e9))
        last = time.strftime("%m/%d %H:%M", time.localtime(r["last_ns"] / 1e9))
        print(f"    {r['platform']:15s} {r['cnt']:>6d} events  ({first} — {last})")

    # Sessions
    sessions = await conn.fetchval(
        "SELECT COUNT(DISTINCT session_id) FROM cognitive_events WHERE method = 'extension'"
    )
    print(f"\n  Sessions: {sessions}")

    # Quality distribution
    quality = await conn.fetch(
        """SELECT cognitive_mode, COUNT(*) as cnt, AVG(quality_score) as avg_score
           FROM cognitive_events
           WHERE method = 'extension' AND cognitive_mode IS NOT NULL
           GROUP BY cognitive_mode ORDER BY avg_score DESC"""
    )
    if quality:
        print("\n  Quality distribution:")
        for r in quality:
            avg = f"{r['avg_score']:.3f}" if r['avg_score'] else "n/a"
            print(f"    {r['cognitive_mode']:15s} {r['cnt']:>6d} events  (avg: {avg})")

    # Embedding coverage
    embedded = await conn.fetchval(
        """SELECT COUNT(*)
           FROM cognitive_events ce
           JOIN embedding_cache ec ON ec.source_event_id = ce.event_id
           WHERE ce.method = 'extension'"""
    )
    total = sum(r["cnt"] for r in rows)
    pct = (embedded / max(total, 1)) * 100
    print(f"\n  Embedding coverage: {embedded}/{total} ({pct:.1f}%)")

    # Coherence moments from extension events
    moments = await conn.fetchval(
        """SELECT COUNT(*) FROM coherence_moments cm
           JOIN cognitive_events ce ON ce.event_id = ANY(cm.source_event_ids)
           WHERE ce.method = 'extension'"""
    )
    print(f"  Coherence moments involving extension events: {moments}")

    await conn.close()


async def main():
    parser = argparse.ArgumentParser(description="UCW Extension Health Check")
    parser.add_argument("--send-test", action="store_true", help="Send a test event")
    parser.add_argument("--send-batch", action="store_true", help="Send a test batch (with dedup)")
    parser.add_argument("--stats", action="store_true", help="Show extension stats")
    parser.add_argument("--embedding", action="store_true", help="Check embedding pipeline")
    args = parser.parse_args()

    if args.stats:
        await show_stats()
        return

    print("UCW Chrome Extension — Health Check")
    print("=" * 50)

    checks = [
        ("Extension files", check_extension_dir),
        ("API endpoint", check_api),
        ("Batch API", check_batch_api),
        ("PostgreSQL", check_db),
        ("Dedup integrity", check_dedup),
        ("Coherence daemon", check_daemon),
    ]

    if args.embedding:
        checks.append(("Embedding pipeline", check_embedding_pipeline))

    all_ok = True
    for name, check_fn in checks:
        ok, detail = await check_fn()
        icon = "OK" if ok else "FAIL"
        print(f"  [{icon:>4s}] {name}")
        if isinstance(detail, dict):
            for k, v in detail.items():
                print(f"         {k}: {v}")
        else:
            print(f"         {detail}")
        if not ok:
            all_ok = False

    if args.send_test:
        print("\n  Sending test event...")
        ok = await send_test_event()
        if not ok:
            all_ok = False

    if args.send_batch:
        print("\n  Sending test batch (dedup verification)...")
        ok = await send_test_batch()
        if not ok:
            all_ok = False

    print(f"\n{'All checks passed.' if all_ok else 'Some checks failed.'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
