#!/usr/bin/env python3
"""
Sync SQLite → Qdrant: rebuild all embeddings with current model.

Usage:
    python3 scripts/sync_qdrant.py              # Incremental (skip if counts match)
    python3 scripts/sync_qdrant.py --rebuild     # Force re-embed everything
"""

import asyncio
import sqlite3
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.qdrant_db import QdrantDB, QDRANT_AVAILABLE

DB_PATH = os.path.expanduser("~/.agent-core/storage/antigravity.db")
BATCH_SIZE = 96  # Cohere limit per request

FORCE_REBUILD = "--rebuild" in sys.argv


def get_sqlite_data(table, columns, where=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    query = f"SELECT {', '.join(columns)} FROM {table}"
    if where:
        query += f" WHERE {where}"
    rows = [dict(r) for r in conn.execute(query).fetchall()]
    conn.close()
    return rows


async def get_qdrant_count(qdrant: QdrantDB, collection: str) -> int:
    try:
        info = await qdrant.async_client.get_collection(collection)
        return info.points_count
    except Exception:
        return 0


async def sync_collection(qdrant, name, sqlite_rows, upsert_fn):
    """Generic sync for any collection."""
    qdrant_count = await get_qdrant_count(qdrant, name)
    delta = len(sqlite_rows) - qdrant_count

    print(f"\n  {name}: SQLite={len(sqlite_rows):,} | Qdrant={qdrant_count:,} | Delta={delta}")

    if not FORCE_REBUILD and len(sqlite_rows) <= qdrant_count:
        print("    Already synced. Use --rebuild to force re-embed.")
        return 0

    if FORCE_REBUILD:
        print(f"    REBUILDING all {len(sqlite_rows):,} embeddings with current model...")

    total = 0
    batches = (len(sqlite_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(sqlite_rows), BATCH_SIZE):
        batch = sqlite_rows[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        try:
            count = await upsert_fn(batch)
            total += count
            pct = total / len(sqlite_rows) * 100
            print(f"    [{batch_num}/{batches}] +{count} ({total:,}/{len(sqlite_rows):,} = {pct:.0f}%)")
        except Exception as e:
            print(f"    [{batch_num}/{batches}] ERROR: {e}")

    return total


async def main():
    print("=" * 60)
    print("  SQLite → Qdrant Embedding Sync")
    if FORCE_REBUILD:
        print("  MODE: FULL REBUILD (re-embed everything)")
    else:
        print("  MODE: Incremental (skip synced collections)")
    print("=" * 60)
    print(f"  Source: {DB_PATH}")
    print(f"  Target: localhost:6333")

    if not QDRANT_AVAILABLE:
        print("ERROR: qdrant-client not installed")
        sys.exit(1)

    qdrant = QdrantDB()
    await qdrant.initialize()

    # Verify Cohere is working
    try:
        test_emb = qdrant.embed("test")
        print(f"  Embedding model: Cohere embed-v4 ({len(test_emb)}d)")
        print(f"  Cohere fallback: {qdrant._use_sbert_fallback}")
    except Exception as e:
        print(f"  WARNING: Cohere failed ({e}), using SBERT fallback")

    start = time.time()
    grand_total = 0

    # Findings (33K)
    findings = get_sqlite_data(
        "findings",
        ["id", "content", "type", "session_id", "project", "confidence"],
        where="content IS NOT NULL AND content != ''"
    )
    grand_total += await sync_collection(qdrant, "findings", findings, qdrant.upsert_findings_batch)

    # Sessions (8K)
    sessions = get_sqlite_data(
        "sessions",
        ["id", "topic", "project", "status", "finding_count", "url_count"],
        where="topic IS NOT NULL AND topic != ''"
    )
    grand_total += await sync_collection(qdrant, "sessions", sessions, qdrant.upsert_sessions_batch)

    # Session outcomes (670)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    outcomes = [dict(r) for r in conn.execute("""
        SELECT session_id, intent, outcome, quality, model_efficiency,
               models_used, date, messages, tools
        FROM session_outcomes WHERE intent IS NOT NULL AND intent != ''
    """).fetchall()]
    conn.close()
    grand_total += await sync_collection(qdrant, "session_outcomes", outcomes, qdrant.upsert_outcomes_batch)

    # Cognitive states (549)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    states = [dict(r) for r in conn.execute("""
        SELECT id, mode, energy_level, flow_score, hour, day, timestamp, predictions
        FROM cognitive_states
    """).fetchall()]
    conn.close()
    grand_total += await sync_collection(qdrant, "cognitive_states", states, qdrant.upsert_cognitive_states_batch)

    # Error patterns (39)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    errors = [dict(r) for r in conn.execute("""
        SELECT id, error_type, context, solution, success_rate
        FROM error_patterns
    """).fetchall()]
    conn.close()
    grand_total += await sync_collection(qdrant, "error_patterns", errors, qdrant.upsert_error_patterns_batch)

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  SYNC COMPLETE")
    print(f"  Total embedded: {grand_total:,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\n  Final Qdrant counts:")
    total_vectors = 0
    for collection in ["findings", "sessions", "session_outcomes", "cognitive_states", "error_patterns"]:
        count = await get_qdrant_count(qdrant, collection)
        total_vectors += count
        print(f"    {collection}: {count:,}")
    print(f"    TOTAL: {total_vectors:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
