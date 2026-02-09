#!/usr/bin/env python3
"""
Migrate embeddings from all-MiniLM-L6-v2 (384d) to nomic-embed-text-v1.5 (768d).

Streams rows in chunks to avoid OOM. Safe to re-run (skips already-migrated rows).
Logs progress to ~/.ucw/logs/embedding_migration.log.

Usage:
    python3 migrate_embeddings.py              # Full migration
    python3 migrate_embeddings.py --limit 1000 # Test with 1000 rows
    python3 migrate_embeddings.py --status     # Check migration progress
"""

import argparse
import asyncio
import os
import sys
import time

os.environ["TQDM_DISABLE"] = "1"

import asyncpg

LOG_PATH = os.path.expanduser("~/.ucw/logs/embedding_migration.log")


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


async def get_status(conn):
    total = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
    migrated = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache WHERE embedding_768 IS NOT NULL")
    remaining = total - migrated
    return total, migrated, remaining


async def migrate(limit: int = 0, batch_size: int = 256, chunk_size: int = 2000):
    conn = await asyncpg.connect("postgresql://localhost/ucw_cognitive")

    total, migrated, remaining = await get_status(conn)
    log(f"Migration: {migrated:,}/{total:,} done, {remaining:,} remaining")

    if remaining == 0:
        log("All embeddings already migrated!")
        await conn.close()
        return

    # Lazy-load model
    log("Loading nomic-embed-text-v1.5...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    log(f"Model loaded in {time.time() - t0:.1f}s")

    target = min(remaining, limit) if limit > 0 else remaining
    t_start = time.time()
    total_processed = 0

    while total_processed < target:
        # Fetch one chunk at a time
        fetch_count = min(chunk_size, target - total_processed)
        rows = await conn.fetch(
            """SELECT content_hash, content_preview
               FROM embedding_cache
               WHERE embedding_768 IS NULL
               LIMIT $1""",
            fetch_count,
        )

        if not rows:
            break

        # Process in encode-batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [f"search_document: {r['content_preview']}" for r in batch]
            hashes = [r["content_hash"] for r in batch]

            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)

            # Batch update using executemany
            updates = []
            for ch, emb in zip(hashes, embeddings):
                vec_str = "[" + ",".join(str(x) for x in emb.tolist()) + "]"
                updates.append((vec_str, ch))

            await conn.executemany(
                """UPDATE embedding_cache
                   SET embedding_768 = $1::vector,
                       model = 'nomic-ai/nomic-embed-text-v1.5',
                       dimensions = 768
                   WHERE content_hash = $2""",
                updates,
            )

            total_processed += len(batch)
            elapsed = time.time() - t_start
            rate = total_processed / elapsed if elapsed > 0 else 0
            eta_s = (target - total_processed) / rate if rate > 0 else 0
            eta_m = eta_s / 60

            if total_processed % 1000 < batch_size:
                log(f"  {total_processed:,}/{target:,} ({total_processed/target*100:.1f}%) | {rate:.0f}/sec | ETA: {eta_m:.0f}min")

    elapsed = time.time() - t_start
    rate = total_processed / elapsed if elapsed > 0 else 0
    log(f"Done: {total_processed:,} embeddings in {elapsed:.0f}s ({rate:.0f}/sec)")

    _, migrated_after, remaining_after = await get_status(conn)
    log(f"Status: {migrated_after:,}/{total:,} migrated, {remaining_after:,} remaining")
    await conn.close()


async def status_only():
    conn = await asyncpg.connect("postgresql://localhost/ucw_cognitive")
    total, migrated, remaining = await get_status(conn)
    pct = (migrated / total * 100) if total > 0 else 0
    print(f"Embedding migration status:")
    print(f"  Total:     {total:,}")
    print(f"  Migrated:  {migrated:,} ({pct:.1f}%)")
    print(f"  Remaining: {remaining:,}")
    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate embeddings to nomic-embed-text-v1.5")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows to migrate (0 = all)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Rows fetched per DB round-trip")
    parser.add_argument("--status", action="store_true", help="Show migration status only")
    args = parser.parse_args()

    if args.status:
        asyncio.run(status_only())
    else:
        asyncio.run(migrate(limit=args.limit, batch_size=args.batch_size, chunk_size=args.chunk_size))
