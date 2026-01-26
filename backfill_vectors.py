#!/usr/bin/env python3
"""
Backfill vectors to Qdrant in batches, respecting Cohere rate limits.

Cohere free tier: 100,000 tokens/minute
- embed-english-v3.0: ~500 tokens per embedding
- Safe batch: 50 embeddings/minute = ~25,000 tokens/minute
"""

import asyncio
import sys
import time
from pathlib import Path
from storage.engine import StorageEngine

# Batch configuration
BATCH_SIZE = 50  # embeddings per batch
DELAY_BETWEEN_BATCHES = 70  # seconds (allow 10s buffer)


async def backfill_findings(engine: StorageEngine, dry_run: bool = False):
    """Backfill finding embeddings to Qdrant."""
    print("\n━━━ Backfilling Findings to Qdrant ━━━━━━━━━━━━━━")

    # Get all findings from SQLite
    async with engine.sqlite.connection() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM findings")
        total = (await cursor.fetchone())[0]
        print(f"Total findings: {total}")

        if dry_run:
            print("DRY RUN - No embeddings will be generated")
            return

        # Process in batches
        offset = 0
        batch_num = 1

        while offset < total:
            cursor = await db.execute("""
                SELECT id, content, type, session_id, project, confidence
                FROM findings
                LIMIT ? OFFSET ?
            """, (BATCH_SIZE, offset))

            rows = await cursor.fetchall()
            if not rows:
                break

            findings = [
                {
                    "id": r[0],
                    "content": r[1],
                    "type": r[2],
                    "session_id": r[3],
                    "project": r[4],
                    "confidence": r[5],
                }
                for r in rows
            ]

            print(f"\nBatch {batch_num}: Processing findings {offset+1}-{offset+len(findings)} of {total}")

            try:
                count = await engine.qdrant.upsert_findings_batch(findings)
                print(f"  ✓ Uploaded {count} vectors to Qdrant")
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    print(f"  ⚠️  Rate limit hit. Waiting 120 seconds...")
                    time.sleep(120)
                    # Retry this batch
                    try:
                        count = await engine.qdrant.upsert_findings_batch(findings)
                        print(f"  ✓ Retry successful: {count} vectors")
                    except Exception as retry_err:
                        print(f"  ✗ Retry failed: {retry_err}")
                        print(f"  Skipping batch {batch_num}")
                else:
                    print(f"  ✗ Error: {e}")

            offset += len(findings)
            batch_num += 1

            # Rate limit pause
            if offset < total:
                print(f"  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
                time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\n✓ Finished backfilling findings")


async def backfill_sessions(engine: StorageEngine, dry_run: bool = False):
    """Backfill session embeddings to Qdrant."""
    print("\n━━━ Backfilling Sessions to Qdrant ━━━━━━━━━━━━━━")

    sessions = await engine.list_sessions(limit=1000)
    print(f"Total sessions: {len(sessions)}")

    if dry_run:
        print("DRY RUN - No embeddings will be generated")
        return

    # Sessions are fewer, can do in one batch
    try:
        count = await engine.qdrant.upsert_sessions_batch(sessions)
        print(f"✓ Uploaded {count} session vectors to Qdrant")
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            print(f"⚠️  Rate limit hit. Waiting 120 seconds...")
            time.sleep(120)
            try:
                count = await engine.qdrant.upsert_sessions_batch(sessions)
                print(f"✓ Retry successful: {count} session vectors")
            except Exception as retry_err:
                print(f"✗ Retry failed: {retry_err}")
        else:
            print(f"✗ Error: {e}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill vectors to Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    parser.add_argument("--findings-only", action="store_true", help="Only backfill findings")
    parser.add_argument("--sessions-only", action="store_true", help="Only backfill sessions")
    args = parser.parse_args()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Qdrant Vector Backfill (Cohere Embeddings)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\nBatch size: {BATCH_SIZE} embeddings")
    print(f"Delay between batches: {DELAY_BETWEEN_BATCHES}s")
    print(f"Estimated rate: ~{int(60/DELAY_BETWEEN_BATCHES * BATCH_SIZE)} embeddings/minute")

    engine = StorageEngine()
    await engine.initialize()

    try:
        if not args.sessions_only:
            await backfill_findings(engine, dry_run=args.dry_run)

        if not args.findings_only:
            await backfill_sessions(engine, dry_run=args.dry_run)

        # Show final stats
        if not args.dry_run:
            print("\n━━━ Final Statistics ━━━━━━━━━━━━━━━━━━━━━━━━━")
            stats = await engine.qdrant.get_stats()
            print(f"Embedding Model: {stats['embedding_model']}")
            print(f"Collections:")
            for name in ['findings', 'sessions']:
                if name in stats:
                    info = stats[name]
                    if 'error' not in info:
                        print(f"  {name}: {info.get('points_count', 0)} vectors")

    finally:
        await engine.close()

    print("\n✓ Backfill complete!")


if __name__ == "__main__":
    asyncio.run(main())
