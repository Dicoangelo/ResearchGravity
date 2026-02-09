#!/usr/bin/env python3
"""
Phase 4 Re-Backfill Script

Fixes hash collision issue by re-loading cognitive states and error patterns
with UUID-based IDs instead of MD5 hashes.

This script:
1. Deletes existing cognitive_states and error_patterns from Qdrant (keeps SQLite)
2. Re-loads them from SQLite using UUID-based IDs
3. Verifies counts match

Usage:
    python3 rebackfill_phase4.py                  # Run re-backfill
    python3 rebackfill_phase4.py --dry-run        # Preview changes
"""

import asyncio
import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

from storage.qdrant_db import QdrantDB


HOME = Path.home()
ANTIGRAVITY_DB = HOME / ".agent-core" / "storage" / "antigravity.db"


async def rebackfill_cognitive_states(dry_run: bool = False) -> int:
    """Re-backfill cognitive states with UUID IDs."""
    print("\n" + "=" * 60)
    print("Re-backfilling Cognitive States (UUID Fix)")
    print("=" * 60)

    # Read from SQLite
    conn = sqlite3.connect(str(ANTIGRAVITY_DB), timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM cognitive_states")
    rows = cursor.fetchall()
    states = [dict(row) for row in rows]
    conn.close()

    print(f"\n📊 Found {len(states)} cognitive states in SQLite")

    if dry_run:
        print("   [DRY RUN] Would re-vectorize with UUID IDs")
        return len(states)

    # Initialize Qdrant
    qdrant = QdrantDB()
    await qdrant.initialize()

    # Delete existing collection
    print("\n🗑️  Deleting old cognitive_states collection from Qdrant...")
    try:
        await qdrant.async_client.delete_collection("cognitive_states")
        print("   ✅ Deleted")
    except Exception as e:
        print(f"   ⚠️  Collection doesn't exist or error: {e}")

    # Recreate collection
    print("\n🔧 Recreating cognitive_states collection...")
    from qdrant_client.http.models import Distance, VectorParams
    await qdrant.async_client.create_collection(
        collection_name="cognitive_states",
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )
    print("   ✅ Collection created")

    # Re-upload with UUID IDs
    print(f"\n📤 Uploading {len(states)} cognitive states with UUID IDs...")
    batch_size = 20
    count = 0

    for i in range(0, len(states), batch_size):
        batch = states[i:i + batch_size]

        # Convert to proper format and generate UUID IDs
        batch_formatted = []
        for state in batch:
            # Generate UUID-based ID
            unique_id = qdrant._generate_unique_id("cognitive")

            # Create searchable context
            mode = state.get("mode", "unknown")
            hour = state.get("hour", 0)
            energy = state.get("energy_level", 0.5)
            context = f"{mode} hour_{hour} energy_{energy:.2f}"

            # Update the ID in the state dict
            state_copy = dict(state)
            state_copy["id"] = unique_id

            batch_formatted.append(state_copy)

        # Batch upload
        try:
            batch_count = await qdrant.upsert_cognitive_states_batch(batch_formatted)
            count += batch_count
            print(f"   Progress: {count}/{len(states)}")
        except Exception as e:
            print(f"   ⚠️  Error on batch {i//batch_size}: {e}")

    await qdrant.close()

    print(f"\n✅ Re-backfilled {count} cognitive states")
    return count


async def rebackfill_error_patterns(dry_run: bool = False) -> int:
    """Re-backfill error patterns with UUID IDs."""
    print("\n" + "=" * 60)
    print("Re-backfilling Error Patterns (UUID Fix)")
    print("=" * 60)

    # Read from SQLite
    conn = sqlite3.connect(str(ANTIGRAVITY_DB), timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM error_patterns")
    rows = cursor.fetchall()
    patterns = [dict(row) for row in rows]
    conn.close()

    print(f"\n📊 Found {len(patterns)} error patterns in SQLite")

    if dry_run:
        print("   [DRY RUN] Would re-vectorize with UUID IDs")
        return len(patterns)

    # Initialize Qdrant
    qdrant = QdrantDB()
    await qdrant.initialize()

    # Delete existing collection
    print("\n🗑️  Deleting old error_patterns collection from Qdrant...")
    try:
        await qdrant.async_client.delete_collection("error_patterns")
        print("   ✅ Deleted")
    except Exception as e:
        print(f"   ⚠️  Collection doesn't exist or error: {e}")

    # Recreate collection
    print("\n🔧 Recreating error_patterns collection...")
    from qdrant_client.http.models import Distance, VectorParams
    await qdrant.async_client.create_collection(
        collection_name="error_patterns",
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )
    print("   ✅ Collection created")

    # Re-upload with UUID IDs
    print(f"\n📤 Uploading {len(patterns)} error patterns with UUID IDs...")
    batch_size = 20
    count = 0

    for i in range(0, len(patterns), batch_size):
        batch = patterns[i:i + batch_size]

        # Convert to proper format and generate UUID IDs
        batch_formatted = []
        for pattern in batch:
            # Generate UUID-based ID
            unique_id = qdrant._generate_unique_id("error")

            # Update the ID in the pattern dict
            pattern_copy = dict(pattern)
            pattern_copy["id"] = unique_id

            batch_formatted.append(pattern_copy)

        # Batch upload
        try:
            batch_count = await qdrant.upsert_error_patterns_batch(batch_formatted)
            count += batch_count
            print(f"   Progress: {count}/{len(patterns)}")
        except Exception as e:
            print(f"   ⚠️  Error on batch {i//batch_size}: {e}")

    await qdrant.close()

    print(f"\n✅ Re-backfilled {count} error patterns")
    return count


async def verify_counts():
    """Verify Qdrant counts match SQLite."""
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    # SQLite counts
    conn = sqlite3.connect(str(ANTIGRAVITY_DB), timeout=30.0)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM cognitive_states")
    sqlite_cognitive = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM error_patterns")
    sqlite_errors = cursor.fetchone()[0]

    conn.close()

    # Qdrant counts
    qdrant = QdrantDB()
    await qdrant.initialize()

    try:
        cognitive_info = await qdrant.async_client.get_collection("cognitive_states")
        qdrant_cognitive = cognitive_info.points_count
    except:
        qdrant_cognitive = 0

    try:
        error_info = await qdrant.async_client.get_collection("error_patterns")
        qdrant_errors = error_info.points_count
    except:
        qdrant_errors = 0

    await qdrant.close()

    print(f"\nCognitive States:")
    print(f"  SQLite:  {sqlite_cognitive}")
    print(f"  Qdrant:  {qdrant_cognitive}")
    print(f"  Match:   {'✅' if sqlite_cognitive == qdrant_cognitive else '❌'}")

    print(f"\nError Patterns:")
    print(f"  SQLite:  {sqlite_errors}")
    print(f"  Qdrant:  {qdrant_errors}")
    print(f"  Match:   {'✅' if sqlite_errors == qdrant_errors else '❌'}")

    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Phase 4 re-backfill with UUID fix")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--cognitive-only", action="store_true", help="Only re-backfill cognitive states")
    parser.add_argument("--errors-only", action="store_true", help="Only re-backfill error patterns")
    args = parser.parse_args()

    if not ANTIGRAVITY_DB.exists():
        print(f"❌ Database not found: {ANTIGRAVITY_DB}")
        return

    try:
        if not args.errors_only:
            await rebackfill_cognitive_states(dry_run=args.dry_run)

        if not args.cognitive_only:
            await rebackfill_error_patterns(dry_run=args.dry_run)

        if not args.dry_run:
            await verify_counts()

        print("\n✅ Phase 4 re-backfill complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
