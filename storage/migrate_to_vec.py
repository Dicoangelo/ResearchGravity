#!/usr/bin/env python3
"""
Migrate Vectors from Qdrant to sqlite-vec
=========================================

Migrates existing vector embeddings from Qdrant to sqlite-vec for:
- Single-file deployment
- No external dependencies
- Offline capability

Phases:
1. Export all vectors from Qdrant collections
2. Import into sqlite-vec tables
3. Validate counts match
4. Switch reads to sqlite-vec
5. (Optional) Remove Qdrant dependency

Usage:
  python3 -m storage.migrate_to_vec --dry-run       # Preview migration
  python3 -m storage.migrate_to_vec                  # Run migration
  python3 -m storage.migrate_to_vec --validate      # Validate only
  python3 -m storage.migrate_to_vec --rollback      # Rollback if needed
"""

import argparse
import asyncio
import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# Paths
AGENT_CORE_DIR = Path.home() / ".agent-core"
MIGRATION_STATE_FILE = AGENT_CORE_DIR / "storage" / "migration_state.json"


@dataclass
class MigrationState:
    """Track migration progress."""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    phase: str = "not_started"  # not_started, exporting, importing, validating, complete
    collections_migrated: Dict[str, int] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.collections_migrated is None:
            self.collections_migrated = {}
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "phase": self.phase,
            "collections_migrated": self.collections_migrated,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationState":
        return cls(
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            phase=data.get("phase", "not_started"),
            collections_migrated=data.get("collections_migrated", {}),
            errors=data.get("errors", []),
        )


def load_migration_state() -> MigrationState:
    """Load migration state from disk."""
    if MIGRATION_STATE_FILE.exists():
        try:
            data = json.loads(MIGRATION_STATE_FILE.read_text())
            return MigrationState.from_dict(data)
        except Exception:
            pass
    return MigrationState()


def save_migration_state(state: MigrationState):
    """Save migration state to disk."""
    MIGRATION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    MIGRATION_STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))


async def export_from_qdrant() -> Dict[str, List[Dict[str, Any]]]:
    """Export all vectors from Qdrant collections."""
    try:
        from storage.qdrant_db import get_qdrant, QDRANT_AVAILABLE
    except ImportError:
        print("Error: Qdrant module not available")
        return {}

    if not QDRANT_AVAILABLE:
        print("Error: Qdrant client not installed or not running")
        return {}

    exported = {}

    try:
        qdrant = await get_qdrant()

        if not await qdrant.health_check():
            print("Error: Qdrant not responding")
            return {}

        # Collections to migrate
        collections = ["findings", "sessions", "packs", "session_outcomes", "cognitive_states", "error_patterns"]

        for collection in collections:
            print(f"Exporting {collection}...")
            try:
                # Get all points from collection
                # Note: This requires access to the raw Qdrant client
                if hasattr(qdrant, '_client'):
                    from qdrant_client.models import ScrollRequest

                    points = []
                    offset = None

                    while True:
                        result = qdrant._client.scroll(
                            collection_name=collection,
                            limit=100,
                            offset=offset,
                            with_vectors=True,
                            with_payload=True,
                        )

                        for point in result[0]:
                            points.append({
                                "id": str(point.id),
                                "vector": point.vector,
                                "payload": point.payload,
                            })

                        offset = result[1]
                        if offset is None:
                            break

                    exported[collection] = points
                    print(f"  Exported {len(points)} points from {collection}")

            except Exception as e:
                print(f"  Warning: Could not export {collection}: {e}")
                exported[collection] = []

        await qdrant.close()

    except Exception as e:
        print(f"Error during export: {e}")

    return exported


async def import_to_sqlite_vec(data: Dict[str, List[Dict[str, Any]]], dry_run: bool = False) -> Dict[str, int]:
    """Import vectors into sqlite-vec."""
    try:
        from storage.sqlite_vec import get_vec_db, SQLITE_VEC_AVAILABLE
    except ImportError:
        print("Error: sqlite-vec module not available")
        return {}

    if not SQLITE_VEC_AVAILABLE:
        print("Warning: sqlite-vec extension not available, using metadata-only mode")

    imported = {}

    try:
        vec_db = await get_vec_db()

        for collection, points in data.items():
            if not points:
                continue

            print(f"Importing {len(points)} points to {collection}...")

            if dry_run:
                print(f"  [DRY RUN] Would import {len(points)} points")
                imported[collection] = len(points)
                continue

            count = 0
            for point in points:
                try:
                    point_id = point["id"]
                    vector = point.get("vector", [])
                    payload = point.get("payload", {})

                    # Map collection to method
                    if collection == "findings":
                        content = payload.get("content", "")
                        await vec_db.upsert_finding(point_id, content, payload)
                    elif collection == "sessions":
                        topic = payload.get("topic", "")
                        await vec_db.upsert_session(point_id, topic, payload)
                    elif collection == "packs":
                        content = payload.get("content", payload.get("name", ""))
                        await vec_db.upsert_pack(point_id, content, payload)
                    # Skip other collections for now (they use different schema)

                    count += 1

                except Exception as e:
                    print(f"  Warning: Could not import {point_id}: {e}")

            imported[collection] = count
            print(f"  Imported {count} points")

    except Exception as e:
        print(f"Error during import: {e}")

    return imported


async def validate_migration() -> Dict[str, Any]:
    """Validate migration by comparing counts."""
    results = {
        "qdrant_counts": {},
        "sqlite_vec_counts": {},
        "match": True,
        "timestamp": datetime.now().isoformat(),
    }

    # Get Qdrant counts
    try:
        from storage.qdrant_db import get_qdrant, QDRANT_AVAILABLE

        if QDRANT_AVAILABLE:
            qdrant = await get_qdrant()
            if await qdrant.health_check():
                stats = await qdrant.get_stats()
                for key, value in stats.items():
                    if key.endswith("_count"):
                        results["qdrant_counts"][key] = value
            await qdrant.close()
    except Exception as e:
        results["qdrant_error"] = str(e)

    # Get sqlite-vec counts
    try:
        from storage.sqlite_vec import get_vec_db

        vec_db = await get_vec_db()
        stats = await vec_db.get_stats()
        for key, value in stats.items():
            if "_count" in key:
                results["sqlite_vec_counts"][key] = value
    except Exception as e:
        results["sqlite_vec_error"] = str(e)

    # Compare counts
    for key in results["qdrant_counts"]:
        qdrant_count = results["qdrant_counts"].get(key, 0)
        vec_key = key.replace("qdrant_", "").replace("_count", "")
        vec_count = results["sqlite_vec_counts"].get(f"{vec_key}_count", 0)

        if qdrant_count != vec_count:
            results["match"] = False
            results[f"mismatch_{key}"] = {"qdrant": qdrant_count, "sqlite_vec": vec_count}

    return results


async def run_migration(dry_run: bool = False):
    """Run the full migration process."""
    print("=" * 60)
    print("  ResearchGravity Vector Migration: Qdrant → sqlite-vec")
    print("=" * 60)
    print()

    if dry_run:
        print("  [DRY RUN MODE - No changes will be made]")
        print()

    state = load_migration_state()

    # Phase 1: Export from Qdrant
    print("Phase 1: Exporting from Qdrant...")
    state.phase = "exporting"
    state.started_at = datetime.now().isoformat()
    if not dry_run:
        save_migration_state(state)

    exported = await export_from_qdrant()

    total_exported = sum(len(v) for v in exported.values())
    print(f"  Total exported: {total_exported} vectors")
    print()

    if total_exported == 0:
        print("  No vectors to migrate. Qdrant may be empty or not available.")
        return

    # Phase 2: Import to sqlite-vec
    print("Phase 2: Importing to sqlite-vec...")
    state.phase = "importing"
    if not dry_run:
        save_migration_state(state)

    imported = await import_to_sqlite_vec(exported, dry_run=dry_run)

    total_imported = sum(imported.values())
    print(f"  Total imported: {total_imported} vectors")
    print()

    state.collections_migrated = imported

    # Phase 3: Validate
    print("Phase 3: Validating migration...")
    state.phase = "validating"
    if not dry_run:
        save_migration_state(state)

    validation = await validate_migration()
    print(f"  Qdrant counts: {validation.get('qdrant_counts', {})}")
    print(f"  sqlite-vec counts: {validation.get('sqlite_vec_counts', {})}")
    print(f"  Match: {validation.get('match', False)}")
    print()

    # Complete
    state.phase = "complete"
    state.completed_at = datetime.now().isoformat()
    if not dry_run:
        save_migration_state(state)

    print("=" * 60)
    if dry_run:
        print("  DRY RUN COMPLETE - No changes made")
    else:
        print("  MIGRATION COMPLETE")
        print(f"  Vectors migrated: {total_imported}")
        print(f"  Collections: {list(imported.keys())}")
    print("=" * 60)


async def show_status():
    """Show current migration status."""
    state = load_migration_state()

    print("=" * 50)
    print("  Migration Status")
    print("=" * 50)
    print(f"Phase: {state.phase}")
    print(f"Started: {state.started_at or 'Not started'}")
    print(f"Completed: {state.completed_at or 'Not completed'}")

    if state.collections_migrated:
        print("\nCollections migrated:")
        for name, count in state.collections_migrated.items():
            print(f"  {name}: {count}")

    if state.errors:
        print("\nErrors:")
        for error in state.errors:
            print(f"  - {error}")

    print()

    # Current validation
    print("Current validation:")
    validation = await validate_migration()
    print(f"  Match: {validation.get('match', 'unknown')}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate vectors from Qdrant to sqlite-vec"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview migration without making changes")
    parser.add_argument("--validate", action="store_true", help="Validate migration only")
    parser.add_argument("--status", action="store_true", help="Show migration status")

    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    elif args.validate:
        asyncio.run(validate_migration())
    else:
        asyncio.run(run_migration(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
