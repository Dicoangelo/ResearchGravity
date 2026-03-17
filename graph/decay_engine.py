"""
Temporal Decay Engine — Automatic knowledge graph edge expiration

Applies MiroFish-inspired fact decay to the ResearchGravity knowledge graph:
- Edges older than a threshold without reinforcement get expired
- Edges that get re-cited (appear in newer sessions) get their valid_at refreshed
- Decay is gradual: weight reduces before full expiration

Decay Rules:
1. STALE (>90 days, no reinforcement): weight halved
2. DECAYED (>180 days, no reinforcement): expired
3. REINFORCED (cited in session <30 days old): weight boosted, valid_at refreshed
4. IMMORTAL (relation=contains): never expires (structural edges)

Usage:
    python3 -m graph.decay_engine --dry-run     # Preview what would decay
    python3 -m graph.decay_engine --apply        # Apply decay
    python3 -m graph.decay_engine --stats        # Show decay statistics
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

DB_PATH = Path.home() / ".agent-core/storage/antigravity.db"

# Decay thresholds (days)
STALE_THRESHOLD = 90
DECAY_THRESHOLD = 180
REINFORCEMENT_WINDOW = 30

# Relations that never expire
IMMORTAL_RELATIONS = {"contains"}

# Weight multipliers
STALE_WEIGHT_FACTOR = 0.5
REINFORCEMENT_WEIGHT_BOOST = 1.5
MAX_WEIGHT = 3.0


@dataclass
class DecayResult:
    """Result of a decay run."""
    edges_analyzed: int
    edges_staled: int
    edges_expired: int
    edges_reinforced: int
    edges_immortal: int
    dry_run: bool
    timestamp: str


class DecayEngine:
    """Applies temporal decay to knowledge graph edges."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_reinforcement_map(self, conn: sqlite3.Connection) -> Dict[str, str]:
        """Build a map of entity_id → most recent session date.

        An edge is reinforced if its target entity appears in a session
        within the reinforcement window.
        """
        cursor = conn.cursor()

        # Findings: check if the finding's session is recent
        cursor.execute("""
            SELECT f.id, s.started_at
            FROM findings f
            JOIN sessions s ON f.session_id = s.id
            WHERE s.started_at IS NOT NULL
            ORDER BY s.started_at DESC
        """)

        reinforcement = {}
        for row in cursor.fetchall():
            fid = f"finding:{row['id']}"
            if fid not in reinforcement:
                reinforcement[fid] = row["started_at"]

        # URLs: check session recency
        cursor.execute("""
            SELECT u.id, u.captured_at
            FROM urls u
            WHERE u.captured_at IS NOT NULL
            ORDER BY u.captured_at DESC
        """)
        for row in cursor.fetchall():
            uid = f"url:{row['id']}"
            if uid not in reinforcement:
                reinforcement[uid] = row["captured_at"]

        return reinforcement

    def analyze(self) -> Dict[str, Any]:
        """Analyze edge decay state without modifying anything."""
        conn = self._get_connection()
        now = datetime.now()

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source_type, source_id, target_type, target_id,
                       relation, weight, valid_at, expired_at, created_at
                FROM lineage
                WHERE expired_at IS NULL
            """)

            reinforcement = self._get_reinforcement_map(conn)

            stats = {
                "total_active": 0,
                "would_stale": 0,
                "would_expire": 0,
                "would_reinforce": 0,
                "immortal": 0,
                "already_stale": 0,
                "by_relation": {},
                "age_distribution": {"<30d": 0, "30-90d": 0, "90-180d": 0, ">180d": 0},
            }

            for row in cursor.fetchall():
                stats["total_active"] += 1
                relation = row["relation"]
                stats["by_relation"][relation] = stats["by_relation"].get(relation, 0) + 1

                # Immortal check
                if relation in IMMORTAL_RELATIONS:
                    stats["immortal"] += 1
                    continue

                # Calculate age
                edge_date = row["valid_at"] or row["created_at"]
                if not edge_date:
                    continue

                try:
                    edge_dt = datetime.fromisoformat(edge_date.replace("Z", "+00:00").split("+")[0])
                except (ValueError, AttributeError):
                    continue

                age_days = (now - edge_dt).days

                # Age bucket
                if age_days < 30:
                    stats["age_distribution"]["<30d"] += 1
                elif age_days < 90:
                    stats["age_distribution"]["30-90d"] += 1
                elif age_days < 180:
                    stats["age_distribution"]["90-180d"] += 1
                else:
                    stats["age_distribution"][">180d"] += 1

                # Check reinforcement
                target_key = f"{row['target_type']}:{row['target_id']}"
                last_cite = reinforcement.get(target_key)
                reinforced = False

                if last_cite:
                    try:
                        cite_dt = datetime.fromisoformat(last_cite.replace("Z", "+00:00").split("+")[0])
                        if (now - cite_dt).days < REINFORCEMENT_WINDOW:
                            reinforced = True
                    except (ValueError, AttributeError):
                        pass

                if reinforced:
                    stats["would_reinforce"] += 1
                elif age_days > DECAY_THRESHOLD:
                    stats["would_expire"] += 1
                elif age_days > STALE_THRESHOLD:
                    stats["would_stale"] += 1
                    if row["weight"] and row["weight"] < 1.0:
                        stats["already_stale"] += 1

            return stats

        finally:
            conn.close()

    def run(self, dry_run: bool = True) -> DecayResult:
        """Execute decay pass on all active edges.

        Args:
            dry_run: If True, only count what would change (default True)

        Returns:
            DecayResult with counts
        """
        conn = self._get_connection()
        now = datetime.now()
        now_iso = now.isoformat()

        staled = 0
        expired = 0
        reinforced = 0
        immortal = 0
        analyzed = 0

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source_type, source_id, target_type, target_id,
                       relation, weight, valid_at, expired_at, created_at
                FROM lineage
                WHERE expired_at IS NULL
            """)
            rows = cursor.fetchall()

            reinforcement = self._get_reinforcement_map(conn)

            for row in rows:
                analyzed += 1
                relation = row["relation"]
                edge_id = row["id"]
                current_weight = row["weight"] or 1.0

                # Immortal: skip
                if relation in IMMORTAL_RELATIONS:
                    immortal += 1
                    continue

                # Calculate age
                edge_date = row["valid_at"] or row["created_at"]
                if not edge_date:
                    continue

                try:
                    edge_dt = datetime.fromisoformat(edge_date.replace("Z", "+00:00").split("+")[0])
                except (ValueError, AttributeError):
                    continue

                age_days = (now - edge_dt).days

                # Check reinforcement
                target_key = f"{row['target_type']}:{row['target_id']}"
                last_cite = reinforcement.get(target_key)
                is_reinforced = False

                if last_cite:
                    try:
                        cite_dt = datetime.fromisoformat(last_cite.replace("Z", "+00:00").split("+")[0])
                        if (now - cite_dt).days < REINFORCEMENT_WINDOW:
                            is_reinforced = True
                    except (ValueError, AttributeError):
                        pass

                if is_reinforced:
                    # Reinforce: refresh valid_at, boost weight
                    new_weight = min(current_weight * REINFORCEMENT_WEIGHT_BOOST, MAX_WEIGHT)
                    if not dry_run:
                        cursor.execute(
                            "UPDATE lineage SET valid_at = ?, weight = ? WHERE id = ?",
                            (now_iso, new_weight, edge_id),
                        )
                    reinforced += 1

                elif age_days > DECAY_THRESHOLD:
                    # Expire: set expired_at
                    if not dry_run:
                        cursor.execute(
                            "UPDATE lineage SET expired_at = ? WHERE id = ?",
                            (now_iso, edge_id),
                        )
                    expired += 1

                elif age_days > STALE_THRESHOLD:
                    # Stale: halve weight (only if not already halved)
                    if current_weight > STALE_WEIGHT_FACTOR:
                        new_weight = current_weight * STALE_WEIGHT_FACTOR
                        if not dry_run:
                            cursor.execute(
                                "UPDATE lineage SET weight = ? WHERE id = ?",
                                (new_weight, edge_id),
                            )
                        staled += 1

            if not dry_run:
                conn.commit()

            return DecayResult(
                edges_analyzed=analyzed,
                edges_staled=staled,
                edges_expired=expired,
                edges_reinforced=reinforced,
                edges_immortal=immortal,
                dry_run=dry_run,
                timestamp=now_iso,
            )

        finally:
            conn.close()


def main():
    """CLI entry point."""
    import sys

    engine = DecayEngine()

    if len(sys.argv) < 2 or sys.argv[1] == "--stats":
        stats = engine.analyze()
        print("=== Knowledge Graph Decay Analysis ===\n")
        print(f"Active edges: {stats['total_active']}")
        print(f"Immortal (contains): {stats['immortal']}")
        print(f"\nAge distribution:")
        for bucket, count in stats["age_distribution"].items():
            bar = "█" * min(count // 10, 40)
            print(f"  {bucket:>8}: {count:>5} {bar}")
        print(f"\nDecay preview:")
        print(f"  Would stale (90-180d):  {stats['would_stale']}")
        print(f"  Would expire (>180d):   {stats['would_expire']}")
        print(f"  Would reinforce (<30d): {stats['would_reinforce']}")
        print(f"  Already stale:          {stats['already_stale']}")
        print(f"\nRelation types:")
        for rel, count in sorted(stats["by_relation"].items(), key=lambda x: -x[1]):
            print(f"  {rel:>15}: {count}")

    elif sys.argv[1] == "--dry-run":
        result = engine.run(dry_run=True)
        print("=== Decay Dry Run ===\n")
        print(f"Analyzed: {result.edges_analyzed}")
        print(f"Would stale:     {result.edges_staled}")
        print(f"Would expire:    {result.edges_expired}")
        print(f"Would reinforce: {result.edges_reinforced}")
        print(f"Immortal:        {result.edges_immortal}")

    elif sys.argv[1] == "--apply":
        result = engine.run(dry_run=False)
        print("=== Decay Applied ===\n")
        print(f"Analyzed: {result.edges_analyzed}")
        print(f"Staled:     {result.edges_staled}")
        print(f"Expired:    {result.edges_expired}")
        print(f"Reinforced: {result.edges_reinforced}")
        print(f"Immortal:   {result.edges_immortal}")

    else:
        print("Usage:")
        print("  python3 -m graph.decay_engine --stats")
        print("  python3 -m graph.decay_engine --dry-run")
        print("  python3 -m graph.decay_engine --apply")


if __name__ == "__main__":
    main()
