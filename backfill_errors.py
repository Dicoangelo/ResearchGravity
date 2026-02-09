#!/usr/bin/env python3
"""
Error Pattern Backfill

Processes error data from multiple sources:
1. Supermemory error_patterns (5 core patterns)
2. ERRORS.md (139 documented errors)
3. recovery-outcomes.jsonl (150 recovery attempts)

Creates searchable error patterns for predictive prevention.

Usage:
    python3 backfill_errors.py                    # Backfill all
    python3 backfill_errors.py --dry-run          # Preview
"""

import asyncio
import json
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse

from storage.qdrant_db import QdrantDB

HOME = Path.home()
SUPERMEMORY_DB = HOME / ".claude" / "memory" / "supermemory.db"
ERRORS_MD = HOME / ".claude" / "ERRORS.md"
RECOVERY_FILE = HOME / ".claude" / "data" / "recovery-outcomes.jsonl"
ANTIGRAVITY_DB = HOME / ".agent-core" / "storage" / "antigravity.db"


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    if not file_path.exists():
        return []

    records = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    return records


def load_supermemory_patterns() -> List[Dict[str, Any]]:
    """Load error patterns from Supermemory."""
    if not SUPERMEMORY_DB.exists():
        return []

    conn = sqlite3.connect(str(SUPERMEMORY_DB))
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM error_patterns")
    rows = cursor.fetchall()
    conn.close()

    patterns = []
    for row in rows:
        # row: (id, category, pattern, occurrences, solution, last_seen, created_at)
        patterns.append({
            "id": f"sm-{row[0]}",
            "error_type": row[1],  # category
            "pattern": row[2],
            "occurrences": row[3],
            "context": row[4][:500] if row[4] else "",  # solution as context
            "solution": row[4] if row[4] else "",
            "success_rate": 0.9,  # High confidence from documented patterns
            "source": "supermemory"
        })

    return patterns


def parse_errors_md() -> List[Dict[str, Any]]:
    """Parse ERRORS.md for documented patterns."""
    if not ERRORS_MD.exists():
        return []

    with open(ERRORS_MD) as f:
        content = f.read()

    patterns = []

    # Parse summary table to get counts
    summary_section = re.search(r'## Summary.*?\n\n(.*?)\n\n---', content, re.DOTALL)
    if summary_section:
        lines = summary_section.group(1).strip().split('\n')[2:]  # Skip header rows

        for line in lines:
            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 4:
                    category = parts[0].replace('**', '')
                    count = int(parts[1])
                    severity = parts[2]
                    status = parts[3]

                    patterns.append({
                        "id": f"md-{category.lower()}",
                        "error_type": category.lower(),
                        "pattern": category,
                        "occurrences": count,
                        "context": f"{severity} severity, status: {status}",
                        "solution": f"See ERRORS.md for {category} prevention strategies",
                        "success_rate": 0.8,  # Documented solutions
                        "source": "errors_md"
                    })

    # Parse specific error sections for detailed solutions

    # Git errors
    if "## Git Errors" in content:
        git_section = re.search(r'## Git Errors.*?(?=\n##|\Z)', content, re.DOTALL)
        if git_section:
            git_text = git_section.group(0)

            # Repository not found
            if "Repository Not Found" in git_text:
                patterns.append({
                    "id": "md-git-repo-not-found",
                    "error_type": "git",
                    "pattern": "repository not found",
                    "occurrences": 80,  # Estimated from count
                    "context": "fatal: repository 'https://github.com/...' not found. Case sensitivity issue with username.",
                    "solution": "Always use exact GitHub username: Dicoangelo (not dicoangelo). Verify repo exists with: gh repo view owner/repo",
                    "success_rate": 0.95,
                    "source": "errors_md_detail"
                })

            # Tag/branch conflicts
            if "Tag/Branch Conflicts" in git_text:
                patterns.append({
                    "id": "md-git-tag-exists",
                    "error_type": "git",
                    "pattern": "tag already exists",
                    "occurrences": 15,
                    "context": "fatal: tag 'v1.1.0' already exists",
                    "solution": "Check before creating: git tag -l | grep <tag>",
                    "success_rate": 0.9,
                    "source": "errors_md_detail"
                })

            # Wrong directory
            if "Wrong Directory" in git_text:
                patterns.append({
                    "id": "md-git-not-repo",
                    "error_type": "git",
                    "pattern": "not a git repository",
                    "occurrences": 17,
                    "context": "fatal: not a git repository (or any of the parent directories): .git",
                    "solution": "Verify with: git rev-parse --git-dir before operations. Use absolute paths.",
                    "success_rate": 0.85,
                    "source": "errors_md_detail"
                })

    # Concurrency errors
    if "## Concurrency Errors" in content:
        patterns.append({
            "id": "md-concurrency-parallel",
            "error_type": "concurrency",
            "pattern": "parallel sessions",
            "occurrences": 11,
            "context": "5+ Claude sessions running simultaneously causing race conditions and data corruption",
            "solution": "Check for other sessions: pgrep -f 'claude' at start. ONE SESSION AT A TIME rule. Use file locks for critical writes.",
            "success_rate": 0.95,
            "source": "errors_md_detail"
        })

    # Permissions errors
    if "## Permissions Errors" in content:
        patterns.append({
            "id": "md-permissions-denied",
            "error_type": "permissions",
            "pattern": "permission denied",
            "occurrences": 8,
            "context": "Permission denied, EACCES when accessing files/directories",
            "solution": "Check permissions: ls -la <file>. Ensure scripts executable: chmod +x. Use sudo when appropriate.",
            "success_rate": 0.9,
            "source": "errors_md_detail"
        })

    return patterns


def parse_recovery_outcomes() -> List[Dict[str, Any]]:
    """Parse recovery outcomes for actual fix patterns."""
    records = read_jsonl(RECOVERY_FILE)

    patterns = []
    seen_actions = {}

    for record in records:
        action = record.get("action", "unknown")
        category = record.get("category", "unknown")
        success = record.get("success", False)

        # Group by action type
        key = f"{category}:{action}"
        if key not in seen_actions:
            seen_actions[key] = {
                "successes": 0,
                "failures": 0,
                "contexts": [],
                "action": action,
                "category": category
            }

        if success:
            seen_actions[key]["successes"] += 1
        else:
            seen_actions[key]["failures"] += 1

        detail = record.get("details", "")
        if detail and len(seen_actions[key]["contexts"]) < 3:
            seen_actions[key]["contexts"].append(detail[:200])

    # Convert to patterns
    for key, data in seen_actions.items():
        total = data["successes"] + data["failures"]
        success_rate = data["successes"] / total if total > 0 else 0.0

        # Only include if we have attempts
        if total > 0:
            patterns.append({
                "id": f"recovery-{key.replace(':', '-')}",
                "error_type": data["category"],
                "pattern": data["action"],
                "occurrences": total,
                "context": " | ".join(data["contexts"]),
                "solution": data["action"].replace("_", " ").title(),
                "success_rate": success_rate,
                "source": "recovery_outcomes"
            })

    return patterns


async def backfill_errors(dry_run: bool = False) -> int:
    """Backfill error patterns to SQLite and Qdrant."""
    print("=" * 60)
    print("Error Pattern Backfill")
    print("=" * 60)

    # Load from all sources
    print("\n1. Loading Supermemory patterns...")
    sm_patterns = load_supermemory_patterns()
    print(f"   ✅ {len(sm_patterns)} patterns from Supermemory")

    print("\n2. Parsing ERRORS.md...")
    md_patterns = parse_errors_md()
    print(f"   ✅ {len(md_patterns)} patterns from ERRORS.md")

    print("\n3. Processing recovery outcomes...")
    recovery_patterns = parse_recovery_outcomes()
    print(f"   ✅ {len(recovery_patterns)} patterns from recovery outcomes")

    # Combine all patterns
    all_patterns = sm_patterns + md_patterns + recovery_patterns

    print(f"\n📊 Total patterns: {len(all_patterns)}")

    # Preview top patterns
    print("\nTop 5 Error Patterns:")
    sorted_patterns = sorted(all_patterns, key=lambda p: p["occurrences"], reverse=True)
    for i, p in enumerate(sorted_patterns[:5], 1):
        print(f"  {i}. {p['error_type']}: {p['pattern']}")
        print(f"     Occurrences: {p['occurrences']} | Success rate: {p['success_rate']:.0%}")
        print(f"     Solution: {p['solution'][:60]}...")

    if dry_run:
        return len(all_patterns)

    # Store in SQLite
    print("\n4. Storing in SQLite...")
    conn = sqlite3.connect(str(ANTIGRAVITY_DB), timeout=30.0)
    cursor = conn.cursor()

    count = 0
    for p in all_patterns:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO error_patterns
                (id, error_type, context, solution, success_rate, occurrences)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                p["id"],
                p["error_type"],
                p["context"][:1000],
                p["solution"][:1000],
                p["success_rate"],
                p.get("occurrences", 1)
            ))
            count += 1
            if count % 20 == 0:
                print(f"   Progress: {count}/{len(all_patterns)}")
                conn.commit()
        except Exception as e:
            print(f"   Error on pattern {p['id']}: {e}")

    conn.commit()
    conn.close()
    print(f"   ✅ Stored {count} patterns in SQLite")

    # Store in Qdrant
    print("\n5. Vectorizing and storing in Qdrant...")
    qdrant = QdrantDB()
    await qdrant.initialize()

    batch_size = 20
    qdrant_count = 0

    for i in range(0, len(all_patterns), batch_size):
        batch = all_patterns[i:i + batch_size]
        try:
            batch_count = await qdrant.upsert_error_patterns_batch(batch)
            qdrant_count += batch_count
            print(f"   Progress: {qdrant_count}/{len(all_patterns)}")
        except Exception as e:
            print(f"   Error on batch {i//batch_size}: {e}")

    await qdrant.close()
    print(f"   ✅ Stored {qdrant_count} patterns in Qdrant")

    print("\n" + "=" * 60)
    print(f"✅ Backfill complete:")
    print(f"   SQLite: {count} patterns")
    print(f"   Qdrant: {qdrant_count} patterns")
    print("=" * 60)

    return count


async def main():
    parser = argparse.ArgumentParser(description="Backfill error patterns")
    parser.add_argument("--dry-run", action="store_true", help="Preview without storing")
    args = parser.parse_args()

    await backfill_errors(args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
