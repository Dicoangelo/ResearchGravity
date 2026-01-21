"""
Migration Script: JSON → SQLite + Qdrant

Migrates existing data from flat files to the storage triad:
- Sessions from ~/.agent-core/sessions/
- Findings from findings_captured.json
- URLs from urls_captured.json
- Papers from research files
- Context packs from context_packs/

Usage:
    python -m storage.migrate                    # Full migration
    python -m storage.migrate --sessions-only   # Just sessions
    python -m storage.migrate --dry-run         # Preview without writing
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_db import DB_PATH
from storage.engine import StorageEngine


# Data locations
AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
RESEARCH_DIR = AGENT_CORE_DIR / "research"
PACKS_DIR = Path.home() / "researchgravity" / "context_packs"
RESEARCHGRAVITY_DIR = Path.home() / "researchgravity"


class Migrator:
    """Handles migration from JSON files to SQLite + Qdrant."""

    def __init__(self, dry_run: bool = False, verbose: bool = True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.engine: Optional[StorageEngine] = None
        self.stats = {
            "sessions": {"found": 0, "migrated": 0, "errors": 0},
            "findings": {"found": 0, "migrated": 0, "errors": 0},
            "urls": {"found": 0, "migrated": 0, "errors": 0},
            "papers": {"found": 0, "migrated": 0, "errors": 0},
            "packs": {"found": 0, "migrated": 0, "errors": 0},
        }

    def log(self, message: str):
        """Print if verbose."""
        if self.verbose:
            print(message)

    async def initialize(self):
        """Initialize storage engine."""
        if not self.dry_run:
            self.engine = StorageEngine()
            await self.engine.initialize()

    async def close(self):
        """Close storage engine."""
        if self.engine:
            await self.engine.close()

    async def migrate_all(self):
        """Run full migration."""
        self.log("=" * 60)
        self.log("MIGRATION: JSON → SQLite + Qdrant")
        self.log("=" * 60)

        if self.dry_run:
            self.log("DRY RUN MODE - No data will be written")
            self.log("")

        await self.initialize()

        # Migrate in order (sessions first, then related data)
        await self.migrate_sessions()
        await self.migrate_findings()
        await self.migrate_urls()
        await self.migrate_papers()
        await self.migrate_packs()

        # Build lineage
        await self.build_lineage()

        await self.close()

        # Print summary
        self.print_summary()

    async def migrate_sessions(self):
        """Migrate sessions from ~/.agent-core/sessions/"""
        self.log("\n📁 Migrating Sessions...")

        if not SESSIONS_DIR.exists():
            self.log(f"  Sessions directory not found: {SESSIONS_DIR}")
            return

        session_dirs = [d for d in SESSIONS_DIR.iterdir() if d.is_dir()]
        self.stats["sessions"]["found"] = len(session_dirs)

        for session_dir in session_dirs:
            try:
                session_id = session_dir.name

                # Load session metadata
                metadata_file = session_dir / "metadata.json"
                summary_file = session_dir / "summary.md"
                findings_file = session_dir / "findings_captured.json"

                session_data = {
                    "id": session_id,
                    "status": "archived",
                }

                # Parse metadata
                if metadata_file.exists():
                    meta = json.loads(metadata_file.read_text())
                    session_data.update({
                        "topic": meta.get("topic") or meta.get("summary", "")[:100],
                        "project": meta.get("project"),
                        "started_at": meta.get("started_at"),
                        "archived_at": meta.get("archived_at"),
                        "transcript_tokens": meta.get("transcript_tokens"),
                    })

                # Get counts
                if findings_file.exists():
                    findings = json.loads(findings_file.read_text())
                    session_data["finding_count"] = len(findings)

                urls_file = session_dir / "urls_captured.json"
                if urls_file.exists():
                    urls = json.loads(urls_file.read_text())
                    session_data["url_count"] = len(urls)

                # Extract topic from summary if not in metadata
                if not session_data.get("topic") and summary_file.exists():
                    summary = summary_file.read_text()
                    # First line often contains topic
                    lines = summary.strip().split("\n")
                    for line in lines:
                        if line.startswith("# "):
                            session_data["topic"] = line[2:].strip()
                            break
                        elif line.startswith("## Topic"):
                            session_data["topic"] = lines[lines.index(line) + 1].strip()
                            break

                # Store
                if not self.dry_run:
                    await self.engine.store_session(session_data, source="migration")

                self.stats["sessions"]["migrated"] += 1
                self.log(f"  ✓ {session_id[:40]}...")

            except Exception as e:
                self.stats["sessions"]["errors"] += 1
                self.log(f"  ✗ {session_dir.name}: {e}")

    async def migrate_findings(self):
        """Migrate findings from all sessions."""
        self.log("\n💡 Migrating Findings...")

        all_findings = []

        # Collect from session directories
        if SESSIONS_DIR.exists():
            for session_dir in SESSIONS_DIR.iterdir():
                if not session_dir.is_dir():
                    continue

                findings_file = session_dir / "findings_captured.json"
                if findings_file.exists():
                    try:
                        findings = json.loads(findings_file.read_text())
                        for f in findings:
                            f["session_id"] = session_dir.name
                        all_findings.extend(findings)
                    except Exception as e:
                        self.log(f"  Warning: Could not read {findings_file}: {e}")

        # Also check unified research index
        unified_index = AGENT_CORE_DIR / "unified_research_index.json"
        if unified_index.exists():
            try:
                data = json.loads(unified_index.read_text())
                for session in data.get("sessions", []):
                    for finding in session.get("findings", []):
                        finding["session_id"] = session.get("session_id")
                        # Avoid duplicates
                        if not any(f.get("id") == finding.get("id") for f in all_findings):
                            all_findings.append(finding)
            except Exception as e:
                self.log(f"  Warning: Could not read unified index: {e}")

        self.stats["findings"]["found"] = len(all_findings)

        # Batch store
        if all_findings and not self.dry_run:
            try:
                # Ensure all findings have IDs
                for i, f in enumerate(all_findings):
                    if "id" not in f:
                        f["id"] = f"finding-migrated-{i}"
                    if "content" not in f and "text" in f:
                        f["content"] = f["text"]
                    if "content" not in f:
                        continue  # Skip invalid findings

                valid_findings = [f for f in all_findings if f.get("content")]
                count = await self.engine.store_findings_batch(valid_findings, source="migration")
                self.stats["findings"]["migrated"] = count
                self.log(f"  ✓ Migrated {count} findings")

            except Exception as e:
                self.stats["findings"]["errors"] += 1
                self.log(f"  ✗ Error migrating findings: {e}")
        else:
            self.stats["findings"]["migrated"] = len(all_findings)
            self.log(f"  Found {len(all_findings)} findings")

    async def migrate_urls(self):
        """Migrate URLs from all sessions."""
        self.log("\n🔗 Migrating URLs...")

        all_urls = []

        if SESSIONS_DIR.exists():
            for session_dir in SESSIONS_DIR.iterdir():
                if not session_dir.is_dir():
                    continue

                urls_file = session_dir / "urls_captured.json"
                if urls_file.exists():
                    try:
                        urls = json.loads(urls_file.read_text())
                        for u in urls:
                            u["session_id"] = session_dir.name
                        all_urls.extend(urls)
                    except Exception as e:
                        self.log(f"  Warning: Could not read {urls_file}: {e}")

        self.stats["urls"]["found"] = len(all_urls)

        if all_urls and not self.dry_run:
            try:
                count = await self.engine.store_urls_batch(all_urls)
                self.stats["urls"]["migrated"] = count
                self.log(f"  ✓ Migrated {count} URLs")
            except Exception as e:
                self.stats["urls"]["errors"] += 1
                self.log(f"  ✗ Error migrating URLs: {e}")
        else:
            self.stats["urls"]["migrated"] = len(all_urls)
            self.log(f"  Found {len(all_urls)} URLs")

    async def migrate_papers(self):
        """Migrate papers from research files."""
        self.log("\n📄 Migrating Papers...")

        papers = {}

        # Check various paper sources
        paper_sources = [
            RESEARCH_DIR / "papers.json",
            AGENT_CORE_DIR / "papers.json",
            RESEARCHGRAVITY_DIR / "papers.json",
        ]

        for source in paper_sources:
            if source.exists():
                try:
                    data = json.loads(source.read_text())
                    paper_list = data if isinstance(data, list) else data.get("papers", [])
                    for p in paper_list:
                        paper_id = p.get("arxiv_id") or p.get("id") or p.get("doi")
                        if paper_id and paper_id not in papers:
                            papers[paper_id] = p
                except Exception as e:
                    self.log(f"  Warning: Could not read {source}: {e}")

        # Also extract from unified index
        unified_index = AGENT_CORE_DIR / "unified_research_index.json"
        if unified_index.exists():
            try:
                data = json.loads(unified_index.read_text())
                for session in data.get("sessions", []):
                    for paper in session.get("papers", []):
                        paper_id = paper.get("arxiv_id") or paper.get("id")
                        if paper_id and paper_id not in papers:
                            papers[paper_id] = paper
            except Exception:
                pass

        self.stats["papers"]["found"] = len(papers)

        if papers and not self.dry_run:
            try:
                async with self.engine.sqlite.connection() as db:
                    for paper_id, paper in papers.items():
                        await db.execute("""
                            INSERT INTO papers (id, title, authors, abstract, url, relevance, applied, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(id) DO UPDATE SET
                                title = excluded.title,
                                url = excluded.url
                        """, (
                            paper_id,
                            paper.get("title"),
                            json.dumps(paper.get("authors", [])),
                            paper.get("abstract"),
                            paper.get("url"),
                            paper.get("relevance", 0),
                            1 if paper.get("applied") else 0,
                            json.dumps(paper)
                        ))
                    await db.commit()

                self.stats["papers"]["migrated"] = len(papers)
                self.log(f"  ✓ Migrated {len(papers)} papers")

            except Exception as e:
                self.stats["papers"]["errors"] += 1
                self.log(f"  ✗ Error migrating papers: {e}")
        else:
            self.stats["papers"]["migrated"] = len(papers)
            self.log(f"  Found {len(papers)} papers")

    async def migrate_packs(self):
        """Migrate context packs."""
        self.log("\n📦 Migrating Context Packs...")

        all_packs = []

        # Check context_packs directory
        if PACKS_DIR.exists():
            for pack_file in PACKS_DIR.glob("*.json"):
                try:
                    pack = json.loads(pack_file.read_text())
                    if "id" not in pack:
                        pack["id"] = pack_file.stem
                    all_packs.append(pack)
                except Exception as e:
                    self.log(f"  Warning: Could not read {pack_file}: {e}")

        # Also check v2 packs
        v2_dir = PACKS_DIR / "v2"
        if v2_dir.exists():
            for pack_file in v2_dir.glob("*.json"):
                try:
                    pack = json.loads(pack_file.read_text())
                    if "id" not in pack:
                        pack["id"] = f"v2-{pack_file.stem}"
                    all_packs.append(pack)
                except Exception as e:
                    self.log(f"  Warning: Could not read {pack_file}: {e}")

        self.stats["packs"]["found"] = len(all_packs)

        if all_packs and not self.dry_run:
            try:
                for pack in all_packs:
                    await self.engine.store_pack(pack, source="migration")
                self.stats["packs"]["migrated"] = len(all_packs)
                self.log(f"  ✓ Migrated {len(all_packs)} packs")
            except Exception as e:
                self.stats["packs"]["errors"] += 1
                self.log(f"  ✗ Error migrating packs: {e}")
        else:
            self.stats["packs"]["migrated"] = len(all_packs)
            self.log(f"  Found {len(all_packs)} packs")

    async def build_lineage(self):
        """Build lineage relationships from migrated data."""
        self.log("\n🔗 Building Lineage...")

        if self.dry_run:
            self.log("  Skipping in dry run mode")
            return

        try:
            # Session → Finding relationships
            sessions = await self.engine.list_sessions(limit=1000)
            for session in sessions:
                findings = await self.engine.get_findings(session_id=session["id"])
                for finding in findings:
                    await self.engine.add_lineage(
                        source_type="session",
                        source_id=session["id"],
                        target_type="finding",
                        target_id=finding["id"],
                        relation="produced"
                    )

            self.log(f"  ✓ Built lineage for {len(sessions)} sessions")

        except Exception as e:
            self.log(f"  ✗ Error building lineage: {e}")

    def print_summary(self):
        """Print migration summary."""
        self.log("\n" + "=" * 60)
        self.log("MIGRATION SUMMARY")
        self.log("=" * 60)

        total_found = 0
        total_migrated = 0
        total_errors = 0

        for entity, counts in self.stats.items():
            total_found += counts["found"]
            total_migrated += counts["migrated"]
            total_errors += counts["errors"]

            status = "✓" if counts["errors"] == 0 else "⚠"
            self.log(f"  {status} {entity.capitalize()}: {counts['migrated']}/{counts['found']} migrated")
            if counts["errors"]:
                self.log(f"      ({counts['errors']} errors)")

        self.log("")
        self.log(f"  Total: {total_migrated}/{total_found} entities migrated")

        if total_errors:
            self.log(f"  Errors: {total_errors}")

        if self.dry_run:
            self.log("\n  DRY RUN - No data was written")
        else:
            self.log(f"\n  Database: {DB_PATH}")


async def migrate_from_json(dry_run: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Main migration function."""
    migrator = Migrator(dry_run=dry_run, verbose=verbose)
    await migrator.migrate_all()
    return migrator.stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate JSON data to SQLite + Qdrant"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without writing data"
    )
    parser.add_argument(
        "--sessions-only",
        action="store_true",
        help="Only migrate sessions"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    async def run():
        migrator = Migrator(dry_run=args.dry_run, verbose=not args.quiet)

        if args.sessions_only:
            await migrator.initialize()
            await migrator.migrate_sessions()
            await migrator.close()
            migrator.print_summary()
        else:
            await migrator.migrate_all()

    asyncio.run(run())


if __name__ == "__main__":
    main()
