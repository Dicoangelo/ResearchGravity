#!/usr/bin/env python3
"""
Archive a completed research session.
Logs all URLs, extracts learnings, validates evidence, and syncs to global storage.

Implements "Agent Maintainability" and "Evidence Required" principles:
- Full session preservation for reinvigoration
- Evidence validation via Writer-Critic (Oracle integration)
- Confidence scoring for all findings

Usage:
  python3 archive_session.py [--extract-learnings] [--clean-local]
  python3 archive_session.py --validate-evidence    # Run evidence validation
  python3 archive_session.py --skip-validation      # Skip validation step
"""

import argparse
import json
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import evidence layer components
try:
    from evidence_extractor import process_session as extract_evidence
    from confidence_scorer import score_session
    from evidence_validator import validate_session
    EVIDENCE_LAYER_AVAILABLE = True
except ImportError:
    EVIDENCE_LAYER_AVAILABLE = False


def get_agent_core_dir() -> Path:
    return Path.home() / ".agent-core"


def get_local_agent_dir() -> Path:
    return Path.cwd() / ".agent"


def get_current_session() -> Optional[dict]:
    local_dir = get_local_agent_dir() / "research"
    session_file = local_dir / "session.json"
    if session_file.exists():
        return json.loads(session_file.read_text())
    return None


def parse_url_table(session_log_path: Path) -> list:
    """Parse URLs from session log markdown table."""
    urls = []
    content = session_log_path.read_text()
    
    # Find the URLs table
    table_pattern = r'\| Time \| Source \| URL.*?\n\|[-\s|]+\n((?:\|.*\n)*)'
    match = re.search(table_pattern, content)
    
    if match:
        rows = match.group(1).strip().split('\n')
        for row in rows:
            cols = [c.strip() for c in row.split('|')[1:-1]]
            if len(cols) >= 6:
                urls.append({
                    "time": cols[0],
                    "source": cols[1],
                    "url": cols[2],
                    "filter": cols[3] if len(cols) > 3 else "",
                    "used": cols[4] if len(cols) > 4 else "",
                    "relevance": cols[5] if len(cols) > 5 else "",
                    "notes": cols[6] if len(cols) > 6 else ""
                })
    
    return urls


def generate_queries_section(session: dict) -> str:
    """Generate queries section, handling missing data gracefully."""
    queries = session.get('queries', {})
    if not queries:
        return "_No search queries recorded for this session._"

    sections = []
    if 'viral' in queries:
        viral_github = queries['viral'].get('github', 'N/A')
        sections.append(f"### Viral Filter\n```\n{viral_github}\n```")
    if 'groundbreaker' in queries:
        gb_github = queries['groundbreaker'].get('github', 'N/A')
        sections.append(f"### Groundbreaker Filter\n```\n{gb_github}\n```")

    return "\n\n".join(sections) if sections else "_No search queries recorded._"


def generate_archive_report(session: dict, urls: list) -> str:
    """Generate the final session archive markdown."""
    now = datetime.now()
    started = datetime.fromisoformat(session["started"])
    duration = (now - started).total_seconds() / 60
    
    used_urls = [u for u in urls if '✓' in u.get("used", "")]
    unused_urls = [u for u in urls if '✗' in u.get("used", "")]
    
    report = f"""# Session Archive: {session['topic']}

**Session ID:** `{session['session_id']}`
**Workflow:** {session['workflow']}
**Environment:** {session['environment']}
**Started:** {session['started']}
**Completed:** {now.isoformat()}
**Duration:** {duration:.1f} minutes

---

## Summary

Total URLs visited: {len(urls)}
- Used in output: {len(used_urls)}
- Visited but not used: {len(unused_urls)}

## Search Queries Used

{generate_queries_section(session)}

---

## URLs Used in Final Output

| Source | URL | Relevance | Contribution |
|--------|-----|-----------|--------------|
"""
    
    for u in used_urls:
        report += f"| {u['source']} | {u['url']} | {u['relevance']} | {u['notes']} |\n"
    
    report += """
---

## URLs Visited But Not Used

| Source | URL | Why Skipped |
|--------|-----|-------------|
"""
    
    for u in unused_urls:
        report += f"| {u['source']} | {u['url']} | {u['notes']} |\n"
    
    report += f"""
---

## Files in Archive

- `session.json` - Session metadata
- `session_log.md` - Full research narrative
- `scratchpad.json` - Machine-readable findings
- `*_report.md` - Final synthesized output
- `*_sources.csv` - Raw source data

---

## Next Steps

To revisit this research:
```
/recall {session['topic']}
```

To continue this research:
```
python3 ~/.agent-core/scripts/init_session.py "{session['topic']}" --continue {session['session_id']}
```

---

*Archived: {now.strftime("%Y-%m-%d %H:%M")}*
"""
    
    return report


def extract_learnings(session: dict, scratchpad_path: Path) -> list:
    """Extract key learnings from scratchpad."""
    learnings = []
    
    if scratchpad_path.exists():
        data = json.loads(scratchpad_path.read_text())
        
        # Extract from viral candidates
        for item in data.get("viral_candidates", []):
            learnings.append({
                "type": "tool",
                "name": item.get("name", "Unknown"),
                "url": item.get("url", ""),
                "insight": f"High-adoption tool: {item.get('why', 'well-maintained')}"
            })
        
        # Extract from groundbreaker candidates
        for item in data.get("groundbreaker_candidates", []):
            learnings.append({
                "type": "innovation",
                "name": item.get("name", "Unknown"),
                "url": item.get("url", ""),
                "insight": f"Novel approach: {item.get('novel', 'emerging')}"
            })
        
        # Extract from arxiv papers
        for item in data.get("arxiv_papers", []):
            learnings.append({
                "type": "paper",
                "name": item.get("title", "Unknown"),
                "url": item.get("url", ""),
                "insight": item.get("insight", "Research finding")
            })
    
    return learnings


def update_learnings_memory(session: dict, learnings: list):
    """Append learnings to global memory."""
    memory_file = get_agent_core_dir() / "memory" / "learnings.md"
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    
    now = datetime.now().strftime("%Y-%m-%d")
    
    entry = f"\n\n## {now} - {session['topic']} (`{session['session_id']}`)\n\n"
    
    for l in learnings:
        entry += f"- **{l['type'].title()}**: [{l['name']}]({l['url']}) — {l['insight']}\n"
    
    with open(memory_file, "a") as f:
        f.write(entry)
    
    return len(learnings)


def update_session_index(session: dict, duration: float):
    """Update the global session index."""
    index_file = get_agent_core_dir() / "sessions" / "index.md"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not index_file.exists():
        index_file.write_text("| Date | Session ID | Topic | Workflow | Duration | Key Finding |\n")
        index_file.write_text(index_file.read_text() + "|------|------------|-------|----------|----------|-------------|\n")
    
    date = datetime.now().strftime("%Y-%m-%d")
    
    # Get key finding from scratchpad
    key_finding = "See report"
    scratchpad = get_local_agent_dir() / "research" / "scratchpad.json"
    if scratchpad.exists():
        data = json.loads(scratchpad.read_text())
        if data.get("viral_candidates"):
            key_finding = data["viral_candidates"][0].get("name", "See report")
    
    entry = f"| {date} | {session['session_id']} | {session['topic']} | {session['workflow']} | {duration:.0f}m | {key_finding} |\n"
    
    with open(index_file, "a") as f:
        f.write(entry)


def archive_session(
    extract_learnings_flag: bool = True,
    clean_local: bool = False,
    validate_evidence_flag: bool = True,
    skip_validation: bool = False
):
    """
    Main archive function with evidence layer integration.

    Args:
        extract_learnings_flag: Extract learnings to memory
        clean_local: Clean local workspace after archiving
        validate_evidence_flag: Run evidence extraction and scoring
        skip_validation: Skip Writer-Critic validation step
    """
    session = get_current_session()
    if not session:
        print("❌ No active session found")
        return False

    local_dir = get_local_agent_dir() / "research"
    global_dir = get_agent_core_dir() / "sessions" / session["session_id"]
    global_dir.mkdir(parents=True, exist_ok=True)

    # Parse URLs from session log
    session_log = local_dir / "session_log.md"
    urls = parse_url_table(session_log) if session_log.exists() else []

    # Generate archive report
    archive_report = generate_archive_report(session, urls)
    archive_path = local_dir / "session_archive.md"
    archive_path.write_text(archive_report)

    # Update session status
    session["status"] = "archived"
    session["completed"] = datetime.now().isoformat()

    # Calculate duration
    started = datetime.fromisoformat(session["started"])
    duration = (datetime.now() - started).total_seconds() / 60

    # Copy all files to global
    import shutil
    for file in local_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, global_dir / file.name)
        elif file.is_dir():
            dest = global_dir / file.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(file, dest)

    # ═══════════════════════════════════════════════════════════════
    # EVIDENCE LAYER: Extract, Score, and Validate
    # Implements "Evidence Required" and "Writer-Critic Validation"
    # ═══════════════════════════════════════════════════════════════
    evidence_stats = {
        "extracted": False,
        "scored": False,
        "validated": False,
        "findings_count": 0,
        "avg_confidence": 0.0,
        "validation_pass_rate": 0.0,
    }

    if EVIDENCE_LAYER_AVAILABLE and validate_evidence_flag:
        print("\n🔬 Evidence Layer Processing...")

        # Step 1: Extract evidence from findings
        try:
            extract_result = extract_evidence(session["session_id"])
            if "error" not in extract_result:
                evidence_stats["extracted"] = True
                evidence_stats["findings_count"] = extract_result.get("findings_processed", 0)
                print(f"   ✓ Evidence extracted: {evidence_stats['findings_count']} findings")
        except Exception as e:
            print(f"   ⚠ Evidence extraction failed: {e}")

        # Step 2: Score confidence
        try:
            score_result = score_session(session["session_id"], update_file=True)
            if "error" not in score_result:
                evidence_stats["scored"] = True
                evidence_stats["avg_confidence"] = score_result.get("avg_confidence", 0)
                print(f"   ✓ Confidence scored: {evidence_stats['avg_confidence']:.2f} avg")
        except Exception as e:
            print(f"   ⚠ Confidence scoring failed: {e}")

        # Step 3: Writer-Critic Validation (Oracle integration)
        if not skip_validation:
            try:
                validate_result = validate_session(
                    session["session_id"],
                    update_file=True,
                    verbose=False
                )
                if "error" not in validate_result:
                    evidence_stats["validated"] = True
                    evidence_stats["validation_pass_rate"] = validate_result.get("pass_rate", 0)
                    print(f"   ✓ Validated: {validate_result.get('passed', 0)}/{validate_result.get('findings_validated', 0)} passed")
            except Exception as e:
                print(f"   ⚠ Validation failed: {e}")

        print()
    elif not EVIDENCE_LAYER_AVAILABLE:
        print("ℹ  Evidence layer not available (import evidence_extractor for full features)")

    # Extract and save learnings
    learnings_count = 0
    if extract_learnings_flag:
        scratchpad = local_dir / "scratchpad.json"
        learnings = extract_learnings(session, scratchpad)
        if learnings:
            learnings_count = update_learnings_memory(session, learnings)

    # Update session index
    update_session_index(session, duration)

    # Save evidence stats to session metadata
    session_meta_path = global_dir / "session.json"
    if session_meta_path.exists():
        try:
            session_meta = json.loads(session_meta_path.read_text())
            session_meta["evidence_stats"] = evidence_stats
            session_meta_path.write_text(json.dumps(session_meta, indent=2))
        except (json.JSONDecodeError, IOError):
            pass

    # Clean local if requested
    if clean_local:
        for file in local_dir.iterdir():
            if file.is_file():
                file.unlink()
        last_session = local_dir / ".last_session"
        last_session.write_text(session["session_id"])

    print(f"✅ Session archived: {session['session_id']}")
    print(f"📁 Location: {global_dir}")
    print(f"📝 Learnings extracted: {learnings_count}")
    print(f"🔗 URLs logged: {len(urls)} total")
    print(f"⏱️  Duration: {duration:.1f} minutes")

    # Evidence summary
    if evidence_stats["extracted"]:
        print(f"🔬 Evidence: {evidence_stats['findings_count']} findings, "
              f"{evidence_stats['avg_confidence']:.2f} confidence, "
              f"{evidence_stats['validation_pass_rate']*100:.0f}% validated")

    print()
    print(f"To revisit: /recall {session['topic']}")
    print(f"To continue: --continue {session['session_id']}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Archive research session")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip extracting learnings to memory")
    parser.add_argument("--clean-local", action="store_true",
                        help="Clean local workspace after archiving")
    parser.add_argument("--validate-evidence", action="store_true",
                        help="Run evidence layer processing (default: True)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip Writer-Critic validation step")
    parser.add_argument("--no-evidence", action="store_true",
                        help="Skip all evidence layer processing")

    args = parser.parse_args()

    archive_session(
        extract_learnings_flag=not args.no_extract,
        clean_local=args.clean_local,
        validate_evidence_flag=not args.no_evidence,
        skip_validation=args.skip_validation
    )


if __name__ == "__main__":
    main()
