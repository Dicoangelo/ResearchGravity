#!/usr/bin/env python3
"""
Archive a completed research session.
Logs all URLs, extracts learnings, and syncs to global storage.

Usage:
  python3 archive_session.py [--extract-learnings] [--clean-local]
"""

import argparse
import json
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


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


def archive_session(extract_learnings_flag: bool = True, clean_local: bool = False):
    """Main archive function."""
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
    
    # Extract and save learnings
    learnings_count = 0
    if extract_learnings_flag:
        scratchpad = local_dir / "scratchpad.json"
        learnings = extract_learnings(session, scratchpad)
        if learnings:
            learnings_count = update_learnings_memory(session, learnings)
    
    # Update session index
    update_session_index(session, duration)
    
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
    
    args = parser.parse_args()
    
    archive_session(
        extract_learnings_flag=not args.no_extract,
        clean_local=args.clean_local
    )


if __name__ == "__main__":
    main()
