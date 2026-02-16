#!/usr/bin/env python3
"""
Backfill learnings from archived sessions into ~/.agent-core/memory/learnings.md

Scans all archived sessions and extracts:
- Findings (thesis, gap, innovation, finding types)
- High-value URLs (Tier 1/2 research sources)
- Project lineage connections
- Evidence chains with confidence scores (new in Evidence Layer)

Implements "Evidence Required" principle by preserving citation chains.

Usage:
  python3 backfill_learnings.py              # Process all sessions
  python3 backfill_learnings.py --since 7    # Only last 7 days
  python3 backfill_learnings.py --session ID # Specific session
  python3 backfill_learnings.py --dry-run    # Preview without writing
  python3 backfill_learnings.py --with-evidence  # Include evidence details
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional


AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
MEMORY_DIR = AGENT_CORE_DIR / "memory"
LEARNINGS_FILE = MEMORY_DIR / "learnings.md"
PROJECTS_FILE = AGENT_CORE_DIR / "projects.json"


def load_projects() -> Dict[str, Any]:
    """Load projects registry for lineage lookups."""
    if PROJECTS_FILE.exists():
        return json.loads(PROJECTS_FILE.read_text())
    return {"projects": {}, "paper_index": {}, "topic_index": {}}


def scan_archived_sessions(since_days: Optional[int] = None) -> List[Path]:
    """
    Scan all archived sessions, optionally filtered by date.
    Returns list of session directories sorted by date (oldest first).
    """
    sessions = []

    if not SESSIONS_DIR.exists():
        return sessions

    cutoff = None
    if since_days:
        cutoff = datetime.now() - timedelta(days=since_days)

    for item in SESSIONS_DIR.iterdir():
        if not item.is_dir():
            continue

        session_file = item / "session.json"
        if not session_file.exists():
            continue

        try:
            data = json.loads(session_file.read_text())

            # Check date filter
            if cutoff:
                date_str = data.get("original_date") or data.get("started") or data.get("backfilled_at")
                if date_str:
                    try:
                        session_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        if session_date.replace(tzinfo=None) < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass

            sessions.append(item)
        except (json.JSONDecodeError, IOError):
            continue

    # Sort by directory name (contains timestamp)
    return sorted(sessions, key=lambda x: x.name)


def extract_session_learnings(session_dir: Path, include_evidence: bool = False) -> Dict[str, Any]:
    """
    Extract learnings from a single session.

    Sources:
    - session.json: metadata (topic, date, status)
    - findings_captured.json: thesis, gap, innovation, finding entries
    - findings_evidenced.json: evidence-enriched findings (new)
    - urls_captured.json: Tier 1/2 research URLs
    - lineage.json: project linkage

    Args:
        session_dir: Path to session directory
        include_evidence: Include evidence chains in output
    """
    result = {
        "session_id": session_dir.name,
        "topic": "",
        "date": "",
        "project": None,
        "findings": [],
        "papers": [],
        "tools": [],
        "insights": [],
        "thesis": None,
        "gap": None,
        "lineage": None,
        # Evidence layer additions
        "evidence_stats": None,
        "evidenced_findings": [],
    }

    # Load session metadata
    session_file = session_dir / "session.json"
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text())
            result["topic"] = data.get("topic", "Unknown")

            # Get date (prefer original_date for backfilled sessions)
            date_str = data.get("original_date") or data.get("started") or data.get("backfilled_at")
            if date_str:
                try:
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    result["date"] = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    result["date"] = date_str[:10] if date_str else ""

            # Load evidence stats if available
            result["evidence_stats"] = data.get("evidence_stats")
        except (json.JSONDecodeError, IOError):
            pass

    # Load evidenced findings if available (new Evidence Layer)
    evidenced_file = session_dir / "findings_evidenced.json"
    if include_evidence and evidenced_file.exists():
        try:
            evidenced = json.loads(evidenced_file.read_text())
            for f in evidenced:
                if "evidence" in f and f["evidence"].get("confidence", 0) >= 0.5:
                    result["evidenced_findings"].append({
                        "content": f.get("content", "")[:200],
                        "type": f.get("type", "finding"),
                        "confidence": f["evidence"].get("confidence", 0),
                        "sources": [s.get("url", "") for s in f["evidence"].get("sources", [])[:3]],
                        "validated": f["evidence"].get("validation", {}).get("validated", False),
                    })
        except (json.JSONDecodeError, IOError):
            pass

    # Load lineage
    lineage_file = session_dir / "lineage.json"
    if lineage_file.exists():
        try:
            lineage = json.loads(lineage_file.read_text())
            result["project"] = lineage.get("impl_project")
            result["lineage"] = lineage
        except (json.JSONDecodeError, IOError):
            pass

    # Load and process findings
    findings_file = session_dir / "findings_captured.json"
    if findings_file.exists():
        try:
            findings = json.loads(findings_file.read_text())

            for f in findings:
                f_type = f.get("type", "finding")
                text = f.get("text", "").strip()

                if not text or len(text) < 20:
                    continue

                # Extract arXiv references
                arxiv_matches = re.findall(r'arXiv[:\s]*(\d{4}\.\d{4,5})', text, re.IGNORECASE)
                for arxiv_id in arxiv_matches:
                    if arxiv_id not in [p.get("id") for p in result["papers"]]:
                        result["papers"].append({
                            "id": arxiv_id,
                            "context": text[:200]
                        })

                # Categorize by type
                if f_type == "thesis" and len(text) > 50:
                    # Look for actual thesis statements
                    if any(kw in text.lower() for kw in ["because", "which means", "therefore", "approach", "sound", "optimize"]):
                        if not result["thesis"] or len(text) > len(result["thesis"]):
                            result["thesis"] = text[:500]

                elif f_type == "gap" and len(text) > 30:
                    if any(kw in text.lower() for kw in ["missing", "gap", "need", "lack", "without"]):
                        if not result["gap"] or len(text) > len(result["gap"]):
                            result["gap"] = text[:500]

                elif f_type == "finding":
                    # Clean up and add meaningful findings
                    if any(kw in text.lower() for kw in ["voting", "consensus", "agent", "performance", "framework", "tool"]):
                        result["findings"].append(text[:300])

                elif f_type == "innovation":
                    result["insights"].append(text[:300])

        except (json.JSONDecodeError, IOError):
            pass

    # Load high-value URLs (Tier 1 and Tier 2)
    urls_file = session_dir / "urls_captured.json"
    if urls_file.exists():
        try:
            urls = json.loads(urls_file.read_text())

            for u in urls:
                tier = u.get("tier", 3)
                url = u.get("url", "")
                source = u.get("source", "")

                # Only include Tier 1/2 URLs
                if tier <= 2:
                    # Extract meaningful URLs
                    if "arxiv.org" in url:
                        arxiv_match = re.search(r'(\d{4}\.\d{4,5})', url)
                        if arxiv_match:
                            arxiv_id = arxiv_match.group(1)
                            if arxiv_id not in [p.get("id") for p in result["papers"]]:
                                result["papers"].append({
                                    "id": arxiv_id,
                                    "url": url,
                                    "context": u.get("context", "")[:100]
                                })

                    elif "github.com" in url and "/blob/" not in url and "/tree/" not in url:
                        # Extract repo info
                        match = re.search(r'github\.com/([^/]+)/([^/\s?#]+)', url)
                        if match:
                            repo_name = f"{match.group(1)}/{match.group(2)}"
                            if repo_name not in [t.get("name") for t in result["tools"]]:
                                result["tools"].append({
                                    "name": repo_name,
                                    "url": url,
                                    "source": source
                                })

        except (json.JSONDecodeError, IOError):
            pass

    # Deduplicate findings
    result["findings"] = list(set(result["findings"]))[:5]
    result["insights"] = list(set(result["insights"]))[:3]

    return result


def format_learning_entry(session_data: Dict[str, Any], include_evidence: bool = False) -> str:
    """Format learnings as a markdown section."""
    lines = []

    # Header
    lines.append(f"\n## {session_data['date']} - {session_data['topic']} (`{session_data['session_id'][:50]}`)")
    lines.append("")

    # Project linkage
    if session_data["project"]:
        lines.append(f"**Project:** {session_data['project']}")
        if session_data["lineage"] and session_data["lineage"].get("key_artifacts"):
            artifacts = ", ".join(session_data["lineage"]["key_artifacts"][:3])
            lines.append(f"**Key Artifacts:** {artifacts}")
        lines.append("")

    # Evidence stats (if available)
    if session_data.get("evidence_stats"):
        stats = session_data["evidence_stats"]
        confidence = stats.get("avg_confidence", 0)
        pass_rate = stats.get("validation_pass_rate", 0)
        if confidence > 0:
            badge = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.5 else "🔴"
            lines.append(f"**Evidence:** {badge} {confidence:.2f} confidence, {pass_rate*100:.0f}% validated")
            lines.append("")

    # Papers (most valuable)
    if session_data["papers"]:
        lines.append("### Research Papers")
        for paper in session_data["papers"][:5]:
            arxiv_id = paper.get("id", "")
            url = paper.get("url", f"https://arxiv.org/abs/{arxiv_id}")
            context = paper.get("context", "")[:80]
            if context:
                lines.append(f"- **[{arxiv_id}]({url})** — {context}")
            else:
                lines.append(f"- **[{arxiv_id}]({url})**")
        lines.append("")

    # Tools/Repos
    if session_data["tools"]:
        lines.append("### Tools & Repositories")
        for tool in session_data["tools"][:5]:
            name = tool.get("name", "Unknown")
            url = tool.get("url", "")
            lines.append(f"- [{name}]({url})")
        lines.append("")

    # Key Findings
    if session_data["findings"]:
        lines.append("### Key Findings")
        for finding in session_data["findings"][:3]:
            # Clean up the finding text
            clean = finding.replace("\n", " ").strip()
            if len(clean) > 200:
                clean = clean[:200] + "..."
            lines.append(f"- {clean}")
        lines.append("")

    # Thesis
    if session_data["thesis"]:
        lines.append("### Thesis")
        thesis = session_data["thesis"].replace("\n", " ").strip()
        if len(thesis) > 300:
            thesis = thesis[:300] + "..."
        lines.append(f"> {thesis}")
        lines.append("")

    # Gap
    if session_data["gap"]:
        lines.append("### Gap Identified")
        gap = session_data["gap"].replace("\n", " ").strip()
        if len(gap) > 200:
            gap = gap[:200] + "..."
        lines.append(f"> {gap}")
        lines.append("")

    # Insights
    if session_data["insights"]:
        lines.append("### Innovation Directions")
        for insight in session_data["insights"][:2]:
            clean = insight.replace("\n", " ").strip()
            if len(clean) > 150:
                clean = clean[:150] + "..."
            lines.append(f"- {clean}")
        lines.append("")

    # Evidenced findings (new Evidence Layer)
    if include_evidence and session_data.get("evidenced_findings"):
        lines.append("### Evidence-Backed Findings")
        for ef in session_data["evidenced_findings"][:3]:
            confidence = ef.get("confidence", 0)
            validated = "✓" if ef.get("validated") else ""
            badge = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.5 else "🔴"
            content = ef.get("content", "")[:150]
            if len(content) >= 150:
                content += "..."
            lines.append(f"- {badge} [{confidence:.2f}]{validated} {content}")

            # Show sources if available
            sources = ef.get("sources", [])
            if sources:
                for src in sources[:2]:
                    lines.append(f"  - Source: {src}")
        lines.append("")

    lines.append("---")

    return "\n".join(lines)


def create_learnings_file(
    all_sessions: List[Dict[str, Any]],
    dry_run: bool = False,
    include_evidence: bool = False
) -> Path:
    """
    Create ~/.agent-core/memory/learnings.md from extracted session data.

    Args:
        all_sessions: List of session data dicts
        dry_run: Preview without writing
        include_evidence: Include evidence chains in output
    """
    # Build header
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    projects_linked = set(s["project"] for s in all_sessions if s["project"])
    total_papers = sum(len(s["papers"]) for s in all_sessions)
    total_findings = sum(len(s["findings"]) for s in all_sessions)

    content = f"""# Learnings Archive

**Last Updated:** {now}
**Total Sessions:** {len(all_sessions)}
**Projects Linked:** {', '.join(sorted(projects_linked)) if projects_linked else 'None'}
**Papers Referenced:** {total_papers}
**Key Findings:** {total_findings}

> This file is auto-generated by `backfill_learnings.py` and updated by `archive_session.py`.
> Use `prefetch.py` to inject relevant learnings into Claude sessions.

---
"""

    # Add each session (most recent first)
    for session in reversed(all_sessions):
        # Skip sessions with no meaningful content
        if not (session["papers"] or session["findings"] or session["thesis"] or session["tools"] or session.get("evidenced_findings")):
            continue

        entry = format_learning_entry(session, include_evidence=include_evidence)
        content += entry

    if dry_run:
        print("=" * 70)
        print("DRY RUN - Would write to:", LEARNINGS_FILE)
        print("=" * 70)
        print(content[:3000])
        if len(content) > 3000:
            print(f"\n... [{len(content) - 3000} more characters]")
        return LEARNINGS_FILE

    # Write file
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    LEARNINGS_FILE.write_text(content)

    return LEARNINGS_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Backfill learnings from archived sessions"
    )
    parser.add_argument("--since", "-s", type=int,
                        help="Only process sessions from last N days")
    parser.add_argument("--session", "-id",
                        help="Process specific session ID only")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Preview output without writing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed extraction info")
    parser.add_argument("--with-evidence", "-e", action="store_true",
                        help="Include evidence chains in output")

    args = parser.parse_args()

    print("ResearchGravity — Learnings Backfill")
    print("=" * 50)

    # Scan sessions
    if args.session:
        session_path = SESSIONS_DIR / args.session
        if not session_path.exists():
            print(f"❌ Session not found: {args.session}")
            return
        sessions = [session_path]
    else:
        sessions = scan_archived_sessions(since_days=args.since)

    print(f"📂 Found {len(sessions)} session(s) to process")

    if not sessions:
        print("No sessions to process")
        return

    # Extract learnings
    all_learnings = []
    for session_dir in sessions:
        print(f"  → Processing: {session_dir.name[:50]}...")

        data = extract_session_learnings(session_dir, include_evidence=args.with_evidence)

        if args.verbose:
            print(f"    Topic: {data['topic']}")
            print(f"    Date: {data['date']}")
            print(f"    Project: {data['project'] or 'None'}")
            print(f"    Papers: {len(data['papers'])}")
            print(f"    Tools: {len(data['tools'])}")
            print(f"    Findings: {len(data['findings'])}")
            if args.with_evidence and data.get("evidenced_findings"):
                print(f"    Evidenced: {len(data['evidenced_findings'])}")

        all_learnings.append(data)

    # Create learnings file
    print()
    output_path = create_learnings_file(
        all_learnings,
        dry_run=args.dry_run,
        include_evidence=args.with_evidence
    )

    if not args.dry_run:
        print(f"✅ Learnings written to: {output_path}")

        # Stats
        total_papers = sum(len(s["papers"]) for s in all_learnings)
        total_findings = sum(len(s["findings"]) for s in all_learnings)
        total_tools = sum(len(s["tools"]) for s in all_learnings)

        print()
        print("📊 Summary:")
        print(f"   Sessions processed: {len(all_learnings)}")
        print(f"   Papers extracted: {total_papers}")
        print(f"   Tools/repos found: {total_tools}")
        print(f"   Key findings: {total_findings}")


if __name__ == "__main__":
    main()
