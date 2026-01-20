#!/usr/bin/env python3
"""
Reinvigoration System - Session Resume with Full Context.

Reconstructs complete agent context for seamless session resumption.
Implements "Agent-Led Construction" and "Reinvigoration Ready" principles.

Features:
- Full transcript preservation
- Finding and evidence loading
- Lineage mapping
- Task identification
- Context pack selection

Usage:
    python3 reinvigorate.py <session-id>             # Get reinvigoration context
    python3 reinvigorate.py <session-id> --inject    # Inject into CLAUDE.md
    python3 reinvigorate.py <session-id> --clipboard # Copy to clipboard
    python3 reinvigorate.py --list                   # List resumable sessions
    python3 reinvigorate.py --verify <session-id>    # Verify context completeness
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
CLAUDE_MD = Path.home() / "CLAUDE.md"


def load_json_safe(path: Path) -> dict | list:
    """Safely load JSON file."""
    if not path.exists():
        return {} if path.suffix == ".json" else []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {} if path.suffix == ".json" else []


def get_session_metadata(session_dir: Path) -> dict:
    """Extract session metadata."""
    session_file = session_dir / "session.json"
    data = load_json_safe(session_file)

    return {
        "id": session_dir.name,
        "topic": data.get("topic", session_dir.name[:50]),
        "status": data.get("status", "archived"),
        "project": data.get("implementation_project"),
        "started_at": data.get("started_at"),
        "archived_at": data.get("archived_at"),
    }


def extract_last_checkpoint(transcript: str, chars: int = 3000) -> str:
    """Extract the last significant checkpoint from transcript."""
    if not transcript:
        return "No transcript available"

    # Get last portion
    recent = transcript[-chars:] if len(transcript) > chars else transcript

    # Try to find a good breaking point
    # Look for common checkpoint markers
    markers = [
        "\n## ", "\n### ",  # Markdown headers
        "\nCheckpoint:", "\nSummary:",
        "\nCompleted:", "\nNext steps:",
        "\n---",  # Section breaks
    ]

    best_start = 0
    for marker in markers:
        idx = recent.rfind(marker)
        if idx > best_start:
            best_start = idx

    if best_start > 0:
        return recent[best_start:].strip()

    # Fall back to last paragraph
    paragraphs = recent.split("\n\n")
    if paragraphs:
        return "\n\n".join(paragraphs[-3:]).strip()

    return recent.strip()


def identify_incomplete_tasks(transcript: str) -> list:
    """Identify potentially incomplete tasks from transcript."""
    if not transcript:
        return []

    tasks = []

    # Look for TODO patterns
    todo_patterns = [
        r'TODO[:\s]+([^\n]+)',
        r'\[ \]\s*([^\n]+)',  # Markdown unchecked
        r'PENDING[:\s]+([^\n]+)',
        r'Next[:\s]+([^\n]+)',
        r'remaining[:\s]+([^\n]+)',
    ]

    for pattern in todo_patterns:
        matches = re.findall(pattern, transcript, re.IGNORECASE)
        tasks.extend(matches[:5])  # Limit per pattern

    # Deduplicate
    seen = set()
    unique_tasks = []
    for task in tasks:
        task_clean = task.strip()[:100]
        if task_clean and task_clean.lower() not in seen:
            seen.add(task_clean.lower())
            unique_tasks.append(task_clean)

    return unique_tasks[:10]  # Limit total


def format_findings(findings: list, limit: int = 10) -> str:
    """Format findings for context injection."""
    if not findings:
        return "No findings captured"

    output = []
    for i, f in enumerate(findings[:limit]):
        finding_type = f.get("type", "finding")
        content = f.get("content", f.get("text", ""))[:200]

        # Get confidence if available
        evidence = f.get("evidence", {})
        confidence = evidence.get("confidence", 0)
        sources_count = len(evidence.get("sources", []))

        line = f"- [{finding_type}] {content}"
        if confidence > 0:
            line += f" (conf: {confidence:.2f}, {sources_count} sources)"

        output.append(line)

    if len(findings) > limit:
        output.append(f"... and {len(findings) - limit} more findings")

    return "\n".join(output)


def format_urls(urls: list, limit: int = 10) -> str:
    """Format URLs for context injection."""
    if not urls:
        return "No URLs captured"

    output = []
    for u in urls[:limit]:
        tier = u.get("tier", 3)
        url = u.get("url", "")[:80]
        category = u.get("category", "")

        line = f"- [Tier {tier}] {url}"
        if category:
            line += f" ({category})"

        output.append(line)

    if len(urls) > limit:
        output.append(f"... and {len(urls) - limit} more URLs")

    return "\n".join(output)


def format_lineage(lineage: dict) -> str:
    """Format lineage connections."""
    if not lineage:
        return "No lineage recorded"

    output = []

    # Parent sessions
    parents = lineage.get("parent_sessions", [])
    if parents:
        output.append(f"Parent sessions: {', '.join(parents[:3])}")

    # Related papers
    papers = lineage.get("papers", [])
    if papers:
        paper_ids = [p.get("arxiv_id", p.get("id", "")) for p in papers[:5]]
        output.append(f"Related papers: {', '.join(paper_ids)}")

    # Implementation project
    project = lineage.get("implementation_project")
    if project:
        output.append(f"Implementation project: {project}")

    return "\n".join(output) if output else "Minimal lineage"


def build_reinvigoration_context(session_id: str) -> str:
    """
    Build complete reinvigoration context for a session.

    Returns a markdown block ready for injection or display.
    """
    session_dir = SESSIONS_DIR / session_id

    if not session_dir.exists():
        return f"## ERROR: Session not found: {session_id}"

    # Load all session data
    metadata = get_session_metadata(session_dir)
    findings_file = session_dir / "findings_evidenced.json"
    if not findings_file.exists():
        findings_file = session_dir / "findings_captured.json"
    findings = load_json_safe(findings_file)

    urls = load_json_safe(session_dir / "urls_captured.json")
    lineage = load_json_safe(session_dir / "lineage.json")

    # Load transcript
    transcript_file = session_dir / "full_transcript.txt"
    transcript = ""
    if transcript_file.exists():
        transcript = transcript_file.read_text()

    # Extract checkpoint and tasks
    last_checkpoint = extract_last_checkpoint(transcript)
    incomplete_tasks = identify_incomplete_tasks(transcript)

    # Build context block
    context = f"""## SESSION REINVIGORATION: {session_id[:50]}

### Session Info
- **Topic:** {metadata.get('topic', 'Unknown')}
- **Project:** {metadata.get('project') or 'None'}
- **Status:** {metadata.get('status', 'archived')}
- **Started:** {metadata.get('started_at', 'Unknown')}

### Key Findings ({len(findings)} total)
{format_findings(findings)}

### URLs Captured ({len(urls) if isinstance(urls, list) else 0} total)
{format_urls(urls if isinstance(urls, list) else [])}

### Lineage & Connections
{format_lineage(lineage if isinstance(lineage, dict) else {})}

### Last Checkpoint
```
{last_checkpoint[:2000]}
```
"""

    if incomplete_tasks:
        context += f"""
### Incomplete Tasks
"""
        for task in incomplete_tasks:
            context += f"- [ ] {task}\n"

    context += """
---
**Resume from this point. Full context preserved.**
"""

    return context


def verify_reinvigoration(session_id: str) -> dict:
    """Verify reinvigoration context completeness."""
    session_dir = SESSIONS_DIR / session_id

    if not session_dir.exists():
        return {"complete": False, "error": "Session not found"}

    checks = {
        "session_json": (session_dir / "session.json").exists(),
        "findings": (session_dir / "findings_captured.json").exists() or
                    (session_dir / "findings_evidenced.json").exists(),
        "urls": (session_dir / "urls_captured.json").exists(),
        "transcript": (session_dir / "full_transcript.txt").exists(),
        "lineage": (session_dir / "lineage.json").exists(),
    }

    # Count findings and URLs
    findings = load_json_safe(session_dir / "findings_captured.json")
    urls = load_json_safe(session_dir / "urls_captured.json")

    completeness = sum(checks.values()) / len(checks)

    return {
        "session_id": session_id,
        "complete": completeness >= 0.6,
        "completeness": completeness,
        "checks": checks,
        "findings_count": len(findings) if isinstance(findings, list) else 0,
        "urls_count": len(urls) if isinstance(urls, list) else 0,
    }


def list_resumable_sessions(limit: int = 20) -> list:
    """List sessions that can be reinvigorated."""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []

    for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue

        verification = verify_reinvigoration(session_dir.name)
        if verification.get("complete"):
            metadata = get_session_metadata(session_dir)
            sessions.append({
                "id": session_dir.name,
                "topic": metadata.get("topic", ""),
                "project": metadata.get("project"),
                "completeness": verification.get("completeness", 0),
                "findings": verification.get("findings_count", 0),
            })

        if len(sessions) >= limit:
            break

    return sessions


def inject_context(context: str) -> bool:
    """Inject reinvigoration context into CLAUDE.md."""
    if not CLAUDE_MD.exists():
        print(f"CLAUDE.md not found at {CLAUDE_MD}")
        return False

    content = CLAUDE_MD.read_text()

    # Find or create injection point
    start_marker = "<!-- REINVIGORATION CONTEXT START -->"
    end_marker = "<!-- REINVIGORATION CONTEXT END -->"

    injection = f"{start_marker}\n{context}\n{end_marker}"

    if start_marker in content:
        # Replace existing
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker) + len(end_marker)
        content = content[:start_idx] + injection + content[end_idx:]
    else:
        # Append
        content += f"\n\n{injection}"

    CLAUDE_MD.write_text(content)
    return True


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard (macOS)."""
    try:
        process = subprocess.Popen(
            ['pbcopy'],
            stdin=subprocess.PIPE,
            env={'LANG': 'en_US.UTF-8'}
        )
        process.communicate(text.encode('utf-8'))
        return process.returncode == 0
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reinvigoration System - Resume sessions with full context"
    )
    parser.add_argument("session_id", nargs="?", help="Session ID to reinvigorate")
    parser.add_argument("--inject", "-i", action="store_true",
                        help="Inject context into CLAUDE.md")
    parser.add_argument("--clipboard", "-c", action="store_true",
                        help="Copy context to clipboard")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List resumable sessions")
    parser.add_argument("--verify", "-v", action="store_true",
                        help="Verify session reinvigoration readiness")
    parser.add_argument("--limit", type=int, default=20,
                        help="Limit for list output")

    args = parser.parse_args()

    if args.list:
        sessions = list_resumable_sessions(args.limit)

        print("Resumable Sessions:")
        print("=" * 60)

        for s in sessions:
            completeness = int(s.get("completeness", 0) * 100)
            print(f"  {s['id'][:50]}")
            print(f"    Topic: {s.get('topic', 'N/A')[:40]}")
            print(f"    Project: {s.get('project') or 'None'}")
            print(f"    Completeness: {completeness}% | Findings: {s.get('findings', 0)}")
            print()

        print(f"Total: {len(sessions)} resumable sessions")
        return 0

    if not args.session_id:
        parser.print_help()
        return 1

    if args.verify:
        result = verify_reinvigoration(args.session_id)

        print(f"Reinvigoration Verification: {args.session_id[:40]}")
        print("=" * 50)
        print(f"Complete: {'✅ Yes' if result['complete'] else '❌ No'}")
        print(f"Completeness: {result.get('completeness', 0):.0%}")
        print(f"Findings: {result.get('findings_count', 0)}")
        print(f"URLs: {result.get('urls_count', 0)}")

        print("\nChecks:")
        for check, passed in result.get("checks", {}).items():
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check}")

        return 0 if result["complete"] else 1

    # Build context
    context = build_reinvigoration_context(args.session_id)

    if args.inject:
        if inject_context(context):
            print(f"✅ Context injected into {CLAUDE_MD}")
        else:
            print("❌ Failed to inject context")
            return 1

    elif args.clipboard:
        if copy_to_clipboard(context):
            print("✅ Context copied to clipboard")
        else:
            print("❌ Failed to copy to clipboard")
            # Still print it
            print("\n" + context)

    else:
        # Just print the context
        print(context)

    return 0


if __name__ == "__main__":
    sys.exit(main())
