#!/usr/bin/env python3
"""
ResearchGravity Session Tracker v2.0
Automatic capture of full research sessions with cross-project lineage.

This script:
1. Links ResearchGravity sessions to Claude Code session files
2. Auto-extracts URLs, findings, summaries from transcripts
3. Tracks project switches and creates lineage links
4. Archives complete research artifacts automatically

Usage:
  python3 session_tracker.py register <topic> [--impl-project PROJECT]
  python3 session_tracker.py capture [--session-id ID]
  python3 session_tracker.py link <research-session> <impl-project>
  python3 session_tracker.py status
"""

import argparse
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


# Paths
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
TRACKER_FILE = AGENT_CORE_DIR / "session_tracker.json"
LOCAL_AGENT_DIR = Path.cwd() / ".agent"


def get_tracker_state() -> Dict[str, Any]:
    """Load or initialize tracker state."""
    if TRACKER_FILE.exists():
        return json.loads(TRACKER_FILE.read_text())
    return {
        "version": "2.0",
        "active_session": None,
        "sessions": {},
        "lineage": [],  # [{research_session, impl_project, impl_session, linked_at}]
        "pending_captures": []
    }


def save_tracker_state(state: Dict[str, Any]):
    """Save tracker state."""
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_FILE.write_text(json.dumps(state, indent=2))


def find_claude_session_file() -> Optional[Path]:
    """Find the most recent Claude Code session file for current directory."""
    cwd = str(Path.cwd())
    # Claude encodes paths in project directory names
    encoded_path = cwd.replace("/", "-")

    project_dir = CLAUDE_PROJECTS_DIR / encoded_path
    if project_dir.exists():
        # Find most recent .jsonl file
        jsonl_files = list(project_dir.glob("*.jsonl"))
        if jsonl_files:
            return max(jsonl_files, key=lambda f: f.stat().st_mtime)

    # Fallback: search all project dirs
    for proj_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if proj_dir.is_dir():
            jsonl_files = list(proj_dir.glob("*.jsonl"))
            if jsonl_files:
                latest = max(jsonl_files, key=lambda f: f.stat().st_mtime)
                # Check if recent (within last hour)
                if (datetime.now().timestamp() - latest.stat().st_mtime) < 3600:
                    return latest
    return None


def extract_urls_from_transcript(transcript_lines: List[str]) -> List[Dict[str, str]]:
    """Extract URLs and context from session transcript."""
    urls = []
    url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

    for i, line in enumerate(transcript_lines):
        matches = url_pattern.findall(line)
        for url in matches:
            # Clean URL
            url = url.rstrip('.,;:)')

            # Get context (surrounding lines)
            start = max(0, i - 2)
            end = min(len(transcript_lines), i + 3)
            context = ' '.join(transcript_lines[start:end])[:500]

            # Detect source type
            source_type = detect_source_type(url)

            urls.append({
                "url": url,
                "source_type": source_type,
                "context": context,
                "line_number": i,
                "timestamp": datetime.now().isoformat()
            })

    # Deduplicate
    seen = set()
    unique_urls = []
    for u in urls:
        if u["url"] not in seen:
            seen.add(u["url"])
            unique_urls.append(u)

    return unique_urls


def detect_source_type(url: str) -> Dict[str, Any]:
    """Detect tier and category from URL."""
    url_lower = url.lower()

    # Tier 1: Research
    if "arxiv.org" in url_lower:
        return {"tier": 1, "category": "research", "source": "arXiv"}
    if "huggingface.co/papers" in url_lower:
        return {"tier": 1, "category": "research", "source": "HuggingFace"}

    # Tier 1: Labs
    if "openai.com" in url_lower:
        return {"tier": 1, "category": "labs", "source": "OpenAI"}
    if "anthropic.com" in url_lower:
        return {"tier": 1, "category": "labs", "source": "Anthropic"}
    if "deepmind.google" in url_lower or "blog.google" in url_lower:
        return {"tier": 1, "category": "labs", "source": "Google AI"}

    # Tier 1: Industry
    if "techcrunch.com" in url_lower:
        return {"tier": 1, "category": "industry", "source": "TechCrunch"}
    if "theverge.com" in url_lower:
        return {"tier": 1, "category": "industry", "source": "The Verge"}

    # Tier 2: GitHub
    if "github.com" in url_lower:
        return {"tier": 2, "category": "github", "source": "GitHub"}

    # Tier 2: Social
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return {"tier": 2, "category": "social", "source": "X/Twitter"}
    if "news.ycombinator.com" in url_lower:
        return {"tier": 2, "category": "social", "source": "Hacker News"}

    # Default
    return {"tier": 3, "category": "other", "source": "Web"}


def extract_key_findings(transcript_lines: List[str]) -> List[Dict[str, str]]:
    """Extract key findings, insights, and summaries from transcript."""
    findings = []

    # Patterns that indicate key findings
    finding_patterns = [
        r"key (finding|insight|takeaway)",
        r"important(ly)?:?\s",
        r"notable:?\s",
        r"significant:?\s",
        r"the main (point|finding|result)",
        r"in summary",
        r"thesis:?\s",
        r"gap:?\s",
        r"innovation (opportunity|direction)",
        r"decision quality",
        r"DQ (score|metric|framework)",
    ]

    combined_pattern = re.compile('|'.join(finding_patterns), re.IGNORECASE)

    for i, line in enumerate(transcript_lines):
        if combined_pattern.search(line):
            # Get surrounding context
            start = max(0, i - 1)
            end = min(len(transcript_lines), i + 5)
            finding_text = ' '.join(transcript_lines[start:end])[:1000]

            findings.append({
                "text": finding_text,
                "line_number": i,
                "pattern_matched": combined_pattern.search(line).group(0),
                "timestamp": datetime.now().isoformat()
            })

    return findings


def parse_claude_session(session_file: Path) -> Dict[str, Any]:
    """Parse a Claude Code session .jsonl file."""
    messages = []
    tool_calls = []

    try:
        with open(session_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "message":
                        messages.append(entry)
                    elif entry.get("type") == "tool_use":
                        tool_calls.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"error": str(e), "messages": [], "tool_calls": []}

    return {
        "session_file": str(session_file),
        "message_count": len(messages),
        "tool_call_count": len(tool_calls),
        "messages": messages,
        "tool_calls": tool_calls
    }


def generate_session_id(topic: str) -> str:
    """Generate unique session ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    topic_hash = hashlib.md5(topic.encode()).hexdigest()[:6]
    safe_topic = re.sub(r'[^a-z0-9]+', '-', topic.lower())[:20]
    return f"{safe_topic}-{timestamp}-{topic_hash}"


def register_session(topic: str, impl_project: Optional[str] = None) -> Dict[str, Any]:
    """Register a new research session with automatic Claude session linking."""
    state = get_tracker_state()

    session_id = generate_session_id(topic)
    claude_session = find_claude_session_file()

    session = {
        "session_id": session_id,
        "topic": topic,
        "started": datetime.now().isoformat(),
        "status": "active",
        "working_directory": str(Path.cwd()),
        "claude_session_file": str(claude_session) if claude_session else None,
        "impl_project_placeholder": impl_project,
        "urls_captured": [],
        "findings_captured": [],
        "checkpoints": [],
        "full_transcript_archived": False
    }

    # Create session directory
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save session metadata
    (session_dir / "session.json").write_text(json.dumps(session, indent=2))

    # Update tracker state
    state["active_session"] = session_id
    state["sessions"][session_id] = session

    # If impl project specified, create pending lineage
    if impl_project:
        state["lineage"].append({
            "research_session": session_id,
            "impl_project": impl_project,
            "impl_session": None,  # Will be filled when detected
            "linked_at": datetime.now().isoformat(),
            "status": "pending"
        })

    save_tracker_state(state)

    print(f"Session registered: {session_id}")
    print(f"  Topic: {topic}")
    print(f"  Claude session: {claude_session.name if claude_session else 'Not detected'}")
    if impl_project:
        print(f"  Implementation target: {impl_project}")

    return session


def capture_session(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Capture current session state - URLs, findings, and transcript."""
    state = get_tracker_state()

    if not session_id:
        session_id = state.get("active_session")

    if not session_id or session_id not in state["sessions"]:
        print("No active session to capture. Run 'register' first.")
        return {}

    session = state["sessions"][session_id]
    session_dir = SESSIONS_DIR / session_id

    # Find Claude session file
    claude_file = session.get("claude_session_file")
    if claude_file:
        claude_file = Path(claude_file)
    else:
        claude_file = find_claude_session_file()

    if not claude_file or not claude_file.exists():
        print("Could not find Claude session file")
        return session

    # Parse Claude session
    parsed = parse_claude_session(claude_file)

    # Extract transcript lines (simplified - get message content)
    transcript_lines = []
    for msg in parsed.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            transcript_lines.extend(content.split('\n'))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    transcript_lines.extend(item["text"].split('\n'))

    # Extract URLs
    urls = extract_urls_from_transcript(transcript_lines)
    session["urls_captured"].extend(urls)

    # Extract findings
    findings = extract_key_findings(transcript_lines)
    session["findings_captured"].extend(findings)

    # Save full transcript
    transcript_file = session_dir / "full_transcript.txt"
    transcript_file.write_text('\n'.join(transcript_lines))
    session["full_transcript_archived"] = True

    # Save URLs
    urls_file = session_dir / "urls_captured.json"
    urls_file.write_text(json.dumps(session["urls_captured"], indent=2))

    # Save findings
    findings_file = session_dir / "findings_captured.json"
    findings_file.write_text(json.dumps(session["findings_captured"], indent=2))

    # Create checkpoint
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "urls_count": len(urls),
        "findings_count": len(findings),
        "transcript_lines": len(transcript_lines)
    }
    session["checkpoints"].append(checkpoint)

    # Update state
    session["last_captured"] = datetime.now().isoformat()
    state["sessions"][session_id] = session
    (session_dir / "session.json").write_text(json.dumps(session, indent=2))
    save_tracker_state(state)

    print(f"Captured session: {session_id}")
    print(f"  URLs extracted: {len(urls)} (total: {len(session['urls_captured'])})")
    print(f"  Findings extracted: {len(findings)} (total: {len(session['findings_captured'])})")
    print(f"  Transcript lines: {len(transcript_lines)}")
    print(f"  Saved to: {session_dir}")

    return session


def link_sessions(research_session: str, impl_project: str, impl_session: Optional[str] = None):
    """Create lineage link between research session and implementation project."""
    state = get_tracker_state()

    # Check if research session exists
    if research_session not in state["sessions"]:
        print(f"Research session not found: {research_session}")
        return

    # Create lineage entry
    lineage_entry = {
        "research_session": research_session,
        "impl_project": impl_project,
        "impl_session": impl_session,
        "linked_at": datetime.now().isoformat(),
        "status": "linked"
    }

    # Check for existing pending link
    for i, link in enumerate(state["lineage"]):
        if link["research_session"] == research_session and link["status"] == "pending":
            state["lineage"][i] = lineage_entry
            break
    else:
        state["lineage"].append(lineage_entry)

    save_tracker_state(state)

    # Also save to research session directory
    session_dir = SESSIONS_DIR / research_session
    lineage_file = session_dir / "lineage.json"
    lineage_file.write_text(json.dumps(lineage_entry, indent=2))

    print(f"Linked: {research_session} → {impl_project}")
    if impl_session:
        print(f"  Implementation session: {impl_session}")


def show_status():
    """Show current tracker status."""
    state = get_tracker_state()

    print("=" * 60)
    print("  ResearchGravity Session Tracker")
    print("=" * 60)
    print()

    # Active session
    active = state.get("active_session")
    if active and active in state["sessions"]:
        session = state["sessions"][active]
        print("ACTIVE SESSION")
        print(f"  ID: {active}")
        print(f"  Topic: {session.get('topic', 'N/A')}")
        print(f"  Started: {session.get('started', 'N/A')}")
        print(f"  URLs captured: {len(session.get('urls_captured', []))}")
        print(f"  Findings captured: {len(session.get('findings_captured', []))}")
        print(f"  Checkpoints: {len(session.get('checkpoints', []))}")
    else:
        print("NO ACTIVE SESSION")

    print()
    print("-" * 60)
    print()

    # Recent sessions
    print(f"RECENT SESSIONS ({len(state['sessions'])} total)")
    sessions = sorted(
        state["sessions"].items(),
        key=lambda x: x[1].get("started", ""),
        reverse=True
    )[:5]

    for sid, sess in sessions:
        status = "" if sid == active else ""
        urls = len(sess.get("urls_captured", []))
        findings = len(sess.get("findings_captured", []))
        print(f"  {status} {sid[:40]}")
        print(f"      Topic: {sess.get('topic', 'N/A')[:50]}")
        print(f"      URLs: {urls} | Findings: {findings}")

    print()
    print("-" * 60)
    print()

    # Lineage
    print(f"LINEAGE LINKS ({len(state['lineage'])} total)")
    for link in state["lineage"][-5:]:
        status_icon = "" if link["status"] == "linked" else ""
        print(f"  {status_icon} {link['research_session'][:30]}")
        print(f"      → {link['impl_project']}")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ResearchGravity Session Tracker - Automatic research capture"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Register
    reg_parser = subparsers.add_parser("register", help="Register a new research session")
    reg_parser.add_argument("topic", help="Research topic")
    reg_parser.add_argument("--impl-project", help="Target implementation project")

    # Capture
    cap_parser = subparsers.add_parser("capture", help="Capture current session state")
    cap_parser.add_argument("--session-id", help="Session ID (default: active)")

    # Link
    link_parser = subparsers.add_parser("link", help="Link research to implementation")
    link_parser.add_argument("research_session", help="Research session ID")
    link_parser.add_argument("impl_project", help="Implementation project path/name")
    link_parser.add_argument("--impl-session", help="Implementation session ID")

    # Status
    subparsers.add_parser("status", help="Show tracker status")

    args = parser.parse_args()

    if args.command == "register":
        register_session(args.topic, args.impl_project)
    elif args.command == "capture":
        capture_session(args.session_id)
    elif args.command == "link":
        link_sessions(args.research_session, args.impl_project, getattr(args, 'impl_session', None))
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
