#!/usr/bin/env python3
"""
ResearchGravity Auto-Capture
Automatically captures research sessions from Claude Code history.

Features:
1. Scans Claude Code session files for research artifacts
2. Extracts URLs, findings, summaries automatically
3. Backfills historical sessions that weren't captured
4. Runs on-demand or via Claude Code hooks

Usage:
  python3 auto_capture.py scan                    # Scan recent sessions
  python3 auto_capture.py backfill <session-file> # Backfill specific session
  python3 auto_capture.py watch                   # Watch for changes (daemon mode)
  python3 auto_capture.py extract-urls <file>    # Extract URLs from any file
"""

import argparse
import json
import os
import re
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


# Paths
CLAUDE_DIR = Path.home() / ".claude"
CLAUDE_PROJECTS_DIR = CLAUDE_DIR / "projects"
CLAUDE_HISTORY = CLAUDE_DIR / "history.jsonl"
AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
AUTO_CAPTURE_LOG = AGENT_CORE_DIR / "auto_capture_log.json"


def get_capture_log() -> Dict[str, Any]:
    """Load or initialize capture log."""
    if AUTO_CAPTURE_LOG.exists():
        return json.loads(AUTO_CAPTURE_LOG.read_text())
    return {
        "last_scan": None,
        "captured_files": {},
        "backfilled_sessions": []
    }


def save_capture_log(log: Dict[str, Any]):
    """Save capture log."""
    AUTO_CAPTURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    AUTO_CAPTURE_LOG.write_text(json.dumps(log, indent=2))


def find_all_claude_sessions() -> List[Path]:
    """Find all Claude Code session files."""
    sessions = []

    # Main history file
    if CLAUDE_HISTORY.exists():
        sessions.append(CLAUDE_HISTORY)

    # Project session files
    if CLAUDE_PROJECTS_DIR.exists():
        for proj_dir in CLAUDE_PROJECTS_DIR.iterdir():
            if proj_dir.is_dir():
                for jsonl in proj_dir.glob("*.jsonl"):
                    sessions.append(jsonl)
                # Also check subagents
                subagents_dir = proj_dir / "subagents"
                if subagents_dir.exists():
                    for jsonl in subagents_dir.glob("*.jsonl"):
                        sessions.append(jsonl)

    return sorted(sessions, key=lambda f: f.stat().st_mtime, reverse=True)


def extract_text_from_jsonl(file_path: Path) -> str:
    """Extract all text content from a JSONL file."""
    text_parts = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    # Extract from various message formats
                    text_parts.extend(extract_text_from_entry(entry))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return '\n'.join(text_parts)


def extract_text_from_entry(entry: Dict) -> List[str]:
    """Extract text from a JSONL entry."""
    texts = []

    # Direct content
    if "content" in entry:
        content = entry["content"]
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        texts.append(item["text"])
                    if "content" in item:
                        texts.append(str(item["content"]))

    # Message content
    if "message" in entry:
        texts.extend(extract_text_from_entry(entry["message"]))

    # Tool results
    if "result" in entry:
        result = entry["result"]
        if isinstance(result, str):
            texts.append(result)
        elif isinstance(result, dict) and "content" in result:
            texts.append(str(result["content"]))

    return texts


def extract_urls(text: str) -> List[Dict[str, Any]]:
    """Extract URLs with metadata from text."""
    url_pattern = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\-._~:/?#\[\]@!$&\'()*+,;=]*'
    )

    urls = []
    seen = set()

    for match in url_pattern.finditer(text):
        url = match.group(0).rstrip('.,;:)"\']')

        if url in seen:
            continue
        seen.add(url)

        # Get context (100 chars before and after)
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end].replace('\n', ' ')

        # Classify URL
        classification = classify_url(url)

        urls.append({
            "url": url,
            "context": context,
            **classification
        })

    return urls


def classify_url(url: str) -> Dict[str, Any]:
    """Classify URL by tier, category, and source."""
    url_lower = url.lower()

    classifications = [
        # Tier 1: Research
        (["arxiv.org"], {"tier": 1, "category": "research", "source": "arXiv"}),
        (["huggingface.co/papers"], {"tier": 1, "category": "research", "source": "HuggingFace"}),
        (["openreview.net"], {"tier": 1, "category": "research", "source": "OpenReview"}),
        (["semanticscholar.org"], {"tier": 1, "category": "research", "source": "Semantic Scholar"}),
        (["aclanthology.org"], {"tier": 1, "category": "research", "source": "ACL Anthology"}),

        # Tier 1: Labs
        (["openai.com"], {"tier": 1, "category": "labs", "source": "OpenAI"}),
        (["anthropic.com"], {"tier": 1, "category": "labs", "source": "Anthropic"}),
        (["deepmind.google", "blog.google/technology/ai"], {"tier": 1, "category": "labs", "source": "Google AI"}),
        (["ai.meta.com"], {"tier": 1, "category": "labs", "source": "Meta AI"}),

        # Tier 1: Industry
        (["techcrunch.com"], {"tier": 1, "category": "industry", "source": "TechCrunch"}),
        (["theverge.com"], {"tier": 1, "category": "industry", "source": "The Verge"}),
        (["arstechnica.com"], {"tier": 1, "category": "industry", "source": "Ars Technica"}),
        (["wired.com"], {"tier": 1, "category": "industry", "source": "Wired"}),

        # Tier 2: GitHub
        (["github.com"], {"tier": 2, "category": "github", "source": "GitHub"}),

        # Tier 2: Benchmarks
        (["metr.org"], {"tier": 2, "category": "benchmarks", "source": "METR"}),
        (["arcprize.org"], {"tier": 2, "category": "benchmarks", "source": "ARC Prize"}),
        (["paperswithcode.com"], {"tier": 2, "category": "benchmarks", "source": "PapersWithCode"}),
        (["lmarena.ai", "lmsys.org"], {"tier": 2, "category": "benchmarks", "source": "LMSYS"}),

        # Tier 2: Social
        (["twitter.com", "x.com"], {"tier": 2, "category": "social", "source": "X/Twitter"}),
        (["news.ycombinator.com"], {"tier": 2, "category": "social", "source": "Hacker News"}),
        (["reddit.com"], {"tier": 2, "category": "social", "source": "Reddit"}),

        # Tier 3: Newsletters/Forums
        (["substack.com"], {"tier": 3, "category": "newsletters", "source": "Substack"}),
        (["lesswrong.com"], {"tier": 3, "category": "forums", "source": "LessWrong"}),
        (["alignmentforum.org"], {"tier": 3, "category": "forums", "source": "Alignment Forum"}),
    ]

    for patterns, classification in classifications:
        if any(p in url_lower for p in patterns):
            return classification

    return {"tier": 3, "category": "other", "source": "Web"}


def extract_research_topics(text: str) -> List[str]:
    """Extract potential research topics from text."""
    topics = []

    # Common research topic patterns
    patterns = [
        r"research(?:ing)?\s+(?:on\s+)?['\"]?([^'\".,\n]{10,100})['\"]?",
        r"investigating\s+([^.,\n]{10,100})",
        r"looking\s+(?:into|at)\s+([^.,\n]{10,100})",
        r"topic[:\s]+([^.,\n]{10,100})",
        r"session[:\s]+([^.,\n]{10,100})",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        topics.extend([m.strip() for m in matches if len(m.strip()) > 10])

    return list(set(topics))[:5]  # Top 5 unique topics


def extract_key_findings(text: str) -> List[Dict[str, str]]:
    """Extract key findings from text."""
    findings = []

    # Patterns indicating findings
    finding_patterns = [
        (r"(?:key|main|important)\s+(?:finding|insight|takeaway)[:\s]+([^\n]{20,500})", "finding"),
        (r"thesis[:\s]+([^\n]{20,500})", "thesis"),
        (r"gap[:\s]+([^\n]{20,500})", "gap"),
        (r"innovation\s+(?:opportunity|direction)[:\s]+([^\n]{20,500})", "innovation"),
        (r"decision\s+quality[:\s]+([^\n]{20,500})", "finding"),
        (r"(?:we\s+found|discovered|identified)[:\s]+([^\n]{20,500})", "finding"),
    ]

    for pattern, finding_type in finding_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            findings.append({
                "type": finding_type,
                "text": match.strip()[:500],
                "extracted_at": datetime.now().isoformat()
            })

    return findings


def generate_backfill_session_id(source_file: Path, topic: str) -> str:
    """Generate session ID for backfilled session."""
    # Use file modification time as base
    mtime = datetime.fromtimestamp(source_file.stat().st_mtime)
    timestamp = mtime.strftime("%Y%m%d-%H%M%S")
    topic_hash = hashlib.md5(topic.encode()).hexdigest()[:6]
    safe_topic = re.sub(r'[^a-z0-9]+', '-', topic.lower())[:20]
    return f"backfill-{safe_topic}-{timestamp}-{topic_hash}"


def backfill_session(source_file: Path, topic: Optional[str] = None) -> Dict[str, Any]:
    """Backfill a historical session from Claude Code files."""
    print(f"Backfilling from: {source_file}")

    # Extract text
    text = extract_text_from_jsonl(source_file)
    if not text:
        print("  No text content found")
        return {}

    # Extract artifacts
    urls = extract_urls(text)
    findings = extract_key_findings(text)
    topics = extract_research_topics(text)

    # Determine topic
    if not topic:
        topic = topics[0] if topics else f"Session from {source_file.name}"

    session_id = generate_backfill_session_id(source_file, topic)

    # Create session
    session = {
        "session_id": session_id,
        "topic": topic,
        "source_file": str(source_file),
        "backfilled": True,
        "backfilled_at": datetime.now().isoformat(),
        "original_date": datetime.fromtimestamp(source_file.stat().st_mtime).isoformat(),
        "status": "archived",
        "urls_captured": urls,
        "findings_captured": findings,
        "detected_topics": topics,
        "stats": {
            "urls_count": len(urls),
            "findings_count": len(findings),
            "text_length": len(text)
        }
    }

    # Save session
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    (session_dir / "session.json").write_text(json.dumps(session, indent=2))
    (session_dir / "urls_captured.json").write_text(json.dumps(urls, indent=2))
    (session_dir / "findings_captured.json").write_text(json.dumps(findings, indent=2))
    (session_dir / "full_transcript.txt").write_text(text)

    # Update capture log
    log = get_capture_log()
    log["captured_files"][str(source_file)] = {
        "session_id": session_id,
        "captured_at": datetime.now().isoformat()
    }
    log["backfilled_sessions"].append(session_id)
    save_capture_log(log)

    print(f"  Session ID: {session_id}")
    print(f"  Topic: {topic}")
    print(f"  URLs: {len(urls)}")
    print(f"  Findings: {len(findings)}")
    print(f"  Saved to: {session_dir}")

    return session


def scan_recent_sessions(hours: int = 24) -> List[Dict[str, Any]]:
    """Scan recent Claude sessions and capture research artifacts."""
    print(f"Scanning sessions from last {hours} hours...")

    cutoff = datetime.now() - timedelta(hours=hours)
    sessions = find_all_claude_sessions()
    log = get_capture_log()

    captured = []
    for session_file in sessions:
        # Check if already captured
        if str(session_file) in log["captured_files"]:
            continue

        # Check if recent enough
        mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
        if mtime < cutoff:
            continue

        # Quick check for research content
        text = extract_text_from_jsonl(session_file)
        urls = extract_urls(text)

        # Only backfill if there are research URLs (Tier 1 or significant Tier 2)
        research_urls = [u for u in urls if u["tier"] == 1 or
                        (u["tier"] == 2 and u["category"] in ["github", "benchmarks"])]

        if len(research_urls) >= 3:  # Threshold: at least 3 research URLs
            print(f"\nFound research session: {session_file.name}")
            session = backfill_session(session_file)
            captured.append(session)

    print(f"\nScan complete. Captured {len(captured)} new sessions.")
    return captured


def watch_mode(interval: int = 300):
    """Watch for new sessions and capture automatically."""
    print(f"Watching for new sessions (interval: {interval}s)")
    print("Press Ctrl+C to stop")

    while True:
        try:
            scan_recent_sessions(hours=1)
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopping watch mode")
            break


def main():
    parser = argparse.ArgumentParser(
        description="ResearchGravity Auto-Capture - Automatic research extraction"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Scan
    scan_parser = subparsers.add_parser("scan", help="Scan recent sessions")
    scan_parser.add_argument("--hours", type=int, default=24, help="Hours to scan back")

    # Backfill
    backfill_parser = subparsers.add_parser("backfill", help="Backfill specific session")
    backfill_parser.add_argument("file", help="Session file path")
    backfill_parser.add_argument("--topic", help="Override topic")

    # Watch
    watch_parser = subparsers.add_parser("watch", help="Watch for new sessions")
    watch_parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")

    # Extract URLs
    extract_parser = subparsers.add_parser("extract-urls", help="Extract URLs from file")
    extract_parser.add_argument("file", help="File to extract from")

    args = parser.parse_args()

    if args.command == "scan":
        scan_recent_sessions(args.hours)
    elif args.command == "backfill":
        backfill_session(Path(args.file), args.topic)
    elif args.command == "watch":
        watch_mode(args.interval)
    elif args.command == "extract-urls":
        text = Path(args.file).read_text()
        urls = extract_urls(text)
        for u in urls:
            print(f"[T{u['tier']}] {u['source']}: {u['url']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
