#!/usr/bin/env python3
"""
Initialize a new Metaventions AI research session.
Multi-source, multi-tier signal capture for meta-invention intelligence.

v2.0: Now with automatic session tracking and cross-project lineage.

Usage:
  python3 init_session.py <topic> [--workflow TYPE] [--env ENV] [--continue SESSION_ID]
  python3 init_session.py <topic> --impl-project OS-App  # Pre-link to implementation
"""

import argparse
import json
import os
import hashlib
import subprocess
from datetime import datetime, timedelta
from pathlib import Path


def get_agent_core_dir() -> Path:
    """Get the global agent-core directory."""
    return Path.home() / ".agent-core"


def register_with_tracker(session_id: str, topic: str, impl_project: str = None):
    """Register session with the auto-capture tracker for lineage tracking."""
    tracker_file = get_agent_core_dir() / "session_tracker.json"

    # Load or create tracker state
    if tracker_file.exists():
        state = json.loads(tracker_file.read_text())
    else:
        state = {
            "version": "2.0",
            "active_session": None,
            "sessions": {},
            "lineage": [],
            "pending_captures": []
        }

    # Find Claude session file for linking
    claude_session = find_claude_session_file()

    # Register session
    state["active_session"] = session_id
    state["sessions"][session_id] = {
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

    # Create lineage entry if impl_project specified
    if impl_project:
        state["lineage"].append({
            "research_session": session_id,
            "impl_project": impl_project,
            "impl_session": None,
            "linked_at": datetime.now().isoformat(),
            "status": "pending"
        })

    # Save tracker state
    tracker_file.parent.mkdir(parents=True, exist_ok=True)
    tracker_file.write_text(json.dumps(state, indent=2))

    return claude_session


def find_claude_session_file() -> Path:
    """Find the most recent Claude Code session file."""
    claude_projects = Path.home() / ".claude" / "projects"
    if not claude_projects.exists():
        return None

    # Find most recent .jsonl file across all projects
    latest = None
    latest_mtime = 0

    for proj_dir in claude_projects.iterdir():
        if proj_dir.is_dir():
            for jsonl in proj_dir.glob("*.jsonl"):
                if jsonl.stat().st_mtime > latest_mtime:
                    latest = jsonl
                    latest_mtime = jsonl.stat().st_mtime

    return latest


def get_local_agent_dir() -> Path:
    """Get the local .agent directory."""
    return Path.cwd() / ".agent"


def detect_environment() -> str:
    """Detect if running in CLI or Antigravity."""
    if os.environ.get("ANTIGRAVITY_SESSION"):
        return "antigravity"
    return "cli"


def generate_session_id(topic: str) -> str:
    """Generate unique session ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    topic_hash = hashlib.md5(topic.encode()).hexdigest()[:6]
    safe_topic = topic.lower().replace(" ", "-")[:20]
    return f"{safe_topic}-{timestamp}-{topic_hash}"


def create_search_queries(topic: str) -> dict:
    """Generate multi-tier search queries for Metaventions-grade research."""
    today = datetime.now()
    year = today.strftime("%Y")
    month = today.strftime("%B")
    viral_cutoff = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    groundbreaker_cutoff = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    recent_cutoff = (today - timedelta(days=2)).strftime("%Y-%m-%d")

    return {
        # Tier 1: Primary Sources
        "tier1": {
            "research": {
                "arxiv_ai": f"site:arxiv.org cs.AI {topic} {year}",
                "arxiv_lg": f"site:arxiv.org cs.LG {topic} {year}",
                "arxiv_se": f"site:arxiv.org cs.SE {topic} {year}",
                "huggingface": f"site:huggingface.co/papers {topic}",
                "description": "Academic research papers"
            },
            "labs": {
                "openai": f"site:openai.com {topic} {month} {year}",
                "anthropic": f"site:anthropic.com {topic} {month} {year}",
                "google_ai": f"site:blog.google/technology/ai {topic} {year}",
                "meta_ai": f"site:ai.meta.com {topic} {year}",
                "deepmind": f"site:deepmind.google {topic} {year}",
                "description": "AI lab announcements"
            },
            "industry": {
                "techcrunch": f"site:techcrunch.com {topic} {month} {year}",
                "verge": f"site:theverge.com {topic} {month} {year}",
                "ars": f"site:arstechnica.com {topic} {month} {year}",
                "description": "Tech industry news"
            }
        },
        # Tier 2: Signal Amplifiers
        "tier2": {
            "github": {
                "viral": f"{topic} stars:>500 pushed:>{viral_cutoff}",
                "groundbreaker": f"{topic} stars:10..200 created:>{groundbreaker_cutoff}",
                "trending": f"{topic} stars:>100 pushed:>{recent_cutoff}",
                "description": "GitHub repositories"
            },
            "benchmarks": {
                "metr": f"site:metr.org {topic}",
                "arcprize": f"site:arcprize.org {topic}",
                "paperswithcode": f"site:paperswithcode.com {topic} {year}",
                "description": "Benchmark and leaderboard updates"
            },
            "social": {
                "hackernews": f"site:news.ycombinator.com {topic}",
                "twitter_search": f"{topic} (from:karpathy OR from:ylecun OR from:sama)",
                "description": "Social signals from key figures"
            }
        },
        # Tier 3: Deep Context
        "tier3": {
            "newsletters": {
                "import_ai": "https://importai.substack.com/",
                "the_batch": "https://www.deeplearning.ai/the-batch/",
                "latent_space": "https://www.latent.space/",
                "description": "Curated newsletters (benchmark calibration)"
            },
            "forums": {
                "lesswrong": f"site:lesswrong.com {topic} {year}",
                "alignmentforum": f"site:alignmentforum.org {topic}",
                "description": "Frontier discourse"
            }
        },
        # Frontier Filter (last 48 hours)
        "frontier": {
            "breaking": f"{topic} {month} {today.day} {year}",
            "protocols": f'"protocol" OR "standard" OR "specification" {topic} {year}',
            "description": "Bleeding edge signals"
        }
    }


def get_scan_urls() -> dict:
    """Get direct URLs for daily scanning."""
    return {
        "daily_scan": [
            "https://arxiv.org/list/cs.AI/new",
            "https://arxiv.org/list/cs.LG/new",
            "https://arxiv.org/list/cs.SE/new",
            "https://huggingface.co/papers/trending",
            "https://news.ycombinator.com/",
            "https://github.com/trending"
        ],
        "lab_blogs": [
            "https://openai.com/news/",
            "https://www.anthropic.com/news",
            "https://blog.google/technology/ai/",
            "https://ai.meta.com/blog/",
            "https://deepmind.google/discover/blog/"
        ],
        "industry_news": [
            "https://techcrunch.com/category/artificial-intelligence/",
            "https://arstechnica.com/ai/",
            "https://www.theverge.com/ai-artificial-intelligence"
        ],
        "benchmarks": [
            "https://metr.org/blog/",
            "https://arcprize.org/blog",
            "https://lmarena.ai/",
            "https://paperswithcode.com/sota"
        ]
    }


def init_session(
    topic: str,
    workflow: str = "research",
    env: str = None,
    continue_session: str = None,
    impl_project: str = None
) -> dict:
    """Initialize a new research session with Metaventions-grade structure."""

    env = env or detect_environment()

    # Handle continuing existing session
    if continue_session:
        return load_session(continue_session)

    session_id = generate_session_id(topic)

    # AUTO-REGISTER with tracker for lineage tracking
    claude_session = register_with_tracker(session_id, topic, impl_project)
    timestamp = datetime.now()

    # Create directories
    local_dir = get_local_agent_dir() / "research"
    global_dir = get_agent_core_dir() / "sessions" / session_id

    local_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    # Session metadata
    session = {
        "session_id": session_id,
        "topic": topic,
        "workflow": workflow,
        "environment": env,
        "started": timestamp.isoformat(),
        "status": "active",
        "queries": create_search_queries(topic),
        "scan_urls": get_scan_urls(),
        "paths": {
            "local": str(local_dir),
            "global": str(global_dir),
            "session_log": str(local_dir / "session_log.md"),
            "scratchpad": str(local_dir / "scratchpad.json"),
            "report": str(local_dir / f"{topic.lower().replace(' ', '-')}_report.md"),
            "sources": str(local_dir / f"{topic.lower().replace(' ', '-')}_sources.csv")
        },
        "stats": {
            "urls_visited": 0,
            "urls_used": 0,
            "checkpoints": 0,
            "last_sync": None
        },
        "quality_standard": "metaventions",
        "impl_project": impl_project,
        "claude_session_linked": str(claude_session) if claude_session else None,
        "auto_tracked": True
    }

    # Create session log
    create_session_log(session)

    # Create scratchpad
    create_scratchpad(session)

    # Create sources CSV header (updated schema)
    sources_path = Path(session["paths"]["sources"])
    sources_path.write_text("name,url,tier,category,signal,relevance,used,notes,timestamp\n")

    # Save session metadata
    metadata_path = local_dir / "session.json"
    metadata_path.write_text(json.dumps(session, indent=2))

    # Also save to global
    global_metadata = global_dir / "session.json"
    global_metadata.write_text(json.dumps(session, indent=2))

    return session


def create_session_log(session: dict):
    """Create the session log markdown file with Metaventions structure."""
    queries = session['queries']
    scan_urls = session['scan_urls']

    content = f"""# Research Session: {session['topic']}

**Session ID:** `{session['session_id']}`
**Workflow:** {session['workflow']}
**Environment:** {session['environment']}
**Started:** {session['started']}
**Quality Standard:** Metaventions-grade

---

## Phase 1: Signal Capture

### Tier 1 Sources (Check First)

#### Research Papers
```
{queries['tier1']['research']['arxiv_ai']}
{queries['tier1']['research']['arxiv_lg']}
{queries['tier1']['research']['huggingface']}
```

#### Lab Blogs
```
{queries['tier1']['labs']['openai']}
{queries['tier1']['labs']['anthropic']}
{queries['tier1']['labs']['google_ai']}
```

#### Industry News
```
{queries['tier1']['industry']['techcrunch']}
{queries['tier1']['industry']['verge']}
```

### Tier 2 Sources (Signal Amplifiers)

#### GitHub
```
Viral:        {queries['tier2']['github']['viral']}
Groundbreaker: {queries['tier2']['github']['groundbreaker']}
Trending:     {queries['tier2']['github']['trending']}
```

#### Benchmarks
```
{queries['tier2']['benchmarks']['metr']}
{queries['tier2']['benchmarks']['paperswithcode']}
```

### Frontier Filter (Last 48h)
```
{queries['frontier']['breaking']}
{queries['frontier']['protocols']}
```

---

## Direct Scan URLs

### Daily Scan
{chr(10).join(f'- {url}' for url in scan_urls['daily_scan'])}

### Lab Blogs
{chr(10).join(f'- {url}' for url in scan_urls['lab_blogs'])}

### Industry News
{chr(10).join(f'- {url}' for url in scan_urls['industry_news'])}

### Benchmarks
{chr(10).join(f'- {url}' for url in scan_urls['benchmarks'])}

---

## URLs Visited

| Time | Tier | Category | URL | Signal | Relevance | Used | Notes |
|------|------|----------|-----|--------|-----------|------|-------|

---

## Phase 2: Synthesis

### Thesis Statement
_What is the pattern across findings?_



### Gap Identified
_What's missing that represents opportunity?_



### Innovation Direction
_Concrete next step_



---

## Key Findings

| # | Finding | Source | Signal | Category |
|---|---------|--------|--------|----------|

---

## Quality Checklist

- [ ] Scanned all Tier 1 sources for timeframe
- [ ] Logged 10+ URLs minimum
- [ ] Identified at least one GAP
- [ ] Wrote thesis statement
- [ ] Each finding has: link + signal + rationale
- [ ] Innovation direction is concrete, not vague

---

## Checkpoints

| Time | URLs Logged | Findings | Notes |
|------|-------------|----------|-------|

"""
    Path(session["paths"]["session_log"]).write_text(content)


def create_scratchpad(session: dict):
    """Create the scratchpad JSON file with multi-tier structure."""
    scratchpad = {
        "session_id": session["session_id"],
        "topic": session["topic"],
        "workflow": session["workflow"],
        "environment": session["environment"],
        "quality_standard": "metaventions",

        # Tier 1 findings
        "tier1": {
            "research": [],
            "labs": [],
            "industry": []
        },

        # Tier 2 findings
        "tier2": {
            "github": {
                "viral": [],
                "groundbreaker": [],
                "trending": []
            },
            "benchmarks": [],
            "social": []
        },

        # Tier 3 context
        "tier3": {
            "newsletters": [],
            "forums": []
        },

        # Frontier signals
        "frontier": [],

        # All URLs visited
        "urls_visited": [],

        # Synthesis
        "synthesis": {
            "thesis": None,
            "gap": None,
            "innovation_direction": None
        },

        # Key findings (top signals)
        "findings": [],

        # Checkpoints
        "checkpoints": [],

        "last_updated": session["started"]
    }
    Path(session["paths"]["scratchpad"]).write_text(json.dumps(scratchpad, indent=2))


def load_session(session_id: str) -> dict:
    """Load an existing session to continue."""
    global_dir = get_agent_core_dir() / "sessions" / session_id
    metadata_path = global_dir / "session.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    session = json.loads(metadata_path.read_text())
    session["status"] = "resumed"

    # Restore to local directory
    local_dir = get_local_agent_dir() / "research"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Copy files from global to local
    for file in global_dir.iterdir():
        if file.is_file():
            dest = local_dir / file.name
            dest.write_text(file.read_text())

    session["paths"]["local"] = str(local_dir)
    return session


def main():
    parser = argparse.ArgumentParser(
        description="Initialize Metaventions-grade research session"
    )
    parser.add_argument("topic", nargs="?", help="Research topic")
    parser.add_argument("--workflow", default="deep-research",
                        choices=["research", "innovation-scout", "deep-research"],
                        help="Workflow type (default: deep-research)")
    parser.add_argument("--env", choices=["cli", "antigravity"],
                        help="Override environment detection")
    parser.add_argument("--continue", dest="continue_session",
                        help="Continue existing session by ID")
    parser.add_argument("--impl-project", dest="impl_project",
                        help="Target implementation project (creates lineage link)")

    args = parser.parse_args()

    if not args.topic and not args.continue_session:
        parser.error("Either topic or --continue SESSION_ID is required")

    try:
        session = init_session(
            topic=args.topic or "",
            workflow=args.workflow,
            env=args.env,
            continue_session=args.continue_session,
            impl_project=args.impl_project
        )

        print(f"Session initialized: {session['session_id']}")
        print(f"   Topic: {session['topic']}")
        print(f"   Workflow: {session['workflow']}")
        print(f"   Quality: Metaventions-grade")
        print(f"   Local: {session['paths']['local']}")
        print()
        print("AUTO-TRACKING ENABLED")
        print(f"   Claude session linked: {session.get('claude_session_linked', 'None')[:50] if session.get('claude_session_linked') else 'Detecting...'}")
        if session.get('impl_project'):
            print(f"   Implementation target: {session['impl_project']}")
        print("   Full transcript will be captured automatically")
        print()
        print("Files created:")
        print("   - session_log.md (with all query templates)")
        print("   - scratchpad.json (multi-tier structure)")
        print("   - sources.csv")
        print()
        print("Quick scan URLs:")
        for url in session['scan_urls']['daily_scan'][:3]:
            print(f"   - {url}")
        print()
        print("Workflow:")
        print("   1. Scan Tier 1 sources (30 min)")
        print("   2. URLs auto-captured from transcript")
        print("   3. Synthesize: thesis + gap + direction")
        print("   4. Archive auto-captures full session")

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
