#!/usr/bin/env python3
"""
ResearchGravity Auto-Capture V2
===============================

Automatically captures research sessions from Claude Code with:
1. Real-time URL extraction from active sessions
2. Intelligent finding detection
3. Session deduplication
4. Direct storage engine integration
5. Hook-based triggers

Target: +70% URL capture rate (from ~40% manual to 90%+ automatic)

Usage:
  python3 auto_capture_v2.py scan                      # Scan recent sessions
  python3 auto_capture_v2.py watch                     # Watch mode (daemon)
  python3 auto_capture_v2.py capture --session ID      # Capture specific session
  python3 auto_capture_v2.py status                    # Show capture stats
  python3 auto_capture_v2.py sync                      # Sync to storage engine
"""

import argparse
import asyncio
import json
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# Pre-compiled Regex Patterns (Performance Optimization)
# =============================================================================

# URL extraction pattern (RFC 3986 compliant)
URL_PATTERN = re.compile(
    r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\-._~:/?#\[\]@!$&\'()*+,;=%]*'
)

# Finding extraction patterns (compiled at module level)
FINDING_PATTERNS = [
    # Thesis patterns
    (re.compile(r"thesis[:\s]+([^\n]{30,500})", re.IGNORECASE), "thesis", 0.9),
    (re.compile(r"main\s+(?:argument|claim|point)[:\s]+([^\n]{30,500})", re.IGNORECASE), "thesis", 0.85),
    (re.compile(r"core\s+insight[:\s]+([^\n]{30,500})", re.IGNORECASE), "thesis", 0.85),

    # Gap patterns
    (re.compile(r"gap[:\s]+([^\n]{30,500})", re.IGNORECASE), "gap", 0.9),
    (re.compile(r"(?:missing|lacking|needs)[:\s]+([^\n]{30,500})", re.IGNORECASE), "gap", 0.75),
    (re.compile(r"opportunity\s+(?:for|to)[:\s]+([^\n]{30,500})", re.IGNORECASE), "gap", 0.8),

    # Innovation patterns
    (re.compile(r"innovation\s+(?:opportunity|direction)[:\s]+([^\n]{30,500})", re.IGNORECASE), "innovation", 0.9),
    (re.compile(r"novel\s+approach[:\s]+([^\n]{30,500})", re.IGNORECASE), "innovation", 0.85),
    (re.compile(r"new\s+method[:\s]+([^\n]{30,500})", re.IGNORECASE), "innovation", 0.8),

    # General findings
    (re.compile(r"key\s+(?:finding|insight|takeaway)[:\s]+([^\n]{30,500})", re.IGNORECASE), "finding", 0.9),
    (re.compile(r"important(?:ly)?[:\s]+([^\n]{30,500})", re.IGNORECASE), "finding", 0.75),
    (re.compile(r"(?:we\s+)?(?:found|discovered|identified)[:\s]+([^\n]{30,500})", re.IGNORECASE), "finding", 0.8),
    (re.compile(r"conclusion[:\s]+([^\n]{30,500})", re.IGNORECASE), "finding", 0.85),
    (re.compile(r"summary[:\s]+([^\n]{30,500})", re.IGNORECASE), "finding", 0.7),

    # Decision quality patterns
    (re.compile(r"DQ\s+(?:score|metric)[:\s]+([^\n]{30,300})", re.IGNORECASE), "finding", 0.95),
    (re.compile(r"decision\s+quality[:\s]+([^\n]{30,300})", re.IGNORECASE), "finding", 0.9),
]

# Topic detection patterns (compiled at module level)
TOPIC_PATTERNS = [
    re.compile(r"research(?:ing)?\s+(?:on\s+)?['\"]?([^'\".\n]{10,80})['\"]?", re.IGNORECASE),
    re.compile(r"topic[:\s]+['\"]?([^'\".\n]{10,80})['\"]?", re.IGNORECASE),
    re.compile(r"session[:\s]+['\"]?([^'\".\n]{10,80})['\"]?", re.IGNORECASE),
    re.compile(r"investigating\s+([^.\n]{10,80})", re.IGNORECASE),
    re.compile(r"exploring\s+([^.\n]{10,80})", re.IGNORECASE),
    re.compile(r"looking\s+(?:into|at)\s+([^.\n]{10,80})", re.IGNORECASE),
]

# URLs to skip (internal/irrelevant)
SKIP_URL_PATTERNS = frozenset([
    "localhost", "127.0.0.1", ".local",
    "chrome://", "file://", "blob:",
    "placeholder", "example.com"
])


# Paths
CLAUDE_DIR = Path.home() / ".claude"
CLAUDE_PROJECTS_DIR = CLAUDE_DIR / "projects"
CLAUDE_HISTORY = CLAUDE_DIR / "history.jsonl"
AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
CAPTURE_STATE_FILE = AGENT_CORE_DIR / "auto_capture_v2_state.json"


class SourceTier(Enum):
    """Source quality tiers."""
    TIER_1 = 1  # Primary: Research papers, labs, industry leaders
    TIER_2 = 2  # Amplifiers: GitHub, benchmarks, social signals
    TIER_3 = 3  # Context: Newsletters, forums, general web


@dataclass
class CapturedURL:
    """A captured URL with metadata."""
    url: str
    tier: int
    category: str
    source: str
    context: str
    captured_at: str
    session_file: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapturedFinding:
    """A captured finding/insight."""
    text: str
    finding_type: str  # thesis, gap, innovation, finding
    confidence: float
    context: str
    captured_at: str
    session_file: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaptureState:
    """Persistent capture state."""
    last_scan: Optional[str] = None
    captured_files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    url_hashes: Set[str] = field(default_factory=set)
    total_urls_captured: int = 0
    total_findings_captured: int = 0
    sessions_created: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_scan": self.last_scan,
            "captured_files": self.captured_files,
            "url_hashes": list(self.url_hashes),
            "total_urls_captured": self.total_urls_captured,
            "total_findings_captured": self.total_findings_captured,
            "sessions_created": self.sessions_created,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptureState":
        return cls(
            last_scan=data.get("last_scan"),
            captured_files=data.get("captured_files", {}),
            url_hashes=set(data.get("url_hashes", [])),
            total_urls_captured=data.get("total_urls_captured", 0),
            total_findings_captured=data.get("total_findings_captured", 0),
            sessions_created=data.get("sessions_created", 0),
        )


# URL classification rules (30+ sources)
URL_CLASSIFICATIONS: List[Tuple[List[str], Dict[str, Any]]] = [
    # Tier 1: Research
    (["arxiv.org"], {"tier": 1, "category": "research", "source": "arXiv"}),
    (["huggingface.co/papers"], {"tier": 1, "category": "research", "source": "HuggingFace Papers"}),
    (["openreview.net"], {"tier": 1, "category": "research", "source": "OpenReview"}),
    (["semanticscholar.org"], {"tier": 1, "category": "research", "source": "Semantic Scholar"}),
    (["aclanthology.org"], {"tier": 1, "category": "research", "source": "ACL Anthology"}),
    (["proceedings.neurips.cc"], {"tier": 1, "category": "research", "source": "NeurIPS"}),
    (["proceedings.mlr.press"], {"tier": 1, "category": "research", "source": "PMLR"}),
    (["jmlr.org"], {"tier": 1, "category": "research", "source": "JMLR"}),
    (["nature.com"], {"tier": 1, "category": "research", "source": "Nature"}),
    (["science.org"], {"tier": 1, "category": "research", "source": "Science"}),

    # Tier 1: Labs
    (["openai.com"], {"tier": 1, "category": "labs", "source": "OpenAI"}),
    (["anthropic.com"], {"tier": 1, "category": "labs", "source": "Anthropic"}),
    (["deepmind.google", "blog.google/technology/ai"], {"tier": 1, "category": "labs", "source": "Google AI"}),
    (["ai.meta.com", "research.facebook.com"], {"tier": 1, "category": "labs", "source": "Meta AI"}),
    (["microsoft.com/research"], {"tier": 1, "category": "labs", "source": "Microsoft Research"}),
    (["ai.google", "research.google"], {"tier": 1, "category": "labs", "source": "Google Research"}),
    (["nvidia.com/research"], {"tier": 1, "category": "labs", "source": "NVIDIA Research"}),

    # Tier 1: Industry
    (["techcrunch.com"], {"tier": 1, "category": "industry", "source": "TechCrunch"}),
    (["theverge.com"], {"tier": 1, "category": "industry", "source": "The Verge"}),
    (["arstechnica.com"], {"tier": 1, "category": "industry", "source": "Ars Technica"}),
    (["wired.com"], {"tier": 1, "category": "industry", "source": "Wired"}),
    (["mit.edu"], {"tier": 1, "category": "industry", "source": "MIT"}),
    (["stanford.edu"], {"tier": 1, "category": "industry", "source": "Stanford"}),
    (["berkeley.edu"], {"tier": 1, "category": "industry", "source": "Berkeley"}),

    # Tier 2: GitHub
    (["github.com"], {"tier": 2, "category": "github", "source": "GitHub"}),
    (["gist.github.com"], {"tier": 2, "category": "github", "source": "GitHub Gist"}),

    # Tier 2: Benchmarks
    (["metr.org"], {"tier": 2, "category": "benchmarks", "source": "METR"}),
    (["arcprize.org"], {"tier": 2, "category": "benchmarks", "source": "ARC Prize"}),
    (["paperswithcode.com"], {"tier": 2, "category": "benchmarks", "source": "Papers With Code"}),
    (["lmarena.ai", "lmsys.org", "chat.lmsys.org"], {"tier": 2, "category": "benchmarks", "source": "LMSYS Arena"}),
    (["huggingface.co/spaces"], {"tier": 2, "category": "benchmarks", "source": "HF Spaces"}),

    # Tier 2: Social
    (["twitter.com", "x.com"], {"tier": 2, "category": "social", "source": "X/Twitter"}),
    (["news.ycombinator.com"], {"tier": 2, "category": "social", "source": "Hacker News"}),
    (["reddit.com/r/MachineLearning", "reddit.com/r/LocalLLaMA"], {"tier": 2, "category": "social", "source": "Reddit ML"}),
    (["linkedin.com"], {"tier": 2, "category": "social", "source": "LinkedIn"}),
    (["youtube.com", "youtu.be"], {"tier": 2, "category": "social", "source": "YouTube"}),

    # Tier 3: Newsletters
    (["substack.com"], {"tier": 3, "category": "newsletters", "source": "Substack"}),
    (["deeplearning.ai"], {"tier": 3, "category": "newsletters", "source": "The Batch"}),
    (["importai.net"], {"tier": 3, "category": "newsletters", "source": "Import AI"}),

    # Tier 3: Forums
    (["lesswrong.com"], {"tier": 3, "category": "forums", "source": "LessWrong"}),
    (["alignmentforum.org"], {"tier": 3, "category": "forums", "source": "Alignment Forum"}),
    (["eaforum.org"], {"tier": 3, "category": "forums", "source": "EA Forum"}),
    (["stackoverflow.com"], {"tier": 3, "category": "forums", "source": "StackOverflow"}),

    # Tier 3: Documentation
    (["docs.python.org"], {"tier": 3, "category": "docs", "source": "Python Docs"}),
    (["pytorch.org/docs"], {"tier": 3, "category": "docs", "source": "PyTorch Docs"}),
    (["tensorflow.org/api_docs"], {"tier": 3, "category": "docs", "source": "TensorFlow Docs"}),
]


def classify_url(url: str) -> Dict[str, Any]:
    """Classify a URL by tier, category, and source."""
    url_lower = url.lower()

    for patterns, classification in URL_CLASSIFICATIONS:
        if any(p in url_lower for p in patterns):
            return classification.copy()

    return {"tier": 3, "category": "other", "source": "Web"}


def hash_url(url: str) -> str:
    """Create a hash for deduplication."""
    # Normalize URL before hashing
    normalized = url.lower().rstrip("/")
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def load_state() -> CaptureState:
    """Load persistent capture state."""
    if CAPTURE_STATE_FILE.exists():
        try:
            data = json.loads(CAPTURE_STATE_FILE.read_text())
            return CaptureState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            pass
    return CaptureState()


def save_state(state: CaptureState):
    """Save capture state."""
    CAPTURE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAPTURE_STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))


def find_claude_sessions(hours: int = 24) -> List[Path]:
    """Find all Claude Code session files modified in the last N hours."""
    sessions = []
    cutoff = datetime.now() - timedelta(hours=hours)

    # Main history file
    if CLAUDE_HISTORY.exists():
        if datetime.fromtimestamp(CLAUDE_HISTORY.stat().st_mtime) > cutoff:
            sessions.append(CLAUDE_HISTORY)

    # Project session files
    if CLAUDE_PROJECTS_DIR.exists():
        for proj_dir in CLAUDE_PROJECTS_DIR.iterdir():
            if proj_dir.is_dir():
                for jsonl in proj_dir.glob("*.jsonl"):
                    if datetime.fromtimestamp(jsonl.stat().st_mtime) > cutoff:
                        sessions.append(jsonl)
                # Check subagents
                subagents_dir = proj_dir / "subagents"
                if subagents_dir.exists():
                    for jsonl in subagents_dir.glob("*.jsonl"):
                        if datetime.fromtimestamp(jsonl.stat().st_mtime) > cutoff:
                            sessions.append(jsonl)

    return sorted(sessions, key=lambda f: f.stat().st_mtime, reverse=True)


def extract_text_from_jsonl(file_path: Path) -> Tuple[str, int]:
    """Extract all text content from a JSONL file.

    Returns (text, message_count).
    """
    text_parts = []
    message_count = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    texts = extract_text_from_entry(entry)
                    if texts:
                        text_parts.extend(texts)
                        message_count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return '\n'.join(text_parts), message_count


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


def extract_urls(text: str, session_file: str, state: CaptureState) -> List[CapturedURL]:
    """Extract URLs with metadata from text.

    Uses pre-compiled URL_PATTERN and SKIP_URL_PATTERNS for performance.
    """
    urls = []

    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip('.,;:)"\']>')

        # Skip internal/irrelevant URLs using pre-compiled patterns
        url_lower = url.lower()
        if any(skip in url_lower for skip in SKIP_URL_PATTERNS):
            continue

        # Check for duplicates
        url_hash = hash_url(url)
        if url_hash in state.url_hashes:
            continue

        # Get context (200 chars before and after)
        start = max(0, match.start() - 200)
        end = min(len(text), match.end() + 200)
        context = text[start:end].replace('\n', ' ').strip()

        # Classify URL
        classification = classify_url(url)

        # Calculate confidence based on context
        confidence = 1.0
        if classification["tier"] == 1:
            confidence = 0.95
        elif classification["tier"] == 2:
            confidence = 0.8
        else:
            confidence = 0.6

        urls.append(CapturedURL(
            url=url,
            tier=classification["tier"],
            category=classification["category"],
            source=classification["source"],
            context=context[:500],
            captured_at=datetime.now().isoformat(),
            session_file=str(session_file),
            confidence=confidence
        ))

        # Mark as seen
        state.url_hashes.add(url_hash)

    return urls


def extract_findings(text: str, session_file: str) -> List[CapturedFinding]:
    """Extract key findings from text.

    Uses pre-compiled FINDING_PATTERNS for performance.
    """
    findings = []
    seen_findings = set()

    # Use pre-compiled patterns from module level
    for pattern, finding_type, base_confidence in FINDING_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            match_text = match.strip()

            # Skip if too short or already captured
            if len(match_text) < 30:
                continue

            # Simple dedup
            match_hash = hashlib.md5(match_text[:100].encode()).hexdigest()[:8]
            if match_hash in seen_findings:
                continue
            seen_findings.add(match_hash)

            # Get broader context using pre-compiled pattern
            pattern_match = pattern.search(text)
            if pattern_match:
                start = max(0, pattern_match.start() - 100)
                end = min(len(text), pattern_match.end() + 100)
                context = text[start:end].replace('\n', ' ').strip()
            else:
                context = match_text

            findings.append(CapturedFinding(
                text=match_text[:500],
                finding_type=finding_type,
                confidence=base_confidence,
                context=context[:500],
                captured_at=datetime.now().isoformat(),
                session_file=str(session_file),
            ))

    return findings


def detect_session_topic(text: str) -> Optional[str]:
    """Detect the research topic from text.

    Uses pre-compiled TOPIC_PATTERNS for performance.
    """
    # Use pre-compiled patterns from module level
    for pattern in TOPIC_PATTERNS:
        matches = pattern.findall(text[:5000])
        if matches:
            # Return first reasonable match
            topic = matches[0].strip()
            if len(topic) >= 10:
                return topic[:80]

    return None


def generate_session_id(topic: str, source_file: Path) -> str:
    """Generate a unique session ID."""
    mtime = datetime.fromtimestamp(source_file.stat().st_mtime)
    timestamp = mtime.strftime("%Y%m%d-%H%M%S")
    topic_hash = hashlib.md5(topic.encode()).hexdigest()[:6]
    safe_topic = re.sub(r'[^a-z0-9]+', '-', topic.lower())[:25]
    return f"auto-{safe_topic}-{timestamp}-{topic_hash}"


def scan_sessions(hours: int = 24, min_research_urls: int = 2) -> Dict[str, Any]:
    """Scan recent Claude sessions and capture research artifacts."""
    print(f"Scanning sessions from last {hours} hours...")

    state = load_state()
    sessions = find_claude_sessions(hours)

    results = {
        "sessions_scanned": 0,
        "sessions_captured": 0,
        "urls_captured": 0,
        "findings_captured": 0,
        "new_sessions": [],
    }

    for session_file in sessions:
        # Skip if already processed recently
        file_key = str(session_file)
        if file_key in state.captured_files:
            file_mtime = session_file.stat().st_mtime
            last_captured = state.captured_files[file_key].get("last_captured_mtime", 0)
            if file_mtime <= last_captured:
                continue

        results["sessions_scanned"] += 1

        # Extract text
        text, message_count = extract_text_from_jsonl(session_file)
        if not text or message_count < 5:
            continue

        # Extract URLs
        urls = extract_urls(text, session_file, state)

        # Filter for research-relevant URLs
        research_urls = [u for u in urls if u.tier <= 2 or
                        (u.tier == 3 and u.category in ["forums", "newsletters"])]

        if len(research_urls) < min_research_urls:
            # Still save the file state to avoid re-scanning
            state.captured_files[file_key] = {
                "last_captured_mtime": session_file.stat().st_mtime,
                "urls_found": len(urls),
                "research_urls": len(research_urls),
                "skipped": True,
            }
            continue

        # Extract findings
        findings = extract_findings(text, session_file)

        # Detect topic
        topic = detect_session_topic(text) or f"Session from {session_file.name}"

        # Generate session ID
        session_id = generate_session_id(topic, session_file)

        # Create session directory
        session_dir = SESSIONS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save session data
        session_data = {
            "session_id": session_id,
            "topic": topic,
            "source_file": str(session_file),
            "auto_captured": True,
            "captured_at": datetime.now().isoformat(),
            "original_date": datetime.fromtimestamp(session_file.stat().st_mtime).isoformat(),
            "status": "archived",
            "message_count": message_count,
            "stats": {
                "urls_count": len(urls),
                "research_urls_count": len(research_urls),
                "findings_count": len(findings),
                "text_length": len(text),
            }
        }

        (session_dir / "session.json").write_text(json.dumps(session_data, indent=2))
        (session_dir / "urls_captured.json").write_text(
            json.dumps([u.to_dict() for u in urls], indent=2)
        )
        (session_dir / "findings_captured.json").write_text(
            json.dumps([f.to_dict() for f in findings], indent=2)
        )

        # Update state
        state.captured_files[file_key] = {
            "session_id": session_id,
            "last_captured_mtime": session_file.stat().st_mtime,
            "urls_captured": len(urls),
            "findings_captured": len(findings),
            "captured_at": datetime.now().isoformat(),
        }
        state.total_urls_captured += len(urls)
        state.total_findings_captured += len(findings)
        state.sessions_created += 1

        # Update results
        results["sessions_captured"] += 1
        results["urls_captured"] += len(urls)
        results["findings_captured"] += len(findings)
        results["new_sessions"].append({
            "session_id": session_id,
            "topic": topic,
            "urls": len(urls),
            "findings": len(findings),
        })

        print(f"  Captured: {session_id}")
        print(f"    Topic: {topic}")
        print(f"    URLs: {len(urls)} ({len(research_urls)} research)")
        print(f"    Findings: {len(findings)}")

    # Save state
    state.last_scan = datetime.now().isoformat()
    save_state(state)

    print(f"\nScan complete:")
    print(f"  Sessions scanned: {results['sessions_scanned']}")
    print(f"  Sessions captured: {results['sessions_captured']}")
    print(f"  URLs captured: {results['urls_captured']}")
    print(f"  Findings captured: {results['findings_captured']}")

    return results


async def sync_to_storage():
    """Sync captured data to the storage engine."""
    try:
        from storage.engine import get_engine
    except ImportError:
        print("Storage engine not available. Run from researchgravity directory.")
        return

    print("Syncing to storage engine...")

    engine = await get_engine()

    # Find sessions to sync
    synced = 0
    for session_dir in SESSIONS_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        session_file = session_dir / "session.json"
        if not session_file.exists():
            continue

        session_data = json.loads(session_file.read_text())

        # Check if already synced
        if session_data.get("synced_to_storage"):
            continue

        # Store session
        session_data["id"] = session_data["session_id"]
        await engine.store_session(session_data, source="auto_capture")

        # Store URLs
        urls_file = session_dir / "urls_captured.json"
        if urls_file.exists():
            urls = json.loads(urls_file.read_text())
            for url_data in urls:
                url_data["session_id"] = session_data["session_id"]
            await engine.store_urls_batch(urls)

        # Store findings
        findings_file = session_dir / "findings_captured.json"
        if findings_file.exists():
            findings = json.loads(findings_file.read_text())
            for f in findings:
                f["session_id"] = session_data["session_id"]
                f["content"] = f.get("text", "")
                f["type"] = f.get("finding_type", "finding")
            await engine.store_findings_batch(findings, source="auto_capture")

        # Mark as synced
        session_data["synced_to_storage"] = True
        session_data["synced_at"] = datetime.now().isoformat()
        session_file.write_text(json.dumps(session_data, indent=2))

        synced += 1
        print(f"  Synced: {session_data['session_id']}")

    await engine.close()
    print(f"\nSynced {synced} sessions to storage engine.")


def show_status():
    """Show capture status."""
    state = load_state()

    print("=" * 60)
    print("  ResearchGravity Auto-Capture V2 Status")
    print("=" * 60)
    print()
    print(f"Last scan: {state.last_scan or 'Never'}")
    print(f"Total sessions created: {state.sessions_created}")
    print(f"Total URLs captured: {state.total_urls_captured}")
    print(f"Total findings captured: {state.total_findings_captured}")
    print(f"Unique URL hashes: {len(state.url_hashes)}")
    print(f"Files tracked: {len(state.captured_files)}")
    print()

    # Show recent captures
    recent = sorted(
        [(k, v) for k, v in state.captured_files.items() if v.get("session_id")],
        key=lambda x: x[1].get("captured_at", ""),
        reverse=True
    )[:5]

    if recent:
        print("Recent captures:")
        for file_path, info in recent:
            print(f"  {info.get('session_id', 'unknown')[:40]}")
            print(f"    URLs: {info.get('urls_captured', 0)} | Findings: {info.get('findings_captured', 0)}")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ResearchGravity Auto-Capture V2"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Scan
    scan_parser = subparsers.add_parser("scan", help="Scan recent sessions")
    scan_parser.add_argument("--hours", type=int, default=24, help="Hours to scan back")
    scan_parser.add_argument("--min-urls", type=int, default=2, help="Minimum research URLs")

    # Watch
    watch_parser = subparsers.add_parser("watch", help="Watch mode (daemon)")
    watch_parser.add_argument("--interval", type=int, default=300, help="Check interval (seconds)")

    # Capture
    capture_parser = subparsers.add_parser("capture", help="Capture specific session")
    capture_parser.add_argument("--session", required=True, help="Session file path")

    # Sync
    subparsers.add_parser("sync", help="Sync to storage engine")

    # Status
    subparsers.add_parser("status", help="Show capture stats")

    args = parser.parse_args()

    if args.command == "scan":
        scan_sessions(args.hours, args.min_urls)
    elif args.command == "watch":
        print(f"Watching for new sessions (interval: {args.interval}s)")
        print("Press Ctrl+C to stop")
        while True:
            try:
                scan_sessions(hours=1, min_research_urls=2)
                import time
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopping watch mode")
                break
    elif args.command == "sync":
        asyncio.run(sync_to_storage())
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
