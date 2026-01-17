"""
UCW Export — Convert agent-core data to Universal Cognitive Wallet format.

This creates a portable, platform-agnostic representation of your AI
interaction history that can be imported into any compatible AI platform.
"""

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

from .schema import (
    CognitiveWallet,
    Concept,
    ConceptType,
    Session,
    URL,
    Connection,
    ConnectionType,
    ValueMetrics,
)
from .value import CognitiveAppreciationEngine
from .history import load_history, record_snapshot


AGENT_CORE = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE / "sessions"
PROJECTS_FILE = AGENT_CORE / "projects.json"


def load_agent_core_data() -> Dict[str, Any]:
    """Load all agent-core data for export."""
    data = {
        "sessions": {},
        "projects": {},
        "papers": {},
    }

    # Load projects.json
    if PROJECTS_FILE.exists():
        try:
            projects_data = json.loads(PROJECTS_FILE.read_text())
            data["projects"] = projects_data.get("projects", {})
            data["papers"] = projects_data.get("paper_index", {})
        except Exception as e:
            print(f"Warning: Could not load projects.json: {e}")

    # Load sessions
    if SESSIONS_DIR.exists():
        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue

            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            try:
                session_data = json.loads(session_file.read_text())
                session_id = session_data.get("session_id", session_dir.name)
                data["sessions"][session_id] = session_data

                # Load additional files if present
                for extra_file in ["urls_captured.json", "findings_captured.json", "lineage.json"]:
                    extra_path = session_dir / extra_file
                    if extra_path.exists():
                        try:
                            extra_data = json.loads(extra_path.read_text())
                            data["sessions"][session_id][extra_file.replace(".json", "")] = extra_data
                        except:
                            pass

            except Exception as e:
                print(f"Warning: Could not load session {session_dir.name}: {e}")

    return data


def extract_concepts_from_sessions(sessions: Dict[str, Any]) -> Dict[str, Concept]:
    """Extract concepts from session findings."""
    concepts = {}
    arxiv_pattern = re.compile(r'(\d{4}\.\d{4,5})')

    for session_id, session_data in sessions.items():
        # Extract from findings_captured
        findings = session_data.get("findings_captured", [])
        if isinstance(findings, list):
            for i, finding in enumerate(findings):
                if isinstance(finding, dict):
                    text = finding.get("text", "")
                    finding_type = finding.get("type", "finding")
                elif isinstance(finding, str):
                    text = finding
                    finding_type = "finding"
                else:
                    continue

                if not text or len(text) < 10:
                    continue

                concept_id = f"concept_{hashlib.md5(text.encode()).hexdigest()[:8]}"

                # Determine concept type
                if "thesis" in finding_type.lower():
                    c_type = ConceptType.THESIS
                elif "gap" in finding_type.lower():
                    c_type = ConceptType.GAP
                elif "innovation" in finding_type.lower():
                    c_type = ConceptType.INNOVATION
                else:
                    c_type = ConceptType.FINDING

                # Extract arXiv references as sources
                sources = [session_id]
                arxiv_matches = arxiv_pattern.findall(text)
                sources.extend(arxiv_matches)

                concepts[concept_id] = Concept(
                    id=concept_id,
                    content=text[:500],  # Truncate long content
                    concept_type=c_type,
                    confidence=0.7,  # Default confidence
                    sources=sources,
                    created_at=datetime.now(),
                )

    return concepts


def extract_papers_from_data(
    projects_data: Dict[str, Any],
    sessions: Dict[str, Any],
) -> Dict[str, Dict]:
    """Extract paper index from projects and sessions."""
    papers = {}
    arxiv_pattern = re.compile(r'(\d{4}\.\d{4,5})')

    # From paper_index in projects
    for arxiv_id, paper_data in projects_data.items():
        papers[arxiv_id] = {
            "id": arxiv_id,
            "projects": paper_data.get("projects", []),
            "sessions": paper_data.get("sessions", []),
        }

    # Extract from session URLs
    for session_id, session_data in sessions.items():
        urls = session_data.get("urls_captured", [])
        if isinstance(urls, list):
            for url_data in urls:
                if isinstance(url_data, dict):
                    url = url_data.get("url", "")
                elif isinstance(url_data, str):
                    url = url_data
                else:
                    continue

                if "arxiv.org" in url:
                    match = arxiv_pattern.search(url)
                    if match:
                        arxiv_id = match.group(1)
                        if arxiv_id not in papers:
                            papers[arxiv_id] = {
                                "id": arxiv_id,
                                "projects": [],
                                "sessions": [],
                            }
                        if session_id not in papers[arxiv_id]["sessions"]:
                            papers[arxiv_id]["sessions"].append(session_id)

    return papers


def convert_session(session_id: str, session_data: Dict[str, Any]) -> Session:
    """Convert agent-core session to UCW Session."""
    # Parse date
    date_str = (
        session_data.get("original_date")
        or session_data.get("started")
        or session_data.get("backfilled_at")
    )
    if date_str:
        try:
            session_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            session_date = datetime.now()
    else:
        session_date = datetime.now()

    # Convert URLs
    urls = []
    urls_data = session_data.get("urls_captured", [])
    if isinstance(urls_data, list):
        for url_data in urls_data:
            if isinstance(url_data, dict):
                urls.append(URL(
                    url=url_data.get("url", ""),
                    tier=url_data.get("tier", 3),
                    category=url_data.get("category", "other"),
                    source=url_data.get("source", "Web"),
                    context=url_data.get("context", "")[:200],
                    captured_at=session_date,
                    relevance=url_data.get("relevance"),
                    signal=url_data.get("signal"),
                ))
            elif isinstance(url_data, str):
                urls.append(URL(
                    url=url_data,
                    tier=3,
                    category="other",
                    source="Web",
                    context="",
                    captured_at=session_date,
                ))

    # Extract paper IDs from URLs
    arxiv_pattern = re.compile(r'(\d{4}\.\d{4,5})')
    papers = []
    for url in urls:
        if "arxiv.org" in url.url:
            match = arxiv_pattern.search(url.url)
            if match:
                papers.append(match.group(1))

    # Get project from lineage
    lineage = session_data.get("lineage", {})
    project = lineage.get("impl_project") if isinstance(lineage, dict) else None

    return Session(
        id=session_id,
        topic=session_data.get("topic", "Unknown"),
        date=session_date,
        findings=[],  # Will be populated with concept IDs
        papers=list(set(papers)),
        urls=urls,
        project=project,
        status=session_data.get("status", "archived"),
        metadata={
            "backfilled": session_data.get("backfilled", False),
            "source_file": session_data.get("source_file"),
        },
    )


def build_wallet_from_agent_core() -> CognitiveWallet:
    """Build a CognitiveWallet from agent-core data."""
    data = load_agent_core_data()

    # Convert sessions
    sessions = {}
    for session_id, session_data in data["sessions"].items():
        sessions[session_id] = convert_session(session_id, session_data)

    # Extract concepts
    concepts = extract_concepts_from_sessions(data["sessions"])

    # Extract papers
    papers = extract_papers_from_data(data["papers"], data["sessions"])

    # Link concepts to sessions
    for concept in concepts.values():
        for source in concept.sources:
            if source in sessions:
                if concept.id not in sessions[source].findings:
                    sessions[source].findings.append(concept.id)

    # Create wallet
    wallet = CognitiveWallet(
        version="1.0",
        created=datetime.now(),
        concepts=concepts,
        sessions=sessions,
        papers=papers,
        connections=[],  # Will be enhanced with connection detection
        metadata={
            "exported_from": "agent-core",
            "export_date": datetime.now().isoformat(),
        },
    )

    # Load existing history
    existing_history = load_history()
    wallet.value_metrics.history = existing_history

    # Calculate value
    engine = CognitiveAppreciationEngine()
    engine.update_wallet_metrics(wallet)

    # Record snapshot to persistent history
    total_urls = sum(len(s.urls) for s in wallet.sessions.values())
    record_snapshot(
        value=wallet.value_metrics.total_value,
        concepts=len(wallet.concepts),
        sessions=len(wallet.sessions),
        papers=len(wallet.papers),
        urls=total_urls,
    )

    # Set integrity hash
    wallet.update_integrity_hash()

    return wallet


def export_wallet(
    wallet: Optional[CognitiveWallet] = None,
    output_path: Optional[str] = None,
    pretty: bool = True,
) -> str:
    """
    Export wallet to UCW JSON format.

    Args:
        wallet: CognitiveWallet to export. If None, builds from agent-core.
        output_path: Path to write JSON file. If None, returns JSON string.
        pretty: Whether to pretty-print JSON.

    Returns:
        JSON string of wallet data.
    """
    if wallet is None:
        wallet = build_wallet_from_agent_core()

    # Ensure integrity hash is current
    wallet.update_integrity_hash()

    # Convert to dict
    data = wallet.to_dict()

    # Serialize
    if pretty:
        json_str = json.dumps(data, indent=2, default=str)
    else:
        json_str = json.dumps(data, default=str)

    # Write to file if path provided
    if output_path:
        Path(output_path).write_text(json_str)
        print(f"Wallet exported to: {output_path}")

    return json_str


def export_wallet_summary(wallet: Optional[CognitiveWallet] = None) -> str:
    """Export a summary of the wallet for quick review."""
    if wallet is None:
        wallet = build_wallet_from_agent_core()

    stats = wallet.get_stats()
    lines = [
        "# Universal Cognitive Wallet Summary",
        "",
        f"**Version:** {wallet.version}",
        f"**Created:** {wallet.created.isoformat()}",
        f"**Integrity:** {wallet.integrity_hash[:16]}...",
        "",
        "## Statistics",
        "",
        f"- Sessions: {stats['sessions']}",
        f"- Concepts: {stats['concepts']}",
        f"- Connections: {stats['connections']}",
        f"- Papers: {stats['papers']}",
        f"- URLs: {stats['urls']}",
        f"- **Value: ${stats['value']:,.2f}**",
        "",
    ]

    if stats['domains']:
        lines.append("## Domains")
        lines.append("")
        for domain, weight in sorted(stats['domains'].items(), key=lambda x: -x[1]):
            lines.append(f"- {domain}: {weight*100:.0f}%")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else None

    print("Building wallet from agent-core...")
    wallet = build_wallet_from_agent_core()

    print("\n" + export_wallet_summary(wallet))

    if output:
        export_wallet(wallet, output)
    else:
        print("\nUse: python3 export.py <output.ucw.json> to save")
