#!/usr/bin/env python3
"""
Evidence Extractor for Antigravity Chief of Staff.

Extracts citations and sources from findings to build evidence chains.
Implements the "Evidence Required" principle.

Sources extracted:
- arXiv paper IDs
- GitHub repository URLs
- Web URLs with domain classification
- Session references

Usage:
    python3 evidence_extractor.py --session <session-id>   # Process specific session
    python3 evidence_extractor.py --all                     # Process all sessions
    python3 evidence_extractor.py --dry-run                 # Preview without writing
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Add parent dir for schema import
sys.path.insert(0, str(Path.home() / ".agent-core" / "schemas"))
try:
    from finding import (
        EvidencedFinding, EvidenceSource, Evidence,
        FindingType, create_finding, migrate_legacy_finding
    )
    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False
    print("Warning: Schema not available, using dict-based approach")


AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"

# URL patterns for extraction
PATTERNS = {
    "arxiv": re.compile(r'(?:arXiv[:\s]*)?(\d{4}\.\d{4,5})', re.IGNORECASE),
    "arxiv_url": re.compile(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})'),
    "github": re.compile(r'github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)'),
    "url": re.compile(r'https?://[^\s\)\]]+'),
    "session_ref": re.compile(r'`([a-z0-9-]+-\d{8}-\d{6}-[a-f0-9]+)`'),
}

# High-value domains for source classification
TIER_1_DOMAINS = [
    "arxiv.org", "openai.com", "anthropic.com", "deepmind.google",
    "huggingface.co", "ai.meta.com", "ai.google", "microsoft.com/research"
]

TIER_2_DOMAINS = [
    "github.com", "paperswithcode.com", "semanticscholar.org",
    "techcrunch.com", "theverge.com", "lmsys.org"
]


def classify_url_tier(url: str) -> int:
    """Classify URL into tier based on domain."""
    url_lower = url.lower()
    for domain in TIER_1_DOMAINS:
        if domain in url_lower:
            return 1
    for domain in TIER_2_DOMAINS:
        if domain in url_lower:
            return 2
    return 3


def extract_evidence_from_text(text: str, session_urls: list = None) -> list:
    """
    Extract evidence sources from finding text.

    Args:
        text: The finding text to analyze
        session_urls: List of URLs captured in the session (for cross-reference)

    Returns:
        List of EvidenceSource dicts
    """
    sources = []
    seen_urls = set()

    # Extract arXiv IDs
    for match in PATTERNS["arxiv"].finditer(text):
        arxiv_id = match.group(1)
        url = f"https://arxiv.org/abs/{arxiv_id}"
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "arxiv_id": arxiv_id,
                "excerpt": text[max(0, match.start()-50):min(len(text), match.end()+50)],
                "relevance_score": 0.8,  # arXiv papers are high-value
                "verified": False,
                "accessed_at": datetime.now().isoformat()
            })

    # Extract arXiv URLs
    for match in PATTERNS["arxiv_url"].finditer(text):
        arxiv_id = match.group(1)
        url = f"https://arxiv.org/abs/{arxiv_id}"
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "arxiv_id": arxiv_id,
                "excerpt": text[max(0, match.start()-50):min(len(text), match.end()+50)],
                "relevance_score": 0.8,
                "verified": False,
                "accessed_at": datetime.now().isoformat()
            })

    # Extract GitHub repos
    for match in PATTERNS["github"].finditer(text):
        repo = match.group(1)
        url = f"https://github.com/{repo}"
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "excerpt": text[max(0, match.start()-30):min(len(text), match.end()+30)],
                "relevance_score": 0.6,  # GitHub repos are useful but need verification
                "verified": False,
                "accessed_at": datetime.now().isoformat()
            })

    # Extract other URLs
    for match in PATTERNS["url"].finditer(text):
        url = match.group(0).rstrip('.,;:')
        if url not in seen_urls and "arxiv.org" not in url and "github.com" not in url:
            seen_urls.add(url)
            tier = classify_url_tier(url)
            relevance = {1: 0.7, 2: 0.5, 3: 0.3}.get(tier, 0.3)
            sources.append({
                "url": url,
                "excerpt": text[max(0, match.start()-30):min(len(text), match.end()+30)],
                "relevance_score": relevance,
                "verified": False,
                "accessed_at": datetime.now().isoformat()
            })

    # Cross-reference with session URLs if provided
    if session_urls:
        for url_data in session_urls:
            url = url_data.get("url", "")
            if url and url not in seen_urls:
                # Check if this URL is mentioned/related to the finding
                # Simple heuristic: check if any keyword from URL appears in text
                url_parts = url.lower().replace("-", " ").replace("_", " ").split("/")
                keywords = [p for p in url_parts if len(p) > 3 and p not in ["https", "http", "www", "com", "org"]]

                text_lower = text.lower()
                if any(kw in text_lower for kw in keywords):
                    tier = url_data.get("tier", 3)
                    relevance = {1: 0.7, 2: 0.5, 3: 0.3}.get(tier, 0.3)
                    sources.append({
                        "url": url,
                        "excerpt": url_data.get("context", "")[:100] or f"Session URL (Tier {tier})",
                        "relevance_score": relevance,
                        "verified": False,
                        "accessed_at": datetime.now().isoformat()
                    })
                    seen_urls.add(url)

    return sources


def process_legacy_finding(finding: dict, session_id: str, session_urls: list = None) -> dict:
    """
    Process a legacy finding and enrich with evidence.

    Args:
        finding: Legacy finding dict with 'type', 'text', 'extracted_at'
        session_id: Session ID for reference
        session_urls: URLs captured in the session

    Returns:
        Enriched finding dict with evidence structure
    """
    text = finding.get("text", "")
    finding_type = finding.get("type", "finding")

    # Extract evidence from text
    sources = extract_evidence_from_text(text, session_urls)

    # Calculate confidence
    if sources:
        confidence = sum(s["relevance_score"] for s in sources) / len(sources)
    else:
        confidence = 0.0

    now = datetime.now().isoformat()

    enriched = {
        "id": f"finding-{int(datetime.now().timestamp())}-{session_id[:8]}",
        "session_id": session_id,
        "content": text,
        "type": finding_type,
        "evidence": {
            "sources": sources,
            "confidence": confidence,
            "reasoning_chain": [],
            "validation": {
                "validated": False
            }
        },
        "created_at": finding.get("extracted_at", now),
        "updated_at": now,
        "needs_review": len(sources) == 0  # Flag findings without evidence
    }

    return enriched


def process_session(session_id: str, dry_run: bool = False) -> dict:
    """
    Process a session and enrich all findings with evidence.

    Args:
        session_id: Session ID to process
        dry_run: If True, don't write changes

    Returns:
        Stats dict with processing results
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return {"error": f"Session not found: {session_id}"}

    stats = {
        "session_id": session_id,
        "findings_processed": 0,
        "sources_extracted": 0,
        "needs_review": 0,
        "avg_confidence": 0.0
    }

    # Load session URLs for cross-reference
    urls_file = session_dir / "urls_captured.json"
    session_urls = []
    if urls_file.exists():
        try:
            session_urls = json.loads(urls_file.read_text())
        except json.JSONDecodeError:
            pass

    # Load findings
    findings_file = session_dir / "findings_captured.json"
    if not findings_file.exists():
        return {"error": "No findings file", "session_id": session_id}

    try:
        findings = json.loads(findings_file.read_text())
    except json.JSONDecodeError:
        return {"error": "Invalid findings JSON", "session_id": session_id}

    if not findings:
        return {"error": "Empty findings", "session_id": session_id}

    # Process each finding
    enriched_findings = []
    total_sources = 0
    total_confidence = 0.0
    needs_review = 0

    for finding in findings:
        enriched = process_legacy_finding(finding, session_id, session_urls)
        enriched_findings.append(enriched)

        # Collect stats
        sources_count = len(enriched["evidence"]["sources"])
        total_sources += sources_count
        total_confidence += enriched["evidence"]["confidence"]
        if enriched.get("needs_review"):
            needs_review += 1

    # Calculate averages
    stats["findings_processed"] = len(enriched_findings)
    stats["sources_extracted"] = total_sources
    stats["needs_review"] = needs_review
    if enriched_findings:
        stats["avg_confidence"] = total_confidence / len(enriched_findings)

    # Write enriched findings
    if not dry_run:
        output_file = session_dir / "findings_evidenced.json"
        output_file.write_text(json.dumps(enriched_findings, indent=2))
        stats["output_file"] = str(output_file)

    return stats


def process_all_sessions(dry_run: bool = False, verbose: bool = False) -> list:
    """Process all sessions and enrich findings."""
    results = []

    if not SESSIONS_DIR.exists():
        return results

    sessions = [d for d in SESSIONS_DIR.iterdir() if d.is_dir()]

    for session_dir in sessions:
        findings_file = session_dir / "findings_captured.json"
        if not findings_file.exists():
            continue

        # Skip already processed sessions
        evidenced_file = session_dir / "findings_evidenced.json"
        if evidenced_file.exists() and not dry_run:
            if verbose:
                print(f"  Skipping (already processed): {session_dir.name[:40]}")
            continue

        if verbose:
            print(f"  Processing: {session_dir.name[:50]}")

        stats = process_session(session_dir.name, dry_run=dry_run)
        results.append(stats)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract evidence from findings"
    )
    parser.add_argument("--session", "-s",
                        help="Process specific session ID")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Process all sessions")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Preview without writing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Reprocess already-processed sessions")

    args = parser.parse_args()

    print("Evidence Extractor — Antigravity Chief of Staff")
    print("=" * 50)

    if args.session:
        stats = process_session(args.session, dry_run=args.dry_run)
        if "error" in stats:
            print(f"❌ {stats['error']}")
        else:
            print(f"✅ Processed: {stats['session_id'][:40]}")
            print(f"   Findings: {stats['findings_processed']}")
            print(f"   Sources extracted: {stats['sources_extracted']}")
            print(f"   Needs review: {stats['needs_review']}")
            print(f"   Avg confidence: {stats['avg_confidence']:.2f}")
            if not args.dry_run:
                print(f"   Output: {stats.get('output_file', 'N/A')}")

    elif args.all:
        results = process_all_sessions(dry_run=args.dry_run, verbose=args.verbose)

        # Aggregate stats
        total_findings = sum(r.get("findings_processed", 0) for r in results if "error" not in r)
        total_sources = sum(r.get("sources_extracted", 0) for r in results if "error" not in r)
        total_review = sum(r.get("needs_review", 0) for r in results if "error" not in r)
        errors = [r for r in results if "error" in r]

        print()
        print("📊 Summary:")
        print(f"   Sessions processed: {len(results) - len(errors)}")
        print(f"   Total findings: {total_findings}")
        print(f"   Sources extracted: {total_sources}")
        print(f"   Findings needing review: {total_review}")
        if errors:
            print(f"   Errors: {len(errors)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
