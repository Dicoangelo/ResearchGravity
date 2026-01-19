#!/usr/bin/env python3
"""
Confidence Scorer for Antigravity Chief of Staff.

Calculates confidence scores for findings based on:
- Source quantity and quality
- Source diversity (different domains)
- Citation freshness
- Cross-validation between sources

Implements the "Evidence Required" principle with quantitative metrics.

Usage:
    python3 confidence_scorer.py --session <session-id>   # Score specific session
    python3 confidence_scorer.py --finding <finding-id>   # Score specific finding
    python3 confidence_scorer.py --stats                   # Show overall stats
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"

# Confidence calculation weights
WEIGHTS = {
    "source_count": 0.25,      # More sources = higher confidence
    "source_quality": 0.30,    # Tier 1 sources weighted higher
    "source_diversity": 0.20,  # Different domains = more robust
    "freshness": 0.15,         # Recent sources preferred
    "verification": 0.10,      # Critic-validated sources
}

# Quality multipliers by source type
QUALITY_MULTIPLIERS = {
    "arxiv": 1.0,              # Academic papers - highest
    "openai": 0.95,            # Major AI labs
    "anthropic": 0.95,
    "deepmind": 0.95,
    "google_ai": 0.90,
    "meta_ai": 0.90,
    "github": 0.70,            # Code repos - need verification
    "huggingface": 0.80,       # ML-specific
    "blog": 0.50,              # Blogs/news - lower
    "unknown": 0.30,           # Unknown sources
}


def classify_source_type(url: str) -> str:
    """Classify source URL by domain/type."""
    url_lower = url.lower()

    if "arxiv.org" in url_lower:
        return "arxiv"
    if "openai.com" in url_lower:
        return "openai"
    if "anthropic.com" in url_lower:
        return "anthropic"
    if "deepmind" in url_lower:
        return "deepmind"
    if "ai.google" in url_lower or "research.google" in url_lower:
        return "google_ai"
    if "ai.meta.com" in url_lower or "research.fb.com" in url_lower:
        return "meta_ai"
    if "github.com" in url_lower:
        return "github"
    if "huggingface.co" in url_lower:
        return "huggingface"
    if any(d in url_lower for d in ["blog", "medium.com", "substack.com"]):
        return "blog"

    return "unknown"


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower().replace("www.", "")
    except Exception:
        return "unknown"


def calculate_source_count_score(sources: list, max_sources: int = 5) -> float:
    """
    Calculate score based on number of sources.
    Logarithmic scaling - diminishing returns after ~5 sources.
    """
    if not sources:
        return 0.0

    count = len(sources)
    # Log scaling with max at ~5 sources
    import math
    score = min(1.0, math.log(count + 1) / math.log(max_sources + 1))
    return score


def calculate_source_quality_score(sources: list) -> float:
    """
    Calculate score based on source quality.
    Higher quality sources (academic, major labs) score higher.
    """
    if not sources:
        return 0.0

    quality_scores = []
    for source in sources:
        url = source.get("url", "")
        source_type = classify_source_type(url)
        multiplier = QUALITY_MULTIPLIERS.get(source_type, 0.3)

        # Factor in the source's own relevance score
        relevance = source.get("relevance_score", 0.5)
        quality_scores.append(multiplier * relevance)

    return sum(quality_scores) / len(quality_scores)


def calculate_diversity_score(sources: list) -> float:
    """
    Calculate score based on source diversity.
    Multiple different domains = more robust evidence.
    """
    if not sources:
        return 0.0

    domains = [get_domain(s.get("url", "")) for s in sources]
    unique_domains = len(set(domains))
    total_domains = len(domains)

    if total_domains == 1:
        return 0.5  # Single source, moderate diversity

    # Ratio of unique to total, with bonus for multiple unique
    diversity = unique_domains / total_domains
    bonus = min(0.3, unique_domains * 0.1)  # Bonus for each unique domain

    return min(1.0, diversity + bonus)


def calculate_freshness_score(sources: list, max_age_days: int = 365) -> float:
    """
    Calculate score based on source freshness.
    More recent sources score higher.
    """
    if not sources:
        return 0.0

    now = datetime.now()
    freshness_scores = []

    for source in sources:
        accessed_at = source.get("accessed_at")
        if accessed_at:
            try:
                access_date = datetime.fromisoformat(accessed_at.replace("Z", "+00:00"))
                access_date = access_date.replace(tzinfo=None)
                age_days = (now - access_date).days

                if age_days < 0:
                    age_days = 0

                # Linear decay over max_age_days
                freshness = max(0, 1 - (age_days / max_age_days))
                freshness_scores.append(freshness)
            except (ValueError, TypeError):
                freshness_scores.append(0.5)  # Default for unparseable dates
        else:
            freshness_scores.append(0.5)  # Default if no date

    return sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0.5


def calculate_verification_score(sources: list) -> float:
    """
    Calculate score based on verification status.
    Critic-validated sources score higher.
    """
    if not sources:
        return 0.0

    verified_count = sum(1 for s in sources if s.get("verified", False))
    return verified_count / len(sources)


def calculate_confidence(evidence: dict) -> dict:
    """
    Calculate comprehensive confidence score for evidence.

    Args:
        evidence: Evidence dict with sources, reasoning_chain, validation

    Returns:
        Dict with overall score and component breakdowns
    """
    sources = evidence.get("sources", [])

    # Calculate component scores
    components = {
        "source_count": calculate_source_count_score(sources),
        "source_quality": calculate_source_quality_score(sources),
        "source_diversity": calculate_diversity_score(sources),
        "freshness": calculate_freshness_score(sources),
        "verification": calculate_verification_score(sources),
    }

    # Calculate weighted overall score
    overall = sum(
        components[key] * WEIGHTS[key]
        for key in WEIGHTS
    )

    # Apply reasoning chain bonus (if present)
    reasoning_chain = evidence.get("reasoning_chain", [])
    if reasoning_chain:
        # Bonus for explicit reasoning (max 10%)
        reasoning_bonus = min(0.1, len(reasoning_chain) * 0.02)
        overall = min(1.0, overall + reasoning_bonus)

    return {
        "overall": round(overall, 3),
        "components": {k: round(v, 3) for k, v in components.items()},
        "source_count": len(sources),
        "source_types": dict(Counter(classify_source_type(s.get("url", "")) for s in sources)),
        "verified_count": sum(1 for s in sources if s.get("verified", False)),
    }


def score_finding(finding: dict) -> dict:
    """
    Score a single finding and return enriched result.

    Args:
        finding: Finding dict with evidence

    Returns:
        Finding dict with updated confidence scores
    """
    evidence = finding.get("evidence", {})
    confidence_result = calculate_confidence(evidence)

    # Update finding with new confidence
    finding["evidence"]["confidence"] = confidence_result["overall"]
    finding["confidence_breakdown"] = confidence_result

    return finding


def score_session(session_id: str, update_file: bool = False) -> dict:
    """
    Score all findings in a session.

    Args:
        session_id: Session ID to process
        update_file: If True, update the findings_evidenced.json file

    Returns:
        Stats dict with scoring results
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return {"error": f"Session not found: {session_id}"}

    # Try evidenced file first, fall back to captured
    findings_file = session_dir / "findings_evidenced.json"
    if not findings_file.exists():
        findings_file = session_dir / "findings_captured.json"

    if not findings_file.exists():
        return {"error": "No findings file", "session_id": session_id}

    try:
        findings = json.loads(findings_file.read_text())
    except json.JSONDecodeError:
        return {"error": "Invalid findings JSON", "session_id": session_id}

    if not findings:
        return {"session_id": session_id, "findings": 0}

    # Score each finding
    scored_findings = []
    confidence_scores = []

    for finding in findings:
        # Skip if no evidence structure (legacy format)
        if "evidence" not in finding:
            scored_findings.append(finding)
            continue

        scored = score_finding(finding)
        scored_findings.append(scored)
        confidence_scores.append(scored["evidence"]["confidence"])

    stats = {
        "session_id": session_id,
        "findings_scored": len(confidence_scores),
        "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
        "min_confidence": min(confidence_scores) if confidence_scores else 0,
        "max_confidence": max(confidence_scores) if confidence_scores else 0,
        "below_threshold": sum(1 for c in confidence_scores if c < 0.7),
    }

    # Update file if requested
    if update_file and confidence_scores:
        output_file = session_dir / "findings_evidenced.json"
        output_file.write_text(json.dumps(scored_findings, indent=2))
        stats["output_file"] = str(output_file)

    return stats


def get_overall_stats() -> dict:
    """Get confidence statistics across all sessions."""
    if not SESSIONS_DIR.exists():
        return {"error": "No sessions directory"}

    all_confidences = []
    session_stats = []

    for session_dir in SESSIONS_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        findings_file = session_dir / "findings_evidenced.json"
        if not findings_file.exists():
            continue

        try:
            findings = json.loads(findings_file.read_text())
            for finding in findings:
                if "evidence" in finding and "confidence" in finding["evidence"]:
                    all_confidences.append(finding["evidence"]["confidence"])
        except (json.JSONDecodeError, IOError):
            continue

        session_stats.append(session_dir.name)

    if not all_confidences:
        return {"error": "No scored findings found"}

    return {
        "total_findings": len(all_confidences),
        "sessions_with_evidence": len(session_stats),
        "avg_confidence": sum(all_confidences) / len(all_confidences),
        "min_confidence": min(all_confidences),
        "max_confidence": max(all_confidences),
        "below_threshold_70": sum(1 for c in all_confidences if c < 0.7),
        "below_threshold_50": sum(1 for c in all_confidences if c < 0.5),
        "distribution": {
            "0.0-0.3": sum(1 for c in all_confidences if c < 0.3),
            "0.3-0.5": sum(1 for c in all_confidences if 0.3 <= c < 0.5),
            "0.5-0.7": sum(1 for c in all_confidences if 0.5 <= c < 0.7),
            "0.7-0.9": sum(1 for c in all_confidences if 0.7 <= c < 0.9),
            "0.9-1.0": sum(1 for c in all_confidences if c >= 0.9),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate confidence scores for findings"
    )
    parser.add_argument("--session", "-s",
                        help="Score specific session")
    parser.add_argument("--update", "-u", action="store_true",
                        help="Update findings file with new scores")
    parser.add_argument("--stats", action="store_true",
                        help="Show overall statistics")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    print("Confidence Scorer — Antigravity Chief of Staff")
    print("=" * 50)

    if args.stats:
        stats = get_overall_stats()
        if "error" in stats:
            print(f"❌ {stats['error']}")
        else:
            print("\n📊 Overall Confidence Statistics:")
            print(f"   Total findings scored: {stats['total_findings']}")
            print(f"   Sessions with evidence: {stats['sessions_with_evidence']}")
            print(f"   Average confidence: {stats['avg_confidence']:.3f}")
            print(f"   Min/Max: {stats['min_confidence']:.3f} / {stats['max_confidence']:.3f}")
            print(f"   Below 0.7 threshold: {stats['below_threshold_70']}")
            print(f"   Below 0.5 threshold: {stats['below_threshold_50']}")
            print("\n   Distribution:")
            for bucket, count in stats["distribution"].items():
                bar = "█" * int(count / max(stats["distribution"].values()) * 20) if stats["distribution"].values() else ""
                print(f"     {bucket}: {count:4d} {bar}")

    elif args.session:
        stats = score_session(args.session, update_file=args.update)
        if "error" in stats:
            print(f"❌ {stats['error']}")
        else:
            print(f"\n✅ Session: {stats['session_id'][:50]}")
            print(f"   Findings scored: {stats['findings_scored']}")
            print(f"   Average confidence: {stats['avg_confidence']:.3f}")
            print(f"   Min/Max: {stats['min_confidence']:.3f} / {stats['max_confidence']:.3f}")
            print(f"   Below threshold (0.7): {stats['below_threshold']}")
            if args.update:
                print(f"   Updated: {stats.get('output_file', 'N/A')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
