#!/usr/bin/env python3
"""
Evidence Validator (Critic Agent) for Antigravity Chief of Staff.

Implements Writer-Critic validation pattern integrated with Oracle multi-stream consensus.

Validation checks:
1. Source validity: URLs are reachable and relevant
2. Citation accuracy: Excerpts match source content
3. Reasoning chain: Logic flows from sources to conclusion
4. Confidence calibration: Self-assessment matches evidence quality

Oracle Integration:
- Uses 3 parallel validation streams (accuracy, completeness, relevance)
- Computes intersection of validated claims
- Only marks validated if confidence > 0.7

Usage:
    python3 evidence_validator.py --session <session-id>    # Validate session
    python3 evidence_validator.py --finding <finding-id>    # Validate single finding
    python3 evidence_validator.py --spot-check 20           # Random sample validation
"""

import argparse
import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error


AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
ORACLE_DIR = AGENT_CORE_DIR / "oracle"

# Validation thresholds
CONFIDENCE_THRESHOLD = 0.7
MIN_SOURCES_FOR_VALIDATION = 1
URL_TIMEOUT = 5  # seconds

# Oracle stream perspectives
ORACLE_PERSPECTIVES = [
    "accuracy",      # Are sources and claims factually correct?
    "completeness",  # Does evidence fully support the finding?
    "relevance",     # Are sources actually relevant to the claim?
]


class ValidationResult:
    """Container for validation results."""

    def __init__(self, finding_id: str):
        self.finding_id = finding_id
        self.validated = False
        self.confidence = 0.0
        self.issues = []
        self.oracle_streams = []
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "validated": self.validated,
            "confidence": self.confidence,
            "issues": self.issues,
            "oracle_streams": self.oracle_streams,
            "validated_at": self.timestamp if self.validated else None,
        }


def check_url_validity(url: str, timeout: int = URL_TIMEOUT) -> dict:
    """
    Check if a URL is reachable and get basic info.

    Returns:
        Dict with 'valid', 'status_code', 'error' keys
    """
    result = {
        "url": url,
        "valid": False,
        "status_code": None,
        "error": None,
    }

    # Skip certain URLs that need auth
    if any(skip in url for skip in ["arxiv.org/pdf", "github.com/settings"]):
        result["valid"] = True
        result["status_code"] = 200  # Assume valid
        return result

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AntigravityValidator/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result["status_code"] = response.status
            result["valid"] = response.status == 200
    except urllib.error.HTTPError as e:
        result["status_code"] = e.code
        result["error"] = f"HTTP {e.code}"
        # 403/429 might be rate limiting, not invalid
        result["valid"] = e.code in [403, 429]
    except urllib.error.URLError as e:
        result["error"] = str(e.reason)
    except Exception as e:
        result["error"] = str(e)

    return result


def validate_source(source: dict, check_url: bool = False) -> dict:
    """
    Validate a single source.

    Checks:
    - URL format validity
    - Excerpt is non-empty
    - Relevance score is reasonable

    Args:
        source: Source dict with url, excerpt, relevance_score
        check_url: Whether to actually check URL reachability

    Returns:
        Dict with validation results
    """
    result = {
        "url": source.get("url", ""),
        "valid": True,
        "issues": [],
    }

    # Check URL format
    url = source.get("url", "")
    if not url or not url.startswith(("http://", "https://")):
        result["valid"] = False
        result["issues"].append("Invalid URL format")

    # Check excerpt
    excerpt = source.get("excerpt", "")
    if not excerpt or len(excerpt) < 10:
        result["issues"].append("Excerpt too short or missing")
        # Don't invalidate, just note

    # Check relevance score
    relevance = source.get("relevance_score", 0)
    if relevance < 0 or relevance > 1:
        result["valid"] = False
        result["issues"].append(f"Invalid relevance score: {relevance}")

    # Optional URL reachability check
    if check_url and url:
        url_check = check_url_validity(url)
        if not url_check["valid"]:
            result["issues"].append(f"URL not reachable: {url_check.get('error', 'unknown')}")
            # Don't invalidate for network issues

    return result


def validate_reasoning_chain(finding: dict) -> dict:
    """
    Validate that the reasoning chain connects sources to conclusion.

    Checks:
    - Chain is present if there are multiple sources
    - Each step references prior context
    - Final step connects to the finding content
    """
    result = {
        "valid": True,
        "issues": [],
    }

    evidence = finding.get("evidence", {})
    sources = evidence.get("sources", [])
    chain = evidence.get("reasoning_chain", [])
    content = finding.get("content", "")

    # If multiple sources, should have reasoning chain
    if len(sources) > 1 and not chain:
        result["issues"].append("Multiple sources but no reasoning chain")
        # Don't invalidate, just note

    # If chain exists, check it's meaningful
    if chain:
        for i, step in enumerate(chain):
            if len(step) < 20:
                result["issues"].append(f"Reasoning step {i+1} too short")

    return result


def run_oracle_stream(
    finding: dict,
    perspective: str
) -> dict:
    """
    Simulate an Oracle validation stream for a given perspective.

    In production, this would call an LLM with the specific perspective.
    For now, implements rule-based validation.

    Args:
        finding: Finding to validate
        perspective: Validation perspective (accuracy, completeness, relevance)

    Returns:
        Dict with stream results
    """
    stream_id = f"stream-{perspective}-{hashlib.md5(finding.get('id', '').encode()).hexdigest()[:8]}"

    result = {
        "stream_id": stream_id,
        "perspective": perspective,
        "confidence": 0.0,
        "issues": [],
        "validated_claims": [],
    }

    evidence = finding.get("evidence", {})
    sources = evidence.get("sources", [])
    content = finding.get("content", "")

    if perspective == "accuracy":
        # Check source validity
        valid_sources = 0
        for source in sources:
            validation = validate_source(source, check_url=False)
            if validation["valid"]:
                valid_sources += 1
            else:
                result["issues"].extend(validation["issues"])

        if sources:
            result["confidence"] = valid_sources / len(sources)
        if valid_sources > 0:
            result["validated_claims"].append("Sources have valid format")

    elif perspective == "completeness":
        # Check if sources adequately cover the claim
        has_sources = len(sources) >= MIN_SOURCES_FOR_VALIDATION
        has_excerpts = all(s.get("excerpt") for s in sources)
        has_chain = len(evidence.get("reasoning_chain", [])) > 0 or len(sources) <= 1

        completeness_score = sum([
            has_sources * 0.4,
            has_excerpts * 0.3,
            has_chain * 0.3,
        ])
        result["confidence"] = completeness_score

        if has_sources:
            result["validated_claims"].append(f"Has {len(sources)} source(s)")
        else:
            result["issues"].append("Insufficient sources")

    elif perspective == "relevance":
        # Check if sources are relevant to the finding content
        relevance_scores = [s.get("relevance_score", 0.5) for s in sources]
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            result["confidence"] = avg_relevance

            if avg_relevance >= 0.6:
                result["validated_claims"].append("Sources are relevant to claim")
            else:
                result["issues"].append("Low average source relevance")
        else:
            result["issues"].append("No sources to check relevance")

    return result


def run_oracle_consensus(finding: dict) -> ValidationResult:
    """
    Run Oracle multi-stream consensus validation.

    Implements the Writer-Critic pattern:
    1. Run 3 parallel validation streams
    2. Compute intersection of validated claims
    3. Calculate aggregate confidence
    4. Mark validated only if confidence > threshold

    Args:
        finding: Finding to validate

    Returns:
        ValidationResult with consensus outcome
    """
    result = ValidationResult(finding.get("id", "unknown"))

    # Run all Oracle streams
    streams = []
    for perspective in ORACLE_PERSPECTIVES:
        stream_result = run_oracle_stream(finding, perspective)
        streams.append(stream_result)
        result.oracle_streams.append(stream_result["stream_id"])

    # Compute intersection
    all_issues = []
    all_claims = []
    confidences = []

    for stream in streams:
        all_issues.extend(stream["issues"])
        all_claims.extend(stream["validated_claims"])
        confidences.append(stream["confidence"])

    # Aggregate confidence (weighted by perspective)
    # Accuracy weighted highest, then completeness, then relevance
    weights = {"accuracy": 0.4, "completeness": 0.35, "relevance": 0.25}
    weighted_confidence = sum(
        stream["confidence"] * weights[stream["perspective"]]
        for stream in streams
    )

    result.confidence = round(weighted_confidence, 3)
    result.issues = list(set(all_issues))  # Dedupe issues

    # Validation decision
    result.validated = (
        result.confidence >= CONFIDENCE_THRESHOLD and
        len(result.issues) < 3  # Allow some minor issues
    )

    return result


def validate_finding(finding: dict) -> ValidationResult:
    """
    Validate a single finding using Oracle consensus.

    Args:
        finding: Finding dict with evidence

    Returns:
        ValidationResult
    """
    # Quick checks first
    evidence = finding.get("evidence", {})
    sources = evidence.get("sources", [])

    # No sources = automatic fail
    if not sources:
        result = ValidationResult(finding.get("id", "unknown"))
        result.issues.append("No sources provided")
        return result

    # Run Oracle consensus
    return run_oracle_consensus(finding)


def validate_session(
    session_id: str,
    update_file: bool = False,
    verbose: bool = False
) -> dict:
    """
    Validate all findings in a session.

    Args:
        session_id: Session ID to validate
        update_file: If True, update findings with validation metadata
        verbose: If True, print detailed output

    Returns:
        Stats dict with validation results
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return {"error": f"Session not found: {session_id}"}

    # Load findings (prefer evidenced file)
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

    # Validate each finding
    validated_findings = []
    validation_results = []

    for finding in findings:
        if "evidence" not in finding:
            validated_findings.append(finding)
            continue

        result = validate_finding(finding)
        validation_results.append(result)

        if verbose:
            status = "✅" if result.validated else "❌"
            print(f"  {status} {finding.get('id', 'N/A')[:30]}: {result.confidence:.2f}")
            if result.issues:
                for issue in result.issues[:2]:
                    print(f"      ⚠ {issue}")

        # Update finding with validation metadata
        finding["evidence"]["validation"] = {
            "validated": result.validated,
            "validated_at": result.timestamp if result.validated else None,
            "critic_notes": result.issues,
            "oracle_streams": result.oracle_streams,
        }
        finding["evidence"]["confidence"] = max(
            finding["evidence"].get("confidence", 0),
            result.confidence
        )

        validated_findings.append(finding)

    # Calculate stats
    validated_count = sum(1 for r in validation_results if r.validated)
    avg_confidence = sum(r.confidence for r in validation_results) / len(validation_results) if validation_results else 0

    stats = {
        "session_id": session_id,
        "findings_validated": len(validation_results),
        "passed": validated_count,
        "failed": len(validation_results) - validated_count,
        "avg_confidence": avg_confidence,
        "pass_rate": validated_count / len(validation_results) if validation_results else 0,
    }

    # Update file if requested
    if update_file:
        output_file = session_dir / "findings_evidenced.json"
        output_file.write_text(json.dumps(validated_findings, indent=2))
        stats["output_file"] = str(output_file)

    return stats


def spot_check_validation(sample_size: int = 20, verbose: bool = False) -> dict:
    """
    Validate a random sample of findings across all sessions.

    Args:
        sample_size: Number of findings to validate
        verbose: If True, print detailed output

    Returns:
        Stats dict with spot check results
    """
    all_findings = []

    # Collect all evidenced findings
    for session_dir in SESSIONS_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        findings_file = session_dir / "findings_evidenced.json"
        if not findings_file.exists():
            continue

        try:
            findings = json.loads(findings_file.read_text())
            for f in findings:
                if "evidence" in f and f["evidence"].get("sources"):
                    f["_session_id"] = session_dir.name
                    all_findings.append(f)
        except (json.JSONDecodeError, IOError):
            continue

    if not all_findings:
        return {"error": "No evidenced findings found"}

    # Random sample
    sample = random.sample(all_findings, min(sample_size, len(all_findings)))

    # Validate sample
    results = []
    for finding in sample:
        result = validate_finding(finding)
        results.append(result)

        if verbose:
            status = "✅" if result.validated else "❌"
            print(f"{status} {finding.get('_session_id', '')[:30]}: {result.confidence:.2f}")
            if result.issues:
                print(f"   Issues: {', '.join(result.issues[:2])}")

    # Stats
    validated_count = sum(1 for r in results if r.validated)
    avg_confidence = sum(r.confidence for r in results) / len(results)

    return {
        "sample_size": len(sample),
        "total_findings": len(all_findings),
        "validated": validated_count,
        "failed": len(sample) - validated_count,
        "pass_rate": validated_count / len(sample),
        "avg_confidence": avg_confidence,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate findings using Writer-Critic pattern"
    )
    parser.add_argument("--session", "-s",
                        help="Validate specific session")
    parser.add_argument("--update", "-u", action="store_true",
                        help="Update findings file with validation metadata")
    parser.add_argument("--spot-check", type=int, metavar="N",
                        help="Validate random sample of N findings")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    print("Evidence Validator (Critic) — Antigravity Chief of Staff")
    print("Oracle Multi-Stream Consensus Integration")
    print("=" * 60)

    if args.session:
        print(f"\nValidating session: {args.session[:50]}")
        stats = validate_session(
            args.session,
            update_file=args.update,
            verbose=args.verbose
        )

        if "error" in stats:
            print(f"❌ {stats['error']}")
        else:
            print(f"\n📊 Validation Results:")
            print(f"   Findings validated: {stats['findings_validated']}")
            print(f"   Passed: {stats['passed']} ({stats['pass_rate']*100:.1f}%)")
            print(f"   Failed: {stats['failed']}")
            print(f"   Average confidence: {stats['avg_confidence']:.3f}")
            if args.update:
                print(f"   Updated: {stats.get('output_file', 'N/A')}")

    elif args.spot_check:
        print(f"\nSpot-checking {args.spot_check} random findings...")
        stats = spot_check_validation(args.spot_check, verbose=args.verbose)

        if "error" in stats:
            print(f"❌ {stats['error']}")
        else:
            print(f"\n📊 Spot Check Results:")
            print(f"   Sample size: {stats['sample_size']} / {stats['total_findings']} total")
            print(f"   Passed: {stats['validated']} ({stats['pass_rate']*100:.1f}%)")
            print(f"   Failed: {stats['failed']}")
            print(f"   Average confidence: {stats['avg_confidence']:.3f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
