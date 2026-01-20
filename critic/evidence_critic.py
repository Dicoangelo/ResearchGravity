#!/usr/bin/env python3
"""
Evidence Critic - Validates citation and source quality.

Checks:
- Evidence presence for findings
- Citation validity (URLs accessible)
- Source tier quality
- Confidence score accuracy

Usage:
    python3 -m critic.evidence_critic --session <session-id>
    python3 -m critic.evidence_critic --finding <finding-id>
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .base import (
    BaseCritic, CriticResult, ValidationIssue,
    IssueSeverity, IssueCategory
)


AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"

# Valid URL patterns
ARXIV_PATTERN = re.compile(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})')
GITHUB_PATTERN = re.compile(r'github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)')


class EvidenceCritic(BaseCritic):
    """
    Validates evidence quality for findings.

    Ensures findings have:
    - At least one source citation
    - Valid URLs
    - Appropriate confidence levels
    - High-quality Tier 1 sources when possible
    """

    # Evidence-specific thresholds
    MIN_SOURCES_FOR_CLAIM = 1           # Claims need at least 1 source
    MIN_CONFIDENCE_FOR_CLAIM = 0.3      # Minimum confidence for unsourced claims
    TIER1_DOMAINS = [
        "arxiv.org", "openai.com", "anthropic.com", "deepmind.google",
        "huggingface.co", "ai.meta.com", "ai.google"
    ]

    def __init__(self):
        super().__init__("EvidenceCritic")

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL has valid structure."""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False

    def _get_url_tier(self, url: str) -> int:
        """Determine source tier from URL."""
        url_lower = url.lower()

        for domain in self.TIER1_DOMAINS:
            if domain in url_lower:
                return 1

        if "github.com" in url_lower:
            return 2

        return 3

    def _validate_source(self, source: dict) -> list[ValidationIssue]:
        """Validate a single source citation."""
        issues = []

        url = source.get("url", "")

        # Check URL validity
        if not url:
            issues.append(ValidationIssue(
                category=IssueCategory.EVIDENCE,
                severity=IssueSeverity.ERROR,
                message="Source missing URL",
                suggestion="Add valid URL to source"
            ))
        elif not self._is_valid_url(url):
            issues.append(ValidationIssue(
                category=IssueCategory.EVIDENCE,
                severity=IssueSeverity.WARNING,
                message=f"Potentially invalid URL: {url[:50]}",
                suggestion="Verify URL is accessible"
            ))

        # Check for arXiv ID consistency
        if "arxiv.org" in url:
            match = ARXIV_PATTERN.search(url)
            if match:
                arxiv_id = match.group(1)
                stored_id = source.get("arxiv_id")
                if stored_id and stored_id != arxiv_id:
                    issues.append(ValidationIssue(
                        category=IssueCategory.CONSISTENCY,
                        severity=IssueSeverity.WARNING,
                        message=f"arXiv ID mismatch: URL has {arxiv_id}, stored is {stored_id}",
                        suggestion="Correct arXiv ID to match URL"
                    ))

        # Check relevance score
        relevance = source.get("relevance_score", 0)
        if relevance > 1.0 or relevance < 0.0:
            issues.append(ValidationIssue(
                category=IssueCategory.ACCURACY,
                severity=IssueSeverity.WARNING,
                message=f"Relevance score out of range: {relevance}",
                suggestion="Relevance should be 0.0-1.0"
            ))

        return issues

    def _validate_finding(self, finding: dict) -> tuple[list[ValidationIssue], float]:
        """Validate evidence for a single finding."""
        issues = []
        confidence_factors = []

        content = finding.get("content", "")
        finding_type = finding.get("type", "finding")
        evidence = finding.get("evidence", {})
        sources = evidence.get("sources", [])
        stated_confidence = evidence.get("confidence", 0.0)

        # Check for sources
        if not sources:
            if finding_type in ["thesis", "gap", "innovation"]:
                # High-value findings need sources
                issues.append(ValidationIssue(
                    category=IssueCategory.EVIDENCE,
                    severity=IssueSeverity.ERROR,
                    message=f"{finding_type.title()} finding has no sources",
                    location=finding.get("id", "unknown"),
                    suggestion="Add source citations for key findings"
                ))
                confidence_factors.append(0.3)
            else:
                issues.append(ValidationIssue(
                    category=IssueCategory.EVIDENCE,
                    severity=IssueSeverity.WARNING,
                    message="Finding has no sources",
                    location=finding.get("id", "unknown"),
                    suggestion="Add source citations if available"
                ))
                confidence_factors.append(0.5)
        else:
            # Validate each source
            tier1_count = 0
            for source in sources:
                source_issues = self._validate_source(source)
                issues.extend(source_issues)

                if not source_issues:
                    confidence_factors.append(1.0)
                    tier = self._get_url_tier(source.get("url", ""))
                    if tier == 1:
                        tier1_count += 1
                else:
                    confidence_factors.append(0.7)

            # Check for Tier 1 sources on important findings
            if finding_type in ["thesis", "gap"] and tier1_count == 0:
                issues.append(ValidationIssue(
                    category=IssueCategory.QUALITY,
                    severity=IssueSeverity.INFO,
                    message=f"{finding_type.title()} has no Tier 1 sources",
                    location=finding.get("id", "unknown"),
                    suggestion="Consider adding arXiv or official documentation sources"
                ))

        # Validate confidence score
        if sources:
            # Calculate expected confidence
            avg_relevance = sum(s.get("relevance_score", 0.5) for s in sources) / len(sources)

            # Check if stated confidence is reasonable
            if abs(stated_confidence - avg_relevance) > 0.3:
                issues.append(ValidationIssue(
                    category=IssueCategory.ACCURACY,
                    severity=IssueSeverity.INFO,
                    message=f"Confidence {stated_confidence:.2f} differs from source average {avg_relevance:.2f}",
                    location=finding.get("id", "unknown"),
                    suggestion="Review confidence calculation"
                ))

        # Calculate overall confidence
        calculated_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0

        return issues, calculated_confidence

    def validate(self, content: dict) -> CriticResult:
        """
        Validate evidence for findings.

        Args:
            content: Dict with 'findings' list or 'session_id'

        Returns:
            CriticResult with evidence quality assessment
        """
        findings = content.get("findings", [])
        session_id = content.get("session_id")

        # Load findings from session if needed
        if not findings and session_id:
            session_dir = SESSIONS_DIR / session_id
            evidenced_file = session_dir / "findings_evidenced.json"
            findings_file = session_dir / "findings_captured.json"

            if evidenced_file.exists():
                try:
                    findings = json.loads(evidenced_file.read_text())
                except json.JSONDecodeError:
                    pass
            elif findings_file.exists():
                try:
                    raw_findings = json.loads(findings_file.read_text())
                    # Convert to standard format
                    findings = [
                        {
                            "id": f"finding-{i}",
                            "content": f.get("text", ""),
                            "type": f.get("type", "finding"),
                            "evidence": {"sources": [], "confidence": 0.0}
                        }
                        for i, f in enumerate(raw_findings)
                    ]
                except json.JSONDecodeError:
                    pass

        if not findings:
            return self._create_result(
                confidence=0.0,
                issues=[ValidationIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.ERROR,
                    message="No findings to validate"
                )],
                summary="No findings provided"
            )

        all_issues = []
        confidence_scores = []
        findings_with_sources = 0
        total_sources = 0

        for finding in findings:
            issues, confidence = self._validate_finding(finding)
            all_issues.extend(issues)
            confidence_scores.append(confidence)

            sources = finding.get("evidence", {}).get("sources", [])
            if sources:
                findings_with_sources += 1
                total_sources += len(sources)

        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        # Summary
        coverage = findings_with_sources / len(findings) if findings else 0
        summary = (
            f"Evidence coverage: {coverage:.0%} ({findings_with_sources}/{len(findings)} findings), "
            f"{total_sources} total sources"
        )

        return self._create_result(
            confidence=overall_confidence,
            issues=all_issues,
            summary=summary,
            metadata={
                "session_id": session_id,
                "findings_count": len(findings),
                "findings_with_sources": findings_with_sources,
                "total_sources": total_sources,
                "coverage": coverage
            }
        )


def validate_evidence(session_id: str, verbose: bool = False) -> CriticResult:
    """Validate evidence for a session."""
    critic = EvidenceCritic()
    result = critic.validate({"session_id": session_id})

    if verbose:
        print(f"\n{'='*50}")
        print(f"Evidence Critic Report: {session_id[:40]}")
        print(f"{'='*50}")
        print(f"Status: {'✅ APPROVED' if result.approved else '❌ REJECTED'}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Summary: {result.summary}")

        meta = result.metadata
        print(f"\nMetrics:")
        print(f"  Findings: {meta.get('findings_count', 0)}")
        print(f"  With sources: {meta.get('findings_with_sources', 0)}")
        print(f"  Total sources: {meta.get('total_sources', 0)}")
        print(f"  Coverage: {meta.get('coverage', 0):.0%}")

        if result.issues:
            errors = [i for i in result.issues if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]]
            warnings = [i for i in result.issues if i.severity == IssueSeverity.WARNING]

            if errors:
                print(f"\nErrors ({len(errors)}):")
                for issue in errors[:5]:
                    print(f"  ❌ {issue.message}")

            if warnings:
                print(f"\nWarnings ({len(warnings)}):")
                for issue in warnings[:5]:
                    print(f"  ⚠️ {issue.message}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Evidence Critic - Validate source citations")
    parser.add_argument("--session", "-s", help="Session ID to validate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--stats", action="store_true", help="Show aggregate stats")

    args = parser.parse_args()

    if args.session:
        result = validate_evidence(args.session, verbose=True)
        return 0 if result.approved else 1

    elif args.stats:
        if not SESSIONS_DIR.exists():
            print("No sessions directory found")
            return 1

        total_findings = 0
        with_sources = 0
        total_sources = 0

        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue

            result = validate_evidence(session_dir.name, verbose=False)
            meta = result.metadata

            total_findings += meta.get("findings_count", 0)
            with_sources += meta.get("findings_with_sources", 0)
            total_sources += meta.get("total_sources", 0)

        print(f"Evidence Stats Across All Sessions:")
        print(f"  Total findings: {total_findings}")
        print(f"  With sources: {with_sources} ({with_sources/total_findings:.0%})" if total_findings else "  With sources: 0")
        print(f"  Total sources: {total_sources}")

    else:
        parser.print_help()


if __name__ == "__main__":
    exit(main() or 0)
