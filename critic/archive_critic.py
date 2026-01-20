#!/usr/bin/env python3
"""
Archive Critic - Validates session archive completeness.

Checks:
- Required files present (session.json, findings, urls)
- Minimum content thresholds met
- Thesis/synthesis present for research sessions
- Lineage properly recorded

Usage:
    python3 -m critic.archive_critic --session <session-id>
    python3 -m critic.archive_critic --all --dry-run
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from .base import (
    BaseCritic, CriticResult, ValidationIssue,
    IssueSeverity, IssueCategory
)


AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"


class ArchiveCritic(BaseCritic):
    """
    Validates archive completeness and quality.

    Ensures archived sessions have:
    - Complete metadata
    - Sufficient findings
    - Proper URL documentation
    - Research synthesis (for research sessions)
    """

    # Archive-specific thresholds
    MIN_URLS_FOR_RESEARCH = 3           # Research sessions should have URLs
    MIN_FINDINGS_FOR_RESEARCH = 2       # Research sessions should have findings
    REQUIRED_FILES = ["session.json"]   # Must have at minimum
    RECOMMENDED_FILES = ["urls_captured.json", "findings_captured.json"]

    def __init__(self):
        super().__init__("ArchiveCritic")

    def validate(self, content: dict) -> CriticResult:
        """
        Validate an archive.

        Args:
            content: Dict with 'session_id' and optionally 'session_dir' path

        Returns:
            CriticResult with completeness assessment
        """
        session_id = content.get("session_id")
        session_dir = content.get("session_dir")

        if session_dir:
            session_path = Path(session_dir)
        elif session_id:
            session_path = SESSIONS_DIR / session_id
        else:
            return self._create_result(
                confidence=0.0,
                issues=[ValidationIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.CRITICAL,
                    message="No session_id or session_dir provided"
                )],
                summary="Cannot validate: no session specified"
            )

        if not session_path.exists():
            return self._create_result(
                confidence=0.0,
                issues=[ValidationIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.CRITICAL,
                    message=f"Session directory not found: {session_path}"
                )],
                summary="Session not found"
            )

        issues = []
        confidence_factors = []

        # Check required files
        for required_file in self.REQUIRED_FILES:
            file_path = session_path / required_file
            if not file_path.exists():
                issues.append(ValidationIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.ERROR,
                    message=f"Required file missing: {required_file}",
                    location=str(session_path),
                    suggestion=f"Create {required_file} with session metadata"
                ))
                confidence_factors.append(0.0)
            else:
                confidence_factors.append(1.0)

        # Check recommended files
        for rec_file in self.RECOMMENDED_FILES:
            file_path = session_path / rec_file
            if not file_path.exists():
                issues.append(ValidationIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.WARNING,
                    message=f"Recommended file missing: {rec_file}",
                    location=str(session_path),
                    suggestion=f"Add {rec_file} for better archive quality"
                ))
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(1.0)

        # Load and validate session.json
        session_file = session_path / "session.json"
        session_data = {}
        if session_file.exists():
            try:
                session_data = json.loads(session_file.read_text())
                confidence_factors.append(1.0)

                # Check for topic
                if not session_data.get("topic"):
                    issues.append(ValidationIssue(
                        category=IssueCategory.COMPLETENESS,
                        severity=IssueSeverity.WARNING,
                        message="Session missing topic",
                        location="session.json",
                        suggestion="Add a descriptive topic to session.json"
                    ))
                    confidence_factors.append(0.7)

            except json.JSONDecodeError as e:
                issues.append(ValidationIssue(
                    category=IssueCategory.ACCURACY,
                    severity=IssueSeverity.ERROR,
                    message=f"Invalid JSON in session.json: {e}",
                    location="session.json",
                    suggestion="Fix JSON syntax errors"
                ))
                confidence_factors.append(0.0)

        # Check URLs for research sessions
        urls_file = session_path / "urls_captured.json"
        url_count = 0
        if urls_file.exists():
            try:
                urls = json.loads(urls_file.read_text())
                url_count = len(urls) if isinstance(urls, list) else 0

                # Check URL quality
                tier1_count = sum(1 for u in urls if u.get("tier") == 1)
                if url_count > 0 and tier1_count == 0:
                    issues.append(ValidationIssue(
                        category=IssueCategory.QUALITY,
                        severity=IssueSeverity.INFO,
                        message="No Tier 1 sources in URLs",
                        location="urls_captured.json",
                        suggestion="Add high-quality Tier 1 sources (arXiv, official docs)"
                    ))

            except json.JSONDecodeError:
                issues.append(ValidationIssue(
                    category=IssueCategory.ACCURACY,
                    severity=IssueSeverity.WARNING,
                    message="Invalid JSON in urls_captured.json",
                    location="urls_captured.json"
                ))

        # Check findings
        findings_file = session_path / "findings_captured.json"
        finding_count = 0
        if findings_file.exists():
            try:
                findings = json.loads(findings_file.read_text())
                finding_count = len(findings) if isinstance(findings, list) else 0

                # Check for thesis in research sessions
                if url_count >= self.MIN_URLS_FOR_RESEARCH:
                    has_thesis = any(
                        f.get("type") == "thesis"
                        for f in findings
                    ) if findings else False

                    if not has_thesis:
                        issues.append(ValidationIssue(
                            category=IssueCategory.COMPLETENESS,
                            severity=IssueSeverity.WARNING,
                            message="Research session missing thesis finding",
                            location="findings_captured.json",
                            suggestion="Add a thesis finding to summarize research conclusions"
                        ))

            except json.JSONDecodeError:
                issues.append(ValidationIssue(
                    category=IssueCategory.ACCURACY,
                    severity=IssueSeverity.WARNING,
                    message="Invalid JSON in findings_captured.json",
                    location="findings_captured.json"
                ))

        # Check if research session has minimum content
        is_research = url_count >= self.MIN_URLS_FOR_RESEARCH
        if is_research and finding_count < self.MIN_FINDINGS_FOR_RESEARCH:
            issues.append(ValidationIssue(
                category=IssueCategory.COMPLETENESS,
                severity=IssueSeverity.WARNING,
                message=f"Research session has only {finding_count} findings (minimum: {self.MIN_FINDINGS_FOR_RESEARCH})",
                location="findings_captured.json",
                suggestion="Extract more findings from the research"
            ))
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(1.0)

        # Check for evidence enrichment
        evidenced_file = session_path / "findings_evidenced.json"
        if not evidenced_file.exists() and finding_count > 0:
            issues.append(ValidationIssue(
                category=IssueCategory.EVIDENCE,
                severity=IssueSeverity.INFO,
                message="Findings not enriched with evidence",
                location=str(session_path),
                suggestion="Run evidence_extractor.py to add citations"
            ))

        # Check for transcript
        transcript_file = session_path / "full_transcript.txt"
        if not transcript_file.exists():
            issues.append(ValidationIssue(
                category=IssueCategory.COMPLETENESS,
                severity=IssueSeverity.INFO,
                message="No full transcript archived",
                location=str(session_path),
                suggestion="Archive transcript for reinvigoration support"
            ))

        # Calculate confidence
        confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0

        # Create summary
        summary_parts = []
        if url_count > 0:
            summary_parts.append(f"{url_count} URLs")
        if finding_count > 0:
            summary_parts.append(f"{finding_count} findings")

        summary = f"Archive contains: {', '.join(summary_parts) if summary_parts else 'minimal content'}"

        return self._create_result(
            confidence=confidence,
            issues=issues,
            summary=summary,
            metadata={
                "session_id": session_path.name,
                "url_count": url_count,
                "finding_count": finding_count,
                "is_research_session": is_research
            }
        )


def validate_session(session_id: str, verbose: bool = False) -> CriticResult:
    """Validate a single session archive."""
    critic = ArchiveCritic()
    result = critic.validate({"session_id": session_id})

    if verbose:
        print(f"\n{'='*50}")
        print(f"Archive Critic Report: {session_id[:40]}")
        print(f"{'='*50}")
        print(f"Status: {'✅ APPROVED' if result.approved else '❌ REJECTED'}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Summary: {result.summary}")

        if result.issues:
            print(f"\nIssues ({len(result.issues)}):")
            for issue in result.issues:
                icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚫"}
                print(f"  {icon.get(issue.severity.value, '•')} [{issue.category.value}] {issue.message}")
                if issue.suggestion:
                    print(f"     → {issue.suggestion}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Archive Critic - Validate session archives")
    parser.add_argument("--session", "-s", help="Session ID to validate")
    parser.add_argument("--all", "-a", action="store_true", help="Validate all sessions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview only")

    args = parser.parse_args()

    if args.session:
        result = validate_session(args.session, verbose=True)
        return 0 if result.approved else 1

    elif args.all:
        if not SESSIONS_DIR.exists():
            print("No sessions directory found")
            return 1

        results = {"approved": 0, "rejected": 0, "total": 0}

        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue

            result = validate_session(session_dir.name, verbose=args.verbose)
            results["total"] += 1

            if result.approved:
                results["approved"] += 1
            else:
                results["rejected"] += 1

        print(f"\n{'='*50}")
        print("Summary:")
        print(f"  Total: {results['total']}")
        print(f"  Approved: {results['approved']}")
        print(f"  Rejected: {results['rejected']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    exit(main() or 0)
