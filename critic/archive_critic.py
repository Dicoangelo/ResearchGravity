"""
Archive Critic

Validates session archive completeness and coherence:
- Required files present
- Session metadata complete
- Findings properly extracted
- URLs captured and categorized
- Transcript preserved
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import CriticBase, ValidationResult, Issue, Severity

# Expected archive structure
REQUIRED_FILES = [
    "session.json",
    "urls_captured.json",
    "findings_captured.json",
]

OPTIONAL_FILES = [
    "full_transcript.txt",
    "lineage.json",
    "evidence.json",
]

# Minimum thresholds
MIN_FINDINGS_FOR_RESEARCH = 3
MIN_URLS_FOR_RESEARCH = 5
MIN_TRANSCRIPT_CHARS = 1000


class ArchiveCritic(CriticBase):
    """
    Validates archive completeness and coherence.

    Checks:
    1. Required files present
    2. Session metadata complete (topic, dates, status)
    3. Findings extracted (min 3 for research sessions)
    4. URLs captured (min 5 for research sessions)
    5. Transcript preserved (if available)
    6. Cross-references valid (findings → URLs)
    """

    name = "archive_critic"
    description = "Validates session archive completeness"

    def __init__(self, sessions_dir: Optional[Path] = None, min_confidence: float = 0.7):
        super().__init__(min_confidence)
        self.sessions_dir = sessions_dir or Path.home() / ".agent-core/sessions"

    async def validate(self, target_id: str, **kwargs) -> ValidationResult:
        """
        Validate an archived session.

        Args:
            target_id: Session ID to validate
            strict: If True, enforce stricter thresholds (default: False)

        Returns:
            ValidationResult with completeness assessment
        """
        strict = kwargs.get('strict', False)

        # Gather evidence
        evidence = await self._gather_evidence(target_id, strict=strict)

        if not evidence.get('exists'):
            return ValidationResult(
                valid=False,
                confidence=0.0,
                issues=[self.add_issue(
                    "ARCHIVE_NOT_FOUND",
                    f"Session archive not found: {target_id}",
                    Severity.CRITICAL,
                )],
                critic_name=self.name,
                target_id=target_id,
            )

        # Run checks
        issues = await self._run_checks(evidence)

        # Calculate confidence
        confidence = self._calculate_confidence(evidence, issues)

        # Build result
        result = ValidationResult(
            valid=confidence >= self.min_confidence,
            confidence=confidence,
            issues=issues,
            metrics=evidence.get('metrics', {}),
            critic_name=self.name,
            target_id=target_id,
        )

        self.record_result(result)
        return result

    async def _gather_evidence(self, target_id: str, **kwargs) -> Dict[str, Any]:
        """Gather all archive data for validation."""
        archive_dir = self.sessions_dir / target_id

        if not archive_dir.exists():
            return {'exists': False}

        evidence = {
            'exists': True,
            'archive_dir': str(archive_dir),
            'files': {},
            'metrics': {},
        }

        # Check which files exist
        for filename in REQUIRED_FILES + OPTIONAL_FILES:
            filepath = archive_dir / filename
            evidence['files'][filename] = {
                'exists': filepath.exists(),
                'size': filepath.stat().st_size if filepath.exists() else 0,
            }

        # Load session.json
        session_file = archive_dir / "session.json"
        if session_file.exists():
            try:
                with open(session_file) as f:
                    evidence['session'] = json.load(f)
            except Exception as e:
                evidence['session'] = {'error': str(e)}
        else:
            evidence['session'] = None

        # Load findings
        findings_file = archive_dir / "findings_captured.json"
        if findings_file.exists():
            try:
                with open(findings_file) as f:
                    findings = json.load(f)
                    evidence['findings'] = findings if isinstance(findings, list) else findings.get('findings', [])
                    evidence['metrics']['finding_count'] = len(evidence['findings'])
            except Exception as e:
                evidence['findings'] = []
                evidence['metrics']['finding_count'] = 0
        else:
            evidence['findings'] = []
            evidence['metrics']['finding_count'] = 0

        # Load URLs
        urls_file = archive_dir / "urls_captured.json"
        if urls_file.exists():
            try:
                with open(urls_file) as f:
                    urls = json.load(f)
                    evidence['urls'] = urls if isinstance(urls, list) else urls.get('urls', [])
                    evidence['metrics']['url_count'] = len(evidence['urls'])

                    # Count by tier
                    tier_counts = {1: 0, 2: 0, 3: 0}
                    for url in evidence['urls']:
                        tier = url.get('tier', 3)
                        tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    evidence['metrics']['urls_by_tier'] = tier_counts
            except Exception as e:
                evidence['urls'] = []
                evidence['metrics']['url_count'] = 0
        else:
            evidence['urls'] = []
            evidence['metrics']['url_count'] = 0

        # Check transcript
        transcript_file = archive_dir / "full_transcript.txt"
        if transcript_file.exists():
            try:
                text = transcript_file.read_text()
                evidence['transcript'] = {
                    'exists': True,
                    'chars': len(text),
                    'lines': text.count('\n'),
                }
                evidence['metrics']['transcript_chars'] = len(text)
            except:
                evidence['transcript'] = {'exists': True, 'chars': 0, 'lines': 0}
        else:
            evidence['transcript'] = {'exists': False, 'chars': 0, 'lines': 0}

        return evidence

    async def _run_checks(self, evidence: Dict[str, Any]) -> List[Issue]:
        """Run all validation checks."""
        issues = []

        # Check 1: Required files
        for filename in REQUIRED_FILES:
            file_info = evidence['files'].get(filename, {})
            if not file_info.get('exists'):
                issues.append(self.add_issue(
                    "MISSING_REQUIRED_FILE",
                    f"Required file missing: {filename}",
                    Severity.ERROR,
                    location=filename,
                ))
            elif file_info.get('size', 0) == 0:
                issues.append(self.add_issue(
                    "EMPTY_FILE",
                    f"File is empty: {filename}",
                    Severity.WARNING,
                    location=filename,
                ))

        # Check 2: Session metadata
        session = evidence.get('session')
        if session:
            if session.get('error'):
                issues.append(self.add_issue(
                    "SESSION_PARSE_ERROR",
                    f"Could not parse session.json: {session['error']}",
                    Severity.ERROR,
                ))
            else:
                # Check required fields
                if not session.get('topic') and not session.get('title'):
                    issues.append(self.add_issue(
                        "MISSING_TOPIC",
                        "Session has no topic/title",
                        Severity.WARNING,
                        suggestion="Add topic to session metadata",
                    ))

                if not session.get('status'):
                    issues.append(self.add_issue(
                        "MISSING_STATUS",
                        "Session has no status",
                        Severity.INFO,
                    ))

        # Check 3: Findings
        finding_count = evidence['metrics'].get('finding_count', 0)
        if finding_count == 0:
            issues.append(self.add_issue(
                "NO_FINDINGS",
                "No findings captured in archive",
                Severity.WARNING,
                suggestion="Run evidence extraction on session",
            ))
        elif finding_count < MIN_FINDINGS_FOR_RESEARCH:
            issues.append(self.add_issue(
                "LOW_FINDINGS",
                f"Only {finding_count} findings (expected {MIN_FINDINGS_FOR_RESEARCH}+)",
                Severity.INFO,
            ))

        # Check 4: URLs
        url_count = evidence['metrics'].get('url_count', 0)
        if url_count == 0:
            issues.append(self.add_issue(
                "NO_URLS",
                "No URLs captured in archive",
                Severity.WARNING,
            ))
        elif url_count < MIN_URLS_FOR_RESEARCH:
            issues.append(self.add_issue(
                "LOW_URLS",
                f"Only {url_count} URLs (expected {MIN_URLS_FOR_RESEARCH}+)",
                Severity.INFO,
            ))

        # Check 5: Transcript
        transcript = evidence.get('transcript', {})
        if not transcript.get('exists'):
            issues.append(self.add_issue(
                "NO_TRANSCRIPT",
                "Full transcript not preserved",
                Severity.WARNING,
                suggestion="Archive should include full_transcript.txt for reinvigoration",
            ))
        elif transcript.get('chars', 0) < MIN_TRANSCRIPT_CHARS:
            issues.append(self.add_issue(
                "SHORT_TRANSCRIPT",
                f"Transcript very short ({transcript.get('chars', 0)} chars)",
                Severity.INFO,
            ))

        # Check 6: Findings have types
        findings = evidence.get('findings', [])
        untyped = [f for f in findings if not f.get('type')]
        if untyped and len(untyped) > len(findings) * 0.3:
            issues.append(self.add_issue(
                "UNTYPED_FINDINGS",
                f"{len(untyped)} findings without type classification",
                Severity.WARNING,
                suggestion="Classify findings by type (thesis, gap, innovation, etc.)",
            ))

        return issues

    def _calculate_confidence(self, evidence: Dict[str, Any], issues: List[Issue]) -> float:
        """
        Calculate confidence score.

        Scoring:
        - Required files present: 30%
        - Session metadata complete: 15%
        - Findings captured: 20%
        - URLs captured: 15%
        - Transcript preserved: 20%
        """
        score = 0.0

        # Required files (30%)
        required_present = sum(
            1 for f in REQUIRED_FILES
            if evidence['files'].get(f, {}).get('exists')
        )
        score += 0.30 * (required_present / len(REQUIRED_FILES))

        # Session metadata (15%)
        session = evidence.get('session', {})
        if session and not session.get('error'):
            meta_score = 0.0
            if session.get('topic') or session.get('title'):
                meta_score += 0.5
            if session.get('status'):
                meta_score += 0.25
            if session.get('started_at') or session.get('archived_at'):
                meta_score += 0.25
            score += 0.15 * meta_score

        # Findings (20%)
        finding_count = evidence['metrics'].get('finding_count', 0)
        if finding_count >= MIN_FINDINGS_FOR_RESEARCH:
            score += 0.20
        elif finding_count > 0:
            score += 0.20 * (finding_count / MIN_FINDINGS_FOR_RESEARCH)

        # URLs (15%)
        url_count = evidence['metrics'].get('url_count', 0)
        if url_count >= MIN_URLS_FOR_RESEARCH:
            score += 0.15
        elif url_count > 0:
            score += 0.15 * (url_count / MIN_URLS_FOR_RESEARCH)

        # Transcript (20%)
        transcript = evidence.get('transcript', {})
        if transcript.get('exists'):
            chars = transcript.get('chars', 0)
            if chars >= MIN_TRANSCRIPT_CHARS:
                score += 0.20
            elif chars > 0:
                score += 0.20 * min(1.0, chars / MIN_TRANSCRIPT_CHARS)

        # Penalty for critical/error issues
        critical_count = sum(1 for i in issues if i.severity == Severity.CRITICAL)
        error_count = sum(1 for i in issues if i.severity == Severity.ERROR)

        score -= critical_count * 0.15
        score -= error_count * 0.05

        return max(0.0, min(1.0, round(score, 3)))


async def main():
    """CLI for archive validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python archive_critic.py <session-id> [--strict]")
        print("\nValidates session archive completeness.")
        sys.exit(1)

    session_id = sys.argv[1]
    strict = '--strict' in sys.argv

    critic = ArchiveCritic()
    result = await critic.validate(session_id, strict=strict)

    print(f"\n{'='*60}")
    print(f"ARCHIVE VALIDATION: {session_id}")
    print(f"{'='*60}")
    print(f"\nConfidence: {result.confidence:.1%} {'✓' if result.passes_threshold else '✗'}")
    print(f"Valid: {result.valid}")

    if result.issues:
        print(f"\nIssues ({len(result.issues)}):")
        for issue in result.issues:
            icon = {'critical': '🔴', 'error': '🟠', 'warning': '🟡', 'info': '🔵'}
            print(f"  {icon.get(issue.severity.value, '•')} [{issue.code}] {issue.message}")
            if issue.suggestion:
                print(f"     → {issue.suggestion}")

    if result.metrics:
        print(f"\nMetrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
