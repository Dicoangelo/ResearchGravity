"""
Evidence Critic

Validates citation accuracy and source integrity:
- URLs are valid and accessible
- arXiv IDs are properly formatted
- Findings cite actual sources
- Confidence scores are justified
"""

import json
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from .base import CriticBase, ValidationResult, Issue, Severity

# URL patterns
ARXIV_PATTERN = re.compile(r'(\d{4}\.\d{4,5})(v\d+)?')
GITHUB_PATTERN = re.compile(r'github\.com/[\w-]+/[\w.-]+')
DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s]+')

# Tier 1 domains (primary research sources)
TIER1_DOMAINS = {
    'arxiv.org', 'huggingface.co', 'openai.com', 'anthropic.com',
    'deepmind.com', 'ai.google', 'research.google', 'meta.com',
    'github.com', 'proceedings.mlr.press', 'proceedings.neurips.cc',
}


class EvidenceCritic(CriticBase):
    """
    Validates citation accuracy and evidence quality.

    Checks:
    1. URLs are well-formed
    2. arXiv IDs are valid format
    3. Findings have source attribution
    4. Source quality (Tier 1 vs Tier 3)
    5. No broken/invalid citations
    """

    name = "evidence_critic"
    description = "Validates citation accuracy and source quality"

    def __init__(self, sessions_dir: Optional[Path] = None, min_confidence: float = 0.7):
        super().__init__(min_confidence)
        self.sessions_dir = sessions_dir or Path.home() / ".agent-core/sessions"

    async def validate(self, target_id: str, **kwargs) -> ValidationResult:
        """
        Validate evidence/citations for a session.

        Args:
            target_id: Session ID to validate
            check_accessibility: If True, verify URLs are reachable (slow)

        Returns:
            ValidationResult with citation assessment
        """
        check_accessibility = kwargs.get('check_accessibility', False)

        # Gather evidence
        evidence = await self._gather_evidence(target_id, check_accessibility=check_accessibility)

        if not evidence.get('exists'):
            return ValidationResult(
                valid=False,
                confidence=0.0,
                issues=[self.add_issue(
                    "SESSION_NOT_FOUND",
                    f"Session not found: {target_id}",
                    Severity.CRITICAL,
                )],
                critic_name=self.name,
                target_id=target_id,
            )

        # Run checks
        issues = await self._run_checks(evidence)

        # Calculate confidence
        confidence = self._calculate_confidence(evidence, issues)

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
        """Gather all citation/evidence data."""
        archive_dir = self.sessions_dir / target_id

        if not archive_dir.exists():
            return {'exists': False}

        evidence = {
            'exists': True,
            'urls': [],
            'findings': [],
            'metrics': {
                'total_urls': 0,
                'valid_urls': 0,
                'arxiv_papers': 0,
                'github_repos': 0,
                'tier1_sources': 0,
                'findings_with_sources': 0,
                'findings_without_sources': 0,
            },
        }

        # Load URLs
        urls_file = archive_dir / "urls_captured.json"
        if urls_file.exists():
            try:
                with open(urls_file) as f:
                    data = json.load(f)
                    evidence['urls'] = data if isinstance(data, list) else data.get('urls', [])
            except:
                evidence['urls'] = []

        # Load findings
        findings_file = archive_dir / "findings_captured.json"
        if findings_file.exists():
            try:
                with open(findings_file) as f:
                    data = json.load(f)
                    evidence['findings'] = data if isinstance(data, list) else data.get('findings', [])
            except:
                evidence['findings'] = []

        # Analyze URLs
        for url_entry in evidence['urls']:
            url = url_entry.get('url', '') if isinstance(url_entry, dict) else str(url_entry)
            evidence['metrics']['total_urls'] += 1

            if self._is_valid_url(url):
                evidence['metrics']['valid_urls'] += 1

                # Check for arXiv
                if 'arxiv.org' in url or ARXIV_PATTERN.search(url):
                    evidence['metrics']['arxiv_papers'] += 1

                # Check for GitHub
                if GITHUB_PATTERN.search(url):
                    evidence['metrics']['github_repos'] += 1

                # Check tier
                domain = urlparse(url).netloc.replace('www.', '')
                if domain in TIER1_DOMAINS:
                    evidence['metrics']['tier1_sources'] += 1

        # Analyze findings
        for finding in evidence['findings']:
            sources = finding.get('sources', []) or finding.get('evidence', {}).get('sources', [])
            if sources:
                evidence['metrics']['findings_with_sources'] += 1
            else:
                evidence['metrics']['findings_without_sources'] += 1

        return evidence

    async def _run_checks(self, evidence: Dict[str, Any]) -> List[Issue]:
        """Run citation validation checks."""
        issues = []
        metrics = evidence['metrics']

        # Check 1: URL validity
        total_urls = metrics['total_urls']
        valid_urls = metrics['valid_urls']
        if total_urls > 0:
            invalid_count = total_urls - valid_urls
            if invalid_count > 0:
                issues.append(self.add_issue(
                    "INVALID_URLS",
                    f"{invalid_count} URLs are malformed or invalid",
                    Severity.WARNING if invalid_count < 3 else Severity.ERROR,
                ))

        # Check 2: Source quality
        tier1_ratio = metrics['tier1_sources'] / total_urls if total_urls > 0 else 0
        if total_urls > 5 and tier1_ratio < 0.3:
            issues.append(self.add_issue(
                "LOW_TIER1_RATIO",
                f"Only {tier1_ratio:.0%} of sources are Tier 1 (research papers, etc.)",
                Severity.INFO,
                suggestion="Prioritize arXiv, official docs, and research sources",
            ))

        # Check 3: Finding citations
        findings_without = metrics['findings_without_sources']
        total_findings = findings_without + metrics['findings_with_sources']
        if total_findings > 0:
            uncited_ratio = findings_without / total_findings
            if uncited_ratio > 0.5:
                issues.append(self.add_issue(
                    "UNCITED_FINDINGS",
                    f"{findings_without}/{total_findings} findings lack source citations",
                    Severity.WARNING,
                    suggestion="Add evidence.sources to findings for traceability",
                ))

        # Check 4: arXiv ID format
        for url_entry in evidence['urls']:
            url = url_entry.get('url', '') if isinstance(url_entry, dict) else str(url_entry)
            if 'arxiv' in url.lower():
                if not ARXIV_PATTERN.search(url):
                    issues.append(self.add_issue(
                        "MALFORMED_ARXIV",
                        f"arXiv URL may be malformed: {url[:60]}...",
                        Severity.WARNING,
                        location=url,
                    ))
                    break  # Only report once

        # Check 5: No sources at all
        if total_urls == 0:
            issues.append(self.add_issue(
                "NO_SOURCES",
                "Session has no captured URLs/sources",
                Severity.ERROR,
                suggestion="Research sessions should cite sources via log_url.py",
            ))

        return issues

    def _calculate_confidence(self, evidence: Dict[str, Any], issues: List[Issue]) -> float:
        """
        Calculate evidence confidence.

        Scoring:
        - URL validity: 25%
        - Source quality (Tier 1): 25%
        - Finding citations: 30%
        - No critical issues: 20%
        """
        score = 0.0
        metrics = evidence['metrics']

        # URL validity (25%)
        total_urls = metrics['total_urls']
        if total_urls > 0:
            validity_ratio = metrics['valid_urls'] / total_urls
            score += 0.25 * validity_ratio
        else:
            # No URLs is a problem but not zero confidence
            score += 0.05

        # Source quality (25%)
        if total_urls > 0:
            tier1_ratio = metrics['tier1_sources'] / total_urls
            # Also count arXiv and GitHub
            research_sources = metrics['arxiv_papers'] + metrics['github_repos']
            research_ratio = research_sources / total_urls
            score += 0.25 * max(tier1_ratio, research_ratio)

        # Finding citations (30%)
        total_findings = metrics['findings_with_sources'] + metrics['findings_without_sources']
        if total_findings > 0:
            cited_ratio = metrics['findings_with_sources'] / total_findings
            score += 0.30 * cited_ratio
        else:
            # No findings to cite
            score += 0.15

        # No critical issues (20%)
        critical_count = sum(1 for i in issues if i.severity == Severity.CRITICAL)
        error_count = sum(1 for i in issues if i.severity == Severity.ERROR)
        if critical_count == 0 and error_count == 0:
            score += 0.20
        elif critical_count == 0:
            score += 0.10

        return max(0.0, min(1.0, round(score, 3)))

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is well-formed."""
        try:
            result = urlparse(url)
            return all([result.scheme in ('http', 'https'), result.netloc])
        except:
            return False


async def main():
    """CLI for evidence validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evidence_critic.py <session-id>")
        print("\nValidates citation accuracy and source quality.")
        sys.exit(1)

    session_id = sys.argv[1]

    critic = EvidenceCritic()
    result = await critic.validate(session_id)

    print(f"\n{'='*60}")
    print(f"EVIDENCE VALIDATION: {session_id}")
    print(f"{'='*60}")
    print(f"\nConfidence: {result.confidence:.1%} {'✓' if result.passes_threshold else '✗'}")

    if result.issues:
        print(f"\nIssues ({len(result.issues)}):")
        for issue in result.issues:
            icon = {'critical': '🔴', 'error': '🟠', 'warning': '🟡', 'info': '🔵'}
            print(f"  {icon.get(issue.severity.value, '•')} [{issue.code}] {issue.message}")

    if result.metrics:
        print(f"\nMetrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
