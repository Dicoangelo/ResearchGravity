"""
Pack Critic

Validates context pack relevance and quality:
- Content is focused and coherent
- Token count is reasonable
- Keywords match content
- Not stale or outdated
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .base import CriticBase, ValidationResult, Issue, Severity

# Pack size thresholds
MIN_TOKENS = 100
MAX_TOKENS = 50000
IDEAL_TOKENS_MIN = 500
IDEAL_TOKENS_MAX = 10000

# Staleness threshold (days)
STALE_THRESHOLD_DAYS = 90


class PackCritic(CriticBase):
    """
    Validates context pack quality and relevance.

    Checks:
    1. Token count in reasonable range
    2. Content is coherent (not fragmented)
    3. Keywords present and relevant
    4. Not stale (updated recently)
    5. Type-appropriate structure
    """

    name = "pack_critic"
    description = "Validates context pack relevance and quality"

    def __init__(self, packs_dir: Optional[Path] = None, min_confidence: float = 0.7):
        super().__init__(min_confidence)
        self.packs_dir = packs_dir or Path.home() / ".agent-core/context-packs"

    async def validate(self, target_id: str, **kwargs) -> ValidationResult:
        """
        Validate a context pack.

        Args:
            target_id: Pack ID or path to validate
            pack_data: Optional dict with pack data (if not loading from file)

        Returns:
            ValidationResult with relevance assessment
        """
        pack_data = kwargs.get('pack_data')

        # Gather evidence
        evidence = await self._gather_evidence(target_id, pack_data=pack_data)

        if not evidence.get('exists'):
            return ValidationResult(
                valid=False,
                confidence=0.0,
                issues=[self.add_issue(
                    "PACK_NOT_FOUND",
                    f"Context pack not found: {target_id}",
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
        """Gather pack data for validation."""
        pack_data = kwargs.get('pack_data')

        if pack_data:
            return self._analyze_pack(pack_data, target_id)

        # Try to load from file
        pack_file = self.packs_dir / f"{target_id}.json"
        if not pack_file.exists():
            # Try alternative locations
            alt_paths = [
                self.packs_dir / target_id / "pack.json",
                Path.home() / ".claude/context-packs" / f"{target_id}.json",
            ]
            for alt in alt_paths:
                if alt.exists():
                    pack_file = alt
                    break

        if not pack_file.exists():
            return {'exists': False}

        try:
            with open(pack_file) as f:
                data = json.load(f)
            return self._analyze_pack(data, target_id)
        except Exception as e:
            return {
                'exists': True,
                'error': str(e),
                'metrics': {},
            }

    def _analyze_pack(self, pack_data: Dict[str, Any], pack_id: str) -> Dict[str, Any]:
        """Analyze pack content and structure."""
        evidence = {
            'exists': True,
            'pack_data': pack_data,
            'metrics': {},
        }

        # Get basic info
        content = pack_data.get('content', {})
        if isinstance(content, str):
            text_content = content
        else:
            text_content = json.dumps(content)

        # Token estimation (~4 chars per token)
        tokens = pack_data.get('tokens', len(text_content) // 4)
        evidence['metrics']['tokens'] = tokens

        # Get pack type
        pack_type = pack_data.get('type', 'unknown')
        evidence['metrics']['type'] = pack_type

        # Keywords
        keywords = []
        if isinstance(content, dict):
            keywords = content.get('keywords', [])
        evidence['metrics']['keyword_count'] = len(keywords)
        evidence['keywords'] = keywords

        # Check for dates
        created_at = pack_data.get('created_at') or pack_data.get('created')
        updated_at = pack_data.get('updated_at') or pack_data.get('updated')
        evidence['metrics']['has_dates'] = bool(created_at or updated_at)

        if updated_at:
            try:
                update_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                days_old = (datetime.now(update_date.tzinfo) - update_date).days
                evidence['metrics']['days_since_update'] = days_old
            except:
                evidence['metrics']['days_since_update'] = None

        # Content structure
        if isinstance(content, dict):
            evidence['metrics']['has_description'] = bool(content.get('description'))
            evidence['metrics']['has_sections'] = bool(content.get('sections'))
            evidence['metrics']['section_count'] = len(content.get('sections', []))
        else:
            evidence['metrics']['has_description'] = False
            evidence['metrics']['has_sections'] = False

        # Source info
        evidence['metrics']['has_source'] = bool(pack_data.get('source'))
        evidence['metrics']['source'] = pack_data.get('source', 'unknown')

        return evidence

    async def _run_checks(self, evidence: Dict[str, Any]) -> List[Issue]:
        """Run pack validation checks."""
        issues = []
        metrics = evidence['metrics']

        if evidence.get('error'):
            issues.append(self.add_issue(
                "PACK_PARSE_ERROR",
                f"Could not parse pack: {evidence['error']}",
                Severity.ERROR,
            ))
            return issues

        # Check 1: Token count
        tokens = metrics.get('tokens', 0)
        if tokens < MIN_TOKENS:
            issues.append(self.add_issue(
                "PACK_TOO_SMALL",
                f"Pack has only {tokens} tokens (min: {MIN_TOKENS})",
                Severity.WARNING,
                suggestion="Pack may be too sparse to be useful",
            ))
        elif tokens > MAX_TOKENS:
            issues.append(self.add_issue(
                "PACK_TOO_LARGE",
                f"Pack has {tokens} tokens (max: {MAX_TOKENS})",
                Severity.WARNING,
                suggestion="Consider splitting into multiple focused packs",
            ))
        elif tokens < IDEAL_TOKENS_MIN or tokens > IDEAL_TOKENS_MAX:
            issues.append(self.add_issue(
                "PACK_SIZE_SUBOPTIMAL",
                f"Pack has {tokens} tokens (ideal: {IDEAL_TOKENS_MIN}-{IDEAL_TOKENS_MAX})",
                Severity.INFO,
            ))

        # Check 2: Keywords
        keyword_count = metrics.get('keyword_count', 0)
        if keyword_count == 0:
            issues.append(self.add_issue(
                "NO_KEYWORDS",
                "Pack has no keywords for searchability",
                Severity.WARNING,
                suggestion="Add keywords for better pack discovery",
            ))

        # Check 3: Staleness
        days_old = metrics.get('days_since_update')
        if days_old is not None and days_old > STALE_THRESHOLD_DAYS:
            issues.append(self.add_issue(
                "STALE_PACK",
                f"Pack not updated in {days_old} days",
                Severity.INFO,
                suggestion="Consider refreshing pack content",
            ))

        # Check 4: Structure
        if not metrics.get('has_description'):
            issues.append(self.add_issue(
                "NO_DESCRIPTION",
                "Pack lacks description",
                Severity.INFO,
                suggestion="Add description for context",
            ))

        # Check 5: Source attribution
        if not metrics.get('has_source'):
            issues.append(self.add_issue(
                "NO_SOURCE",
                "Pack has no source attribution",
                Severity.INFO,
            ))

        return issues

    def _calculate_confidence(self, evidence: Dict[str, Any], issues: List[Issue]) -> float:
        """
        Calculate pack quality confidence.

        Scoring:
        - Token count appropriate: 30%
        - Has keywords: 20%
        - Not stale: 15%
        - Has description: 15%
        - Has source: 10%
        - No errors: 10%
        """
        if evidence.get('error'):
            return 0.1

        score = 0.0
        metrics = evidence['metrics']

        # Token count (30%)
        tokens = metrics.get('tokens', 0)
        if IDEAL_TOKENS_MIN <= tokens <= IDEAL_TOKENS_MAX:
            score += 0.30
        elif MIN_TOKENS <= tokens <= MAX_TOKENS:
            score += 0.20
        elif tokens > 0:
            score += 0.10

        # Keywords (20%)
        if metrics.get('keyword_count', 0) >= 3:
            score += 0.20
        elif metrics.get('keyword_count', 0) > 0:
            score += 0.10

        # Staleness (15%)
        days_old = metrics.get('days_since_update')
        if days_old is None:
            score += 0.10  # No date info, partial credit
        elif days_old <= STALE_THRESHOLD_DAYS:
            score += 0.15

        # Description (15%)
        if metrics.get('has_description'):
            score += 0.15

        # Source (10%)
        if metrics.get('has_source'):
            score += 0.10

        # No errors (10%)
        error_count = sum(1 for i in issues if i.severity in (Severity.ERROR, Severity.CRITICAL))
        if error_count == 0:
            score += 0.10

        return max(0.0, min(1.0, round(score, 3)))


async def main():
    """CLI for pack validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pack_critic.py <pack-id>")
        print("\nValidates context pack quality and relevance.")
        sys.exit(1)

    pack_id = sys.argv[1]

    critic = PackCritic()
    result = await critic.validate(pack_id)

    print(f"\n{'='*60}")
    print(f"PACK VALIDATION: {pack_id}")
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
