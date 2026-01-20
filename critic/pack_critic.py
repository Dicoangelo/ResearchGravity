#!/usr/bin/env python3
"""
Pack Critic - Validates context pack relevance and quality.

Checks:
- Content relevance to pack type/topic
- Token budget compliance
- Recency of included sessions
- Balance across sources

Usage:
    python3 -m critic.pack_critic --pack <pack-id>
    python3 -m critic.pack_critic --all
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .base import (
    BaseCritic, CriticResult, ValidationIssue,
    IssueSeverity, IssueCategory
)


AGENT_CORE_DIR = Path.home() / ".agent-core"
PACKS_DIR = AGENT_CORE_DIR / "packs"


class PackCritic(BaseCritic):
    """
    Validates context pack quality and relevance.

    Ensures packs have:
    - Relevant content for the stated topic/project
    - Reasonable token counts
    - Recent sessions (not stale)
    - Balanced source distribution
    """

    # Pack-specific thresholds
    MAX_PACK_TOKENS = 50000             # Warn if pack exceeds this
    MIN_PACK_TOKENS = 100               # Error if pack too small
    STALE_DAYS = 30                     # Warn if all content older than this
    MIN_SESSIONS_FOR_DOMAIN = 2         # Domain packs need multiple sessions

    def __init__(self):
        super().__init__("PackCritic")

    def _check_content_relevance(self, pack: dict) -> list[ValidationIssue]:
        """Check if pack content matches its stated purpose."""
        issues = []

        pack_type = pack.get("type", "unknown")
        pack_topic = pack.get("topic", pack.get("pattern", ""))
        content = pack.get("content", "")

        # Simple keyword relevance check
        if pack_topic and content:
            topic_words = pack_topic.lower().replace("-", " ").split()
            content_lower = content.lower()

            # Check if topic keywords appear in content
            matches = sum(1 for word in topic_words if word in content_lower and len(word) > 3)
            relevance_ratio = matches / len(topic_words) if topic_words else 0

            if relevance_ratio < 0.3:
                issues.append(ValidationIssue(
                    category=IssueCategory.RELEVANCE,
                    severity=IssueSeverity.WARNING,
                    message=f"Pack content may not match topic '{pack_topic}'",
                    suggestion="Review pack content for relevance"
                ))

        return issues

    def _check_token_budget(self, pack: dict) -> list[ValidationIssue]:
        """Validate token count is within acceptable range."""
        issues = []

        tokens = pack.get("tokens", 0)

        if tokens < self.MIN_PACK_TOKENS:
            issues.append(ValidationIssue(
                category=IssueCategory.COMPLETENESS,
                severity=IssueSeverity.ERROR,
                message=f"Pack too small: {tokens} tokens (minimum: {self.MIN_PACK_TOKENS})",
                suggestion="Add more content to pack"
            ))
        elif tokens > self.MAX_PACK_TOKENS:
            issues.append(ValidationIssue(
                category=IssueCategory.QUALITY,
                severity=IssueSeverity.WARNING,
                message=f"Pack exceeds budget: {tokens} tokens (target: {self.MAX_PACK_TOKENS})",
                suggestion="Consider compressing or splitting pack"
            ))

        return issues

    def _check_recency(self, pack: dict) -> list[ValidationIssue]:
        """Check if pack content is recent enough."""
        issues = []

        created_at = pack.get("created_at")
        sessions = pack.get("sessions", [])

        # Check pack creation date
        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_days = (datetime.now(created_date.tzinfo) - created_date).days

                if age_days > self.STALE_DAYS:
                    issues.append(ValidationIssue(
                        category=IssueCategory.QUALITY,
                        severity=IssueSeverity.INFO,
                        message=f"Pack is {age_days} days old (threshold: {self.STALE_DAYS})",
                        suggestion="Consider regenerating pack with recent sessions"
                    ))
            except:
                pass

        # Check session dates if available
        if sessions:
            # Sessions typically have timestamps in their IDs
            recent_count = 0
            cutoff = datetime.now() - timedelta(days=self.STALE_DAYS)

            for session in sessions:
                session_id = session if isinstance(session, str) else session.get("id", "")
                # Try to extract date from session ID (format: topic-YYYYMMDD-HHMMSS-xxxx)
                import re
                date_match = re.search(r'(\d{8})-\d{6}', session_id)
                if date_match:
                    try:
                        session_date = datetime.strptime(date_match.group(1), "%Y%m%d")
                        if session_date > cutoff:
                            recent_count += 1
                    except:
                        pass

            if sessions and recent_count == 0:
                issues.append(ValidationIssue(
                    category=IssueCategory.QUALITY,
                    severity=IssueSeverity.WARNING,
                    message=f"No sessions from last {self.STALE_DAYS} days",
                    suggestion="Regenerate pack to include recent work"
                ))

        return issues

    def _check_balance(self, pack: dict) -> list[ValidationIssue]:
        """Check for balanced source distribution."""
        issues = []

        pack_type = pack.get("type", "")
        sessions = pack.get("sessions", [])

        if pack_type == "domain" and len(sessions) < self.MIN_SESSIONS_FOR_DOMAIN:
            issues.append(ValidationIssue(
                category=IssueCategory.COMPLETENESS,
                severity=IssueSeverity.WARNING,
                message=f"Domain pack has only {len(sessions)} sessions (minimum: {self.MIN_SESSIONS_FOR_DOMAIN})",
                suggestion="Include more sessions for comprehensive domain coverage"
            ))

        return issues

    def validate(self, content: dict) -> CriticResult:
        """
        Validate a context pack.

        Args:
            content: Dict with pack data or 'pack_id' to load

        Returns:
            CriticResult with pack quality assessment
        """
        pack = content.get("pack")
        pack_id = content.get("pack_id")

        # Load pack if needed
        if not pack and pack_id:
            pack_file = PACKS_DIR / f"{pack_id}.json"
            if pack_file.exists():
                try:
                    pack = json.loads(pack_file.read_text())
                except json.JSONDecodeError:
                    return self._create_result(
                        confidence=0.0,
                        issues=[ValidationIssue(
                            category=IssueCategory.ACCURACY,
                            severity=IssueSeverity.CRITICAL,
                            message=f"Invalid JSON in pack file: {pack_id}"
                        )],
                        summary="Cannot parse pack file"
                    )
            else:
                return self._create_result(
                    confidence=0.0,
                    issues=[ValidationIssue(
                        category=IssueCategory.COMPLETENESS,
                        severity=IssueSeverity.CRITICAL,
                        message=f"Pack not found: {pack_id}"
                    )],
                    summary="Pack not found"
                )

        if not pack:
            return self._create_result(
                confidence=0.0,
                issues=[ValidationIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.CRITICAL,
                    message="No pack data provided"
                )],
                summary="No pack to validate"
            )

        # Run all checks
        issues = []
        issues.extend(self._check_content_relevance(pack))
        issues.extend(self._check_token_budget(pack))
        issues.extend(self._check_recency(pack))
        issues.extend(self._check_balance(pack))

        # Calculate confidence
        error_count = sum(1 for i in issues if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL])
        warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

        if error_count > 0:
            confidence = 0.3
        elif warning_count > 2:
            confidence = 0.6
        elif warning_count > 0:
            confidence = 0.8
        else:
            confidence = 0.95

        # Summary
        tokens = pack.get("tokens", 0)
        sessions = pack.get("sessions", [])
        pack_type = pack.get("type", "unknown")

        summary = f"{pack_type} pack: {tokens} tokens, {len(sessions)} sessions"

        return self._create_result(
            confidence=confidence,
            issues=issues,
            summary=summary,
            metadata={
                "pack_id": pack_id or pack.get("id"),
                "pack_type": pack_type,
                "tokens": tokens,
                "session_count": len(sessions)
            }
        )


def validate_pack(pack_id: str, verbose: bool = False) -> CriticResult:
    """Validate a single pack."""
    critic = PackCritic()
    result = critic.validate({"pack_id": pack_id})

    if verbose:
        print(f"\n{'='*50}")
        print(f"Pack Critic Report: {pack_id}")
        print(f"{'='*50}")
        print(f"Status: {'✅ APPROVED' if result.approved else '❌ REJECTED'}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Summary: {result.summary}")

        if result.issues:
            print(f"\nIssues ({len(result.issues)}):")
            for issue in result.issues:
                icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚫"}
                print(f"  {icon.get(issue.severity.value, '•')} {issue.message}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Pack Critic - Validate context packs")
    parser.add_argument("--pack", "-p", help="Pack ID to validate")
    parser.add_argument("--all", "-a", action="store_true", help="Validate all packs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.pack:
        result = validate_pack(args.pack, verbose=True)
        return 0 if result.approved else 1

    elif args.all:
        if not PACKS_DIR.exists():
            print("No packs directory found")
            return 1

        results = {"approved": 0, "rejected": 0}

        for pack_file in PACKS_DIR.glob("*.json"):
            result = validate_pack(pack_file.stem, verbose=args.verbose)

            if result.approved:
                results["approved"] += 1
            else:
                results["rejected"] += 1

        print(f"\nSummary:")
        print(f"  Approved: {results['approved']}")
        print(f"  Rejected: {results['rejected']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    exit(main() or 0)
