#!/usr/bin/env python3
"""
Oracle Adapter - Connects Writer-Critic system to Oracle Multi-Stream Consensus.

Implements the Oracle validation pattern:
1. Run 3 parallel validation streams (different perspectives)
2. Compute intersection of validated claims
3. Generate consensus result with aggregate confidence
4. Only approve if confidence > threshold (default 0.7)

Oracle Perspectives:
- Accuracy: Are claims factually correct?
- Completeness: Is the content thorough?
- Relevance: Is content appropriate for purpose?

Usage:
    from critic.oracle_adapter import OracleValidator, run_oracle_consensus

    # Single critic with Oracle consensus
    result = run_oracle_consensus(archive_critic, content)

    # Multi-critic with Oracle synthesis
    validator = OracleValidator([archive_critic, evidence_critic])
    result = validator.validate(content)
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any

from .base import (
    BaseCritic, CriticResult, ValidationIssue,
    IssueSeverity, IssueCategory
)


class OraclePerspective(Enum):
    """Validation perspectives for Oracle streams."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"


@dataclass
class OracleStream:
    """Result from a single Oracle validation stream."""
    stream_id: str
    perspective: OraclePerspective
    confidence: float
    validated_claims: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "perspective": self.perspective.value,
            "confidence": self.confidence,
            "validated_claims": self.validated_claims,
            "issues": self.issues,
            "timestamp": self.timestamp
        }


@dataclass
class OracleConsensus:
    """Consensus result from Oracle multi-stream validation."""
    consensus_id: str
    approved: bool
    confidence: float
    streams: List[OracleStream]
    intersection: List[str]  # Claims validated by all streams
    issues: List[str]
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "consensus_id": self.consensus_id,
            "approved": self.approved,
            "confidence": self.confidence,
            "streams": [s.to_dict() for s in self.streams],
            "intersection": self.intersection,
            "issues": self.issues,
            "summary": self.summary,
            "timestamp": self.timestamp
        }


# Perspective weights for confidence calculation
PERSPECTIVE_WEIGHTS = {
    OraclePerspective.ACCURACY: 0.40,
    OraclePerspective.COMPLETENESS: 0.35,
    OraclePerspective.RELEVANCE: 0.25,
}

# Default thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MAX_ISSUES = 5


def _generate_stream_id(content: Any, perspective: OraclePerspective) -> str:
    """Generate unique stream ID."""
    content_hash = hashlib.md5(str(content).encode()).hexdigest()[:8]
    return f"stream-{perspective.value}-{content_hash}"


def _generate_consensus_id(streams: List[OracleStream]) -> str:
    """Generate unique consensus ID."""
    stream_ids = "-".join([s.stream_id[:8] for s in streams])
    return f"consensus-{hashlib.md5(stream_ids.encode()).hexdigest()[:12]}"


def run_perspective_stream(
    critic: BaseCritic,
    content: dict,
    perspective: OraclePerspective
) -> OracleStream:
    """
    Run a single Oracle stream with a specific perspective.

    The critic is run normally, then results are interpreted through
    the lens of the given perspective.

    Args:
        critic: Critic to run validation
        content: Content to validate
        perspective: Perspective to evaluate from

    Returns:
        OracleStream with perspective-specific results
    """
    stream_id = _generate_stream_id(content, perspective)

    # Run critic
    result = critic.validate(content)

    # Interpret results through perspective lens
    perspective_issues = []
    validated_claims = []
    confidence_adjustment = 1.0

    for issue in result.issues:
        # Map issue categories to perspectives
        if perspective == OraclePerspective.ACCURACY:
            if issue.category in [IssueCategory.ACCURACY, IssueCategory.CONSISTENCY]:
                perspective_issues.append(issue.message)
                if issue.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]:
                    confidence_adjustment *= 0.5
                elif issue.severity == IssueSeverity.WARNING:
                    confidence_adjustment *= 0.8

        elif perspective == OraclePerspective.COMPLETENESS:
            if issue.category == IssueCategory.COMPLETENESS:
                perspective_issues.append(issue.message)
                if issue.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]:
                    confidence_adjustment *= 0.5
                elif issue.severity == IssueSeverity.WARNING:
                    confidence_adjustment *= 0.8

        elif perspective == OraclePerspective.RELEVANCE:
            if issue.category in [IssueCategory.RELEVANCE, IssueCategory.QUALITY]:
                perspective_issues.append(issue.message)
                if issue.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]:
                    confidence_adjustment *= 0.6
                elif issue.severity == IssueSeverity.WARNING:
                    confidence_adjustment *= 0.85

    # Calculate perspective confidence
    perspective_confidence = result.confidence * confidence_adjustment

    # Generate validated claims based on perspective
    if perspective_confidence >= 0.7:
        validated_claims.append(f"Passed {perspective.value} check")
    if perspective_confidence >= 0.5:
        validated_claims.append(f"Acceptable {perspective.value}")

    return OracleStream(
        stream_id=stream_id,
        perspective=perspective,
        confidence=min(perspective_confidence, 1.0),
        validated_claims=validated_claims,
        issues=perspective_issues
    )


def run_oracle_consensus(
    critic: BaseCritic,
    content: dict,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_issues: int = DEFAULT_MAX_ISSUES
) -> OracleConsensus:
    """
    Run Oracle multi-stream consensus validation.

    Implements the Writer-Critic pattern with Oracle consensus:
    1. Run 3 parallel validation streams (accuracy, completeness, relevance)
    2. Compute intersection of validated claims
    3. Calculate weighted aggregate confidence
    4. Approve only if confidence > threshold

    Args:
        critic: Critic to use for validation
        content: Content to validate
        confidence_threshold: Minimum confidence for approval (default 0.7)
        max_issues: Maximum allowed issues for approval

    Returns:
        OracleConsensus with multi-stream validation result
    """
    # Run all perspective streams
    streams = []
    for perspective in OraclePerspective:
        stream = run_perspective_stream(critic, content, perspective)
        streams.append(stream)

    # Compute intersection of validated claims
    if streams:
        claim_sets = [set(s.validated_claims) for s in streams]
        intersection = list(claim_sets[0].intersection(*claim_sets[1:]))
    else:
        intersection = []

    # Aggregate issues
    all_issues = []
    for stream in streams:
        all_issues.extend(stream.issues)
    unique_issues = list(set(all_issues))

    # Calculate weighted confidence
    weighted_confidence = sum(
        stream.confidence * PERSPECTIVE_WEIGHTS[stream.perspective]
        for stream in streams
    )

    # Determine approval
    approved = (
        weighted_confidence >= confidence_threshold and
        len(unique_issues) <= max_issues
    )

    # Generate summary
    stream_status = ", ".join([
        f"{s.perspective.value}: {s.confidence:.2f}"
        for s in streams
    ])

    if approved:
        summary = f"Oracle approved (confidence: {weighted_confidence:.2f}). Streams: {stream_status}"
    else:
        summary = f"Oracle rejected (confidence: {weighted_confidence:.2f}, issues: {len(unique_issues)}). Streams: {stream_status}"

    consensus_id = _generate_consensus_id(streams)

    return OracleConsensus(
        consensus_id=consensus_id,
        approved=approved,
        confidence=weighted_confidence,
        streams=streams,
        intersection=intersection,
        issues=unique_issues,
        summary=summary
    )


class OracleValidator:
    """
    Oracle-enhanced validator that combines multiple critics.

    Runs each critic through Oracle consensus, then synthesizes
    results into a final approval decision.
    """

    def __init__(
        self,
        critics: List[BaseCritic],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        require_all_approved: bool = False
    ):
        """
        Initialize Oracle validator.

        Args:
            critics: List of critics to run
            confidence_threshold: Minimum confidence for approval
            require_all_approved: If True, all critics must approve
        """
        self.critics = critics
        self.confidence_threshold = confidence_threshold
        self.require_all_approved = require_all_approved

    def validate(self, content: dict) -> Dict[str, Any]:
        """
        Run Oracle consensus validation with all critics.

        Args:
            content: Content to validate

        Returns:
            Dict with overall result and per-critic consensus
        """
        critic_results = {}
        confidences = []
        all_approved = True

        for critic in self.critics:
            consensus = run_oracle_consensus(
                critic,
                content,
                self.confidence_threshold
            )

            critic_results[critic.name] = consensus.to_dict()
            confidences.append(consensus.confidence)

            if not consensus.approved:
                all_approved = False

        # Calculate aggregate confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Determine final approval
        if self.require_all_approved:
            approved = all_approved
        else:
            approved = avg_confidence >= self.confidence_threshold

        return {
            "approved": approved,
            "confidence": avg_confidence,
            "critic_count": len(self.critics),
            "all_approved": all_approved,
            "critic_results": critic_results,
            "timestamp": datetime.now().isoformat()
        }

    def validate_with_critic_result(self, content: dict) -> CriticResult:
        """
        Run validation and return as CriticResult for compatibility.

        Args:
            content: Content to validate

        Returns:
            CriticResult compatible with existing critic infrastructure
        """
        result = self.validate(content)

        # Convert to CriticResult
        issues = []
        for critic_name, consensus in result["critic_results"].items():
            for issue_msg in consensus.get("issues", []):
                issues.append(ValidationIssue(
                    category=IssueCategory.QUALITY,
                    severity=IssueSeverity.WARNING,
                    message=f"[{critic_name}] {issue_msg}"
                ))

        summary = f"Oracle validated {result['critic_count']} critics, confidence: {result['confidence']:.2f}"

        return CriticResult(
            critic_name="OracleValidator",
            approved=result["approved"],
            confidence=result["confidence"],
            issues=issues,
            summary=summary,
            metadata={
                "all_approved": result["all_approved"],
                "critic_results": result["critic_results"]
            }
        )


def validate_with_oracle(
    content: dict,
    critics: List[BaseCritic] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> CriticResult:
    """
    Convenience function for Oracle validation.

    Args:
        content: Content to validate
        critics: Critics to use (defaults to standard set)
        confidence_threshold: Minimum confidence for approval

    Returns:
        CriticResult with Oracle consensus
    """
    if critics is None:
        # Import standard critics
        from .archive_critic import ArchiveCritic
        from .evidence_critic import EvidenceCritic

        critics = [ArchiveCritic(), EvidenceCritic()]

    validator = OracleValidator(critics, confidence_threshold)
    return validator.validate_with_critic_result(content)
