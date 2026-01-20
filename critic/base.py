#!/usr/bin/env python3
"""
Base Critic Class for Writer-Critic Validation.

Provides the foundation for all critics with:
- Standard validation interface
- Confidence thresholds
- Issue categorization
- Approval/rejection logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"           # Informational, doesn't affect approval
    WARNING = "warning"     # Should be reviewed but can proceed
    ERROR = "error"         # Must be fixed before approval
    CRITICAL = "critical"   # Blocks approval, requires immediate attention


class IssueCategory(Enum):
    """Categories of validation issues."""
    COMPLETENESS = "completeness"       # Missing required elements
    ACCURACY = "accuracy"               # Factual or citation errors
    RELEVANCE = "relevance"             # Off-topic or irrelevant content
    QUALITY = "quality"                 # Style, clarity, structure issues
    EVIDENCE = "evidence"               # Citation or source problems
    CONSISTENCY = "consistency"         # Internal contradictions


@dataclass
class ValidationIssue:
    """A single validation issue found by the critic."""
    category: IssueCategory
    severity: IssueSeverity
    message: str
    location: Optional[str] = None      # Where in the content (e.g., "finding #3")
    suggestion: Optional[str] = None    # How to fix
    evidence: Optional[str] = None      # Supporting evidence for the issue


@dataclass
class CriticResult:
    """Result of a critic validation."""
    critic_name: str
    approved: bool
    confidence: float                   # 0.0 - 1.0
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "critic_name": self.critic_name,
            "approved": self.approved,
            "confidence": self.confidence,
            "issues": [
                {
                    "category": i.category.value,
                    "severity": i.severity.value,
                    "message": i.message,
                    "location": i.location,
                    "suggestion": i.suggestion,
                    "evidence": i.evidence
                }
                for i in self.issues
            ],
            "summary": self.summary,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @property
    def error_count(self) -> int:
        """Count of ERROR or CRITICAL issues."""
        return sum(1 for i in self.issues
                   if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL])

    @property
    def warning_count(self) -> int:
        """Count of WARNING issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)


class BaseCritic(ABC):
    """
    Abstract base class for all critics.

    Critics validate outputs from the "Writer" (whatever produced the content)
    and determine whether the output meets quality standards.
    """

    # Default thresholds (can be overridden by subclasses)
    MIN_CONFIDENCE_THRESHOLD = 0.7      # Below this, cannot approve
    MAX_ERRORS_FOR_APPROVAL = 0         # Any errors block approval
    MAX_WARNINGS_FOR_APPROVAL = 3       # Too many warnings block approval

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def validate(self, content: dict) -> CriticResult:
        """
        Validate content and return result.

        Args:
            content: The content to validate (structure depends on critic type)

        Returns:
            CriticResult with approval status, confidence, and issues
        """
        pass

    def _should_approve(self, confidence: float, issues: list[ValidationIssue]) -> bool:
        """
        Determine if content should be approved based on confidence and issues.

        Override in subclasses for custom approval logic.
        """
        # Check confidence threshold
        if confidence < self.MIN_CONFIDENCE_THRESHOLD:
            return False

        # Count errors and warnings
        error_count = sum(1 for i in issues
                         if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL])
        warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

        # Check error threshold
        if error_count > self.MAX_ERRORS_FOR_APPROVAL:
            return False

        # Check warning threshold
        if warning_count > self.MAX_WARNINGS_FOR_APPROVAL:
            return False

        return True

    def _create_result(
        self,
        confidence: float,
        issues: list[ValidationIssue],
        summary: str = "",
        metadata: dict = None
    ) -> CriticResult:
        """Helper to create a CriticResult with automatic approval calculation."""
        approved = self._should_approve(confidence, issues)

        if not summary:
            if approved:
                summary = f"Approved with confidence {confidence:.2f}"
                if issues:
                    summary += f" ({len(issues)} minor issues)"
            else:
                error_count = sum(1 for i in issues
                                  if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL])
                summary = f"Rejected: {error_count} errors found, confidence {confidence:.2f}"

        return CriticResult(
            critic_name=self.name,
            approved=approved,
            confidence=confidence,
            issues=issues,
            summary=summary,
            metadata=metadata or {}
        )


class CompositeCritic(BaseCritic):
    """
    A critic that combines multiple critics.

    Useful for running several validation passes on the same content.
    """

    def __init__(self, critics: list[BaseCritic], name: str = "CompositeCritic"):
        super().__init__(name)
        self.critics = critics

    def validate(self, content: dict) -> CriticResult:
        """Run all critics and combine results."""
        all_issues = []
        confidences = []
        metadata = {"critic_results": {}}

        for critic in self.critics:
            result = critic.validate(content)

            all_issues.extend(result.issues)
            confidences.append(result.confidence)
            metadata["critic_results"][critic.name] = {
                "approved": result.approved,
                "confidence": result.confidence,
                "issue_count": len(result.issues)
            }

        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Create combined result
        return self._create_result(
            confidence=avg_confidence,
            issues=all_issues,
            summary=f"Combined validation from {len(self.critics)} critics",
            metadata=metadata
        )
