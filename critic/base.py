"""
Critic Base Class

Provides the foundation for all critic validators:
- Standard validation interface
- Confidence scoring
- Issue tracking
- Oracle multi-stream integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class Severity(Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Issue:
    """Represents a validation issue found by a critic."""
    code: str
    message: str
    severity: Severity
    location: Optional[str] = None
    suggestion: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "location": self.location,
            "suggestion": self.suggestion,
            "context": self.context,
        }


@dataclass
class ValidationResult:
    """Result of a critic validation pass."""
    valid: bool
    confidence: float  # 0.0 - 1.0
    issues: List[Issue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    critic_name: str = ""
    target_id: str = ""

    @property
    def passes_threshold(self) -> bool:
        """Check if confidence meets minimum threshold (0.7)."""
        return self.confidence >= 0.7

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity in (Severity.ERROR, Severity.CRITICAL))

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "confidence": self.confidence,
            "passes_threshold": self.passes_threshold,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "critic_name": self.critic_name,
            "target_id": self.target_id,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


class CriticBase(ABC):
    """
    Base class for all critics.

    Critics validate high-stakes outputs using the Writer-Critic pattern:
    1. Writer produces output (already done)
    2. Critic validates against criteria
    3. Issues flagged for revision if confidence < 0.7
    """

    name: str = "base_critic"
    description: str = "Base critic class"
    min_confidence: float = 0.7

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self._validation_history: List[ValidationResult] = []

    @abstractmethod
    async def validate(self, target_id: str, **kwargs) -> ValidationResult:
        """
        Validate the target and return results.

        Args:
            target_id: Identifier for what's being validated (session_id, pack_id, etc.)
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with confidence score and any issues found
        """
        pass

    @abstractmethod
    async def _gather_evidence(self, target_id: str, **kwargs) -> Dict[str, Any]:
        """
        Gather all data needed for validation.

        Returns dict with all relevant data for the validation checks.
        """
        pass

    @abstractmethod
    async def _run_checks(self, evidence: Dict[str, Any]) -> List[Issue]:
        """
        Run validation checks on gathered evidence.

        Returns list of issues found.
        """
        pass

    @abstractmethod
    def _calculate_confidence(self, evidence: Dict[str, Any], issues: List[Issue]) -> float:
        """
        Calculate confidence score based on evidence and issues.

        Returns float between 0.0 and 1.0
        """
        pass

    def add_issue(
        self,
        code: str,
        message: str,
        severity: Severity = Severity.WARNING,
        location: Optional[str] = None,
        suggestion: Optional[str] = None,
        **context
    ) -> Issue:
        """Helper to create an issue."""
        return Issue(
            code=code,
            message=message,
            severity=severity,
            location=location,
            suggestion=suggestion,
            context=context,
        )

    def record_result(self, result: ValidationResult):
        """Record validation result for history tracking."""
        self._validation_history.append(result)

    def get_history(self, limit: int = 10) -> List[ValidationResult]:
        """Get recent validation history."""
        return self._validation_history[-limit:]

    async def validate_batch(self, target_ids: List[str], **kwargs) -> List[ValidationResult]:
        """Validate multiple targets."""
        results = []
        for target_id in target_ids:
            result = await self.validate(target_id, **kwargs)
            results.append(result)
        return results


class OracleConsensus:
    """
    Multi-stream consensus for critical validations.

    Runs 3 parallel validation perspectives and finds intersection.
    Based on Oracle multi-stream pattern.
    """

    def __init__(self, critics: List[CriticBase]):
        """
        Initialize with multiple critics for consensus.

        Args:
            critics: List of 3 critics for multi-perspective validation
        """
        if len(critics) != 3:
            raise ValueError("Oracle consensus requires exactly 3 critics")
        self.critics = critics

    async def validate_with_consensus(
        self,
        target_id: str,
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Run all critics and compute consensus.

        Args:
            target_id: Target to validate
            weights: Optional weights for each critic (default: equal)
            **kwargs: Passed to each critic

        Returns:
            Combined ValidationResult with consensus confidence
        """
        # Default weights: Accuracy 40%, Completeness 35%, Relevance 25%
        if weights is None:
            weights = {
                self.critics[0].name: 0.40,
                self.critics[1].name: 0.35,
                self.critics[2].name: 0.25,
            }

        # Run all critics
        results = []
        for critic in self.critics:
            result = await critic.validate(target_id, **kwargs)
            results.append(result)

        # Compute weighted confidence
        total_weight = sum(weights.get(c.name, 1/3) for c in self.critics)
        weighted_confidence = sum(
            r.confidence * weights.get(c.name, 1/3)
            for r, c in zip(results, self.critics)
        ) / total_weight

        # Collect all issues
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)

        # Find consensus issues (appeared in 2+ critics)
        issue_codes = {}
        for issue in all_issues:
            issue_codes[issue.code] = issue_codes.get(issue.code, 0) + 1

        consensus_issues = [
            i for i in all_issues
            if issue_codes[i.code] >= 2
        ]

        # Deduplicate
        seen_codes = set()
        unique_issues = []
        for issue in consensus_issues:
            if issue.code not in seen_codes:
                seen_codes.add(issue.code)
                unique_issues.append(issue)

        # Build consensus result
        return ValidationResult(
            valid=weighted_confidence >= 0.7 and len([i for i in unique_issues if i.severity == Severity.CRITICAL]) == 0,
            confidence=round(weighted_confidence, 3),
            issues=unique_issues,
            metrics={
                "critics_run": len(self.critics),
                "individual_confidences": [r.confidence for r in results],
                "total_issues": len(all_issues),
                "consensus_issues": len(unique_issues),
                "weights": weights,
            },
            critic_name="oracle_consensus",
            target_id=target_id,
        )
