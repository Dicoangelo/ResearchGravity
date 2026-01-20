"""
Writer-Critic Validation System.

Implements dual-agent validation for high-stakes outputs:
- Archive synthesis validation
- Evidence quality validation
- Context pack relevance validation
- Oracle multi-stream consensus

Based on the "Writer-Critic Validation" principle from the 4 Patterns framework.
"""

from .base import (
    BaseCritic,
    CriticResult,
    CompositeCritic,
    ValidationIssue,
    IssueSeverity,
    IssueCategory
)
from .archive_critic import ArchiveCritic
from .evidence_critic import EvidenceCritic
from .pack_critic import PackCritic
from .oracle_adapter import (
    OracleValidator,
    OracleConsensus,
    OracleStream,
    OraclePerspective,
    run_oracle_consensus,
    validate_with_oracle
)

__all__ = [
    # Base classes
    "BaseCritic",
    "CriticResult",
    "CompositeCritic",
    "ValidationIssue",
    "IssueSeverity",
    "IssueCategory",
    # Critics
    "ArchiveCritic",
    "EvidenceCritic",
    "PackCritic",
    # Oracle integration
    "OracleValidator",
    "OracleConsensus",
    "OracleStream",
    "OraclePerspective",
    "run_oracle_consensus",
    "validate_with_oracle",
]
