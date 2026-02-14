"""
Delegation Data Models

Core dataclasses for intelligent delegation system based on arXiv:2602.11865.
Uses dataclasses (not Pydantic) to match existing codebase conventions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class VerificationMethod(str, Enum):
    """Subtask verification methods"""
    AUTOMATED_TEST = "automated_test"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HUMAN_REVIEW = "human_review"
    GROUND_TRUTH = "ground_truth"


@dataclass
class TaskProfile:
    """
    11-dimensional task profile from arXiv:2602.11865.

    All dimensions are normalized to [0.0, 1.0] for consistent scoring.

    Dimensions:
    - complexity: Computational/cognitive complexity (0=trivial, 1=extremely complex)
    - criticality: Impact of failure (0=low, 1=mission-critical)
    - uncertainty: Ambiguity in requirements/approach (0=clear, 1=highly ambiguous)
    - duration: Estimated time to complete (0=instant, 1=very long)
    - cost: Resource consumption (0=free, 1=very expensive)
    - resource_requirements: Dependencies on external resources (0=none, 1=many)
    - constraints: Hard constraints on approach/solution (0=none, 1=many)
    - verifiability: Ease of verifying correctness (0=hard, 1=easy)
    - reversibility: Ease of undoing/rolling back (0=irreversible, 1=easily reversible)
    - contextuality: Dependence on external context (0=context-free, 1=highly contextual)
    - subjectivity: Degree of subjective judgment needed (0=objective, 1=highly subjective)
    """
    complexity: float = 0.5
    criticality: float = 0.5
    uncertainty: float = 0.5
    duration: float = 0.5
    cost: float = 0.5
    resource_requirements: float = 0.5
    constraints: float = 0.5
    verifiability: float = 0.5
    reversibility: float = 0.5
    contextuality: float = 0.5
    subjectivity: float = 0.5

    def __post_init__(self):
        """Validate all dimensions are in [0.0, 1.0]"""
        for field_name in [
            'complexity', 'criticality', 'uncertainty', 'duration', 'cost',
            'resource_requirements', 'constraints', 'verifiability',
            'reversibility', 'contextuality', 'subjectivity'
        ]:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0.0, 1.0], got {value}")


@dataclass
class SubTask:
    """
    Decomposed subtask ready for delegation.

    Fields:
    - id: Unique subtask identifier
    - description: Human-readable task description
    - verification_method: How to verify completion
    - estimated_cost: Estimated resource cost (0.0-1.0)
    - estimated_duration: Estimated time to complete (0.0-1.0)
    - parallel_safe: Can be executed in parallel with other subtasks
    - parent_task_id: ID of parent task (for hierarchical decomposition)
    - dependencies: List of subtask IDs that must complete first
    - profile: TaskProfile for this subtask
    """
    id: str
    description: str
    verification_method: VerificationMethod
    estimated_cost: float
    estimated_duration: float
    parallel_safe: bool
    parent_task_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    profile: Optional[TaskProfile] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate cost and duration are in [0.0, 1.0]"""
        if not 0.0 <= self.estimated_cost <= 1.0:
            raise ValueError(f"estimated_cost must be in [0.0, 1.0], got {self.estimated_cost}")
        if not 0.0 <= self.estimated_duration <= 1.0:
            raise ValueError(f"estimated_duration must be in [0.0, 1.0], got {self.estimated_duration}")


@dataclass
class Assignment:
    """
    Agent assignment for a subtask.

    Fields:
    - subtask_id: ID of the subtask being assigned
    - agent_id: ID of the agent assigned
    - trust_score: Current trust score for this agent (0.0-1.0)
    - capability_match: How well agent capabilities match task requirements (0.0-1.0)
    - timestamp: Unix timestamp of assignment
    - assignment_reasoning: Why this agent was chosen
    """
    subtask_id: str
    agent_id: str
    trust_score: float
    capability_match: float
    timestamp: float
    assignment_reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trust_score and capability_match are in [0.0, 1.0]"""
        if not 0.0 <= self.trust_score <= 1.0:
            raise ValueError(f"trust_score must be in [0.0, 1.0], got {self.trust_score}")
        if not 0.0 <= self.capability_match <= 1.0:
            raise ValueError(f"capability_match must be in [0.0, 1.0], got {self.capability_match}")


@dataclass
class TrustEntry:
    """
    Trust ledger entry tracking agent performance.

    Fields:
    - agent_id: ID of the agent
    - task_id: ID of the task performed
    - timestamp: Unix timestamp of entry
    - success: Whether task was completed successfully
    - quality_score: Quality of the output (0.0-1.0)
    - trust_delta: Change in trust score (-1.0 to +1.0)
    - updated_trust_score: New trust score after this entry
    """
    agent_id: str
    task_id: str
    timestamp: float
    success: bool
    quality_score: float
    trust_delta: float
    updated_trust_score: float
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate scores are in valid ranges"""
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(f"quality_score must be in [0.0, 1.0], got {self.quality_score}")
        if not -1.0 <= self.trust_delta <= 1.0:
            raise ValueError(f"trust_delta must be in [-1.0, 1.0], got {self.trust_delta}")
        if not 0.0 <= self.updated_trust_score <= 1.0:
            raise ValueError(f"updated_trust_score must be in [0.0, 1.0], got {self.updated_trust_score}")


@dataclass
class DelegationEvent:
    """
    Event in a delegation chain.

    Tracks the full lifecycle of a delegated task from creation through completion.

    Fields:
    - event_id: Unique event identifier
    - delegation_id: ID of the delegation chain this belongs to
    - timestamp: Unix timestamp of event
    - event_type: Type of event (created, assigned, started, completed, failed, verified)
    - agent_id: Agent involved in this event
    - task_id: Task being delegated
    - status: Current status of the task
    - details: Additional event details
    """
    event_id: str
    delegation_id: str
    timestamp: float
    event_type: str  # created, assigned, started, completed, failed, verified
    agent_id: str
    task_id: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """
    Result of subtask verification.

    Fields:
    - subtask_id: ID of the verified subtask
    - timestamp: Unix timestamp of verification
    - method: Verification method used
    - passed: Whether verification passed
    - quality_score: Quality assessment (0.0-1.0)
    - feedback: Human-readable feedback
    - evidence: Supporting evidence for verification decision
    """
    subtask_id: str
    timestamp: float
    method: VerificationMethod
    passed: bool
    quality_score: float
    feedback: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate quality_score is in [0.0, 1.0]"""
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(f"quality_score must be in [0.0, 1.0], got {self.quality_score}")
