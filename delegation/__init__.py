"""
Intelligent Delegation Module — Multi-Agent Task Decomposition & Trust

Implements the intelligent delegation framework from arXiv:2602.11865.
Provides task taxonomy, decomposition, routing, trust tracking, coordination,
and verification for sovereign multi-agent orchestration.

Core Components:
- taxonomy: 11-dimensional task profiling
- decomposer: Intelligent task decomposition
- router: Agent selection and assignment
- trust_ledger: Trust score tracking and updates
- coordinator: Multi-agent orchestration
- four_ds: 4D framework (Decompose, Delegate, Develop, Deliver)
- memory_bleed: Cross-conversation knowledge transfer
- evolution: Learning from delegation outcomes
- verifier: Result verification and quality assessment
- permissions: Risk-adaptive permission handling (Section 4.7)
- ethical_delegation: Human oversight, accountability chains, service tiers (Sections 5.1-5.3)

Usage:
    from delegation import TaskProfile, decompose_task, assign_agent

    profile = TaskProfile(complexity=0.8, criticality=0.9, ...)
    subtasks = decompose_task(task, profile)
    assignment = assign_agent(subtask, trust_ledger)
"""

__version__ = "0.1.0"

from .models import (
    TaskProfile,
    SubTask,
    Assignment,
    TrustEntry,
    DelegationEvent,
    VerificationResult,
    VerificationMethod,
)
from .taxonomy import classify_task
from .decomposer import decompose_task
from .router import route_subtask, route_batch, load_agent_registry, AgentCapability
from .four_ds import (
    FourDsGate,
    delegation_gate,
    description_gate,
    discernment_gate,
    diligence_gate,
)
from .coordinator import DelegationCoordinator, TriggerType, ResponseAction
from .memory_bleed import (
    get_relevant_context,
    get_error_patterns,
    get_domain_expertise,
    write_delegation_outcome,
    inject_context,
    MemoryContext,
    ErrorPattern,
)
from .verifier import (
    verify_completion,
    feed_to_trust_ledger,
    feed_to_memory_bleed,
)
from .permissions import (
    PermissionManager,
    PermissionScope,
    AccessLevel,
    PermissionState,
)
from .ethical_delegation import (
    HumanOversight,
    OversightDecision,
    OversightLevel,
    AccountabilityChain,
    HandoffRecord,
    ServiceTier,
    ServiceTierName,
    get_service_tier,
    enforce_safety_floor,
)

__all__ = [
    "TaskProfile",
    "SubTask",
    "Assignment",
    "TrustEntry",
    "DelegationEvent",
    "VerificationResult",
    "VerificationMethod",
    "classify_task",
    "decompose_task",
    "route_subtask",
    "route_batch",
    "load_agent_registry",
    "AgentCapability",
    "FourDsGate",
    "delegation_gate",
    "description_gate",
    "discernment_gate",
    "diligence_gate",
    "DelegationCoordinator",
    "TriggerType",
    "ResponseAction",
    "get_relevant_context",
    "get_error_patterns",
    "get_domain_expertise",
    "write_delegation_outcome",
    "inject_context",
    "MemoryContext",
    "ErrorPattern",
    "verify_completion",
    "feed_to_trust_ledger",
    "feed_to_memory_bleed",
    "PermissionManager",
    "PermissionScope",
    "AccessLevel",
    "PermissionState",
    "HumanOversight",
    "OversightDecision",
    "OversightLevel",
    "AccountabilityChain",
    "HandoffRecord",
    "ServiceTier",
    "ServiceTierName",
    "get_service_tier",
    "enforce_safety_floor",
]
