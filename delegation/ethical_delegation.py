"""
Ethical Delegation Framework — Meaningful Human Control, Accountability, and Service Levels

Implements arXiv:2602.11865 Sections 5.1-5.3:
- Section 5.1: Meaningful Human Control — cognitive friction that dynamically
  increases oversight requirements based on criticality and uncertainty.
  Prevents "rubber-stamping" through zone of indifference tracking.
- Section 5.2: Accountability in Long Delegation Chains — liability firebreaks
  with immutable provenance. Chain-of-custody transparency for A->B->C chains.
- Section 5.3: Tiered Service Levels — low-cost delegation for routine tasks,
  high-assurance delegation for critical functions, safety floor enforcement.

Usage:
    from delegation.ethical_delegation import (
        HumanOversight,
        AccountabilityChain,
        ServiceTier,
        get_service_tier,
    )

    # Check if human oversight is required
    oversight = HumanOversight()
    decision = oversight.evaluate(task_profile, chain_depth=2)
    if decision.requires_human:
        print(f"Human review required: {decision.reason}")

    # Track accountability through delegation chains
    chain = AccountabilityChain(chain_id="chain-abc123")
    chain.record_handoff("agent-A", "agent-B", subtask, permissions)
    provenance = chain.get_provenance()

    # Get service tier for a task
    tier = get_service_tier(task_profile)
    print(f"Service tier: {tier.name}, safety floor: {tier.safety_floor}")
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .models import TaskProfile, SubTask


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5.1: MEANINGFUL HUMAN CONTROL
# ═══════════════════════════════════════════════════════════════════════════


class OversightLevel(str, Enum):
    """Human oversight intensity levels."""
    NONE = "none"                  # Fully autonomous (low-risk, high-trust)
    NOTIFICATION = "notification"  # Human notified after-the-fact
    APPROVAL = "approval"          # Human must approve before execution
    SUPERVISION = "supervision"    # Human must actively supervise execution
    DIRECT = "direct"              # Human must perform the action themselves


@dataclass
class OversightDecision:
    """Result of human oversight evaluation."""
    requires_human: bool
    oversight_level: OversightLevel
    reason: str
    cognitive_friction_score: float  # 0.0-1.0: how much friction to apply
    zone_of_indifference: bool      # True if task is in "auto-approve" zone
    metadata: Dict = field(default_factory=dict)


# Cognitive friction thresholds (from paper Section 5.1)
# Higher friction = more human involvement required
FRICTION_THRESHOLDS = {
    OversightLevel.NONE: 0.0,
    OversightLevel.NOTIFICATION: 0.2,
    OversightLevel.APPROVAL: 0.5,
    OversightLevel.SUPERVISION: 0.7,
    OversightLevel.DIRECT: 0.9,
}

# Maximum chain depth before mandatory human checkpoint
MAX_AUTONOMOUS_DEPTH = 3


class HumanOversight:
    """
    Meaningful Human Control system (Section 5.1).

    Dynamically adjusts oversight requirements based on:
    - Task criticality and uncertainty
    - Delegation chain depth (deeper = more oversight)
    - Historical rubber-stamping detection
    - Zone of indifference (routine tasks that don't need review)

    The key insight from the paper: oversight must impose "cognitive friction"
    proportional to the stakes. Too little friction → rubber-stamping.
    Too much → bottleneck. The system calibrates dynamically.
    """

    def __init__(self):
        self._approval_history: List[Dict] = []
        self._rubber_stamp_threshold = 0.5  # seconds — faster = likely rubber-stamp

    def evaluate(
        self,
        profile: TaskProfile,
        chain_depth: int = 0,
        agent_trust: float = 0.5,
    ) -> OversightDecision:
        """
        Evaluate whether human oversight is required for a task.

        Cognitive friction formula:
            friction = criticality * 0.4 + (1 - reversibility) * 0.3
                     + uncertainty * 0.2 + depth_penalty * 0.1

        Args:
            profile: 11-dimensional task profile
            chain_depth: Current depth in delegation chain (0 = top level)
            agent_trust: Trust score of the agent being delegated to

        Returns:
            OversightDecision with level, friction score, and reasoning
        """
        # Depth penalty: deeper chains need more oversight
        depth_penalty = min(1.0, chain_depth / MAX_AUTONOMOUS_DEPTH)

        # Cognitive friction calculation
        friction = (
            profile.criticality * 0.4 +
            (1.0 - profile.reversibility) * 0.3 +
            profile.uncertainty * 0.2 +
            depth_penalty * 0.1
        )
        friction = max(0.0, min(1.0, friction))

        # Trust discount: high-trust agents reduce friction
        trust_discount = max(0.0, (agent_trust - 0.7) * 0.3)  # Up to 0.09 discount
        friction = max(0.0, friction - trust_discount)

        # Zone of indifference: routine tasks (low criticality + high reversibility)
        zone_of_indifference = (
            profile.criticality < 0.3 and
            profile.reversibility > 0.7 and
            profile.uncertainty < 0.3 and
            chain_depth <= 1
        )

        # Determine oversight level from friction score
        if zone_of_indifference:
            level = OversightLevel.NONE
        elif friction >= 0.9:
            level = OversightLevel.DIRECT
        elif friction >= 0.7:
            level = OversightLevel.SUPERVISION
        elif friction >= 0.5:
            level = OversightLevel.APPROVAL
        elif friction >= 0.2:
            level = OversightLevel.NOTIFICATION
        else:
            level = OversightLevel.NONE

        # Mandatory checkpoint at max depth regardless of friction
        if chain_depth >= MAX_AUTONOMOUS_DEPTH and level == OversightLevel.NONE:
            level = OversightLevel.APPROVAL
            friction = max(friction, 0.5)

        requires_human = level in (
            OversightLevel.APPROVAL,
            OversightLevel.SUPERVISION,
            OversightLevel.DIRECT,
        )

        # Build reasoning
        reasons = []
        if profile.criticality > 0.7:
            reasons.append(f"high criticality ({profile.criticality:.2f})")
        if profile.reversibility < 0.3:
            reasons.append(f"low reversibility ({profile.reversibility:.2f})")
        if profile.uncertainty > 0.7:
            reasons.append(f"high uncertainty ({profile.uncertainty:.2f})")
        if chain_depth >= MAX_AUTONOMOUS_DEPTH:
            reasons.append(f"chain depth {chain_depth} >= max {MAX_AUTONOMOUS_DEPTH}")
        if zone_of_indifference:
            reasons.append("zone of indifference (routine task)")

        reason = "; ".join(reasons) if reasons else "standard oversight level"

        return OversightDecision(
            requires_human=requires_human,
            oversight_level=level,
            reason=reason,
            cognitive_friction_score=round(friction, 3),
            zone_of_indifference=zone_of_indifference,
            metadata={
                "criticality": profile.criticality,
                "reversibility": profile.reversibility,
                "uncertainty": profile.uncertainty,
                "chain_depth": chain_depth,
                "agent_trust": agent_trust,
                "trust_discount": round(trust_discount, 3),
                "depth_penalty": round(depth_penalty, 3),
            },
        )

    def record_approval(self, decision_time_seconds: float, approved: bool):
        """
        Record a human approval decision for rubber-stamp detection.

        If humans consistently approve in < 0.5s, they're rubber-stamping
        and the system should increase friction or consolidate approvals.

        Args:
            decision_time_seconds: How long the human took to decide
            approved: Whether they approved or rejected
        """
        self._approval_history.append({
            "timestamp": time.time(),
            "decision_time": decision_time_seconds,
            "approved": approved,
        })

        # Keep last 50 decisions for analysis
        if len(self._approval_history) > 50:
            self._approval_history = self._approval_history[-50:]

    def detect_rubber_stamping(self) -> Tuple[bool, float]:
        """
        Detect if human is rubber-stamping approvals.

        Returns:
            Tuple of (is_rubber_stamping: bool, avg_decision_time: float)
        """
        if len(self._approval_history) < 5:
            return False, 0.0

        recent = self._approval_history[-10:]
        avg_time = sum(d["decision_time"] for d in recent) / len(recent)
        approval_rate = sum(1 for d in recent if d["approved"]) / len(recent)

        # Rubber-stamping: fast approvals with high approval rate
        is_rubber_stamping = (
            avg_time < self._rubber_stamp_threshold and
            approval_rate > 0.95
        )

        return is_rubber_stamping, avg_time


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5.2: ACCOUNTABILITY IN LONG DELEGATION CHAINS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HandoffRecord:
    """Immutable record of a delegation handoff between agents."""
    handoff_id: str
    chain_id: str
    from_agent: str
    to_agent: str
    subtask_id: str
    timestamp: float
    permissions_granted: List[str]    # List of tool names granted
    access_level: str                 # AccessLevel value
    rationale: str                    # Why this delegation happened
    depth: int                        # Position in chain (0 = originator)


@dataclass
class LiabilityFirebreak:
    """
    Liability boundary in a delegation chain.

    Firebreaks mark points where:
    - Accountability transfers from delegator to delegatee
    - Human approval was obtained
    - Permissions were attenuated
    """
    firebreak_id: str
    chain_id: str
    position: int                     # Depth in chain where firebreak sits
    agent_id: str                     # Agent at this boundary
    human_approved: bool              # Was human approval obtained here?
    permissions_before: List[str]     # Permissions going in
    permissions_after: List[str]      # Permissions going out (attenuated)
    timestamp: float


class AccountabilityChain:
    """
    Accountability tracking for delegation chains (Section 5.2).

    Maintains immutable provenance records for every handoff in a
    delegation chain. Implements liability firebreaks at key points
    where accountability transfers between agents.

    Key principle from the paper: "If agent A delegates to B who delegates
    to C, and C causes harm, the accountability chain must be traceable
    back to A with clear liability boundaries at each handoff."
    """

    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.handoffs: List[HandoffRecord] = []
        self.firebreaks: List[LiabilityFirebreak] = []
        self.created_at = time.time()

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        subtask_id: str,
        permissions: List[str],
        access_level: str = "execute",
        rationale: str = "",
    ) -> HandoffRecord:
        """
        Record a delegation handoff between agents.

        Args:
            from_agent: Delegating agent ID
            to_agent: Receiving agent ID
            subtask_id: Subtask being delegated
            permissions: Tools/capabilities granted
            access_level: Access level granted
            rationale: Why this delegation happened

        Returns:
            HandoffRecord for the handoff
        """
        depth = len(self.handoffs)

        record = HandoffRecord(
            handoff_id=f"handoff-{uuid.uuid4().hex[:8]}",
            chain_id=self.chain_id,
            from_agent=from_agent,
            to_agent=to_agent,
            subtask_id=subtask_id,
            timestamp=time.time(),
            permissions_granted=permissions,
            access_level=access_level,
            rationale=rationale,
            depth=depth,
        )

        self.handoffs.append(record)

        # Auto-create firebreak at depth thresholds
        if depth > 0 and depth % MAX_AUTONOMOUS_DEPTH == 0:
            self._create_firebreak(to_agent, depth, permissions)

        return record

    def _create_firebreak(
        self,
        agent_id: str,
        position: int,
        permissions: List[str],
    ):
        """Create a liability firebreak at this position in the chain."""
        # Attenuate permissions at firebreak
        attenuated = permissions[:max(1, len(permissions) // 2)]

        firebreak = LiabilityFirebreak(
            firebreak_id=f"firebreak-{uuid.uuid4().hex[:8]}",
            chain_id=self.chain_id,
            position=position,
            agent_id=agent_id,
            human_approved=False,  # Will be updated when human approves
            permissions_before=permissions,
            permissions_after=attenuated,
            timestamp=time.time(),
        )

        self.firebreaks.append(firebreak)

    def get_provenance(self) -> Dict:
        """
        Get full provenance record for this delegation chain.

        Returns:
            Dict with chain_id, handoffs, firebreaks, depth, duration
        """
        return {
            "chain_id": self.chain_id,
            "total_handoffs": len(self.handoffs),
            "total_firebreaks": len(self.firebreaks),
            "max_depth": max((h.depth for h in self.handoffs), default=0),
            "handoffs": [
                {
                    "handoff_id": h.handoff_id,
                    "from": h.from_agent,
                    "to": h.to_agent,
                    "subtask": h.subtask_id,
                    "depth": h.depth,
                    "permissions": h.permissions_granted,
                    "access_level": h.access_level,
                    "timestamp": h.timestamp,
                }
                for h in self.handoffs
            ],
            "firebreaks": [
                {
                    "firebreak_id": f.firebreak_id,
                    "position": f.position,
                    "agent": f.agent_id,
                    "human_approved": f.human_approved,
                    "permissions_attenuated": len(f.permissions_before) > len(f.permissions_after),
                }
                for f in self.firebreaks
            ],
            "created_at": self.created_at,
            "duration": time.time() - self.created_at,
        }

    def get_liable_agent(self, depth: int) -> Optional[str]:
        """
        Determine which agent is liable at a given depth.

        Liability flows UP the chain to the nearest firebreak.
        If no firebreak exists above, the originator is liable.

        Args:
            depth: Depth in the chain where an issue occurred

        Returns:
            Agent ID of the liable party, or None if chain is empty
        """
        if not self.handoffs:
            return None

        # Find the nearest firebreak at or below this depth
        relevant_firebreaks = [
            f for f in self.firebreaks
            if f.position <= depth and f.human_approved
        ]

        if relevant_firebreaks:
            # Liability sits at the most recent approved firebreak
            nearest = max(relevant_firebreaks, key=lambda f: f.position)
            return nearest.agent_id

        # No firebreak → originator is liable
        return self.handoffs[0].from_agent


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5.3: TIERED SERVICE LEVELS
# ═══════════════════════════════════════════════════════════════════════════


class ServiceTierName(str, Enum):
    """Service tier levels for delegation."""
    BASIC = "basic"            # Low-cost, routine tasks
    STANDARD = "standard"      # Normal delegation with verification
    PREMIUM = "premium"        # Enhanced oversight and verification
    CRITICAL = "critical"      # Maximum assurance, human-gated


@dataclass
class ServiceTier:
    """
    Service tier configuration for a delegation task.

    Different tiers provide different levels of:
    - Verification rigor
    - Human oversight
    - Retry budget
    - Quality thresholds
    """
    name: ServiceTierName
    safety_floor: float           # Minimum quality score to accept (0.0-1.0)
    max_retries: int              # Maximum retry attempts
    verification_required: bool   # Whether verification is mandatory
    human_gate: bool              # Whether human approval is required
    oversight_level: OversightLevel
    estimated_cost_multiplier: float  # Cost multiplier vs basic tier


# Service tier definitions (Section 5.3)
SERVICE_TIERS = {
    ServiceTierName.BASIC: ServiceTier(
        name=ServiceTierName.BASIC,
        safety_floor=0.3,
        max_retries=1,
        verification_required=False,
        human_gate=False,
        oversight_level=OversightLevel.NONE,
        estimated_cost_multiplier=1.0,
    ),
    ServiceTierName.STANDARD: ServiceTier(
        name=ServiceTierName.STANDARD,
        safety_floor=0.5,
        max_retries=2,
        verification_required=True,
        human_gate=False,
        oversight_level=OversightLevel.NOTIFICATION,
        estimated_cost_multiplier=1.5,
    ),
    ServiceTierName.PREMIUM: ServiceTier(
        name=ServiceTierName.PREMIUM,
        safety_floor=0.7,
        max_retries=3,
        verification_required=True,
        human_gate=False,
        oversight_level=OversightLevel.APPROVAL,
        estimated_cost_multiplier=2.5,
    ),
    ServiceTierName.CRITICAL: ServiceTier(
        name=ServiceTierName.CRITICAL,
        safety_floor=0.9,
        max_retries=3,
        verification_required=True,
        human_gate=True,
        oversight_level=OversightLevel.SUPERVISION,
        estimated_cost_multiplier=4.0,
    ),
}


def get_service_tier(profile: TaskProfile) -> ServiceTier:
    """
    Determine appropriate service tier based on task profile.

    Tier selection logic:
    - CRITICAL: criticality > 0.8 OR reversibility < 0.2
    - PREMIUM: criticality > 0.6 OR complexity > 0.7
    - STANDARD: criticality > 0.3 OR complexity > 0.4
    - BASIC: everything else (routine, low-stakes)

    Args:
        profile: 11-dimensional task profile

    Returns:
        ServiceTier configuration for this task
    """
    # Critical tier: mission-critical or irreversible
    if profile.criticality > 0.8 or profile.reversibility < 0.2:
        return SERVICE_TIERS[ServiceTierName.CRITICAL]

    # Premium tier: high criticality or high complexity
    if profile.criticality > 0.6 or profile.complexity > 0.7:
        return SERVICE_TIERS[ServiceTierName.PREMIUM]

    # Standard tier: moderate stakes
    if profile.criticality > 0.3 or profile.complexity > 0.4:
        return SERVICE_TIERS[ServiceTierName.STANDARD]

    # Basic tier: routine tasks
    return SERVICE_TIERS[ServiceTierName.BASIC]


def enforce_safety_floor(
    quality_score: float,
    tier: ServiceTier,
) -> Tuple[bool, str]:
    """
    Enforce safety floor for a given service tier.

    Args:
        quality_score: Quality score from verification [0.0, 1.0]
        tier: Service tier with safety floor

    Returns:
        Tuple of (passes: bool, reason: str)
    """
    if quality_score >= tier.safety_floor:
        return True, f"Quality {quality_score:.3f} meets {tier.name.value} floor {tier.safety_floor}"

    return False, (
        f"Quality {quality_score:.3f} below {tier.name.value} "
        f"safety floor {tier.safety_floor} — requires re-execution or escalation"
    )
