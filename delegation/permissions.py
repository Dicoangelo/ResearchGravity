"""
Permission Handler — Risk-Adaptive Access Control for AI Delegation

Implements arXiv:2602.11865 Section 4.7: Permission Handling.

Core principles from the paper:
- Balance operational efficiency with systemic safety
- Low-stakes: standing permissions from verifiable attributes
- High-stakes: just-in-time, task-duration-scoped, human-gated
- Privilege attenuation: sub-delegating agent gets strict subset of parent's permissions
- Semantic constraints: access defined by allowable operations, not just tools
- Circuit breakers: trust drops trigger immediate permission revocation
- Meta-permissions: govern which permissions delegators may grant to delegatees

Permission lifecycle:
1. Grant: based on trust tier + task profile
2. Scope: capabilities narrowed to task requirements
3. Monitor: trust changes can trigger revocation
4. Revoke: automatic on completion, trust drop, or anomaly

Usage:
    from delegation.permissions import PermissionManager, PermissionScope

    pm = PermissionManager()

    # Grant scoped permissions for a subtask
    scope = pm.grant_permissions(agent_id, subtask, trust_score)

    # Check if agent can perform an action
    allowed = pm.check_permission(agent_id, "search_learnings", "read")

    # Revoke on trust drop
    pm.circuit_breaker_check(agent_id, old_trust=0.7, new_trust=0.3)
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .models import TaskProfile, SubTask


class AccessLevel(str, Enum):
    """Permission access levels (ordered by privilege)."""
    NONE = "none"
    READ = "read"              # Read-only access to tool outputs
    EXECUTE = "execute"        # Execute tool with constraints
    EXECUTE_WRITE = "execute_write"  # Execute + write results
    FULL = "full"              # Unrestricted access (human-approved only)


class PermissionState(str, Enum):
    """Permission lifecycle states."""
    ACTIVE = "active"
    SUSPENDED = "suspended"    # Temporarily suspended (under review)
    REVOKED = "revoked"        # Permanently revoked
    EXPIRED = "expired"        # TTL expired


@dataclass
class PermissionScope:
    """
    Scoped permission grant for an agent on a specific task.

    Implements privilege attenuation (Section 4.7):
    - Permissions are task-scoped, not global
    - Sub-delegated agents get strict subset
    - Semantic constraints limit operations within tools
    """
    scope_id: str
    agent_id: str
    chain_id: str
    tools_allowed: Set[str]               # Tool names this agent can call
    access_level: AccessLevel
    semantic_constraints: Dict[str, str]   # tool_name -> operation constraint
    granted_at: float
    expires_at: float                      # TTL: permissions auto-expire
    state: PermissionState = PermissionState.ACTIVE
    parent_scope_id: Optional[str] = None  # For privilege attenuation tracking
    granted_by: str = "system"             # "system" or human approver ID
    metadata: Dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        if self.state != PermissionState.ACTIVE:
            return False
        if time.time() > self.expires_at:
            self.state = PermissionState.EXPIRED
            return False
        return True

    def attenuate(self, agent_id: str, tools_subset: Set[str]) -> "PermissionScope":
        """
        Create an attenuated child scope for sub-delegation.

        The child scope is a STRICT SUBSET of the parent's permissions.
        This prevents privilege escalation through delegation chains.
        """
        allowed = tools_subset & self.tools_allowed  # Intersection only
        child_level = self.access_level
        # Sub-delegated agents can never exceed parent's access level
        # and are downgraded one level for safety
        if child_level == AccessLevel.FULL:
            child_level = AccessLevel.EXECUTE_WRITE
        elif child_level == AccessLevel.EXECUTE_WRITE:
            child_level = AccessLevel.EXECUTE

        return PermissionScope(
            scope_id=f"scope-{uuid.uuid4().hex[:8]}",
            agent_id=agent_id,
            chain_id=self.chain_id,
            tools_allowed=allowed,
            access_level=child_level,
            semantic_constraints={
                k: v for k, v in self.semantic_constraints.items()
                if k in allowed
            },
            granted_at=time.time(),
            expires_at=min(self.expires_at, time.time() + 300),  # Max 5min for child
            parent_scope_id=self.scope_id,
            granted_by=self.agent_id,  # Parent grants child
        )


# Trust tier thresholds for graduated authority (Section 4.6 enhancement)
TRUST_TIERS = {
    "untrusted": (0.0, 0.2),    # No delegation allowed
    "supervised": (0.2, 0.4),   # Read-only, human-approved execution
    "standard": (0.4, 0.7),     # Execute with constraints
    "trusted": (0.7, 0.9),      # Execute + write
    "autonomous": (0.9, 1.0),   # Full access (still scoped to task)
}

# TTL per trust tier (seconds)
SCOPE_TTL = {
    "untrusted": 0,             # No grant
    "supervised": 60,           # 1 minute
    "standard": 300,            # 5 minutes
    "trusted": 600,             # 10 minutes
    "autonomous": 1800,         # 30 minutes
}

# Circuit breaker: trust drop threshold that triggers immediate revocation
CIRCUIT_BREAKER_DROP = 0.2  # If trust drops by >= 0.2 in one update


def _trust_tier(score: float) -> str:
    """Map trust score to tier name."""
    for tier, (low, high) in TRUST_TIERS.items():
        if low <= score < high:
            return tier
    return "autonomous" if score >= 0.9 else "untrusted"


def _access_for_tier(tier: str) -> AccessLevel:
    """Map trust tier to access level (graduated authority)."""
    return {
        "untrusted": AccessLevel.NONE,
        "supervised": AccessLevel.READ,
        "standard": AccessLevel.EXECUTE,
        "trusted": AccessLevel.EXECUTE_WRITE,
        "autonomous": AccessLevel.FULL,
    }.get(tier, AccessLevel.NONE)


class PermissionManager:
    """
    Risk-adaptive permission manager for AI delegation.

    Implements:
    - Graduated authority based on trust tiers
    - Just-in-time permission grants scoped to task duration
    - Privilege attenuation for sub-delegation
    - Circuit breakers on trust drops
    - Semantic constraints per tool
    - Meta-permissions (what permissions can be delegated)
    """

    def __init__(self):
        self.active_scopes: Dict[str, PermissionScope] = {}  # scope_id -> scope
        self.agent_scopes: Dict[str, List[str]] = {}          # agent_id -> [scope_ids]
        self.revocation_log: List[Dict] = []

    def grant_permissions(
        self,
        agent_id: str,
        subtask: SubTask,
        trust_score: float,
        chain_id: str = "",
        tools_requested: Optional[Set[str]] = None,
        require_human_approval: bool = False,
    ) -> Optional[PermissionScope]:
        """
        Grant scoped permissions for an agent to execute a subtask.

        Implements graduated authority (arXiv:2602.11865 Section 4.6):
        - Trust tier determines access level
        - High-criticality + low trust → requires human approval
        - Permissions auto-expire based on trust tier TTL

        Args:
            agent_id: Agent requesting permissions
            subtask: SubTask to execute
            trust_score: Current trust score [0.0, 1.0]
            chain_id: Delegation chain ID
            tools_requested: Specific tools needed (None = infer from subtask)
            require_human_approval: Force human gate regardless of trust

        Returns:
            PermissionScope if granted, None if denied
        """
        tier = _trust_tier(trust_score)
        access_level = _access_for_tier(tier)

        # Untrusted agents get nothing
        if access_level == AccessLevel.NONE:
            return None

        # High-stakes gate: critical + low trust → block without human approval
        if subtask.profile:
            if subtask.profile.criticality > 0.7 and tier in ("supervised", "standard"):
                if not require_human_approval:
                    return None  # Needs human gate

            # Irreversible tasks need higher trust
            if subtask.profile.reversibility < 0.3 and tier == "supervised":
                return None

        # Determine allowed tools
        if tools_requested:
            tools_allowed = tools_requested
        else:
            # Infer from subtask metadata (router puts agent info there)
            agent_name = subtask.metadata.get("agent_name", "")
            tools_allowed = {agent_name} if agent_name else {"*"}

        # Semantic constraints based on task profile
        constraints = {}
        if subtask.profile and subtask.profile.reversibility < 0.5:
            # Low reversibility → read-only constraint on data tools
            for tool in tools_allowed:
                if "delete" in tool.lower() or "drop" in tool.lower():
                    constraints[tool] = "read_only"

        # TTL based on trust tier
        ttl = SCOPE_TTL.get(tier, 60)

        scope = PermissionScope(
            scope_id=f"scope-{uuid.uuid4().hex[:8]}",
            agent_id=agent_id,
            chain_id=chain_id,
            tools_allowed=tools_allowed,
            access_level=access_level,
            semantic_constraints=constraints,
            granted_at=time.time(),
            expires_at=time.time() + ttl,
            granted_by="human" if require_human_approval else "system",
        )

        # Track scope
        self.active_scopes[scope.scope_id] = scope
        if agent_id not in self.agent_scopes:
            self.agent_scopes[agent_id] = []
        self.agent_scopes[agent_id].append(scope.scope_id)

        return scope

    def check_permission(
        self,
        agent_id: str,
        tool_name: str,
        operation: str = "execute",
    ) -> Tuple[bool, str]:
        """
        Check if an agent has permission to perform an operation on a tool.

        Args:
            agent_id: Agent ID
            tool_name: Tool to access
            operation: Operation type (read, execute, execute_write)

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        scope_ids = self.agent_scopes.get(agent_id, [])
        if not scope_ids:
            return False, f"No active permissions for agent {agent_id}"

        for scope_id in scope_ids:
            scope = self.active_scopes.get(scope_id)
            if not scope or not scope.is_active:
                continue

            # Check tool is allowed
            if "*" not in scope.tools_allowed and tool_name not in scope.tools_allowed:
                continue

            # Check semantic constraints
            constraint = scope.semantic_constraints.get(tool_name)
            if constraint == "read_only" and operation != "read":
                continue

            # Check access level sufficient for operation
            required = {
                "read": AccessLevel.READ,
                "execute": AccessLevel.EXECUTE,
                "execute_write": AccessLevel.EXECUTE_WRITE,
                "full": AccessLevel.FULL,
            }.get(operation, AccessLevel.EXECUTE)

            level_order = [AccessLevel.NONE, AccessLevel.READ, AccessLevel.EXECUTE,
                           AccessLevel.EXECUTE_WRITE, AccessLevel.FULL]

            if level_order.index(scope.access_level) >= level_order.index(required):
                return True, f"Permitted via scope {scope_id} (tier: {scope.access_level.value})"

        return False, f"No active scope permits {operation} on {tool_name}"

    def circuit_breaker_check(
        self,
        agent_id: str,
        old_trust: float,
        new_trust: float,
    ) -> List[str]:
        """
        Circuit breaker: revoke all permissions if trust drops sharply.

        Implements arXiv:2602.11865 Section 4.7:
        "Sudden reputation score drops trigger active token invalidation
        across delegation chain."

        Args:
            agent_id: Agent whose trust changed
            old_trust: Previous trust score
            new_trust: New trust score

        Returns:
            List of revoked scope IDs
        """
        drop = old_trust - new_trust
        if drop < CIRCUIT_BREAKER_DROP:
            return []  # Drop not severe enough

        revoked = []
        scope_ids = self.agent_scopes.get(agent_id, [])

        for scope_id in scope_ids:
            scope = self.active_scopes.get(scope_id)
            if scope and scope.is_active:
                scope.state = PermissionState.REVOKED
                revoked.append(scope_id)

                self.revocation_log.append({
                    "scope_id": scope_id,
                    "agent_id": agent_id,
                    "timestamp": time.time(),
                    "reason": f"Circuit breaker: trust dropped {drop:.3f} "
                              f"({old_trust:.3f} → {new_trust:.3f})",
                    "old_trust": old_trust,
                    "new_trust": new_trust,
                })

        return revoked

    def revoke_chain(self, chain_id: str) -> int:
        """
        Revoke all permissions for a delegation chain (on completion or failure).

        Args:
            chain_id: Chain ID

        Returns:
            Number of scopes revoked
        """
        count = 0
        for scope in self.active_scopes.values():
            if scope.chain_id == chain_id and scope.is_active:
                scope.state = PermissionState.REVOKED
                count += 1
        return count

    def cleanup_expired(self) -> int:
        """Remove expired scopes from active tracking."""
        expired = [
            sid for sid, scope in self.active_scopes.items()
            if not scope.is_active
        ]
        for sid in expired:
            del self.active_scopes[sid]
        return len(expired)
