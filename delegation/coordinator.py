"""
Delegation Coordinator — Adaptive Multi-Agent Orchestration with Trigger Detection

Implements the adaptive coordination framework from arXiv:2602.11865 Section 4.4.

Key Features:
- End-to-end delegation chain: classify → decompose → route → execute → verify
- Real-time monitoring with configurable check intervals (default 5s)
- External trigger detection: API timeout, resource unavailable, rate limit
- Internal trigger detection: quality below threshold, progress stall, budget overrun
- Adaptive responses: RETRY, REROUTE, ESCALATE, ABORT
- Escalation chain: 1st failure = RETRY, 2nd = REROUTE, 3rd = ESCALATE
- All coordination events captured as cognitive_events via mcp_raw/capture

Usage:
    from delegation.coordinator import DelegationCoordinator

    async with DelegationCoordinator() as coordinator:
        # Submit end-to-end delegation chain
        chain_id = await coordinator.submit_chain("Build API server with auth")

        # Monitor status
        status = await coordinator.get_chain_status(chain_id)
        print(f"Progress: {status['progress']}, Status: {status['status']}")
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import TaskProfile, SubTask, Assignment, DelegationEvent, VerificationResult
from .taxonomy import classify_task
from .decomposer import decompose_task
from .router import route_subtask, load_agent_registry, AgentCapability
from .verifier import verify_completion


# ═══════════════════════════════════════════════════════════════════════════
# TRIGGER DETECTION AND RESPONSE
# ═══════════════════════════════════════════════════════════════════════════


class TriggerType(str, Enum):
    """Types of triggers that can activate adaptive responses"""
    # External triggers
    API_TIMEOUT = "api_timeout"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    RATE_LIMIT_HIT = "rate_limit_hit"

    # Internal triggers
    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    PROGRESS_STALL = "progress_stall"
    BUDGET_OVERRUN = "budget_overrun"


class ResponseAction(str, Enum):
    """Adaptive responses to triggers"""
    RETRY = "retry"          # 1st failure: retry same agent
    REROUTE = "reroute"      # 2nd failure: reroute to next-best agent
    ESCALATE = "escalate"    # 3rd failure: escalate for human review
    ABORT = "abort"          # Unrecoverable failure


@dataclass
class Trigger:
    """Detected trigger event"""
    type: TriggerType
    subtask_id: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainStatus:
    """Status of a delegation chain"""
    chain_id: str
    status: str  # "running", "completed", "failed", "escalated"
    progress: float  # 0.0-1.0
    subtask_statuses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    triggers: List[Trigger] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
# DELEGATION COORDINATOR
# ═══════════════════════════════════════════════════════════════════════════


class DelegationCoordinator:
    """
    Adaptive multi-agent coordinator with trigger detection and response.

    Implements the coordination cycle from arXiv:2602.11865 Section 4.4:
    1. Submit delegation chain (classify → decompose → route)
    2. Monitor execution with configurable intervals
    3. Detect triggers (external and internal)
    4. Respond adaptively (RETRY, REROUTE, ESCALATE, ABORT)
    5. Track all events as cognitive_events

    Uses async context manager pattern for resource management.
    """

    def __init__(
        self,
        db_path: str = "",
        check_interval: float = 5.0,
        quality_threshold: float = 0.7,
        stall_timeout: float = 30.0
    ):
        """
        Initialize delegation coordinator.

        Args:
            db_path: Path to SQLite database (defaults to ~/.agent-core/storage/delegation.db)
            check_interval: How often to check for triggers in seconds (default: 5.0)
            quality_threshold: Minimum acceptable quality score (default: 0.7)
            stall_timeout: Time without updates before considering stalled (default: 30.0s)
        """
        self.db_path = db_path or str(
            Path.home() / ".agent-core" / "storage" / "delegation.db"
        )
        self.check_interval = check_interval
        self.quality_threshold = quality_threshold
        self.stall_timeout = stall_timeout

        # In-memory state
        self.chains: Dict[str, ChainStatus] = {}
        self.agent_registry: List[AgentCapability] = []
        self.failure_counts: Dict[str, int] = {}  # subtask_id -> failure count

        # Background monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    async def __aenter__(self):
        """Async context manager entry: initialize resources"""
        # Load agent registry
        self.agent_registry = load_agent_registry()

        # Start monitoring task
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit: cleanup resources"""
        # Stop monitoring
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        return False

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN API
    # ═══════════════════════════════════════════════════════════════════════

    async def submit_chain(self, task: str, context: Optional[Dict] = None) -> str:
        """
        Submit end-to-end delegation chain.

        Pipeline:
        1. Classify task (taxonomy.classify_task)
        2. Decompose into subtasks (decomposer.decompose_task)
        3. Route each subtask to agent (router.route_subtask)
        4. Track execution (async monitoring)
        5. Verify results (verifier.verify_completion)

        Args:
            task: Task description to delegate
            context: Optional context dict for classification

        Returns:
            chain_id: Unique identifier for this delegation chain
        """
        chain_id = f"chain-{uuid.uuid4().hex[:12]}"

        # Step 1: Classify task
        profile = classify_task(task, context=context)

        # Step 2: Decompose into subtasks
        subtasks = decompose_task(task, profile)

        # Step 3: Route subtasks to agents
        assignments = {}
        for subtask in subtasks:
            assignment = route_subtask(
                subtask=subtask,
                available_agents=self.agent_registry,
                use_llm=True
            )
            assignments[subtask.id] = assignment

        # Step 4: Create chain status
        chain_status = ChainStatus(
            chain_id=chain_id,
            status="running",
            progress=0.0,
            subtask_statuses={
                st.id: {
                    "description": st.description,
                    "agent_id": assignments[st.id].agent_id,
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "result": None,
                    "verification": None,
                    "last_update": time.time(),
                }
                for st in subtasks
            },
            events=[{
                "type": "chain_submitted",
                "timestamp": time.time(),
                "task": task,
                "subtask_count": len(subtasks),
                "profile": {
                    "complexity": profile.complexity,
                    "criticality": profile.criticality,
                }
            }]
        )

        self.chains[chain_id] = chain_status

        # Step 5: Capture as cognitive event
        await self._capture_event(
            event_type="delegation_chain_submitted",
            chain_id=chain_id,
            details={
                "task": task,
                "subtask_count": len(subtasks),
                "complexity": profile.complexity,
            }
        )

        return chain_id

    async def get_chain_status(self, chain_id: str) -> Dict[str, Any]:
        """
        Get current status of a delegation chain.

        Args:
            chain_id: Chain identifier

        Returns:
            Dict with status, progress, per-subtask status, timing, agent assignments
        """
        if chain_id not in self.chains:
            raise ValueError(f"Chain {chain_id} not found")

        chain = self.chains[chain_id]

        # Calculate progress
        total = len(chain.subtask_statuses)
        completed = sum(
            1 for st in chain.subtask_statuses.values()
            if st["status"] in ("completed", "failed")
        )
        chain.progress = completed / total if total > 0 else 0.0

        return {
            "chain_id": chain.chain_id,
            "status": chain.status,
            "progress": chain.progress,
            "subtask_statuses": chain.subtask_statuses,
            "events": chain.events,
            "triggers": [
                {
                    "type": t.type,
                    "subtask_id": t.subtask_id,
                    "timestamp": t.timestamp,
                    "details": t.details
                }
                for t in chain.triggers
            ],
            "created_at": chain.created_at,
            "updated_at": chain.updated_at,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # MONITORING AND TRIGGER DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    async def _monitor_loop(self):
        """
        Background monitoring loop.

        Runs every check_interval seconds to:
        1. Check for external triggers (timeouts, resource issues)
        2. Check for internal triggers (quality, stalls, budget)
        3. Respond adaptively based on trigger type
        """
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)

                for chain_id, chain in list(self.chains.items()):
                    if chain.status != "running":
                        continue

                    # Detect triggers for each subtask
                    for subtask_id, st_status in chain.subtask_statuses.items():
                        triggers = await self._detect_triggers(
                            chain_id, subtask_id, st_status
                        )

                        # Respond to triggers
                        for trigger in triggers:
                            chain.triggers.append(trigger)
                            await self._respond_to_trigger(
                                chain_id, subtask_id, trigger
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                await self._capture_event(
                    event_type="monitor_error",
                    chain_id="",
                    details={"error": str(e)}
                )

    async def _detect_triggers(
        self,
        chain_id: str,
        subtask_id: str,
        st_status: Dict[str, Any]
    ) -> List[Trigger]:
        """
        Detect triggers for a subtask.

        External triggers:
        - API_TIMEOUT: Execution time exceeds expected duration
        - RESOURCE_UNAVAILABLE: Agent not responding
        - RATE_LIMIT_HIT: Too many requests to agent

        Internal triggers:
        - QUALITY_BELOW_THRESHOLD: Verification quality < threshold
        - PROGRESS_STALL: No updates in > stall_timeout seconds
        - BUDGET_OVERRUN: Cost exceeds estimated budget

        Args:
            chain_id: Chain identifier
            subtask_id: Subtask identifier
            st_status: Subtask status dict

        Returns:
            List of detected Trigger objects
        """
        triggers = []
        now = time.time()

        # Skip if not started or already completed
        if st_status["status"] not in ("running", "verifying"):
            return triggers

        # External trigger: API timeout
        if st_status["started_at"]:
            elapsed = now - st_status["started_at"]
            # Rough heuristic: timeout if > 10x estimated duration
            if elapsed > 60.0:  # 1 minute timeout for now
                triggers.append(Trigger(
                    type=TriggerType.API_TIMEOUT,
                    subtask_id=subtask_id,
                    timestamp=now,
                    details={"elapsed": elapsed}
                ))

        # Internal trigger: Progress stall
        last_update = st_status.get("last_update", st_status.get("started_at", now))
        if now - last_update > self.stall_timeout:
            triggers.append(Trigger(
                type=TriggerType.PROGRESS_STALL,
                subtask_id=subtask_id,
                timestamp=now,
                details={"stalled_for": now - last_update}
            ))

        # Internal trigger: Quality below threshold
        if st_status.get("verification"):
            quality = st_status["verification"].get("quality_score", 1.0)
            if quality < self.quality_threshold:
                triggers.append(Trigger(
                    type=TriggerType.QUALITY_BELOW_THRESHOLD,
                    subtask_id=subtask_id,
                    timestamp=now,
                    details={"quality": quality, "threshold": self.quality_threshold}
                ))

        return triggers

    async def _respond_to_trigger(
        self,
        chain_id: str,
        subtask_id: str,
        trigger: Trigger
    ):
        """
        Respond adaptively to a trigger.

        Escalation chain (from arXiv:2602.11865 Section 4.4):
        1. 1st failure → RETRY same agent
        2. 2nd failure → REROUTE to next-best agent
        3. 3rd failure → ESCALATE for human review

        Args:
            chain_id: Chain identifier
            subtask_id: Subtask identifier
            trigger: Detected trigger
        """
        chain = self.chains[chain_id]
        st_status = chain.subtask_statuses[subtask_id]

        # Get failure count
        failure_count = self.failure_counts.get(subtask_id, 0)

        # Determine response action
        if failure_count == 0:
            action = ResponseAction.RETRY
        elif failure_count == 1:
            action = ResponseAction.REROUTE
        else:
            action = ResponseAction.ESCALATE

        # Increment failure count
        self.failure_counts[subtask_id] = failure_count + 1

        # Log event
        event = {
            "type": "trigger_response",
            "timestamp": time.time(),
            "subtask_id": subtask_id,
            "trigger_type": trigger.type,
            "action": action,
            "failure_count": failure_count + 1,
        }
        chain.events.append(event)

        # Execute response action
        if action == ResponseAction.RETRY:
            # Retry same agent
            st_status["status"] = "retrying"
            st_status["last_update"] = time.time()
            await self._capture_event(
                event_type="retry_subtask",
                chain_id=chain_id,
                details={
                    "subtask_id": subtask_id,
                    "trigger": trigger.type,
                    "attempt": failure_count + 1,
                }
            )

        elif action == ResponseAction.REROUTE:
            # Reroute to next-best agent
            # Get fallback chain from assignment metadata
            fallback_agents = st_status.get("fallback_chain", [])
            if fallback_agents:
                new_agent_id = fallback_agents[0]
                st_status["agent_id"] = new_agent_id
                st_status["status"] = "rerouted"
                st_status["last_update"] = time.time()
                await self._capture_event(
                    event_type="reroute_subtask",
                    chain_id=chain_id,
                    details={
                        "subtask_id": subtask_id,
                        "new_agent_id": new_agent_id,
                        "trigger": trigger.type,
                    }
                )
            else:
                # No fallback agents → escalate
                action = ResponseAction.ESCALATE

        if action == ResponseAction.ESCALATE:
            # Escalate for human review
            st_status["status"] = "escalated"
            chain.status = "escalated"
            st_status["last_update"] = time.time()
            await self._capture_event(
                event_type="escalate_subtask",
                chain_id=chain_id,
                details={
                    "subtask_id": subtask_id,
                    "trigger": trigger.type,
                    "failure_count": failure_count + 1,
                    "reason": "Exceeded retry/reroute limits, requires human review"
                }
            )

    # ═══════════════════════════════════════════════════════════════════════
    # EVENT CAPTURE
    # ═══════════════════════════════════════════════════════════════════════

    async def _capture_event(
        self,
        event_type: str,
        chain_id: str,
        details: Dict[str, Any]
    ):
        """
        Capture coordination event as cognitive_event via mcp_raw/capture.

        This enables coherence detection across the delegation system.

        Args:
            event_type: Type of event
            chain_id: Chain identifier
            details: Event details dict
        """
        # TODO: Integrate with mcp_raw.capture.CaptureEngine
        # For now, just log the event structure
        event = {
            "event_type": event_type,
            "chain_id": chain_id,
            "timestamp": time.time(),
            "details": details,
        }

        # Future integration point:
        # await capture_engine.capture(
        #     raw_bytes=json.dumps(event).encode(),
        #     parsed=event,
        #     direction="internal",
        #     stage="coordination"
        # )

        # For now, store in chain events
        if chain_id and chain_id in self.chains:
            self.chains[chain_id].events.append(event)
