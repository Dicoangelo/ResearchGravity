"""
Tests for Delegation Coordinator — Adaptive Multi-Agent Orchestration

Tests cover:
- Chain submission (classify → decompose → route pipeline)
- Status tracking (progress calculation, per-subtask status)
- Trigger detection (external and internal)
- Adaptive responses (RETRY → REROUTE → ESCALATE)
- Event capture (cognitive event logging)
- Async context manager pattern
"""

import asyncio
import pytest
import time
from delegation.coordinator import (
    DelegationCoordinator,
    TriggerType,
    ResponseAction,
    Trigger,
    ChainStatus,
)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Chain Submission
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_submit_chain_returns_chain_id():
    """Test that submit_chain returns a valid chain_id"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Build API server with auth")
        assert chain_id.startswith("chain-")
        assert len(chain_id) == 18  # "chain-" + 12 hex chars


@pytest.mark.asyncio
async def test_submit_chain_creates_subtasks():
    """Test that chain submission creates subtasks via decomposition"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Implement user authentication")
        status = await coordinator.get_chain_status(chain_id)

        # Should have subtasks from decomposition
        assert len(status["subtask_statuses"]) > 0
        assert status["status"] == "running"


@pytest.mark.asyncio
async def test_submit_chain_routes_to_agents():
    """Test that each subtask is assigned to an agent"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Research multi-agent systems")
        status = await coordinator.get_chain_status(chain_id)

        # Each subtask should have an agent_id
        for subtask_status in status["subtask_statuses"].values():
            assert subtask_status["agent_id"] is not None
            assert subtask_status["status"] == "pending"


@pytest.mark.asyncio
async def test_submit_chain_captures_event():
    """Test that chain submission captures a cognitive event"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Add new feature")
        status = await coordinator.get_chain_status(chain_id)

        # Should have at least the chain_submitted event
        assert len(status["events"]) >= 1
        assert status["events"][0]["type"] == "chain_submitted"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Status Tracking
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_chain_status_returns_all_fields():
    """Test that get_chain_status returns all required fields"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Test task")
        status = await coordinator.get_chain_status(chain_id)

        # Required fields
        assert "chain_id" in status
        assert "status" in status
        assert "progress" in status
        assert "subtask_statuses" in status
        assert "events" in status
        assert "triggers" in status
        assert "created_at" in status
        assert "updated_at" in status


@pytest.mark.asyncio
async def test_get_chain_status_calculates_progress():
    """Test that progress is calculated correctly"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Multi-step task")
        status = await coordinator.get_chain_status(chain_id)

        # Initial progress should be 0.0 (nothing completed)
        assert status["progress"] == 0.0

        # Simulate completing a subtask
        chain = coordinator.chains[chain_id]
        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            chain.subtask_statuses[subtask_ids[0]]["status"] = "completed"

            # Recalculate progress
            status = await coordinator.get_chain_status(chain_id)
            expected = 1.0 / len(subtask_ids)
            assert abs(status["progress"] - expected) < 0.01


@pytest.mark.asyncio
async def test_get_chain_status_raises_on_invalid_id():
    """Test that get_chain_status raises ValueError for invalid chain_id"""
    async with DelegationCoordinator() as coordinator:
        with pytest.raises(ValueError, match="Chain invalid-id not found"):
            await coordinator.get_chain_status("invalid-id")


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Trigger Detection
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_detect_trigger_progress_stall():
    """Test detection of progress stall trigger"""
    async with DelegationCoordinator(stall_timeout=0.1) as coordinator:
        chain_id = await coordinator.submit_chain("Test task")
        chain = coordinator.chains[chain_id]

        # Simulate a running subtask
        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            st_status = chain.subtask_statuses[subtask_ids[0]]
            st_status["status"] = "running"
            st_status["started_at"] = time.time()
            st_status["last_update"] = time.time() - 1.0  # Stalled 1 second ago

            # Detect triggers
            triggers = await coordinator._detect_triggers(
                chain_id, subtask_ids[0], st_status
            )

            # Should detect progress stall
            assert len(triggers) > 0
            assert any(t.type == TriggerType.PROGRESS_STALL for t in triggers)


@pytest.mark.asyncio
async def test_detect_trigger_api_timeout():
    """Test detection of API timeout trigger"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Test task")
        chain = coordinator.chains[chain_id]

        # Simulate a running subtask that's been running too long
        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            st_status = chain.subtask_statuses[subtask_ids[0]]
            st_status["status"] = "running"
            st_status["started_at"] = time.time() - 120.0  # Started 2 min ago
            st_status["last_update"] = time.time()

            # Detect triggers
            triggers = await coordinator._detect_triggers(
                chain_id, subtask_ids[0], st_status
            )

            # Should detect API timeout
            assert len(triggers) > 0
            assert any(t.type == TriggerType.API_TIMEOUT for t in triggers)


@pytest.mark.asyncio
async def test_detect_trigger_quality_below_threshold():
    """Test detection of quality below threshold trigger"""
    async with DelegationCoordinator(quality_threshold=0.7) as coordinator:
        chain_id = await coordinator.submit_chain("Test task")
        chain = coordinator.chains[chain_id]

        # Simulate a subtask with low quality verification
        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            st_status = chain.subtask_statuses[subtask_ids[0]]
            st_status["status"] = "verifying"
            st_status["verification"] = {"quality_score": 0.5}
            st_status["last_update"] = time.time()

            # Detect triggers
            triggers = await coordinator._detect_triggers(
                chain_id, subtask_ids[0], st_status
            )

            # Should detect quality issue
            assert len(triggers) > 0
            assert any(t.type == TriggerType.QUALITY_BELOW_THRESHOLD for t in triggers)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Adaptive Responses
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_respond_to_trigger_first_failure_retry():
    """Test that first failure triggers RETRY"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Test task")
        chain = coordinator.chains[chain_id]

        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            subtask_id = subtask_ids[0]
            trigger = Trigger(
                type=TriggerType.PROGRESS_STALL,
                subtask_id=subtask_id,
                timestamp=time.time(),
                details={}
            )

            # First failure should RETRY
            await coordinator._respond_to_trigger(chain_id, subtask_id, trigger)

            # Check status updated to retrying
            assert chain.subtask_statuses[subtask_id]["status"] == "retrying"
            assert coordinator.failure_counts[subtask_id] == 1


@pytest.mark.asyncio
async def test_respond_to_trigger_second_failure_reroute():
    """Test that second failure triggers REROUTE"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Test task")
        chain = coordinator.chains[chain_id]

        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            subtask_id = subtask_ids[0]

            # Simulate first failure already happened
            coordinator.failure_counts[subtask_id] = 1

            # Add fallback chain to metadata
            chain.subtask_statuses[subtask_id]["fallback_chain"] = ["agent-fallback-1"]

            trigger = Trigger(
                type=TriggerType.API_TIMEOUT,
                subtask_id=subtask_id,
                timestamp=time.time(),
                details={}
            )

            # Second failure should REROUTE
            await coordinator._respond_to_trigger(chain_id, subtask_id, trigger)

            # Check status updated to rerouted with new agent
            assert chain.subtask_statuses[subtask_id]["status"] == "rerouted"
            assert chain.subtask_statuses[subtask_id]["agent_id"] == "agent-fallback-1"
            assert coordinator.failure_counts[subtask_id] == 2


@pytest.mark.asyncio
async def test_respond_to_trigger_third_failure_escalate():
    """Test that third failure triggers ESCALATE"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Test task")
        chain = coordinator.chains[chain_id]

        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            subtask_id = subtask_ids[0]

            # Simulate two failures already happened
            coordinator.failure_counts[subtask_id] = 2

            trigger = Trigger(
                type=TriggerType.QUALITY_BELOW_THRESHOLD,
                subtask_id=subtask_id,
                timestamp=time.time(),
                details={}
            )

            # Third failure should ESCALATE
            await coordinator._respond_to_trigger(chain_id, subtask_id, trigger)

            # Check status escalated
            assert chain.subtask_statuses[subtask_id]["status"] == "escalated"
            assert chain.status == "escalated"
            assert coordinator.failure_counts[subtask_id] == 3


@pytest.mark.asyncio
async def test_respond_to_trigger_logs_event():
    """Test that trigger response logs event"""
    async with DelegationCoordinator() as coordinator:
        chain_id = await coordinator.submit_chain("Test task")
        chain = coordinator.chains[chain_id]

        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            subtask_id = subtask_ids[0]
            trigger = Trigger(
                type=TriggerType.PROGRESS_STALL,
                subtask_id=subtask_id,
                timestamp=time.time(),
                details={}
            )

            initial_event_count = len(chain.events)

            await coordinator._respond_to_trigger(chain_id, subtask_id, trigger)

            # Should have added trigger_response event
            assert len(chain.events) > initial_event_count
            latest_events = [e for e in chain.events if e.get("type") == "trigger_response"]
            assert len(latest_events) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Async Context Manager
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_context_manager_initializes_registry():
    """Test that context manager loads agent registry"""
    async with DelegationCoordinator() as coordinator:
        assert len(coordinator.agent_registry) > 0


@pytest.mark.asyncio
async def test_context_manager_starts_monitoring():
    """Test that context manager starts background monitoring"""
    async with DelegationCoordinator() as coordinator:
        assert coordinator._running is True
        assert coordinator._monitor_task is not None


@pytest.mark.asyncio
async def test_context_manager_cleanup():
    """Test that context manager stops monitoring on exit"""
    coordinator = DelegationCoordinator()
    async with coordinator:
        assert coordinator._running is True

    # After exit, monitoring should be stopped
    assert coordinator._running is False


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Integration Scenarios
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_integration_simulate_failure_and_recovery():
    """Integration test: simulate a failure, verify retry then reroute behavior"""
    async with DelegationCoordinator(check_interval=0.5, stall_timeout=0.2) as coordinator:
        # Submit chain
        chain_id = await coordinator.submit_chain("Build feature X")
        chain = coordinator.chains[chain_id]

        # Simulate a subtask that stalls
        subtask_ids = list(chain.subtask_statuses.keys())
        if subtask_ids:
            subtask_id = subtask_ids[0]
            st_status = chain.subtask_statuses[subtask_id]
            st_status["status"] = "running"
            st_status["started_at"] = time.time()
            st_status["last_update"] = time.time() - 1.0  # Stalled
            st_status["fallback_chain"] = ["agent-backup"]

            # Wait for monitoring to detect and respond
            await asyncio.sleep(0.7)  # Let monitor run

            # Check that trigger was detected and response logged
            status = await coordinator.get_chain_status(chain_id)
            assert len(status["triggers"]) > 0 or len(status["events"]) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
