"""
Tests for delegation.trust_ledger

Covers:
- Bayesian trust score calculation
- Record outcome and trust updates
- New agent initialization (uninformative prior)
- Trust score convergence with multiple outcomes
- Time decay for stale entries
- Top agents ranking
- Database operations
"""

import pytest
import asyncio
import time
import tempfile
import os
from pathlib import Path
from delegation.trust_ledger import TrustLedger, AgentTrustScore


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


class TestTrustLedgerBasics:
    """Test basic trust ledger functionality"""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_db):
        """Test trust ledger initialization"""
        async with TrustLedger(db_path=temp_db) as ledger:
            assert ledger.db_path == Path(temp_db)
            assert ledger._db is not None

    @pytest.mark.asyncio
    async def test_new_agent_uninformative_prior(self, temp_db):
        """Test new agents start at 0.5 trust (uninformative prior)"""
        async with TrustLedger(db_path=temp_db) as ledger:
            score = await ledger.get_trust_score("new-agent", "code_generation")
            assert score == 0.5

    @pytest.mark.asyncio
    async def test_record_first_outcome_success(self, temp_db):
        """Test recording first successful outcome"""
        async with TrustLedger(db_path=temp_db) as ledger:
            score = await ledger.record_outcome(
                agent_id="agent-1",
                task_type="code_generation",
                success=True,
                quality=0.9,
                duration=120.0
            )
            # Beta(2, 1) → 2/3 = 0.667
            assert abs(score - 0.667) < 0.001

    @pytest.mark.asyncio
    async def test_record_first_outcome_failure(self, temp_db):
        """Test recording first failed outcome"""
        async with TrustLedger(db_path=temp_db) as ledger:
            score = await ledger.record_outcome(
                agent_id="agent-2",
                task_type="code_generation",
                success=False,
                quality=0.3,
                duration=180.0
            )
            # Beta(1, 2) → 1/3 = 0.333
            assert abs(score - 0.333) < 0.001


class TestBayesianUpdates:
    """Test Bayesian trust score updates"""

    @pytest.mark.asyncio
    async def test_ten_successes_zero_failures(self, temp_db):
        """Test trust converges to ~0.917 after 10 successes, 0 failures"""
        async with TrustLedger(db_path=temp_db) as ledger:
            agent_id = "expert-agent"
            task_type = "code_generation"

            for i in range(10):
                score = await ledger.record_outcome(
                    agent_id=agent_id,
                    task_type=task_type,
                    success=True,
                    quality=0.9 + i * 0.01,  # Vary quality slightly
                    duration=100.0 + i * 5
                )

            # Beta(11, 1) → 11/12 = 0.917
            assert abs(score - 0.917) < 0.001

    @pytest.mark.asyncio
    async def test_five_successes_five_failures(self, temp_db):
        """Test trust stays at 0.5 with equal successes and failures"""
        async with TrustLedger(db_path=temp_db) as ledger:
            agent_id = "mediocre-agent"
            task_type = "code_generation"

            for i in range(10):
                score = await ledger.record_outcome(
                    agent_id=agent_id,
                    task_type=task_type,
                    success=(i % 2 == 0),  # Alternate success/failure
                    quality=0.5,
                    duration=100.0
                )

            # Beta(6, 6) → 6/12 = 0.5
            assert abs(score - 0.5) < 0.001

    @pytest.mark.asyncio
    async def test_eight_successes_two_failures(self, temp_db):
        """Test trust converges to 0.75 with 8 successes, 2 failures"""
        async with TrustLedger(db_path=temp_db) as ledger:
            agent_id = "good-agent"
            task_type = "code_generation"

            # 8 successes
            for i in range(8):
                await ledger.record_outcome(
                    agent_id=agent_id,
                    task_type=task_type,
                    success=True,
                    quality=0.8,
                    duration=100.0
                )

            # 2 failures
            for i in range(2):
                score = await ledger.record_outcome(
                    agent_id=agent_id,
                    task_type=task_type,
                    success=False,
                    quality=0.4,
                    duration=150.0
                )

            # Beta(9, 3) → 9/12 = 0.75
            assert abs(score - 0.75) < 0.001

    @pytest.mark.asyncio
    async def test_trust_converges_correctly(self, temp_db):
        """Test trust score converges correctly over multiple outcomes"""
        async with TrustLedger(db_path=temp_db) as ledger:
            agent_id = "converging-agent"
            task_type = "research"

            # Initial success
            score1 = await ledger.record_outcome(agent_id, task_type, True, 0.9, 100.0)
            assert abs(score1 - 0.667) < 0.001  # Beta(2,1) = 2/3

            # Second success
            score2 = await ledger.record_outcome(agent_id, task_type, True, 0.85, 120.0)
            assert abs(score2 - 0.75) < 0.001  # Beta(3,1) = 3/4

            # First failure
            score3 = await ledger.record_outcome(agent_id, task_type, False, 0.4, 200.0)
            assert abs(score3 - 0.6) < 0.001  # Beta(3,2) = 3/5

            # Third success
            score4 = await ledger.record_outcome(agent_id, task_type, True, 0.88, 110.0)
            assert abs(score4 - 0.667) < 0.001  # Beta(4,2) = 4/6


class TestDecayFunction:
    """Test time decay for stale entries"""

    @pytest.mark.asyncio
    async def test_no_decay_for_recent_entries(self, temp_db):
        """Test no decay applied to recently updated entries"""
        async with TrustLedger(db_path=temp_db) as ledger:
            agent_id = "recent-agent"
            task_type = "code_generation"

            # Record outcome
            original_score = await ledger.record_outcome(agent_id, task_type, True, 0.9, 100.0)

            # Retrieve immediately
            retrieved_score = await ledger.get_trust_score(agent_id, task_type)

            assert abs(retrieved_score - original_score) < 0.001

    @pytest.mark.asyncio
    async def test_decay_applied_to_stale_entries(self, temp_db):
        """Test decay applied to entries not updated in 7+ days"""
        async with TrustLedger(db_path=temp_db) as ledger:
            agent_id = "stale-agent"
            task_type = "code_generation"

            # Record outcome with 10 successes → trust = 0.917
            for _ in range(10):
                await ledger.record_outcome(agent_id, task_type, True, 0.9, 100.0)

            # Manually update last_updated to 8 days ago
            eight_days_ago = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.gmtime(time.time() - 8 * 24 * 3600)
            )
            await ledger._db.execute(
                "UPDATE trust_entries SET last_updated = ? WHERE agent_id = ? AND task_type = ?",
                (eight_days_ago, agent_id, task_type)
            )
            await ledger._db.commit()

            # Retrieve trust score (should have decay applied)
            decayed_score = await ledger.get_trust_score(agent_id, task_type)

            # Original: 0.917, after decay: 0.917 * 0.95 = 0.871
            expected_decayed = 0.917 * 0.95
            assert abs(decayed_score - expected_decayed) < 0.01


class TestTopAgents:
    """Test top agents ranking"""

    @pytest.mark.asyncio
    async def test_get_top_agents_empty(self, temp_db):
        """Test get_top_agents with no agents"""
        async with TrustLedger(db_path=temp_db) as ledger:
            top = await ledger.get_top_agents("code_generation", limit=5)
            assert top == []

    @pytest.mark.asyncio
    async def test_get_top_agents_single_task_type(self, temp_db):
        """Test get_top_agents for a single task type"""
        async with TrustLedger(db_path=temp_db) as ledger:
            task_type = "code_generation"

            # Agent 1: 8 successes → trust = 9/10 = 0.9
            for _ in range(8):
                await ledger.record_outcome("agent-1", task_type, True, 0.9, 100.0)

            # Agent 2: 5 successes → trust = 6/7 = 0.857
            for _ in range(5):
                await ledger.record_outcome("agent-2", task_type, True, 0.85, 120.0)

            # Agent 3: 3 successes, 2 failures → trust = 4/6 = 0.667
            for _ in range(3):
                await ledger.record_outcome("agent-3", task_type, True, 0.8, 110.0)
            for _ in range(2):
                await ledger.record_outcome("agent-3", task_type, False, 0.5, 150.0)

            # Get top agents
            top = await ledger.get_top_agents(task_type, limit=5)

            assert len(top) == 3
            assert top[0][0] == "agent-1"  # Highest trust
            assert abs(top[0][1] - 0.9) < 0.001
            assert top[1][0] == "agent-2"
            assert abs(top[1][1] - 0.857) < 0.01
            assert top[2][0] == "agent-3"
            # Beta(4,3) = 4/7 = 0.571
            assert abs(top[2][1] - 0.571) < 0.01

    @pytest.mark.asyncio
    async def test_get_top_agents_respects_limit(self, temp_db):
        """Test get_top_agents respects limit parameter"""
        async with TrustLedger(db_path=temp_db) as ledger:
            task_type = "research"

            # Create 10 agents with varying trust
            for i in range(10):
                for _ in range(i + 1):  # More successes = higher trust
                    await ledger.record_outcome(f"agent-{i}", task_type, True, 0.8, 100.0)

            # Get top 3
            top = await ledger.get_top_agents(task_type, limit=3)

            assert len(top) == 3
            assert top[0][0] == "agent-9"  # Highest trust (10 successes)
            assert top[1][0] == "agent-8"
            assert top[2][0] == "agent-7"


class TestTaskTypeSpecificity:
    """Test task-type-specific trust tracking"""

    @pytest.mark.asyncio
    async def test_different_trust_per_task_type(self, temp_db):
        """Test agents can have different trust scores for different task types"""
        async with TrustLedger(db_path=temp_db) as ledger:
            agent_id = "specialist-agent"

            # High performance on code_generation
            for _ in range(8):
                await ledger.record_outcome(agent_id, "code_generation", True, 0.9, 100.0)

            # Low performance on research
            for _ in range(6):
                await ledger.record_outcome(agent_id, "research", False, 0.4, 150.0)

            # Check scores
            code_score = await ledger.get_trust_score(agent_id, "code_generation")
            research_score = await ledger.get_trust_score(agent_id, "research")

            # code_generation: Beta(9,1) = 0.9
            assert abs(code_score - 0.9) < 0.001

            # research: Beta(1,7) = 0.125
            assert abs(research_score - 0.125) < 0.001


class TestAgentStats:
    """Test agent statistics retrieval"""

    @pytest.mark.asyncio
    async def test_get_agent_stats(self, temp_db):
        """Test get_agent_stats returns correct statistics"""
        async with TrustLedger(db_path=temp_db) as ledger:
            agent_id = "stats-agent"
            task_type = "code_generation"

            # Record multiple outcomes
            await ledger.record_outcome(agent_id, task_type, True, 0.9, 100.0)
            await ledger.record_outcome(agent_id, task_type, True, 0.85, 120.0)
            await ledger.record_outcome(agent_id, task_type, False, 0.5, 200.0)

            stats = await ledger.get_agent_stats(agent_id, task_type)

            assert stats is not None
            assert stats.agent_id == agent_id
            assert stats.task_type == task_type
            assert stats.success_count == 2
            assert stats.failure_count == 1
            assert abs(stats.trust_score - 0.6) < 0.001  # Beta(3,2) = 3/5
            assert abs(stats.avg_quality - 0.75) < 0.01  # (0.9 + 0.85 + 0.5) / 3
            assert abs(stats.avg_duration - 140.0) < 1.0  # (100 + 120 + 200) / 3

    @pytest.mark.asyncio
    async def test_get_agent_stats_nonexistent(self, temp_db):
        """Test get_agent_stats returns None for nonexistent agent"""
        async with TrustLedger(db_path=temp_db) as ledger:
            stats = await ledger.get_agent_stats("nonexistent", "code_generation")
            assert stats is None


class TestValidation:
    """Test input validation"""

    @pytest.mark.asyncio
    async def test_invalid_quality_score(self, temp_db):
        """Test record_outcome rejects invalid quality score"""
        async with TrustLedger(db_path=temp_db) as ledger:
            with pytest.raises(ValueError, match="quality must be in"):
                await ledger.record_outcome("agent", "task", True, 1.5, 100.0)

            with pytest.raises(ValueError, match="quality must be in"):
                await ledger.record_outcome("agent", "task", True, -0.1, 100.0)

    @pytest.mark.asyncio
    async def test_invalid_duration(self, temp_db):
        """Test record_outcome rejects invalid duration"""
        async with TrustLedger(db_path=temp_db) as ledger:
            with pytest.raises(ValueError, match="duration must be"):
                await ledger.record_outcome("agent", "task", True, 0.8, -10.0)


class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_record_outcome_performance(self, temp_db):
        """Test record_outcome completes quickly"""
        async with TrustLedger(db_path=temp_db) as ledger:
            start = time.time()
            await ledger.record_outcome("agent", "task", True, 0.9, 100.0)
            elapsed = time.time() - start

            assert elapsed < 0.1  # Should complete in < 100ms

    @pytest.mark.asyncio
    async def test_get_trust_score_performance(self, temp_db):
        """Test get_trust_score completes quickly"""
        async with TrustLedger(db_path=temp_db) as ledger:
            await ledger.record_outcome("agent", "task", True, 0.9, 100.0)

            start = time.time()
            await ledger.get_trust_score("agent", "task")
            elapsed = time.time() - start

            assert elapsed < 0.05  # Should complete in < 50ms
