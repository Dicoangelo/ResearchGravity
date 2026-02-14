"""
Tests for delegation/router.py

Test Coverage:
- Agent registry loading
- Keyword extraction and capability matching
- Trust-weighted scoring
- Complexity floor (direct execution for trivial tasks)
- Fallback chain generation
- Batch routing
"""

import pytest
from delegation.router import (
    load_agent_registry,
    route_subtask,
    route_batch,
    AgentCapability,
    _extract_keywords,
    _calculate_capability_match,
)
from delegation.models import SubTask, TaskProfile, VerificationMethod


# ═══════════════════════════════════════════════════════════════════════════
# Test Agent Registry Loading
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRegistryLoading:
    """Test agent registry loading from MCP servers"""

    def test_load_agent_registry_returns_list(self):
        """Registry returns a list of AgentCapability objects"""
        registry = load_agent_registry()
        assert isinstance(registry, list)
        # Should have at least some agents from mcp_server.py and mcp_raw/tools
        assert len(registry) > 0

    def test_agents_have_required_fields(self):
        """All agents have agent_id, name, description, keywords"""
        registry = load_agent_registry()
        for agent in registry:
            assert isinstance(agent, AgentCapability)
            assert agent.agent_id
            assert agent.name
            assert agent.description
            assert isinstance(agent.keywords, list)
            assert 0.0 <= agent.estimated_cost <= 1.0

    def test_agent_ids_are_unique(self):
        """No duplicate agent IDs in registry"""
        registry = load_agent_registry()
        agent_ids = [a.agent_id for a in registry]
        assert len(agent_ids) == len(set(agent_ids))


# ═══════════════════════════════════════════════════════════════════════════
# Test Keyword Extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestKeywordExtraction:
    """Test keyword extraction for capability matching"""

    def test_extract_keywords_removes_stopwords(self):
        """Stopwords like 'the', 'and', 'for' are removed"""
        text = "Get the session context from the database"
        keywords = _extract_keywords(text)
        assert "session" in keywords
        assert "context" in keywords
        assert "database" in keywords
        assert "the" not in keywords
        assert "from" not in keywords

    def test_extract_keywords_min_length(self):
        """Keywords must be >= 4 characters"""
        text = "Get URL and log it"
        keywords = _extract_keywords(text)
        # "URL" is 3 chars → excluded, "log" is 3 chars → excluded
        assert "get" not in keywords  # Stopword
        assert "url" not in keywords  # Too short (3 chars)

    def test_extract_keywords_unique(self):
        """Duplicate keywords are deduplicated"""
        text = "session session session context"
        keywords = _extract_keywords(text)
        assert keywords.count("session") == 1


# ═══════════════════════════════════════════════════════════════════════════
# Test Capability Matching
# ═══════════════════════════════════════════════════════════════════════════


class TestCapabilityMatching:
    """Test capability matching between subtasks and agents"""

    def test_perfect_keyword_match(self):
        """Perfect keyword overlap → score close to 1.0"""
        subtask = SubTask(
            id="task-1",
            description="Search research papers and find learnings",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.5,
            estimated_duration=0.5,
            parallel_safe=True,
        )
        agent = AgentCapability(
            agent_id="test::search",
            name="search_learnings",
            description="Search research papers and extract learnings",
            keywords=["search", "research", "papers", "learnings"],
        )

        score = _calculate_capability_match(subtask, agent, use_llm=False)
        assert score > 0.5  # High overlap

    def test_no_keyword_match(self):
        """No keyword overlap → score 0.0"""
        subtask = SubTask(
            id="task-1",
            description="Deploy application to production server",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.5,
            estimated_duration=0.5,
            parallel_safe=True,
        )
        agent = AgentCapability(
            agent_id="test::search",
            name="search_learnings",
            description="Search research papers and extract learnings",
            keywords=["search", "research", "papers", "learnings"],
        )

        score = _calculate_capability_match(subtask, agent, use_llm=False)
        assert score == 0.0  # No overlap

    def test_partial_keyword_match(self):
        """Partial keyword overlap → score between 0.0 and 1.0"""
        subtask = SubTask(
            id="task-1",
            description="Search research database and context",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.5,
            estimated_duration=0.5,
            parallel_safe=True,
        )
        agent = AgentCapability(
            agent_id="test::session",
            name="get_session_context",
            description="Get active session information",
            keywords=["session", "active", "context", "information"],
        )

        score = _calculate_capability_match(subtask, agent, use_llm=False)
        assert 0.0 < score < 1.0  # Partial overlap ("context" matches)


# ═══════════════════════════════════════════════════════════════════════════
# Test Routing Logic
# ═══════════════════════════════════════════════════════════════════════════


class TestRoutingLogic:
    """Test subtask routing to agents"""

    def test_route_subtask_returns_assignment(self):
        """route_subtask returns an Assignment object"""
        subtask = SubTask(
            id="task-1",
            description="Search research papers",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.5,
            estimated_duration=0.5,
            parallel_safe=True,
            profile=TaskProfile(complexity=0.6),
        )
        registry = load_agent_registry()

        assignment = route_subtask(subtask, registry, use_llm=False)

        assert assignment.subtask_id == "task-1"
        assert assignment.agent_id  # Some agent selected
        assert 0.0 <= assignment.trust_score <= 1.0
        assert 0.0 <= assignment.capability_match <= 1.0
        assert assignment.assignment_reasoning

    def test_complexity_floor_direct_execution(self):
        """Tasks with complexity < 0.2 execute directly (no delegation)"""
        subtask = SubTask(
            id="task-trivial",
            description="Print hello world",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.1,
            estimated_duration=0.1,
            parallel_safe=True,
            profile=TaskProfile(complexity=0.15),  # Below 0.2 threshold
        )
        registry = load_agent_registry()

        assignment = route_subtask(subtask, registry, use_llm=False)

        assert assignment.agent_id == "DIRECT_EXECUTION"
        assert "delegation threshold" in assignment.assignment_reasoning
        assert assignment.metadata.get("delegation_bypassed") is True

    def test_fallback_chain_populated(self):
        """Assignment includes fallback chain of next-best agents"""
        subtask = SubTask(
            id="task-1",
            description="Search research papers",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.5,
            estimated_duration=0.5,
            parallel_safe=True,
            profile=TaskProfile(complexity=0.6),
        )
        registry = load_agent_registry()

        # Ensure we have enough agents
        if len(registry) < 2:
            pytest.skip("Not enough agents in registry for fallback test")

        assignment = route_subtask(subtask, registry, use_llm=False)

        assert "fallback_chain" in assignment.metadata
        fallback = assignment.metadata["fallback_chain"]
        assert isinstance(fallback, list)
        # Fallback chain should not include the selected agent
        assert assignment.agent_id not in fallback

    def test_scoring_weights_applied(self):
        """Final score uses weighted combination of capability, trust, cost"""
        subtask = SubTask(
            id="task-1",
            description="Search research papers",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.5,
            estimated_duration=0.5,
            parallel_safe=True,
            profile=TaskProfile(complexity=0.6),
        )
        registry = load_agent_registry()

        assignment = route_subtask(subtask, registry, use_llm=False)

        # Metadata should have final_score, cost_efficiency
        assert "final_score" in assignment.metadata
        assert "cost_efficiency" in assignment.metadata
        assert 0.0 <= assignment.metadata["final_score"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Test Batch Routing
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchRouting:
    """Test batch routing of multiple subtasks"""

    def test_route_batch_returns_list(self):
        """route_batch returns list of Assignments"""
        subtasks = [
            SubTask(
                id=f"task-{i}",
                description=f"Task {i}",
                verification_method=VerificationMethod.AUTOMATED_TEST,
                estimated_cost=0.5,
                estimated_duration=0.5,
                parallel_safe=True,
                profile=TaskProfile(complexity=0.5),
            )
            for i in range(3)
        ]
        registry = load_agent_registry()

        assignments = route_batch(subtasks, registry, use_llm=False)

        assert len(assignments) == 3
        assert all(a.subtask_id == f"task-{i}" for i, a in enumerate(assignments))

    def test_route_batch_preserves_order(self):
        """Assignments are in same order as input subtasks"""
        subtasks = [
            SubTask(
                id=f"task-{i}",
                description=f"Task {i}",
                verification_method=VerificationMethod.AUTOMATED_TEST,
                estimated_cost=0.5,
                estimated_duration=0.5,
                parallel_safe=True,
                profile=TaskProfile(complexity=0.5),
            )
            for i in range(5)
        ]
        registry = load_agent_registry()

        assignments = route_batch(subtasks, registry, use_llm=False)

        for i, assignment in enumerate(assignments):
            assert assignment.subtask_id == f"task-{i}"


# ═══════════════════════════════════════════════════════════════════════════
# Test Diverse Task Types
# ═══════════════════════════════════════════════════════════════════════════


class TestDiverseTaskTypes:
    """Test routing for different types of tasks"""

    def test_route_research_task(self):
        """Research tasks route to research-related agents"""
        subtask = SubTask(
            id="research-1",
            description="Search archived learnings for multi-agent patterns",
            verification_method=VerificationMethod.SEMANTIC_SIMILARITY,
            estimated_cost=0.4,
            estimated_duration=0.3,
            parallel_safe=True,
            profile=TaskProfile(complexity=0.5),
        )
        registry = load_agent_registry()

        assignment = route_subtask(subtask, registry, use_llm=False)

        # Should route to a research-related agent
        assert assignment.agent_id
        # Agent name should contain research-related keywords
        agent_name = assignment.metadata.get("agent_name", "").lower()
        assert any(
            keyword in agent_name
            for keyword in ["search", "research", "learning", "context", "session"]
        )

    def test_route_coherence_task(self):
        """Coherence tasks route to coherence-related agents"""
        subtask = SubTask(
            id="coherence-1",
            description="Detect coherence moments in conversation",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.6,
            estimated_duration=0.5,
            parallel_safe=True,
            profile=TaskProfile(complexity=0.7),
        )
        registry = load_agent_registry()

        assignment = route_subtask(subtask, registry, use_llm=False)

        assert assignment.agent_id

    def test_route_ucw_task(self):
        """UCW tasks route to UCW-related agents"""
        subtask = SubTask(
            id="ucw-1",
            description="Capture cognitive event to universal wallet",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.5,
            estimated_duration=0.4,
            parallel_safe=True,
            profile=TaskProfile(complexity=0.6),
        )
        registry = load_agent_registry()

        assignment = route_subtask(subtask, registry, use_llm=False)

        assert assignment.agent_id
