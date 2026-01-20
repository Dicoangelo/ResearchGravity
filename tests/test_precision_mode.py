#!/usr/bin/env python3
"""
Tests for CPB Precision Mode

Unit and integration tests for:
- precision_config.py
- rg_adapter.py
- critic_verifier.py
- precision_orchestrator.py
- precision_cli.py

Run with: python3 -m pytest tests/test_precision_mode.py -v
"""

import asyncio
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpb.precision_config import (
    PRECISION_CONFIG, PrecisionConfig, PrecisionACEConfig,
    PRECISION_AGENT_PERSONAS, PRECISION_DQ_WEIGHTS,
    get_precision_agent_prompts, calculate_precision_dq,
    validate_precision_config, get_agent_by_role
)
from cpb.rg_adapter import (
    RGAdapter, RGContext, ConnectionMode, SearchResult,
    rg_adapter, get_context, search_learnings
)
from cpb.critic_verifier import (
    CriticVerifier, VerificationResult, ConfidenceScorer,
    CitationExtractor, verify, format_critic_feedback
)
from cpb.precision_orchestrator import (
    PrecisionOrchestrator, PrecisionResult, PrecisionStatus,
    execute_precision, get_precision_status
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_query():
    return "What are best practices for multi-agent orchestration?"


@pytest.fixture
def sample_context():
    return """
    ## Research Context
    Multi-agent systems (arXiv:2508.17536) show that voting-based consensus
    outperforms debate in most scenarios. The ACE framework provides adaptive
    consensus mechanisms.
    """


@pytest.fixture
def sample_response():
    return """
    # Multi-Agent Orchestration Best Practices

    Based on recent research (arXiv:2508.17536), here are the key practices:

    ## 1. Consensus Mechanisms
    - Voting-based approaches work better than debate for most decisions
    - Use 3-5 agents for optimal balance between diversity and convergence

    ## 2. Agent Specialization
    - Assign distinct roles (Analyst, Skeptic, Synthesizer)
    - Ensure complementary perspectives

    ## 3. Quality Verification
    - Implement DQ scoring (arXiv:2511.15755) for response validation
    - Use writer-critic patterns for high-stakes outputs

    Sources:
    - arXiv:2508.17536 (Voting vs Debate)
    - arXiv:2511.15755 (DQ Framework)
    """


@pytest.fixture
def sample_sources():
    return [
        {'type': 'arxiv', 'arxiv_id': '2508.17536', 'url': 'https://arxiv.org/abs/2508.17536'},
        {'type': 'arxiv', 'arxiv_id': '2511.15755', 'url': 'https://arxiv.org/abs/2511.15755'},
    ]


# =============================================================================
# PRECISION CONFIG TESTS
# =============================================================================

class TestPrecisionConfig:
    """Tests for precision_config.py"""

    def test_precision_config_defaults(self):
        """Test PRECISION_CONFIG default values."""
        assert PRECISION_CONFIG.dq_threshold == 0.95
        assert PRECISION_CONFIG.force_cascade is True
        assert PRECISION_CONFIG.critic_validation is True
        assert PRECISION_CONFIG.max_retries == 5
        assert PRECISION_CONFIG.ace_config.agent_count == 7

    def test_precision_ace_config(self):
        """Test PrecisionACEConfig."""
        config = PrecisionACEConfig()
        assert config.max_rounds == 25
        assert config.agent_count == 7
        assert config.require_citations is True

    def test_precision_agent_personas(self):
        """Test 7-agent personas are defined."""
        assert len(PRECISION_AGENT_PERSONAS) == 7

        names = [a['name'] for a in PRECISION_AGENT_PERSONAS]
        assert 'Analyst' in names
        assert 'Skeptic' in names
        assert 'Synthesizer' in names
        assert 'Pragmatist' in names
        assert 'Visionary' in names
        assert 'Historian' in names
        assert 'Innovator' in names

        for agent in PRECISION_AGENT_PERSONAS:
            assert 'name' in agent
            assert 'role' in agent
            assert 'prompt' in agent

    def test_precision_dq_weights(self):
        """Test DQ weights sum to 1.0."""
        total = sum(PRECISION_DQ_WEIGHTS.values())
        assert total == 1.0

        assert PRECISION_DQ_WEIGHTS['validity'] == 0.30
        assert PRECISION_DQ_WEIGHTS['specificity'] == 0.25
        assert PRECISION_DQ_WEIGHTS['correctness'] == 0.45

    def test_calculate_precision_dq(self):
        """Test precision DQ calculation."""
        # Perfect scores
        dq = calculate_precision_dq(1.0, 1.0, 1.0)
        assert dq == 1.0

        # Zero scores
        dq = calculate_precision_dq(0.0, 0.0, 0.0)
        assert dq == 0.0

        # Weighted calculation
        dq = calculate_precision_dq(0.8, 0.8, 0.8)
        assert dq == pytest.approx(0.8)

        # Correctness weighted higher
        dq_high_correct = calculate_precision_dq(0.5, 0.5, 1.0)
        dq_low_correct = calculate_precision_dq(0.5, 0.5, 0.0)
        assert dq_high_correct > dq_low_correct

    def test_get_precision_agent_prompts(self, sample_query, sample_context):
        """Test agent prompt generation."""
        prompts = get_precision_agent_prompts(sample_query, sample_context)

        assert len(prompts) == 7

        for prompt in prompts:
            assert 'agent' in prompt
            assert 'role' in prompt
            assert 'system_prompt' in prompt
            assert 'user_prompt' in prompt
            assert 'full_prompt' in prompt

            # Check query is in user prompt
            assert sample_query in prompt['user_prompt']

    def test_validate_precision_config(self):
        """Test configuration validation."""
        # Valid config should have no warnings
        warnings = validate_precision_config(PRECISION_CONFIG)
        assert isinstance(warnings, list)
        assert len(warnings) == 0

        # Invalid config should have warnings
        invalid_config = PrecisionConfig(
            dq_threshold=0.5,
            max_retries=1,
            critic_validation=False
        )
        warnings = validate_precision_config(invalid_config)
        assert len(warnings) > 0

    def test_get_agent_by_role(self):
        """Test getting agent by role."""
        agent = get_agent_by_role('evidence')
        assert agent is not None
        assert agent['name'] == 'Analyst'

        agent = get_agent_by_role('critique')
        assert agent is not None
        assert agent['name'] == 'Skeptic'

        agent = get_agent_by_role('nonexistent')
        assert agent is None


# =============================================================================
# RG ADAPTER TESTS
# =============================================================================

class TestRGAdapter:
    """Tests for rg_adapter.py"""

    def test_connection_modes(self):
        """Test ConnectionMode enum."""
        assert ConnectionMode.MCP.value == 'mcp'
        assert ConnectionMode.REST.value == 'rest'
        assert ConnectionMode.FILE.value == 'file'
        assert ConnectionMode.DEGRADED.value == 'degraded'

    def test_rg_context_dataclass(self):
        """Test RGContext dataclass."""
        context = RGContext(
            learnings="Test learnings",
            connection_mode=ConnectionMode.FILE
        )
        assert context.learnings == "Test learnings"
        assert context.connection_mode == ConnectionMode.FILE
        assert context.packs == []
        assert context.sessions == []
        assert context.warnings == []

    def test_rg_context_to_enriched_context(self):
        """Test context enrichment formatting."""
        context = RGContext(
            learnings="# Important Learning\nKey insight here.",
            sessions=[
                {'id': 'test-session-123', 'topic': 'Multi-agent', 'findings': []}
            ],
            packs=[
                {'name': 'test-pack', 'description': 'A test pack'}
            ]
        )

        enriched = context.to_enriched_context(budget=5000)
        assert '# Important Learning' in enriched or 'Recent Learnings' in enriched

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass."""
        result = SearchResult(
            content="Test content",
            source="test.md",
            relevance_score=0.85
        )
        assert result.content == "Test content"
        assert result.relevance_score == 0.85

    @pytest.mark.asyncio
    async def test_adapter_initialization(self):
        """Test adapter can be initialized."""
        adapter = RGAdapter()
        status = adapter.get_status()

        assert 'connection_mode' in status
        assert 'agent_core_exists' in status

    @pytest.mark.asyncio
    async def test_get_context_degraded(self):
        """Test context retrieval in degraded mode."""
        adapter = RGAdapter()
        adapter._connection_mode = ConnectionMode.DEGRADED

        context = await adapter.get_context("test query")
        assert context.connection_mode == ConnectionMode.DEGRADED
        assert len(context.warnings) > 0


# =============================================================================
# CRITIC VERIFIER TESTS
# =============================================================================

class TestCriticVerifier:
    """Tests for critic_verifier.py"""

    def test_verification_result_dataclass(self):
        """Test VerificationResult dataclass."""
        result = VerificationResult(
            dq_score=0.92,
            passed=True,
            evidence_score=0.88,
            oracle_score=0.90,
            confidence_score=0.85
        )

        assert result.dq_score == 0.92
        assert result.passed is True

        result_dict = result.to_dict()
        assert 'dq_score' in result_dict
        assert 'passed' in result_dict

    def test_confidence_scorer(self, sample_response):
        """Test ConfidenceScorer."""
        scorer = ConfidenceScorer()

        score = scorer.calculate(
            response=sample_response,
            citations_verified=2,
            citations_total=3,
            evidence_score=0.8
        )

        assert 0.0 <= score <= 1.0

    def test_confidence_scorer_hedging(self):
        """Test hedging detection."""
        scorer = ConfidenceScorer()

        # Response with hedging
        hedging_response = "I think maybe this might possibly work, but I'm not sure."
        score_hedging = scorer.calculate(hedging_response, 0, 0, 0.5)

        # Response without hedging
        confident_response = "This clearly works. Evidence shows it is proven."
        score_confident = scorer.calculate(confident_response, 1, 1, 0.5)

        assert score_confident > score_hedging

    def test_citation_extractor(self, sample_response):
        """Test CitationExtractor."""
        extractor = CitationExtractor()
        citations = extractor.extract_citations(sample_response)

        # Should find arXiv citations
        arxiv_citations = [c for c in citations if c['type'] == 'arxiv']
        assert len(arxiv_citations) >= 2

        # Check arxiv IDs
        arxiv_ids = [c['id'] for c in arxiv_citations]
        assert '2508.17536' in arxiv_ids
        assert '2511.15755' in arxiv_ids

    def test_citation_extractor_claim_count(self, sample_response):
        """Test claim counting."""
        extractor = CitationExtractor()
        claim_count = extractor.count_claims(sample_response)

        assert claim_count >= 1

    @pytest.mark.asyncio
    async def test_verifier_full_pipeline(self, sample_response, sample_sources, sample_query):
        """Test full verification pipeline."""
        verifier = CriticVerifier()

        result = await verifier.verify(
            response=sample_response,
            sources=sample_sources,
            query=sample_query
        )

        assert isinstance(result, VerificationResult)
        assert 0.0 <= result.dq_score <= 1.0
        assert 0.0 <= result.validity <= 1.0
        assert 0.0 <= result.specificity <= 1.0
        assert 0.0 <= result.correctness <= 1.0
        assert result.citations_found >= 0

    @pytest.mark.asyncio
    async def test_verify_convenience_function(self, sample_response, sample_sources):
        """Test verify() convenience function."""
        result = await verify(sample_response, sample_sources)

        assert isinstance(result, VerificationResult)

    def test_format_critic_feedback(self):
        """Test feedback formatting."""
        issues = [
            {'code': 'LOW_VALIDITY', 'message': 'Low validity', 'suggestion': 'Improve validity'},
            {'code': 'NO_CITATIONS', 'message': 'No citations', 'suggestion': 'Add citations'}
        ]

        feedback = format_critic_feedback(issues)
        assert 'Improve validity' in feedback
        assert 'Add citations' in feedback


# =============================================================================
# PRECISION ORCHESTRATOR TESTS
# =============================================================================

class TestPrecisionOrchestrator:
    """Tests for precision_orchestrator.py"""

    def test_precision_result_dataclass(self):
        """Test PrecisionResult dataclass."""
        result = PrecisionResult(
            output="Test output",
            dq_score=0.93,
            verified=True
        )

        assert result.output == "Test output"
        assert result.dq_score == 0.93
        assert result.verified is True

        result_dict = result.to_dict()
        assert 'output' in result_dict
        assert 'dq_score' in result_dict

    def test_precision_status_dataclass(self):
        """Test PrecisionStatus dataclass."""
        status = PrecisionStatus(
            phase="verifying",
            progress=60,
            current_step="Running critics"
        )

        assert status.phase == "verifying"
        assert status.progress == 60

    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        orchestrator = PrecisionOrchestrator()

        status = orchestrator.get_status()
        assert status['mode'] == 'precision'
        assert 'config' in status
        assert status['config']['dq_threshold'] == 0.95

    def test_orchestrator_agent_info(self):
        """Test get_agent_info."""
        orchestrator = PrecisionOrchestrator()
        agents = orchestrator.get_agent_info()

        assert len(agents) == 7
        for agent in agents:
            assert 'name' in agent
            assert 'role' in agent

    @pytest.mark.asyncio
    async def test_orchestrator_execute(self, sample_query):
        """Test execute method."""
        orchestrator = PrecisionOrchestrator()

        # Track status updates
        statuses = []
        def on_status(status):
            statuses.append(status)

        result = await orchestrator.execute(
            sample_query,
            on_status=on_status
        )

        assert isinstance(result, PrecisionResult)
        assert result.output != ""
        assert result.path == CPBPath.CASCADE

        # Should have received status updates
        assert len(statuses) > 0

    @pytest.mark.asyncio
    async def test_execute_precision_convenience(self, sample_query):
        """Test execute_precision convenience function."""
        result = await execute_precision(sample_query)

        assert isinstance(result, PrecisionResult)

    def test_get_precision_status_convenience(self):
        """Test get_precision_status convenience function."""
        status = get_precision_status()

        assert 'mode' in status
        assert status['mode'] == 'precision'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full precision pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_query, sample_context):
        """Test complete precision pipeline."""
        orchestrator = PrecisionOrchestrator()

        result = await orchestrator.execute(
            sample_query,
            context=sample_context
        )

        # Basic result validation
        assert isinstance(result, PrecisionResult)
        assert result.output != ""
        assert result.agent_count == 7

        # Verification should have run
        assert result.verification is not None or result.dq_score > 0

    @pytest.mark.asyncio
    async def test_context_enrichment(self, sample_query):
        """Test RG context enrichment."""
        orchestrator = PrecisionOrchestrator()

        # Get context
        rg_context = await orchestrator._enrich_context(sample_query, None)

        assert isinstance(rg_context, RGContext)
        assert rg_context.connection_mode in ConnectionMode

    @pytest.mark.asyncio
    async def test_retry_loop_behavior(self):
        """Test that retry loop works correctly."""
        orchestrator = PrecisionOrchestrator()

        # Use a query that might need retries
        result = await orchestrator.execute(
            "What is X?",  # Vague query
        )

        # Result should be returned regardless of quality
        assert isinstance(result, PrecisionResult)
        assert result.retry_count >= 0
        assert result.retry_count <= orchestrator.config.max_retries


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test handling of empty query."""
        orchestrator = PrecisionOrchestrator()

        result = await orchestrator.execute("")

        assert isinstance(result, PrecisionResult)

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test handling of very long query."""
        orchestrator = PrecisionOrchestrator()

        long_query = "What is the best approach? " * 100
        result = await orchestrator.execute(long_query)

        assert isinstance(result, PrecisionResult)

    def test_empty_response_verification(self):
        """Test verifying empty response."""
        extractor = CitationExtractor()
        citations = extractor.extract_citations("")

        assert citations == []

    @pytest.mark.asyncio
    async def test_no_sources_verification(self, sample_response):
        """Test verification with no sources."""
        result = await verify(sample_response, [])

        assert isinstance(result, VerificationResult)
        assert result.citations_verified == 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
