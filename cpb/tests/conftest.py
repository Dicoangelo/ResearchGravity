#!/usr/bin/env python3
"""
CPB Test Fixtures - Shared test configuration and fixtures.

Provides:
- Mock LLM clients
- Sample search results
- Mock deep research providers
- Test data factories
"""

import pytest
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# ASYNC FIXTURES
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# MOCK DATA FACTORIES
# =============================================================================

@pytest.fixture
def sample_search_results():
    """Factory for sample SearchResult objects."""
    from cpb.search_layer import SearchResult, SourceTier, SourceCategory

    def _create_results(count: int = 5, tier1_count: int = 2):
        results = []
        for i in range(count):
            tier = SourceTier.TIER_1 if i < tier1_count else SourceTier.TIER_2
            results.append(SearchResult(
                url=f"https://arxiv.org/abs/2501.{10000+i}",
                title=f"Test Paper {i+1}: Multi-Agent Patterns",
                content=f"Abstract content for paper {i+1} about multi-agent orchestration.",
                tier=tier,
                category=SourceCategory.RESEARCH if i < tier1_count else SourceCategory.GITHUB,
                source_name="arXiv" if i < tier1_count else "GitHub",
                published_date=datetime.now(),
                base_relevance=0.85 - (i * 0.05),
            ))
        return results

    return _create_results


@pytest.fixture
def sample_sources():
    """Factory for sample source dictionaries."""
    def _create_sources(count: int = 5):
        return [
            {
                'type': 'arxiv',
                'tier': 'TIER_1',
                'url': f'https://arxiv.org/abs/2501.{10000+i}',
                'title': f'Test Paper {i+1}',
                'signal': '★2.3k, 7d ago',
                'score': 0.85 - (i * 0.05),
                'content': f'Paper content {i+1}',
            }
            for i in range(count)
        ]
    return _create_sources


@pytest.fixture
def sample_deep_research_result():
    """Factory for DeepResearchResult objects."""
    from cpb.deep_research import DeepResearchResult

    def _create_result(
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        citations_count: int = 5
    ):
        return DeepResearchResult(
            content="Deep research synthesized content about multi-agent systems.",
            citations=[
                {'url': f'https://example.com/{i}', 'title': f'Source {i}', 'snippet': f'Snippet {i}'}
                for i in range(citations_count)
            ],
            provider=provider,
            model=model,
            query="test query",
            search_time_ms=1500,
            token_count=1000,
            cost_usd=0.002,
        )

    return _create_result


@pytest.fixture
def sample_enhanced_query():
    """Factory for EnhancedQuery objects."""
    from cpb.query_enhancer import EnhancedQuery

    def _create_enhanced(
        original: str = "test query",
        enhanced: str = "Enhanced research query",
        suggest_pioneer: bool = False,
        pioneer_signals: Optional[List[str]] = None
    ):
        return EnhancedQuery(
            original=original,
            enhanced=enhanced,
            reasoning="Added specificity and temporal context",
            follow_ups=["Follow-up 1", "Follow-up 2"],
            dimensions=["architecture", "performance"],
            was_enhanced=True,
            suggest_pioneer=suggest_pioneer,
            pioneer_signals=pioneer_signals or [],
        )

    return _create_enhanced


@pytest.fixture
def sample_precision_result():
    """Factory for PrecisionResult objects."""
    from cpb.precision_orchestrator import PrecisionResult
    from cpb.types import CPBPath

    def _create_result(**kwargs):
        defaults = {
            'output': "Test output with citations [1], [2]",
            'confidence': 0.85,
            'dq_score': 0.88,
            'validity': 0.90,
            'specificity': 0.85,
            'correctness': 0.88,
            'path': CPBPath.CASCADE,
            'execution_time_ms': 5000,
        }
        defaults.update(kwargs)
        return PrecisionResult(**defaults)

    return _create_result


# =============================================================================
# MOCK CLIENTS
# =============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without API calls."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=MagicMock(
        content="Mocked LLM response",
        model="mock",
        cost_usd=0.001,
    ))
    mock.get_available_providers.return_value = ['mock']
    return mock


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for deep research tests."""
    mock = MagicMock()

    async def mock_research(query, model=None, system_prompt=None):
        from cpb.deep_research import DeepResearchResult
        return DeepResearchResult(
            content="Gemini research content",
            citations=[
                {'url': 'https://example.com/1', 'title': 'Source 1', 'snippet': 'Snippet 1'},
                {'url': 'https://example.com/2', 'title': 'Source 2', 'snippet': 'Snippet 2'},
            ],
            provider="gemini",
            model=model or "gemini-2.0-flash",
            query=query,
            search_time_ms=1000,
            token_count=500,
            cost_usd=0.001,
        )

    mock.research = mock_research
    return mock


@pytest.fixture
def mock_perplexity_client():
    """Mock Perplexity client for deep research tests."""
    mock = MagicMock()

    async def mock_research(query, model=None, system_prompt=None):
        from cpb.deep_research import DeepResearchResult
        return DeepResearchResult(
            content="Perplexity research content",
            citations=[
                {'url': 'https://example.com/1', 'title': 'Source 1', 'snippet': 'Snippet 1'},
            ],
            provider="perplexity",
            model=model or "sonar",
            query=query,
            search_time_ms=800,
            token_count=400,
            cost_usd=0.0005,
        )

    mock.research = mock_research
    return mock


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@pytest.fixture
def disable_llm_calls():
    """Context manager to disable all LLM API calls."""
    with patch('cpb.llm_client.complete') as mock:
        mock.return_value = MagicMock(content="Mock response", model="mock", cost_usd=0)
        yield mock


@pytest.fixture
def disable_deep_research():
    """Context manager to disable deep research API calls."""
    with patch('cpb.deep_research.deep_research') as mock:
        async def mock_deep(*args, **kwargs):
            from cpb.deep_research import DeepResearchResult
            return (
                DeepResearchResult(
                    content="Mock deep research",
                    citations=[],
                    provider="mock",
                    model="mock",
                    query=args[0] if args else "test",
                    search_time_ms=0,
                    token_count=0,
                    cost_usd=0,
                ),
                []
            )
        mock.side_effect = mock_deep
        yield mock


# =============================================================================
# TEST DATA
# =============================================================================

@pytest.fixture
def pioneer_query():
    """Query that should trigger pioneer mode detection."""
    return "What are the TDP patterns from January 2026 papers?"


@pytest.fixture
def standard_query():
    """Standard research query."""
    return "What are best practices for multi-agent orchestration?"


@pytest.fixture
def user_context():
    """Sample user-provided context."""
    return """
    ## Research Summary

    Based on my analysis of recent papers:
    1. Supervisor-executor patterns show 15% improvement
    2. DAG-based decomposition handles complex tasks better
    3. Token efficiency gains of 2.3x observed

    Sources: arXiv:2501.12345, arXiv:2501.23456
    """
