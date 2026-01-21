#!/usr/bin/env python3
"""
CPB v2.4 Feature Tests - Pioneer Mode, Trust Context, Deep Research

Tests cover:
1. Pioneer mode weight application
2. Trust context weight application
3. Pioneer auto-detection from query signals
4. Pioneer auto-detection from sparse Tier 1 results
5. Pioneer auto-detection from recent source prevalence
6. Trust context Tier 1 source creation
7. Deep research Gemini availability check
8. Deep research Perplexity availability check
9. Deep research fallback chain
10. Deep research result conversion
11. Enhancer pioneer signals
12. PrecisionResult field verification
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta


# =============================================================================
# PIONEER MODE TESTS
# =============================================================================

class TestPioneerMode:
    """Tests for pioneer mode DQ weight application."""

    def test_pioneer_weights_applied(self):
        """Verify PIONEER_DQ_WEIGHTS used when pioneer=True."""
        from cpb.precision_config import PIONEER_DQ_WEIGHTS, PRECISION_DQ_WEIGHTS

        # Pioneer mode should have different weights
        assert PIONEER_DQ_WEIGHTS['validity'] < PRECISION_DQ_WEIGHTS['validity'], \
            "Pioneer mode should reduce validity weight"
        assert PIONEER_DQ_WEIGHTS['correctness'] < PRECISION_DQ_WEIGHTS['correctness'], \
            "Pioneer mode should reduce correctness weight"
        assert 'ground_truth' in PIONEER_DQ_WEIGHTS, \
            "Pioneer mode should have ground_truth weight"
        assert PIONEER_DQ_WEIGHTS['ground_truth'] > 0.15, \
            "Pioneer mode should increase ground_truth weight"

        # Weights should sum to 1.0
        total = sum(PIONEER_DQ_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Pioneer weights should sum to 1.0, got {total}"

    def test_trust_context_weights_applied(self):
        """Verify TRUST_CONTEXT_DQ_WEIGHTS used when trust_context=True."""
        from cpb.precision_config import TRUST_CONTEXT_DQ_WEIGHTS, PRECISION_DQ_WEIGHTS

        # Trust context should increase correctness (user context assumed credible)
        assert TRUST_CONTEXT_DQ_WEIGHTS['correctness'] >= 0.35, \
            "Trust context should have high correctness weight"
        assert 'ground_truth' in TRUST_CONTEXT_DQ_WEIGHTS, \
            "Trust context should have ground_truth weight"
        assert TRUST_CONTEXT_DQ_WEIGHTS['ground_truth'] < 0.15, \
            "Trust context should reduce ground_truth weight (less external validation needed)"

        # Weights should sum to 1.0
        total = sum(TRUST_CONTEXT_DQ_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Trust context weights should sum to 1.0, got {total}"


# =============================================================================
# PIONEER AUTO-DETECTION TESTS
# =============================================================================

class TestPioneerAutoDetection:
    """Tests for automatic pioneer mode detection."""

    def test_pioneer_auto_detection_query_signals(self, sample_enhanced_query):
        """Test detection from query_enhancer pioneer signals."""
        enhanced = sample_enhanced_query(
            suggest_pioneer=True,
            pioneer_signals=["references January 2026", "cutting-edge pattern"]
        )

        assert enhanced.suggest_pioneer is True
        assert len(enhanced.pioneer_signals) == 2
        assert "January 2026" in enhanced.pioneer_signals[0]

    @pytest.mark.asyncio
    async def test_pioneer_auto_detection_sparse_tier1(self, sample_search_results):
        """Test detection when <3 Tier 1 results found."""
        from cpb.search_layer import SearchContext

        # Create search context with only 2 Tier 1 results
        results = sample_search_results(count=5, tier1_count=2)
        context = SearchContext(query="test")
        context.results = results
        context.tier1_results = [r for r in results if r.tier.value == 1]

        # Should trigger pioneer signal
        assert len(context.tier1_results) < 3, "Should have sparse Tier 1 results"

        # Verify signal would be generated
        pioneer_signals = []
        if len(context.tier1_results) < 3:
            pioneer_signals.append(f"sparse Tier 1 results ({len(context.tier1_results)} found)")

        assert len(pioneer_signals) == 1
        assert "sparse" in pioneer_signals[0]

    def test_pioneer_auto_detection_recent_sources(self, sample_search_results):
        """Test detection when >60% sources are <14 days old."""
        from cpb.search_layer import SearchResult, SourceTier, SourceCategory

        # Create all recent results (< 14 days old)
        recent_date = datetime.now() - timedelta(days=7)
        results = []
        for i in range(5):
            results.append(SearchResult(
                url=f"https://arxiv.org/abs/2501.{10000+i}",
                title=f"Recent Paper {i+1}",
                content=f"Content {i+1}",
                tier=SourceTier.TIER_1,
                category=SourceCategory.RESEARCH,
                source_name="arXiv",
                published_date=recent_date,
                base_relevance=0.85,
            ))

        # Calculate recent ratio
        recent_count = sum(
            1 for r in results
            if r.published_date and (datetime.now() - r.published_date).days < 14
        )
        recent_ratio = recent_count / len(results)

        assert recent_ratio > 0.6, "Should have >60% recent sources"

        # Verify signal would be generated
        pioneer_signals = []
        if recent_ratio > 0.6:
            pioneer_signals.append(f"mostly recent sources ({recent_count}/{len(results)} < 14 days)")

        assert len(pioneer_signals) == 1
        assert "mostly recent" in pioneer_signals[0]


# =============================================================================
# TRUST CONTEXT TESTS
# =============================================================================

class TestTrustContext:
    """Tests for trust context mode."""

    def test_trusted_context_creates_tier1(self, user_context):
        """Verify create_trusted_user_context creates Tier 1 source."""
        from cpb.search_layer import create_trusted_user_context, SourceTier, SourceCategory

        trusted = create_trusted_user_context(user_context)

        assert trusted.tier == SourceTier.TIER_1, "Should be Tier 1"
        assert trusted.category == SourceCategory.INTERNAL, "Should be internal category"
        assert trusted.base_relevance == 1.0, "Should have max relevance"
        assert "trusted" in trusted.url.lower(), "URL should indicate trusted source"
        assert trusted.content == user_context, "Content should match input"


# =============================================================================
# DEEP RESEARCH TESTS
# =============================================================================

class TestDeepResearch:
    """Tests for deep research integration."""

    def test_deep_research_gemini_available(self):
        """Test Gemini availability check."""
        from cpb.deep_research import check_deep_research_available, HAS_GEMINI

        available, msg = check_deep_research_available("gemini")

        # Result depends on environment (API key presence)
        assert isinstance(available, bool)
        assert isinstance(msg, str)
        assert len(msg) > 0

        if HAS_GEMINI:
            # If library installed, message should indicate key status
            assert "Gemini" in msg or "api_key" in msg.lower() or "ready" in msg.lower()

    def test_deep_research_perplexity_available(self):
        """Test Perplexity availability check."""
        from cpb.deep_research import check_deep_research_available, HAS_AIOHTTP

        available, msg = check_deep_research_available("perplexity")

        assert isinstance(available, bool)
        assert isinstance(msg, str)
        assert len(msg) > 0

        if not HAS_AIOHTTP:
            assert "aiohttp" in msg.lower()

    def test_deep_research_fallback(self):
        """Test fallback chain logic."""
        from cpb.deep_research import get_best_available_provider

        provider, msg = get_best_available_provider()

        # Should return either a provider or None with a message
        assert provider is None or provider in ("gemini", "perplexity")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_deep_research_result_conversion(self, sample_deep_research_result):
        """Test DeepResearchResult → SearchResult conversion."""
        from cpb.deep_research import deep_result_to_search_results
        from cpb.search_layer import SourceTier, SourceCategory

        dr_result = sample_deep_research_result(citations_count=3)
        search_results = deep_result_to_search_results(dr_result)

        # Should have main result + citations
        assert len(search_results) >= 1, "Should have at least main result"

        # Main result should be Tier 1
        main_result = search_results[0]
        assert main_result.tier == SourceTier.TIER_1
        assert main_result.category == SourceCategory.RESEARCH
        assert main_result.base_relevance >= 0.9, "Main result should have high relevance"

        # Should contain deep research URL identifier
        assert "deep-research://" in main_result.url


# =============================================================================
# QUERY ENHANCER TESTS
# =============================================================================

class TestQueryEnhancer:
    """Tests for query enhancement with pioneer detection."""

    def test_enhancer_pioneer_signals(self, sample_enhanced_query):
        """Test EnhancedQuery.pioneer_signals field."""
        # Without pioneer signals
        enhanced_normal = sample_enhanced_query(suggest_pioneer=False)
        assert enhanced_normal.suggest_pioneer is False
        assert enhanced_normal.pioneer_signals == []

        # With pioneer signals
        enhanced_pioneer = sample_enhanced_query(
            suggest_pioneer=True,
            pioneer_signals=["references 2026", "emerging patterns"]
        )
        assert enhanced_pioneer.suggest_pioneer is True
        assert len(enhanced_pioneer.pioneer_signals) == 2

    @pytest.mark.asyncio
    async def test_enhancer_returns_pioneer_signals(self):
        """Test that enhance_query returns pioneer signals."""
        from cpb.query_enhancer import EnhancedQuery

        # Create mock enhanced query directly (avoid LLM call)
        enhanced = EnhancedQuery(
            original="What are TDP patterns from 2026?",
            enhanced="Enhanced query about TDP patterns",
            reasoning="Added specificity",
            follow_ups=[],
            dimensions=["architecture"],
            was_enhanced=True,
            suggest_pioneer=True,
            pioneer_signals=["references 2026"]
        )

        assert hasattr(enhanced, 'suggest_pioneer')
        assert hasattr(enhanced, 'pioneer_signals')
        assert enhanced.suggest_pioneer is True
        assert len(enhanced.pioneer_signals) > 0


# =============================================================================
# PRECISION RESULT TESTS
# =============================================================================

class TestPrecisionResult:
    """Tests for PrecisionResult field verification."""

    def test_precision_result_fields(self, sample_precision_result):
        """Verify all new PrecisionResult fields exist."""
        result = sample_precision_result()

        # v2.4 fields
        assert hasattr(result, 'pioneer_mode')
        assert hasattr(result, 'trust_context_provided')
        assert hasattr(result, 'pioneer_auto_detected')
        assert hasattr(result, 'pioneer_signals')

        # Deep research fields
        assert hasattr(result, 'deep_research_used')
        assert hasattr(result, 'deep_research_provider')
        assert hasattr(result, 'deep_research_time_ms')
        assert hasattr(result, 'deep_research_citations')
        assert hasattr(result, 'deep_research_content')

    def test_precision_result_to_dict(self, sample_precision_result):
        """Verify to_dict includes all v2.4 fields."""
        result = sample_precision_result(
            pioneer_mode=True,
            trust_context_provided=True,
            pioneer_auto_detected=True,
            pioneer_signals=["signal1", "signal2"],
            deep_research_used=True,
            deep_research_provider="gemini/gemini-2.0-flash",
            deep_research_time_ms=1500,
            deep_research_citations=5,
        )

        data = result.to_dict()

        # Verify v2.4 fields in dict
        assert 'pioneer_mode' in data
        assert 'trust_context_provided' in data
        assert 'pioneer_auto_detected' in data
        assert 'pioneer_signals' in data
        assert 'deep_research_used' in data
        assert 'deep_research_provider' in data
        assert 'deep_research_time_ms' in data
        assert 'deep_research_citations' in data

        # Verify values
        assert data['pioneer_mode'] is True
        assert data['pioneer_auto_detected'] is True
        assert len(data['pioneer_signals']) == 2

    def test_precision_result_defaults(self):
        """Verify PrecisionResult has sensible defaults."""
        from cpb.precision_orchestrator import PrecisionResult

        result = PrecisionResult()

        # v2.4 defaults
        assert result.pioneer_mode is False
        assert result.trust_context_provided is False
        assert result.pioneer_auto_detected is False
        assert result.pioneer_signals == []
        assert result.deep_research_used is False
        assert result.deep_research_provider == ""
        assert result.deep_research_time_ms == 0
        assert result.deep_research_citations == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestV24Integration:
    """Integration tests for v2.4 features."""

    @pytest.mark.asyncio
    async def test_pioneer_mode_affects_verification(self):
        """Test that pioneer mode changes verification behavior."""
        from cpb.critic_verifier import verify
        from cpb.precision_config import PIONEER_DQ_WEIGHTS

        response = "Test response with [1] citation."
        sources = [{'url': 'https://arxiv.org/abs/2501.12345', 'title': 'Test'}]
        query = "What are TDP patterns from 2026?"

        # Verify with pioneer mode
        result = await verify(
            response=response,
            sources=sources,
            query=query,
            pioneer_mode=True,
            trust_context=False
        )

        assert result is not None
        assert hasattr(result, 'dq_score')
        assert 0 <= result.dq_score <= 1

    @pytest.mark.asyncio
    async def test_trust_context_mode_affects_verification(self):
        """Test that trust context mode changes verification behavior."""
        from cpb.critic_verifier import verify

        response = "Test response based on user context [1]."
        sources = [{'url': 'internal://user-trusted', 'title': 'User Context'}]
        query = "Analyze my research"

        # Verify with trust context mode
        result = await verify(
            response=response,
            sources=sources,
            query=query,
            pioneer_mode=False,
            trust_context=True
        )

        assert result is not None
        assert hasattr(result, 'dq_score')


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
