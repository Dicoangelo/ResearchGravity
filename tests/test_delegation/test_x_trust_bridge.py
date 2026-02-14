"""
Tests for delegation.x_trust_bridge

Covers:
- Author data parsing (X format → delegation format)
- Trust score synchronization accuracy
- Bayesian Beta parameter mapping (alpha/beta → success/failure)
- Combined trust view
- Edge cases (no data, zero followers, single observation)
"""

import pytest
import asyncio
from delegation.x_trust_bridge import XTrustBridge, AGENT_PREFIX, TASK_TYPE
from delegation.trust_ledger import TrustLedger


@pytest.fixture
def sample_authors():
    """Sample X top_authors data"""
    return {
        "top_authors": [
            {
                "username": "TestAuthor1",
                "trust_score": 0.8,
                "total_tweets_seen": 3,
                "avg_quality": 0.72,
                "deep_signal_count": 1,
                "noise_count": 0,
                "followers": 5000,
                "alpha": 4.0,
                "beta_param": 1.0,
                "trust_level": "trusted",
            },
            {
                "username": "TestAuthor2",
                "trust_score": 0.5,
                "total_tweets_seen": 4,
                "avg_quality": 0.45,
                "deep_signal_count": 0,
                "noise_count": 2,
                "followers": 200,
                "alpha": 2.0,
                "beta_param": 2.0,
                "trust_level": "neutral",
            },
        ]
    }


class TestAgentIdMapping:
    def test_prefix(self):
        assert XTrustBridge._agent_id("karpathy") == f"{AGENT_PREFIX}:karpathy"

    def test_lowercase(self):
        assert XTrustBridge._agent_id("AnthropicAI") == f"{AGENT_PREFIX}:anthropicai"


class TestParseAuthor:
    def test_basic_parse(self):
        author = {
            "username": "TestUser",
            "trust_score": 0.75,
            "alpha": 3.0,
            "beta_param": 1.0,
            "avg_quality": 0.65,
            "followers": 1000,
            "trust_level": "trusted",
            "deep_signal_count": 2,
            "noise_count": 0,
        }
        parsed = XTrustBridge._parse_author(author)

        assert parsed["agent_id"] == f"{AGENT_PREFIX}:testuser"
        assert parsed["success_count"] == 2  # alpha - 1
        assert parsed["failure_count"] == 0  # beta_param - 1
        assert parsed["avg_quality"] == 0.65
        assert parsed["trust_score"] == 0.75

    def test_equal_alpha_beta(self):
        author = {
            "username": "NeutralUser",
            "trust_score": 0.5,
            "alpha": 3.0,
            "beta_param": 3.0,
            "avg_quality": 0.5,
        }
        parsed = XTrustBridge._parse_author(author)

        assert parsed["success_count"] == 2
        assert parsed["failure_count"] == 2

    def test_uninformative_prior(self):
        author = {
            "username": "NewUser",
            "trust_score": 0.5,
            "alpha": 1.0,
            "beta_param": 1.0,
            "avg_quality": 0.5,
        }
        parsed = XTrustBridge._parse_author(author)

        assert parsed["success_count"] == 0
        assert parsed["failure_count"] == 0


class TestSyncAuthors:
    @pytest.mark.asyncio
    async def test_sync_preserves_trust(self, sample_authors):
        ledger = TrustLedger(":memory:")
        async with XTrustBridge(trust_ledger=ledger) as bridge:
            results = await bridge.sync_top_authors(sample_authors)

            assert len(results) == 2
            # Trust scores should match X values
            assert abs(results[0]["delegation_trust"] - 0.8) < 0.01
            assert abs(results[1]["delegation_trust"] - 0.5) < 0.01

    @pytest.mark.asyncio
    async def test_sync_returns_metadata(self, sample_authors):
        ledger = TrustLedger(":memory:")
        async with XTrustBridge(trust_ledger=ledger) as bridge:
            results = await bridge.sync_top_authors(sample_authors)

            for r in results:
                assert "agent_id" in r
                assert "username" in r
                assert "x_trust" in r
                assert "delegation_trust" in r
                assert "synced_at" in r

    @pytest.mark.asyncio
    async def test_sync_empty_list(self):
        ledger = TrustLedger(":memory:")
        async with XTrustBridge(trust_ledger=ledger) as bridge:
            results = await bridge.sync_top_authors({"top_authors": []})
            assert results == []


class TestCombinedTrust:
    @pytest.mark.asyncio
    async def test_unknown_author(self):
        ledger = TrustLedger(":memory:")
        async with XTrustBridge(trust_ledger=ledger) as bridge:
            combined = await bridge.get_combined_trust("nonexistent_user")

            assert combined["delegation_trust"] == 0.5  # uninformative prior
            assert combined["has_delegation_history"] is False

    @pytest.mark.asyncio
    async def test_synced_author(self, sample_authors):
        ledger = TrustLedger(":memory:")
        async with XTrustBridge(trust_ledger=ledger) as bridge:
            await bridge.sync_top_authors(sample_authors)
            combined = await bridge.get_combined_trust("TestAuthor1")

            assert combined["has_delegation_history"] is True
            assert combined["stats"]["success_count"] == 3
            assert combined["stats"]["failure_count"] == 0
