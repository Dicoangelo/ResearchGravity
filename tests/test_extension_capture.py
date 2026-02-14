"""
Tests for the Chrome extension capture API endpoints.

Tests /api/v2/coherence/capture/extension (single event).
Covers: event building, UCW layers, dedup logic, HTTP endpoint, DB integration,
quality scoring, and embedding trigger.
"""

import asyncio
import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# -- Helpers ------------------------------------------------------------------


def make_event(
    platform="chatgpt",
    content="Hello, this is a test message with enough content",
    direction="out",
    url="https://chatgpt.com/c/test-123",
    topic="test topic",
    session_hint="chatgpt-test-123",
    **kwargs,
):
    return {
        "platform": platform,
        "content": content,
        "direction": direction,
        "url": url,
        "topic": topic,
        "session_hint": session_hint,
        **kwargs,
    }


def _make_mock_conn(*, dedup_hit=False):
    """
    Create a mock asyncpg connection.

    Args:
        dedup_hit: If True, fetchval returns a fake event_id (coherence_sig dedup hit).
                   If False, fetchval returns None (no existing event).
    """
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetchval = AsyncMock(return_value="ext-existing-001" if dedup_hit else None)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    return conn


def _make_mock_pool(conn=None, *, dedup_hit=False):
    """Create a mock asyncpg pool that yields a mock connection.

    pool.acquire() returns a sync-call async context manager (not a coroutine),
    matching asyncpg's PoolAcquireContext pattern.
    """
    if conn is None:
        conn = _make_mock_conn(dedup_hit=dedup_hit)
    pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


# -- Test _build_extension_event ----------------------------------------------


class TestBuildExtensionEvent:
    """Test the event builder helper."""

    def test_deterministic_event_id(self):
        """Same content produces same event_id (dedup key)."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        f1 = _build_extension_event(ev)
        f2 = _build_extension_event(ev)

        assert f1["event_id"] == f2["event_id"]
        assert f1["coherence_sig"] == f2["coherence_sig"]

    def test_different_content_different_id(self):
        """Different content produces different event_id."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev1 = ExtensionEvent(**make_event(content="first message content here"))
        ev2 = ExtensionEvent(**make_event(content="second different message here"))
        f1 = _build_extension_event(ev1)
        f2 = _build_extension_event(ev2)

        assert f1["event_id"] != f2["event_id"]

    def test_session_hint_passthrough(self):
        """session_hint is used as session_id when provided."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event(session_hint="chatgpt-abc123"))
        fields = _build_extension_event(ev)
        assert fields["session_id"] == "chatgpt-abc123"

    def test_session_id_generated_when_no_hint(self):
        """Without session_hint, generates ext-{platform}-{date} ID."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event(session_hint=None))
        fields = _build_extension_event(ev)
        assert fields["session_id"].startswith("ext-chatgpt-")

    def test_coherence_potential_scaling(self):
        """Longer content gets higher coherence_potential."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        short = ExtensionEvent(**make_event(content="short msg here ok"))
        long_content = "x" * 3000
        long = ExtensionEvent(**make_event(content=long_content))

        f_short = _build_extension_event(short)
        f_long = _build_extension_event(long)

        short_potential = json.loads(f_short["instinct_layer"])["coherence_potential"]
        long_potential = json.loads(f_long["instinct_layer"])["coherence_potential"]

        assert long_potential > short_potential

    def test_coherence_potential_capped(self):
        """Coherence potential never exceeds 0.9."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event(
            content="x" * 20000,
            concepts=["a", "b", "c"],
        ))
        fields = _build_extension_event(ev)
        potential = json.loads(fields["instinct_layer"])["coherence_potential"]
        assert potential <= 0.9

    def test_semantic_layers_populated(self):
        """All 3 UCW semantic layers are populated as valid JSON."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(
            platform="chatgpt",
            content="Analyzing multi-agent orchestration patterns for coherence",
            direction="out",
            topic="AI research",
            intent="analyze",
            concepts=["mcp", "coherence"],
            session_hint="chatgpt-test-123",
        )
        fields = _build_extension_event(ev)

        data = json.loads(fields["data_layer"])
        assert "content" in data
        assert "source_url" in data

        light = json.loads(fields["light_layer"])
        assert light["topic"] == "AI research"
        assert light["intent"] == "analyze"
        assert light["concepts"] == ["mcp", "coherence"]

        instinct = json.loads(fields["instinct_layer"])
        assert "coherence_potential" in instinct
        assert instinct["gut_signal"] == "extension_capture"

    def test_content_truncated_at_10k(self):
        """Content is capped at 10,000 characters."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event(content="x" * 20000))
        fields = _build_extension_event(ev)

        data = json.loads(fields["data_layer"])
        assert len(data["content"]) == 10000

    def test_platforms(self):
        """All 5 supported platforms produce valid events."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        for platform in ["chatgpt", "grok", "gemini", "notebooklm", "youtube"]:
            ev = ExtensionEvent(**make_event(platform=platform))
            fields = _build_extension_event(ev)
            assert fields["platform"] == platform
            assert fields["event_id"].startswith("ext-")

    def test_coherence_sig_is_hex(self):
        """coherence_sig is a valid hex string."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)
        sig = fields["coherence_sig"]
        assert len(sig) == 32
        assert all(c in "0123456789abcdef" for c in sig)

    def test_event_id_prefix(self):
        """Event IDs start with 'ext-' for extension events."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)
        assert fields["event_id"].startswith("ext-")

    def test_timestamp_ns_is_recent(self):
        """Generated timestamp is within a reasonable range."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        before = time.time_ns()
        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)
        after = time.time_ns()

        assert before <= fields["now_ns"] <= after

    def test_quality_score_included(self):
        """Build result includes quality_score and cognitive_mode."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event(
            content="Implementing a distributed database schema with coherent architecture",
        ))
        fields = _build_extension_event(ev)
        assert "quality_score" in fields
        assert "cognitive_mode" in fields
        assert isinstance(fields["quality_score"], float)
        assert fields["cognitive_mode"] in ("deep_work", "exploration", "casual", "garbage")


# -- Test ExtensionEvent model ------------------------------------------------


class TestExtensionEventModel:
    """Test the Pydantic model validation."""

    def test_minimal_event(self):
        from api.routes.coherence import ExtensionEvent

        ev = ExtensionEvent(platform="chatgpt", content="hello world test msg")
        assert ev.direction == "in"  # default
        assert ev.url is None

    def test_full_event(self):
        from api.routes.coherence import ExtensionEvent

        ev = ExtensionEvent(
            platform="grok",
            content="analyzing quantum computing research",
            direction="out",
            url="https://grok.x.ai/chat/123",
            topic="quantum computing",
            intent="research",
            concepts=["quantum", "computing"],
            session_hint="grok-123",
            metadata={"model": "grok-3"},
        )
        assert ev.platform == "grok"
        assert ev.direction == "out"
        assert len(ev.concepts) == 2


# -- Test _store_extension_event (DB integration) ----------------------------


class TestStoreExtensionEvent:
    """Test DB store logic with mocked asyncpg connection."""

    @pytest.mark.asyncio
    async def test_store_returns_captured_on_insert(self):
        """New event returns status='captured'."""
        from api.routes.coherence import _store_extension_event, _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)

        conn = _make_mock_conn(dedup_hit=False)

        result = await _store_extension_event(conn, fields)

        assert result["status"] == "captured"
        assert result["event_id"] == fields["event_id"]
        assert result["platform"] == "chatgpt"

    @pytest.mark.asyncio
    async def test_store_returns_duplicate_on_coherence_sig_hit(self):
        """Existing coherence_sig returns status='duplicate' before insert."""
        from api.routes.coherence import _store_extension_event, _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)

        conn = _make_mock_conn(dedup_hit=True)

        result = await _store_extension_event(conn, fields)
        assert result["status"] == "duplicate"
        # Should NOT attempt insert (early return after fetchval)
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_returns_duplicate_on_event_id_conflict(self):
        """ON CONFLICT DO NOTHING returns status='duplicate'."""
        from api.routes.coherence import _store_extension_event, _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)

        conn = _make_mock_conn(dedup_hit=False)
        conn.execute = AsyncMock(return_value="INSERT 0 0")

        result = await _store_extension_event(conn, fields)
        assert result["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_store_upserts_session(self):
        """Session is upserted with ON CONFLICT before event insert."""
        from api.routes.coherence import _store_extension_event, _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)

        conn = _make_mock_conn(dedup_hit=False)

        await _store_extension_event(conn, fields)

        # fetchval for dedup + two execute calls (session + event)
        assert conn.execute.call_count == 2
        session_call = conn.execute.call_args_list[0]
        assert "cognitive_sessions" in session_call.args[0]
        assert "ON CONFLICT" in session_call.args[0]

    @pytest.mark.asyncio
    async def test_store_event_insert_has_on_conflict(self):
        """Event INSERT uses ON CONFLICT (event_id) DO NOTHING."""
        from api.routes.coherence import _store_extension_event, _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)

        conn = _make_mock_conn(dedup_hit=False)

        await _store_extension_event(conn, fields)

        event_call = conn.execute.call_args_list[1]
        sql = event_call.args[0]
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_store_includes_quality_score(self):
        """Stored event includes quality_score and cognitive_mode."""
        from api.routes.coherence import _store_extension_event, _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event(
            content="Implementing sovereign AI architecture with coherence detection",
        ))
        fields = _build_extension_event(ev)

        conn = _make_mock_conn(dedup_hit=False)

        result = await _store_extension_event(conn, fields)

        assert "quality_score" in result
        assert "cognitive_mode" in result
        assert isinstance(result["quality_score"], float)

    @pytest.mark.asyncio
    async def test_store_coherence_sig_dedup_query(self):
        """Dedup check queries coherence_sig before insert."""
        from api.routes.coherence import _store_extension_event, _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        fields = _build_extension_event(ev)

        conn = _make_mock_conn(dedup_hit=False)

        await _store_extension_event(conn, fields)

        # fetchval was called with coherence_sig
        conn.fetchval.assert_called_once()
        sql = conn.fetchval.call_args.args[0]
        assert "coherence_sig" in sql


# -- Test HTTP endpoint -------------------------------------------------------


class TestCaptureEndpointHTTP:
    """Test the /capture/extension endpoint via mocked pool."""

    @pytest.mark.asyncio
    async def test_single_capture_success(self):
        """POST /capture/extension returns captured status."""
        from api.routes.coherence import capture_extension_event, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task"):  # suppress embedding trigger
            ev = ExtensionEvent(**make_event())
            result = await capture_extension_event(ev)

        assert result["status"] == "captured"
        assert result["event_id"].startswith("ext-")
        assert result["platform"] == "chatgpt"

    @pytest.mark.asyncio
    async def test_single_capture_duplicate(self):
        """Duplicate event is detected via coherence_sig dedup."""
        from api.routes.coherence import capture_extension_event, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=True)

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool):
            ev = ExtensionEvent(**make_event())
            result = await capture_extension_event(ev)

        assert result["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_capture_returns_timestamp(self):
        """Response includes timestamp_ns for the client."""
        from api.routes.coherence import capture_extension_event, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task"):
            ev = ExtensionEvent(**make_event())
            result = await capture_extension_event(ev)

        assert "timestamp_ns" in result
        assert isinstance(result["timestamp_ns"], int)

    @pytest.mark.asyncio
    async def test_capture_triggers_embedding(self):
        """Captured events trigger async embedding."""
        from api.routes.coherence import capture_extension_event, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task") as mock_task:
            ev = ExtensionEvent(**make_event())
            result = await capture_extension_event(ev)

        assert result["status"] == "captured"
        mock_task.assert_called_once()  # embedding trigger fired

    @pytest.mark.asyncio
    async def test_duplicate_does_not_trigger_embedding(self):
        """Duplicate events do NOT trigger embedding."""
        from api.routes.coherence import capture_extension_event, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=True)

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task") as mock_task:
            ev = ExtensionEvent(**make_event())
            result = await capture_extension_event(ev)

        assert result["status"] == "duplicate"
        mock_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_all_platforms(self):
        """Each platform is accepted and stored correctly."""
        from api.routes.coherence import capture_extension_event, ExtensionEvent

        for platform in ["chatgpt", "grok", "gemini", "notebooklm", "youtube"]:
            pool, conn = _make_mock_pool(dedup_hit=False)
            with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
                 patch("asyncio.create_task"):
                ev = ExtensionEvent(**make_event(platform=platform))
                result = await capture_extension_event(ev)
            assert result["platform"] == platform


# -- Test quality scoring integration ----------------------------------------


class TestQualityScoringIntegration:
    """Test quality scoring from capture.quality module."""

    def test_deep_work_content(self):
        """Technical content with keywords scores as deep_work or exploration."""
        from capture.quality import score_event

        score, mode = score_event(
            "Implementing a distributed database schema with coherent architecture "
            "using the sovereign cognitive framework for multi-agent optimization "
            "and system performance analysis",
            role="out",
            platform="chatgpt",
        )
        assert score >= 0.50
        assert mode in ("deep_work", "exploration")

    def test_garbage_content(self):
        """Trivial content scores as garbage or casual."""
        from capture.quality import score_event

        score, mode = score_event("ok", role="out", platform="chatgpt")
        assert score < 0.30
        assert mode == "garbage"

    def test_platform_bonus(self):
        """claude-desktop gets +0.10 platform bonus."""
        from capture.quality import score_event

        content = "Working on the implementation of a new feature for analysis"
        score_chatgpt, _ = score_event(content, "out", "chatgpt")
        score_claude, _ = score_event(content, "out", "claude-desktop")
        assert score_claude >= score_chatgpt + 0.09  # allow float margin

    def test_score_bounds(self):
        """Score is always between 0.0 and 1.0."""
        from capture.quality import score_event

        for content in ["x", "x" * 10000, "", "hello", "complex architecture analysis"]:
            score, mode = score_event(content or "x", "out", "chatgpt")
            assert 0.0 <= score <= 1.0
            assert mode in ("deep_work", "exploration", "casual", "garbage")


# -- Test dedup engine (unit) ------------------------------------------------


class TestDeduplicationEngine:
    """Test the in-memory dedup engine."""

    def test_exact_event_id_match(self):
        """Exact event_id is detected as duplicate."""
        from capture.dedup import DeduplicationEngine

        engine = DeduplicationEngine()
        engine._initialized = True
        engine._seen_event_ids.add("ext-abc123")

        assert engine.is_duplicate("ext-abc123", "hash1", "session1")
        assert not engine.is_duplicate("ext-xyz789", "hash2", "session2")

    def test_content_hash_match(self):
        """Same content hash is detected as duplicate."""
        from capture.dedup import DeduplicationEngine

        engine = DeduplicationEngine()
        engine._initialized = True
        engine._seen_hashes.add("deadbeef" * 8)

        assert engine.is_duplicate("new-id", "deadbeef" * 8, "session1")

    def test_mark_seen(self):
        """mark_seen adds to all tracking sets."""
        from capture.dedup import DeduplicationEngine

        engine = DeduplicationEngine()
        engine._initialized = True
        engine.mark_seen("ext-001", "hash001", "sess-001", "chatgpt")

        assert "ext-001" in engine._seen_event_ids
        assert "hash001" in engine._seen_hashes
        assert "chatgpt:sess-001" in engine._seen_session_keys

    def test_stats(self):
        """Stats reflect internal set sizes."""
        from capture.dedup import DeduplicationEngine

        engine = DeduplicationEngine()
        engine._initialized = True
        engine.mark_seen("id1", "hash1", "s1", "chatgpt")
        engine.mark_seen("id2", "hash2", "s2", "grok")

        stats = engine.stats
        assert stats["event_ids"] == 2
        assert stats["content_hashes"] == 2
        assert stats["session_keys"] == 2
        assert stats["initialized"] is True
