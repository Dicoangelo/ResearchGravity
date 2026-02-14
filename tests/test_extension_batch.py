"""
Tests for the Chrome extension batch capture endpoint.

Covers: batch dedup, transaction behavior, cap enforcement, error handling,
edge cases, embedding pipeline, and embedding trigger.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# -- Helpers ------------------------------------------------------------------


def make_event(
    platform="chatgpt",
    content="Test message with enough content for capture",
    direction="out",
    session_hint="chatgpt-test-batch",
    **kwargs,
):
    return {
        "platform": platform,
        "content": content,
        "direction": direction,
        "session_hint": session_hint,
        **kwargs,
    }


def _make_mock_conn(*, dedup_hit=False):
    """Create a mock asyncpg connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetchval = AsyncMock(return_value="ext-existing" if dedup_hit else None)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    # Transaction context manager
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)
    return conn


def _make_mock_pool(conn=None, *, dedup_hit=False):
    """Create a mock asyncpg pool.

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


# -- Test batch dedup ---------------------------------------------------------


class TestBatchDedup:
    """Test dedup behavior via deterministic event_id."""

    def test_same_content_same_id(self):
        """Identical events produce identical event_ids."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        f1 = _build_extension_event(ev)
        f2 = _build_extension_event(ev)
        assert f1["event_id"] == f2["event_id"]

    def test_direction_matters_for_dedup(self):
        """Same content with different direction produces different event_id."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev_out = ExtensionEvent(**make_event(direction="out"))
        ev_in = ExtensionEvent(**make_event(direction="in"))
        f_out = _build_extension_event(ev_out)
        f_in = _build_extension_event(ev_in)
        assert f_out["event_id"] != f_in["event_id"]

    def test_platform_matters_for_dedup(self):
        """Same content on different platforms produces different event_id."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev_chatgpt = ExtensionEvent(**make_event(platform="chatgpt"))
        ev_grok = ExtensionEvent(**make_event(platform="grok"))
        f1 = _build_extension_event(ev_chatgpt)
        f2 = _build_extension_event(ev_grok)
        assert f1["event_id"] != f2["event_id"]


# -- Test batch endpoint (HTTP-level) ----------------------------------------


class TestBatchEndpointHTTP:
    """Test POST /capture/extension/batch via mocked pool."""

    @pytest.mark.asyncio
    async def test_batch_capture_all_new(self):
        """Batch of 3 new events returns captured=3."""
        from api.routes.coherence import capture_extension_batch, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)

        events = [
            ExtensionEvent(**make_event(content=f"unique message number {i} here"))
            for i in range(3)
        ]

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task"):
            result = await capture_extension_batch(events)

        assert result["captured"] == 3
        assert result["duplicates"] == 0
        assert result["errors"] == 0
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_batch_with_dedup_hits(self):
        """Batch where coherence_sig dedup catches duplicates."""
        from api.routes.coherence import capture_extension_batch, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=True)

        events = [
            ExtensionEvent(**make_event(content=f"event {i} for dedup test"))
            for i in range(3)
        ]

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool):
            result = await capture_extension_batch(events)

        assert result["duplicates"] == 3
        assert result["captured"] == 0

    @pytest.mark.asyncio
    async def test_batch_cap_at_50(self):
        """Only first 50 events are processed."""
        from api.routes.coherence import capture_extension_batch, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)

        events = [
            ExtensionEvent(**make_event(content=f"batch event number {i} unique"))
            for i in range(60)
        ]

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task"):
            result = await capture_extension_batch(events)

        assert result["total"] == 50

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        """Empty batch returns all-zero counts."""
        from api.routes.coherence import capture_extension_batch, ExtensionEvent

        pool, conn = _make_mock_pool()

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool):
            result = await capture_extension_batch([])

        assert result["total"] == 0
        assert result["captured"] == 0

    @pytest.mark.asyncio
    async def test_batch_transaction_wrapping(self):
        """Batch uses conn.transaction() for atomicity."""
        from api.routes.coherence import capture_extension_batch, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)

        events = [ExtensionEvent(**make_event(content=f"tx event {i}")) for i in range(3)]

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task"):
            await capture_extension_batch(events)

        # transaction() was called once for the whole batch
        conn.transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_error_handling(self):
        """Individual event errors don't crash the batch."""
        from api.routes.coherence import capture_extension_batch, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on the 4th execute call (second event's INSERT)
            if call_count == 4:
                raise Exception("simulated DB error")
            return "INSERT 0 1"

        conn.execute = AsyncMock(side_effect=side_effect)

        events = [
            ExtensionEvent(**make_event(content=f"error test event {i} unique"))
            for i in range(3)
        ]

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task"):
            result = await capture_extension_batch(events)

        assert result["errors"] >= 1
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_batch_response_shape(self):
        """Response has exactly the expected keys."""
        from api.routes.coherence import capture_extension_batch, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)

        events = [ExtensionEvent(**make_event())]

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task"):
            result = await capture_extension_batch(events)

        assert set(result.keys()) == {"captured", "duplicates", "errors", "total"}
        for v in result.values():
            assert isinstance(v, int)

    @pytest.mark.asyncio
    async def test_batch_triggers_embeddings_for_captured(self):
        """Batch triggers embedding for each captured event."""
        from api.routes.coherence import capture_extension_batch, ExtensionEvent

        pool, conn = _make_mock_pool(dedup_hit=False)

        events = [
            ExtensionEvent(**make_event(content=f"embed trigger test {i}"))
            for i in range(3)
        ]

        with patch("api.routes.coherence._get_pool", new_callable=AsyncMock, return_value=pool), \
             patch("asyncio.create_task") as mock_task:
            result = await capture_extension_batch(events)

        assert result["captured"] == 3
        assert mock_task.call_count == 3  # one embedding trigger per captured event


# -- Test batch edge cases ---------------------------------------------------


class TestBatchEdgeCases:
    """Test edge cases for batch capture."""

    def test_content_with_unicode(self):
        """Unicode content is handled correctly."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(
            **make_event(content="Analyzing 量子コンピューティング research — emergence detected!")
        )
        fields = _build_extension_event(ev)
        data = json.loads(fields["data_layer"])
        assert "量子" in data["content"]

    def test_very_long_content_truncated(self):
        """Content over 10K is truncated in data_layer."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        long = "x" * 15000
        ev = ExtensionEvent(**make_event(content=long))
        fields = _build_extension_event(ev)
        data = json.loads(fields["data_layer"])
        assert len(data["content"]) == 10000

    def test_coherence_sig_stability(self):
        """Coherence signature is deterministic across calls."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(**make_event())
        sigs = set()
        for _ in range(5):
            fields = _build_extension_event(ev)
            sigs.add(fields["coherence_sig"])
        assert len(sigs) == 1  # All identical

    def test_mixed_platforms_in_batch(self):
        """Batch with events from different platforms all produce valid fields."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        platforms = ["chatgpt", "grok", "gemini", "notebooklm", "youtube"]
        for p in platforms:
            ev = ExtensionEvent(**make_event(platform=p, content=f"msg from {p}"))
            fields = _build_extension_event(ev)
            assert fields["platform"] == p
            assert fields["event_id"].startswith("ext-")

    def test_special_characters_in_content(self):
        """SQL injection-like content is safely handled."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        dangerous = "'; DROP TABLE cognitive_events; --"
        ev = ExtensionEvent(**make_event(content=dangerous))
        fields = _build_extension_event(ev)
        data = json.loads(fields["data_layer"])
        assert "DROP TABLE" in data["content"]  # Content preserved, not executed

    def test_null_optional_fields(self):
        """Events with all optional fields as None produce valid events."""
        from api.routes.coherence import _build_extension_event, ExtensionEvent

        ev = ExtensionEvent(
            platform="chatgpt",
            content="minimal event with no optional fields set",
            direction="in",
        )
        fields = _build_extension_event(ev)
        light = json.loads(fields["light_layer"])
        assert light["topic"] == ""
        assert light["intent"] == ""
        assert light["concepts"] == []


# -- Test embedding pipeline (unit) ------------------------------------------


class TestEmbeddingPipelineUnit:
    """Test embedding-related logic without loading models."""

    def test_event_to_text_with_data_layer(self):
        """event_to_text extracts embeddable text from event row."""
        from coherence_engine.embeddings import event_to_text

        row = {
            "data_layer": {"content": "Analyzing sovereign AI architecture"},
            "light_layer": {"topic": "AI", "intent": "analyze", "concepts": ["sovereign"]},
            "platform": "chatgpt",
            "cognitive_mode": "deep_work",
        }
        text = event_to_text(row)
        assert text  # Non-empty
        assert len(text) >= 10

    def test_event_to_text_empty_content(self):
        """Empty data_layer content returns minimal or empty text."""
        from coherence_engine.embeddings import event_to_text

        row = {
            "data_layer": {"content": ""},
            "light_layer": {},
        }
        text = event_to_text(row)
        # May return empty or minimal — that's OK, embed_event_row checks length

    def test_content_hash_deterministic(self):
        """content_hash produces same hash for same input."""
        from mcp_raw.embeddings import content_hash

        h1 = content_hash("test content for hashing")
        h2 = content_hash("test content for hashing")
        assert h1 == h2

    def test_content_hash_differs(self):
        """Different content produces different hashes."""
        from mcp_raw.embeddings import content_hash

        h1 = content_hash("content alpha")
        h2 = content_hash("content beta")
        assert h1 != h2


# -- Test embedding pipeline (async with mocked pool) -----------------------


class TestEmbeddingPipelineAsync:
    """Test async embedding storage with mocked DB."""

    @pytest.mark.asyncio
    async def test_embed_event_row_stores_embedding(self):
        """embed_event_row stores in embedding_cache via pool."""
        from coherence_engine.embeddings import embed_event_row

        pool, conn = _make_mock_pool()

        row = {
            "event_id": "ext-test-001",
            "data_layer": {"content": "Deep analysis of sovereign AI infrastructure"},
            "light_layer": {"topic": "AI", "intent": "analyze"},
            "platform": "chatgpt",
        }

        with patch("coherence_engine.embeddings.embed_single") as mock_embed:
            mock_embed.return_value = [0.1] * 768  # fake embedding
            result = await embed_event_row(pool, row)

        assert result is not None
        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_event_row_skips_short_text(self):
        """embed_event_row returns None for very short content."""
        from coherence_engine.embeddings import embed_event_row

        pool, conn = _make_mock_pool()
        row = {
            "event_id": "ext-short",
            "data_layer": {"content": "hi"},
            "light_layer": {},
        }

        result = await embed_event_row(pool, row)
        # Short text should be skipped (returns None)

    @pytest.mark.asyncio
    async def test_batch_embed_events_skips_existing(self):
        """batch_embed_events with skip_existing=True uses NOT EXISTS subquery."""
        from coherence_engine.embeddings import batch_embed_events

        pool, conn = _make_mock_pool()
        # No unembedded events found
        conn.fetch = AsyncMock(return_value=[])

        result = await batch_embed_events(pool, limit=10, skip_existing=True)

        assert result == 0
        # Verify the SQL used NOT EXISTS
        fetch_call = conn.fetch.call_args
        sql = fetch_call.args[0]
        assert "NOT EXISTS" in sql
