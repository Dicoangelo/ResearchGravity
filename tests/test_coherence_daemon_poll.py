"""
Tests for the coherence daemon's poll loop.

Regression coverage for the embedding starvation bug: the poll query used to
INNER JOIN embedding_cache, so an event was only eligible for coherence
scanning if something else had already embedded it. The daemon is what creates
embeddings, and the only method that persisted them was unreachable — so the
result set was empty by construction for every event that did not arrive
through the extension capture endpoint. 75,124 events were invisible and the
daemon logged nothing for six months.

Two invariants keep it fixed:
  1. The poll must not gate on a pre-existing embedding.
  2. Embeddings computed in the loop must be persisted, because
     SimilarityIndex searches embedding_cache — an unpersisted event can never
     be found as a neighbour of a later one.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_conn(rows=None):
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 1")
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=rows if rows is not None else [])
    return conn


def _make_mock_pool(conn):
    pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool


def _event_row(event_id="ev-1", platform="claude-code"):
    return {
        "event_id": event_id,
        "session_id": "sess-1",
        "timestamp_ns": 1_785_000_000_000_000_000,
        "platform": platform,
        "data_layer": json.dumps(
            {"content": "A sufficiently long piece of content to survive the "
                        "minimum length filter applied before embedding."}
        ),
        "light_layer": json.dumps({"topic": "testing", "intent": "verify"}),
        "instinct_layer": json.dumps({"coherence_potential": 0.5}),
        "coherence_sig": None,
    }


class TestPollQueryIsNotGatedOnEmbeddings:
    """Invariant 1: the poll must not require a pre-existing embedding."""

    @pytest.mark.asyncio
    async def test_poll_query_does_not_join_embedding_cache(self):
        from coherence_engine.daemon import CoherenceDaemon

        conn = _make_mock_conn(rows=[])
        daemon = CoherenceDaemon(pool=_make_mock_pool(conn))

        await daemon._poll_and_process()

        assert conn.fetch.await_count >= 1
        sql = conn.fetch.await_args_list[0].args[0]
        normalized = " ".join(sql.split()).lower()

        assert "from cognitive_events" in normalized
        assert "coherence_scanned_at is null" in normalized
        # The regression: gating the candidate set on embedding_cache made it
        # empty by construction.
        assert "join embedding_cache" not in normalized


class TestComputedEmbeddingsArePersisted:
    """Invariant 2: embeddings computed in the loop reach embedding_cache."""

    @pytest.mark.asyncio
    async def test_poll_persists_embeddings_it_computes(self):
        from coherence_engine.daemon import CoherenceDaemon

        conn = _make_mock_conn(rows=[_event_row()])
        daemon = CoherenceDaemon(pool=_make_mock_pool(conn))
        daemon._multi_scale = AsyncMock()
        daemon._multi_scale.detect_multi_scale = AsyncMock(return_value=[])

        with patch(
            "coherence_engine.daemon.embed_texts", return_value=[[0.1] * 768]
        ), patch(
            "coherence_engine.daemon.store_embeddings", new=AsyncMock(return_value=1)
        ) as store:
            await daemon._poll_and_process()

        store.assert_awaited_once()
        items = store.await_args.args[1]
        assert len(items) == 1
        event_id, text, embedding = items[0]
        assert event_id == "ev-1"
        assert text
        assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_events_are_marked_scanned(self):
        from coherence_engine.daemon import CoherenceDaemon

        conn = _make_mock_conn(rows=[_event_row()])
        daemon = CoherenceDaemon(pool=_make_mock_pool(conn))
        daemon._multi_scale = AsyncMock()
        daemon._multi_scale.detect_multi_scale = AsyncMock(return_value=[])

        with patch(
            "coherence_engine.daemon.embed_texts", return_value=[[0.1] * 768]
        ), patch(
            "coherence_engine.daemon.store_embeddings", new=AsyncMock(return_value=1)
        ):
            processed = await daemon._poll_and_process()

        assert processed == 1
        update_sql = " ".join(conn.execute.await_args.args[0].split()).lower()
        assert "update cognitive_events" in update_sql
        assert "coherence_scanned_at" in update_sql

    @pytest.mark.asyncio
    async def test_no_persist_call_when_all_content_filtered(self):
        """Noise/short events yield no embeddings, so nothing is stored.

        Note event_to_text builds *contextual* text (platform, topic, intent),
        so the layers must be empty too — content alone being short is not
        enough to fall under MIN_CONTENT_LENGTH.
        """
        from coherence_engine.daemon import CoherenceDaemon

        row = _event_row()
        row["data_layer"] = json.dumps({"content": "hi"})
        row["light_layer"] = json.dumps({})
        row["instinct_layer"] = json.dumps({})
        conn = _make_mock_conn(rows=[row])
        daemon = CoherenceDaemon(pool=_make_mock_pool(conn))
        daemon._multi_scale = AsyncMock()
        daemon._multi_scale.detect_multi_scale = AsyncMock(return_value=[])

        with patch(
            "coherence_engine.daemon.store_embeddings", new=AsyncMock(return_value=0)
        ) as store:
            processed = await daemon._poll_and_process()

        store.assert_not_awaited()
        # Still marked scanned — otherwise short events are retried forever.
        assert processed == 1
