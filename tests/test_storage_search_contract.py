"""
Contract tests for the QdrantDB search layer.

These characterize the exact search behavior — collection targeted, filter
structure, fetch limits, score thresholds, result shapes, and rerank wiring —
for every search_* method, using a fake qdrant_client/cohere injected into
sys.modules. They run without the real SDKs or a live Qdrant server, so they
hold as a net for refactors of the search layer.

Written against the pre-refactor implementation (2026-07-18) and required to
pass unchanged afterward.
"""

import asyncio
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

# ─── Fake qdrant_client SDK ──────────────────────────────────────────────────


@dataclass
class FakeMatchValue:
    value: Any = None


@dataclass
class FakeRange:
    gte: Any = None
    lte: Any = None


@dataclass
class FakeFieldCondition:
    key: str = ""
    match: Any = None
    range: Any = None


@dataclass
class FakeFilter:
    must: list = field(default_factory=list)


@dataclass
class FakePoint:
    payload: Dict[str, Any]
    score: float


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


class FakeAsyncClient:
    """Records query_points calls and returns canned points."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.canned_points: List[FakePoint] = []

    async def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return FakeQueryResult(self.canned_points)


def _install_fake_sdks():
    """Inject fake qdrant_client + cohere so storage.qdrant_db imports cleanly."""
    qc = types.ModuleType("qdrant_client")
    qc.QdrantClient = object
    qc.AsyncQdrantClient = object

    http = types.ModuleType("qdrant_client.http")
    models = types.ModuleType("qdrant_client.http.models")
    models.Distance = types.SimpleNamespace(COSINE="cosine")
    models.VectorParams = object
    models.PointStruct = object
    models.Filter = FakeFilter
    models.FieldCondition = FakeFieldCondition
    models.MatchValue = FakeMatchValue
    models.Range = FakeRange
    models.SearchRequest = object
    models.UpdateStatus = object
    http.models = models
    qc.http = http

    cohere_mod = types.ModuleType("cohere")
    cohere_mod.Client = object

    sys.modules.setdefault("qdrant_client", qc)
    sys.modules.setdefault("qdrant_client.http", http)
    sys.modules.setdefault("qdrant_client.http.models", models)
    sys.modules.setdefault("cohere", cohere_mod)


_install_fake_sdks()

# storage.qdrant_db may already be cached from another test module's import
# chain (with QDRANT_AVAILABLE=False); reload so it binds the fake SDKs.
import importlib  # noqa: E402

import storage.qdrant_db as _qdrant_db_module  # noqa: E402

_qdrant_db_module = importlib.reload(_qdrant_db_module)
QdrantDB = _qdrant_db_module.QdrantDB

QUERY_VEC = [0.1, 0.2, 0.3]


@pytest.fixture
def db(monkeypatch):
    """QdrantDB wired to a fake async client, fake embedder, spy reranker."""
    d = QdrantDB()
    fake = FakeAsyncClient()
    d._async_client = fake
    d.fake = fake

    async def fake_embed_query(query):
        return QUERY_VEC

    rerank_calls = []

    async def fake_rerank(query, documents, top_n=10, content_key="content"):
        rerank_calls.append({"top_n": top_n, "content_key": content_key})
        # reverse order so reranking observably reorders
        out = list(reversed(documents))[:top_n]
        for doc in out:
            doc["relevance_score"] = 0.99
        return out

    monkeypatch.setattr(d, "embed_query_async", fake_embed_query)
    monkeypatch.setattr(d, "rerank_async", fake_rerank)
    d.rerank_calls = rerank_calls
    return d


def _run(coro):
    return asyncio.run(coro)


def _conditions(call) -> Dict[str, Any]:
    """Extract {key: (match_value, range_gte)} from a recorded filter."""
    f = call.get("query_filter")
    if f is None:
        return {}
    out = {}
    for c in f.must:
        out[c.key] = (
            c.match.value if c.match else None,
            c.range.gte if c.range else None,
        )
    return out


# ─── search_findings ─────────────────────────────────────────────────────────


def test_findings_no_filters_no_rerank(db):
    db.fake.canned_points = [
        FakePoint(
            payload={
                "finding_id": "f1", "content": "alpha", "type": "general",
                "session_id": "s1", "project": "p1", "confidence": 0.9,
                "extra_ignored": "x",
            },
            score=0.8,
        )
    ]
    results = _run(db.search_findings("q", limit=5, rerank=False))
    call = db.fake.calls[0]
    assert call["collection_name"] == "findings"
    assert call["query"] == QUERY_VEC
    assert call["limit"] == 5  # no rerank -> no fetch inflation
    assert call["score_threshold"] == 0.5  # default min_score
    assert call["query_filter"] is None
    assert results == [
        {
            "finding_id": "f1", "content": "alpha", "type": "general",
            "session_id": "s1", "project": "p1", "confidence": 0.9,
            "score": 0.8,
        }
    ]
    assert db.rerank_calls == []


def test_findings_filters_and_rerank(db):
    db.fake.canned_points = [
        FakePoint(payload={"finding_id": f"f{i}", "content": f"c{i}"}, score=0.9)
        for i in range(4)
    ]
    results = _run(
        db.search_findings(
            "q", limit=2, filter_type="innovation", filter_project="os-app",
            filter_session="sess-1", rerank=True,
        )
    )
    call = db.fake.calls[0]
    assert call["limit"] == 6  # limit * 3 for rerank
    assert _conditions(call) == {
        "type": ("innovation", None),
        "project": ("os-app", None),
        "session_id": ("sess-1", None),
    }
    assert db.rerank_calls == [{"top_n": 2, "content_key": "content"}]
    assert len(results) == 2
    assert results[0]["finding_id"] == "f3"  # fake rerank reverses


def test_findings_rerank_top_n_override(db):
    db.fake.canned_points = [
        FakePoint(payload={"finding_id": f"f{i}", "content": "c"}, score=0.9)
        for i in range(6)
    ]
    results = _run(db.search_findings("q", limit=3, rerank=True, rerank_top_n=5))
    assert db.rerank_calls == [{"top_n": 5, "content_key": "content"}]
    assert len(results) == 3  # still truncated to limit


# ─── search_sessions ─────────────────────────────────────────────────────────


def test_sessions_shape_and_defaults(db):
    db.fake.canned_points = [
        FakePoint(
            payload={
                "session_id": "s1", "topic": "agents", "project": "p",
                "status": "archived", "finding_count": 7, "url_count": 3,
            },
            score=0.7,
        )
    ]
    results = _run(db.search_sessions("q", rerank=False, filter_project="p"))
    call = db.fake.calls[0]
    assert call["collection_name"] == "sessions"
    assert call["score_threshold"] == 0.4
    assert _conditions(call) == {"project": ("p", None)}
    assert results == [
        {
            "session_id": "s1", "topic": "agents", "project": "p",
            "status": "archived", "finding_count": 7,
            "content": "agents",  # topic aliased for reranking
            "score": 0.7,
        }
    ]


def test_sessions_rerank_uses_topic(db):
    db.fake.canned_points = [
        FakePoint(payload={"session_id": "s", "topic": "t"}, score=0.9)
    ]
    _run(db.search_sessions("q", limit=4, rerank=True))
    assert db.fake.calls[0]["limit"] == 12
    assert db.rerank_calls == [{"top_n": 4, "content_key": "topic"}]


# ─── search_packs ────────────────────────────────────────────────────────────


def test_packs_shape_filters_rerank(db):
    db.fake.canned_points = [
        FakePoint(
            payload={
                "pack_id": "pk1", "name": "debug pack", "type": "workflow",
                "source": "local", "tokens": 1200,
            },
            score=0.6,
        )
    ]
    results = _run(
        db.search_packs("q", filter_type="workflow", filter_source="local", rerank=True)
    )
    call = db.fake.calls[0]
    assert call["collection_name"] == "packs"
    assert call["score_threshold"] == 0.4
    assert _conditions(call) == {"type": ("workflow", None), "source": ("local", None)}
    assert db.rerank_calls == [{"top_n": 10, "content_key": "name"}]
    assert results[0]["pack_id"] == "pk1"
    assert results[0]["content"] == "debug pack"  # name aliased


# ─── search_papers ───────────────────────────────────────────────────────────


def test_papers_range_and_match_filters(db):
    db.fake.canned_points = [
        FakePoint(
            payload={
                "paper_id": "2501.1", "title": "T", "content": "T\n\nabs",
                "authors": "A", "relevance": 5, "source": "hf", "url": "u",
                "ai_keywords": ["k"], "upvotes": 10, "github_repo": "g",
            },
            score=0.9,
        )
    ]
    results = _run(
        db.search_papers(
            "q", limit=3, min_relevance=4, filter_source="hf",
            rerank=True, rerank_top_n=7,
        )
    )
    call = db.fake.calls[0]
    assert call["collection_name"] == "papers"
    assert call["score_threshold"] == 0.4
    assert call["limit"] == 9
    assert _conditions(call) == {"relevance": (None, 4), "source": ("hf", None)}
    assert db.rerank_calls == [{"top_n": 7, "content_key": "content"}]
    assert set(results[0]) == {
        "paper_id", "title", "content", "authors", "relevance", "source",
        "url", "ai_keywords", "upvotes", "github_repo", "score",
        "relevance_score",  # added by rerank
    }


def test_papers_min_relevance_zero_still_filters(db):
    # min_relevance uses `is not None`, so 0 must produce a range filter
    db.fake.canned_points = []
    _run(db.search_papers("q", min_relevance=0, rerank=False))
    assert _conditions(db.fake.calls[0]) == {"relevance": (None, 0)}


# ─── search_outcomes ─────────────────────────────────────────────────────────


def test_outcomes_filters_shape_rerank(db):
    db.fake.canned_points = [
        FakePoint(
            payload={
                "outcome_id": "o1", "session_id": "s1", "intent": "build X",
                "outcome": "success", "quality": 4.5, "model_efficiency": 0.8,
                "models_used": ["opus"], "date": "2026-07-01", "messages": 20,
                "tools": 5,
            },
            score=0.9,
        )
    ]
    results = _run(
        db.search_outcomes("q", filter_outcome="success", min_quality=4, rerank=True)
    )
    call = db.fake.calls[0]
    assert call["collection_name"] == "session_outcomes"
    assert call["score_threshold"] == 0.5
    assert _conditions(call) == {"outcome": ("success", None), "quality": (None, 4)}
    assert db.rerank_calls == [{"top_n": 10, "content_key": "intent"}]
    assert results[0]["content"] == "build X"  # intent aliased


def test_outcomes_min_quality_zero_is_no_filter(db):
    # min_quality uses truthiness, so 0 must NOT add a filter
    db.fake.canned_points = []
    _run(db.search_outcomes("q", min_quality=0, rerank=False))
    assert db.fake.calls[0]["query_filter"] is None


# ─── search_cognitive_states ─────────────────────────────────────────────────


def test_cognitive_states_no_rerank_no_inflation(db):
    db.fake.canned_points = [
        FakePoint(
            payload={
                "state_id": "st1", "mode": "peak", "energy_level": 0.9,
                "flow_score": 0.8, "hour": 15, "day": "mon",
                "timestamp": "t", "predictions": {},
            },
            score=0.7,
        )
    ]
    results = _run(db.search_cognitive_states("q", limit=4))
    call = db.fake.calls[0]
    assert call["collection_name"] == "cognitive_states"
    assert call["limit"] == 4  # never inflated — this method has no rerank
    assert call["score_threshold"] == 0.5
    assert call.get("query_filter") is None
    assert db.rerank_calls == []
    assert results == [
        {
            "state_id": "st1", "mode": "peak", "energy_level": 0.9,
            "flow_score": 0.8, "hour": 15, "day": "mon", "timestamp": "t",
            "predictions": {}, "score": 0.7,
        }
    ]


# ─── search_error_patterns ───────────────────────────────────────────────────


def test_error_patterns_range_filter_no_rerank(db):
    db.fake.canned_points = [
        FakePoint(
            payload={
                "error_id": "e1", "error_type": "ImportError", "context": "ctx",
                "solution": "pip install", "success_rate": 0.9,
            },
            score=0.8,
        )
    ]
    results = _run(db.search_error_patterns("q", min_success_rate=0.5))
    call = db.fake.calls[0]
    assert call["collection_name"] == "error_patterns"
    assert call["limit"] == 10  # no inflation — no rerank on this method
    assert _conditions(call) == {"success_rate": (None, 0.5)}
    assert db.rerank_calls == []
    assert results[0]["solution"] == "pip install"


def test_error_patterns_zero_success_rate_is_no_filter(db):
    db.fake.canned_points = []
    _run(db.search_error_patterns("q", min_success_rate=0.0))
    assert db.fake.calls[0]["query_filter"] is None


# ─── semantic_search (multi-collection) ──────────────────────────────────────


def test_semantic_search_default_collections_and_content_keys(db):
    db.fake.canned_points = [FakePoint(payload={"content": "c", "topic": "t"}, score=0.9)]
    results = _run(db.semantic_search("q", limit=2, rerank=True))
    assert [c["collection_name"] for c in db.fake.calls] == [
        "findings", "sessions", "packs", "papers",
    ]
    assert all(c["limit"] == 6 for c in db.fake.calls)
    assert [r["content_key"] for r in db.rerank_calls] == [
        "content", "topic", "name", "content",
    ]
    assert set(results) == {"findings", "sessions", "packs", "papers"}


def test_semantic_search_skips_unknown_collections(db):
    db.fake.canned_points = []
    results = _run(db.semantic_search("q", collections=["findings", "nonexistent"]))
    assert [c["collection_name"] for c in db.fake.calls] == ["findings"]
    assert "nonexistent" not in results
