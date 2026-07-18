"""
Contract tests for the SqliteVecDB search layer.

Characterize search_findings / search_sessions / search_packs — SQL targeted,
params, score conversion, post-hoc metadata filtering, result shapes, and the
FTS LIKE fallback — with a fake connection, so they run without the sqlite-vec
extension and hold as a net for refactors.

Written against the pre-refactor implementation (2026-07-18) and required to
pass unchanged afterward.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import pytest

import storage.sqlite_vec as sv
from storage.sqlite_vec import SqliteVecDB

QUERY_VEC = [0.1] * 4


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class FakeConn:
    """Records execute() calls; returns rows canned per SQL substring."""

    def __init__(self, canned: Dict[str, List[tuple]]):
        self.canned = canned
        self.calls: List[Dict[str, Any]] = []

    async def execute(self, sql, params=()):
        self.calls.append({"sql": " ".join(sql.split()), "params": params})
        for key, rows in self.canned.items():
            if key in sql:
                return FakeCursor(rows)
        return FakeCursor([])


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(sv, "SQLITE_VEC_AVAILABLE", True)
    d = SqliteVecDB.__new__(SqliteVecDB)  # skip __init__ (no cohere/db setup)

    async def fake_embed_query(text, dimension=None):
        return QUERY_VEC

    d.embed_query = fake_embed_query

    conn_holder = {}

    @asynccontextmanager
    async def fake_connection():
        yield conn_holder["conn"]

    d.connection = fake_connection
    d._set_conn = lambda c: conn_holder.__setitem__("conn", c)
    return d


def _run(coro):
    return asyncio.run(coro)


def _meta(**kw):
    return json.dumps(kw)


# ─── search_findings ─────────────────────────────────────────────────────────


def test_findings_vector_path_shape_and_score(db):
    conn = FakeConn(
        {
            "vec_findings": [
                ("f1", "alpha", _meta(type="general", project="p1"), 0.2),
                ("f2", "beta", _meta(type="general"), 0.7),  # score 0.3 < 0.5 min
            ]
        }
    )
    db._set_conn(conn)
    results = _run(db.search_findings("q", limit=5))

    sql = conn.calls[0]["sql"]
    assert "vec_findings" in sql
    assert "vec_distance_cosine" in sql
    assert "vm.entity_type = 'finding'" in sql
    assert conn.calls[0]["params"][-1] == 10  # fetches limit * 2

    # distance 0.2 -> score 0.8; second row filtered by min_score
    assert results == [
        {
            "id": "f1",
            "content": "alpha",
            "score": 0.8,
            "relevance_score": 0.8,
            "type": "general",
            "project": "p1",
        }
    ]


def test_findings_metadata_filters_applied_post_query(db):
    conn = FakeConn(
        {
            "vec_findings": [
                ("f1", "a", _meta(type="innovation", project="os-app"), 0.1),
                ("f2", "b", _meta(type="general", project="os-app"), 0.1),
                ("f3", "c", _meta(type="innovation", project="other"), 0.1),
            ]
        }
    )
    db._set_conn(conn)
    results = _run(
        db.search_findings("q", filter_type="innovation", filter_project="os-app")
    )
    assert [r["id"] for r in results] == ["f1"]


def test_findings_truncates_at_limit(db):
    rows = [(f"f{i}", "c", _meta(), 0.1) for i in range(8)]
    conn = FakeConn({"vec_findings": rows})
    db._set_conn(conn)
    results = _run(db.search_findings("q", limit=3))
    assert len(results) == 3


def test_findings_like_fallback_when_no_vector_results(db):
    conn = FakeConn(
        {
            "vec_findings": [],
            "content LIKE": [("f9", "fallback hit", _meta(type="general"))],
        }
    )
    db._set_conn(conn)
    results = _run(db.search_findings("needle", limit=5))

    like_call = next(c for c in conn.calls if "LIKE" in c["sql"])
    assert like_call["params"] == ("%needle%", 5)
    assert results == [
        {"id": "f9", "content": "fallback hit", "score": 0.5, "type": "general"}
    ]


def test_findings_no_embedding_goes_straight_to_fallback(db):
    async def no_embed(text, dimension=None):
        return None

    db.embed_query = no_embed
    conn = FakeConn({"content LIKE": []})
    db._set_conn(conn)
    results = _run(db.search_findings("q"))
    assert results == []
    assert all("vec_findings" not in c["sql"] for c in conn.calls)


# ─── search_sessions ─────────────────────────────────────────────────────────


def test_sessions_shape_and_project_filter(db):
    conn = FakeConn(
        {
            "vec_sessions": [
                ("s1", "agents research", _meta(project="p1", status="archived"), 0.3),
                ("s2", "other", _meta(project="p2"), 0.3),
            ]
        }
    )
    db._set_conn(conn)
    results = _run(db.search_sessions("q", limit=5, filter_project="p1"))

    sql = conn.calls[0]["sql"]
    assert "vec_sessions" in sql
    assert "vm.entity_type = 'session'" in sql
    assert conn.calls[0]["params"][-1] == 5  # no fetch inflation for sessions

    assert results == [
        {
            "id": "s1",
            "topic": "agents research",  # sessions use "topic", not "content"
            "score": 0.7,
            "project": "p1",
            "status": "archived",
        }
    ]


def test_sessions_no_fallback_on_empty(db):
    conn = FakeConn({"vec_sessions": []})
    db._set_conn(conn)
    results = _run(db.search_sessions("q"))
    assert results == []
    assert all("LIKE" not in c["sql"] for c in conn.calls)


# ─── search_packs ────────────────────────────────────────────────────────────


def test_packs_shape_and_filters(db):
    conn = FakeConn(
        {
            "vec_packs": [
                ("pk1", "debug pack", _meta(type="workflow", source="local"), 0.4),
                ("pk2", "other", _meta(type="workflow", source="remote"), 0.4),
                ("pk3", "third", _meta(type="notes", source="local"), 0.4),
            ]
        }
    )
    db._set_conn(conn)
    results = _run(
        db.search_packs("q", filter_type="workflow", filter_source="local")
    )

    sql = conn.calls[0]["sql"]
    assert "vec_packs" in sql
    assert "vm.entity_type = 'pack'" in sql

    assert results == [
        {
            "id": "pk1",
            "content": "debug pack",
            "score": 0.6,
            "type": "workflow",
            "source": "local",
        }
    ]


def test_packs_min_score_default(db):
    conn = FakeConn({"vec_packs": [("pk1", "c", _meta(), 0.65)]})  # score 0.35
    db._set_conn(conn)
    assert _run(db.search_packs("q")) == []  # default min_score 0.4 filters it
