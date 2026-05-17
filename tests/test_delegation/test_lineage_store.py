"""Tests for delegation.lineage_store (E5 keystone)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from delegation.lineage_store import (
    LineageNode,
    LineageStore,
    LineageStoreError,
    VALID_EDGE_TYPES,
    VALID_NODE_TYPES,
)


@pytest.fixture
def store(tmp_path: Path) -> LineageStore:
    return LineageStore(db_path=tmp_path / "lineage.db")


def test_schema_creates_tables_and_indexes(store: LineageStore) -> None:
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        indexes = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        views = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            ).fetchall()
        }
    assert "delegation_lineage" in tables
    assert "delegation_lineage_edges" in tables
    assert "idx_lineage_root" in indexes
    assert "idx_lineage_path" in indexes
    assert "lineage_active_subtree" in views


def test_ensure_schema_is_idempotent(store: LineageStore) -> None:
    store.ensure_schema()
    store.ensure_schema()
    assert store.stats()["total_nodes"] == 0


def test_add_root_node(store: LineageStore) -> None:
    node = store.add_node("root-1", node_type="session")
    assert node.parent_id is None
    assert node.root_id == "root-1"
    assert node.depth == 0
    assert node.path == "/root-1"


def test_add_child_inherits_root_and_increments_depth(store: LineageStore) -> None:
    store.add_node("root-1", node_type="session")
    child = store.add_node("child-1", parent_id="root-1", node_type="delegation")
    grandchild = store.add_node(
        "grand-1", parent_id="child-1", node_type="finding"
    )
    assert child.root_id == "root-1"
    assert child.depth == 1
    assert child.path == "/root-1/child-1"
    assert grandchild.depth == 2
    assert grandchild.path == "/root-1/child-1/grand-1"


def test_unknown_parent_raises(store: LineageStore) -> None:
    with pytest.raises(LineageStoreError, match="parent_id"):
        store.add_node("orphan", parent_id="nope")


def test_invalid_node_type_raises(store: LineageStore) -> None:
    with pytest.raises(LineageStoreError, match="node_type"):
        store.add_node("bad", node_type="martian")


def test_get_node_roundtrip(store: LineageStore) -> None:
    store.add_node("a", node_type="session", metadata={"k": "v"})
    loaded = store.get_node("a")
    assert loaded is not None
    assert loaded.metadata == {"k": "v"}


def test_get_node_missing_returns_none(store: LineageStore) -> None:
    assert store.get_node("ghost") is None


def test_get_ancestors_climbs_from_root(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    store.add_node("a", parent_id="r", node_type="delegation")
    store.add_node("b", parent_id="a", node_type="delegation")
    store.add_node("c", parent_id="b", node_type="finding")

    ancestors = store.get_ancestors("c")
    ids = [n.node_id for n in ancestors]
    assert ids == ["r", "a", "b"]


def test_get_ancestors_of_root_is_empty(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    assert store.get_ancestors("r") == []


def test_get_descendants_strict(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    store.add_node("a", parent_id="r", node_type="delegation")
    store.add_node("b", parent_id="r", node_type="delegation")
    store.add_node("aa", parent_id="a", node_type="finding")

    desc = store.get_descendants("r")
    ids = {n.node_id for n in desc}
    assert ids == {"a", "b", "aa"}
    assert "r" not in ids


def test_get_subtree_includes_root(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    store.add_node("a", parent_id="r", node_type="delegation")
    subtree = store.get_subtree("r")
    ids = {n.node_id for n in subtree}
    assert ids == {"r", "a"}


def test_expire_hides_from_active_queries(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    store.add_node("a", parent_id="r", node_type="delegation")
    assert store.expire_node("a")

    active = store.get_descendants("r")
    assert active == []

    all_nodes = store.get_descendants("r", include_expired=True)
    assert [n.node_id for n in all_nodes] == ["a"]


def test_expire_node_returns_false_on_second_call(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    assert store.expire_node("r") is True
    assert store.expire_node("r") is False


def test_get_roots_lists_only_depth_zero(store: LineageStore) -> None:
    store.add_node("r1", node_type="session")
    store.add_node("r2", node_type="session")
    store.add_node("c", parent_id="r1", node_type="delegation")

    roots = store.get_roots()
    ids = {n.node_id for n in roots}
    assert ids == {"r1", "r2"}


def test_add_edge_and_neighbors(store: LineageStore) -> None:
    store.add_node("a", node_type="finding")
    store.add_node("b", node_type="paper")
    edge = store.add_edge("a", "b", edge_type="cites", weight=0.7)

    assert edge.source_id == "a"
    assert edge.target_id == "b"

    neigh = store.get_neighbors("a")
    assert len(neigh) == 1
    assert neigh[0].edge_type == "cites"
    assert neigh[0].weight == 0.7


def test_expire_edge(store: LineageStore) -> None:
    store.add_node("a", node_type="finding")
    store.add_node("b", node_type="paper")
    edge = store.add_edge("a", "b", edge_type="cites")

    assert store.expire_edge(edge.edge_id) is True
    assert store.get_neighbors("a") == []
    assert len(store.get_neighbors("a", include_expired=True)) == 1


def test_invalid_edge_type_raises(store: LineageStore) -> None:
    store.add_node("a", node_type="finding")
    store.add_node("b", node_type="paper")
    with pytest.raises(LineageStoreError, match="edge_type"):
        store.add_edge("a", "b", edge_type="telepathy")


def test_stats_reports_structure(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    store.add_node("a", parent_id="r", node_type="delegation")
    store.add_node("aa", parent_id="a", node_type="finding")
    store.add_edge("r", "a", edge_type="contains")

    s = store.stats()
    assert s["total_nodes"] == 3
    assert s["active_nodes"] == 3
    assert s["roots"] == 1
    assert s["max_depth"] == 2
    assert s["total_edges"] == 1
    assert s["active_edges"] == 1


def test_bulk_add_respects_parent_order(store: LineageStore) -> None:
    rows = [
        ("r", None, "session"),
        ("a", "r", "delegation"),
        ("b", "r", "delegation"),
        ("aa", "a", "finding"),
    ]
    inserted = store.bulk_add_nodes(rows)
    assert inserted == 4
    assert store.stats()["total_nodes"] == 4
    assert store.get_node("aa").depth == 2


def test_primary_key_prevents_duplicate_node(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_node("r", node_type="session")


def test_valid_type_enums_are_populated() -> None:
    assert "delegation" in VALID_NODE_TYPES
    assert "finding" in VALID_NODE_TYPES
    assert "cites" in VALID_EDGE_TYPES
    assert "spawned" in VALID_EDGE_TYPES


def test_path_uses_slash_separator(store: LineageStore) -> None:
    store.add_node("r", node_type="session")
    a = store.add_node("a", parent_id="r", node_type="delegation")
    assert a.path.startswith("/")
    assert "/" in a.path[1:]


def test_descendants_do_not_leak_between_roots(store: LineageStore) -> None:
    store.add_node("r1", node_type="session")
    store.add_node("r2", node_type="session")
    store.add_node("a", parent_id="r1", node_type="delegation")
    store.add_node("b", parent_id="r2", node_type="delegation")

    d1 = {n.node_id for n in store.get_descendants("r1")}
    d2 = {n.node_id for n in store.get_descendants("r2")}
    assert d1 == {"a"}
    assert d2 == {"b"}


def test_node_and_edge_types_are_persisted_verbatim(store: LineageStore) -> None:
    store.add_node("a", node_type="concept")
    store.add_node("b", node_type="paper")
    store.add_edge("a", "b", edge_type="informs")

    assert store.get_node("a").node_type == "concept"
    edges = store.get_neighbors("a")
    assert edges[0].edge_type == "informs"
