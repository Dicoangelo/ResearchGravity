"""E5 integration tests: 4Ds delegation gate → lineage store auto-record."""

from __future__ import annotations

from pathlib import Path

import pytest

from delegation.four_ds import FourDsGate
from delegation.lineage_store import LineageStore
from delegation.models import TaskProfile


@pytest.fixture
def gate_and_store(tmp_path: Path):
    db = tmp_path / "delegation_events.db"
    gate = FourDsGate(db_path=str(db))
    store = LineageStore(db_path=db)
    return gate, store


def test_approved_delegation_creates_root_lineage_node(gate_and_store):
    gate, store = gate_and_store
    approved, _ = gate.delegation_gate("do safe thing", TaskProfile())
    assert approved is True

    node = store.get_node(gate._hash_task("do safe thing"))
    assert node is not None
    assert node.node_type == "delegation"
    assert node.parent_id is None
    assert node.depth == 0
    assert node.metadata["status"] == "approved"
    assert node.metadata["gate"] == "delegation"


def test_blocked_delegation_still_creates_lineage_node(gate_and_store):
    gate, store = gate_and_store
    profile = TaskProfile(
        subjectivity=0.9, criticality=0.9, reversibility=0.1
    )
    approved, _ = gate.delegation_gate("risky judgment call", profile)
    assert approved is False

    node = store.get_node(gate._hash_task("risky judgment call"))
    assert node is not None
    assert node.metadata["status"] == "blocked"


def test_parent_task_id_wires_hierarchy(gate_and_store, monkeypatch):
    gate, store = gate_and_store
    root_id = gate._hash_task("parent task")
    store.add_node(root_id, node_type="delegation")

    # Inject parent_task_id by calling the log path directly with details.
    child_task = "child task"
    child_id = gate._hash_task(child_task)
    gate._log_event(
        task_id=child_id,
        event_type="delegation_gate",
        status="approved",
        details={"gate": "delegation", "parent_task_id": root_id},
    )

    child = store.get_node(child_id)
    assert child is not None
    assert child.parent_id == root_id
    assert child.depth == 1
    assert child.root_id == root_id


def test_missing_parent_degrades_to_root(gate_and_store):
    gate, store = gate_and_store
    tid = gate._hash_task("orphan task")
    gate._log_event(
        task_id=tid,
        event_type="delegation_gate",
        status="approved",
        details={"gate": "delegation", "parent_task_id": "ghost"},
    )

    node = store.get_node(tid)
    assert node is not None
    assert node.parent_id is None
    assert node.depth == 0
    assert node.metadata.get("degraded") == "parent_not_found"


def test_duplicate_invocation_is_idempotent(gate_and_store):
    gate, store = gate_and_store
    task = "repeat task"
    gate.delegation_gate(task, TaskProfile())
    gate.delegation_gate(task, TaskProfile())
    gate.delegation_gate(task, TaskProfile())

    assert store.stats()["total_nodes"] == 1


def test_non_delegation_events_do_not_record_lineage(gate_and_store):
    gate, store = gate_and_store
    # description/discernment/diligence gates call _log_event with other types.
    gate._log_event(
        task_id="abc12345",
        event_type="description_gate",
        status="scored",
        details={"gate": "description", "score": 0.7},
    )
    assert store.get_node("abc12345") is None
    assert store.stats()["total_nodes"] == 0
