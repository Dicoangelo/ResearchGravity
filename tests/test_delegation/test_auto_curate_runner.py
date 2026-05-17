"""E7 tests: auto-curate runner closes coherence → notebook lineage loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from delegation.auto_curate_gate import CurateGate, CurateGatePolicy
from delegation.auto_curate_runner import (
    AutoCurateRunner,
    RunOutcome,
    RunReport,
    run_auto_curate_cycle,
)
from delegation.lineage_store import LineageStore


@pytest.fixture
def store(tmp_path: Path) -> LineageStore:
    return LineageStore(db_path=tmp_path / "lineage.db")


@pytest.fixture
def gate(store: LineageStore) -> CurateGate:
    return CurateGate(store, CurateGatePolicy(budget_per_cycle=5))


def _arc(arc_id: str = "a1", significance: float = 0.9) -> dict:
    return {
        "arc_id": arc_id,
        "topic": "convergence",
        "significance": significance,
        "platforms": 3,
        "moments": 5,
    }


def _make_executor(result_by_arc: dict[str, Optional[dict[str, Any]]]):
    calls: list[str] = []

    async def executor(arc_id: str) -> Optional[dict[str, Any]]:
        calls.append(arc_id)
        return result_by_arc.get(arc_id)

    executor.calls = calls  # type: ignore[attr-defined]
    return executor


@pytest.mark.asyncio
async def test_success_records_notebook_node_and_edge(store, gate):
    executor = _make_executor(
        {
            "a1": {
                "notebook_id": "nb-123",
                "title": "Arc: convergence",
                "sources_added": 4,
                "moments_included": 5,
            }
        }
    )
    runner = AutoCurateRunner(gate, store, executor)
    report = await runner.run_cycle([_arc("a1")])

    assert report.approved_count == 1
    assert report.succeeded_count == 1
    assert report.failed_count == 0
    outcome = report.outcomes[0]
    assert outcome.succeeded
    assert outcome.notebook_id == "nb-123"
    assert outcome.lineage_node_id == "notebook:nb-123"
    assert outcome.error is None

    notebook_node = store.get_node("notebook:nb-123")
    assert notebook_node is not None
    assert notebook_node.node_type == "session"
    assert notebook_node.parent_id == "arc:a1"
    assert notebook_node.metadata["notebook_id"] == "nb-123"
    assert notebook_node.metadata["sources_added"] == 4


@pytest.mark.asyncio
async def test_executor_exception_captured_and_cycle_continues(store, gate):
    async def executor(arc_id: str):
        if arc_id == "a1":
            raise RuntimeError("boom")
        return {"notebook_id": "nb-ok"}

    runner = AutoCurateRunner(gate, store, executor)
    report = await runner.run_cycle([_arc("a1"), _arc("a2")])

    assert report.approved_count == 2
    assert report.succeeded_count == 1
    assert report.failed_count == 1
    errors = {o.arc_id: o.error for o in report.outcomes if o.error}
    assert "RuntimeError: boom" in errors["a1"]
    assert store.get_node("notebook:nb-ok") is not None


@pytest.mark.asyncio
async def test_executor_returns_none(store, gate):
    executor = _make_executor({"a1": None})
    runner = AutoCurateRunner(gate, store, executor)
    report = await runner.run_cycle([_arc("a1")])

    outcome = report.outcomes[0]
    assert not outcome.succeeded
    assert outcome.error == "executor_returned_none"
    assert outcome.notebook_id is None


@pytest.mark.asyncio
async def test_executor_returns_dict_without_notebook_id(store, gate):
    executor = _make_executor({"a1": {"title": "orphan"}})
    runner = AutoCurateRunner(gate, store, executor)
    report = await runner.run_cycle([_arc("a1")])

    outcome = report.outcomes[0]
    assert not outcome.succeeded
    assert outcome.error == "missing_notebook_id"
    assert outcome.executor_result == {"title": "orphan"}


@pytest.mark.asyncio
async def test_idempotent_replay_does_not_duplicate_notebook_node(store, gate):
    executor = _make_executor({"a1": {"notebook_id": "nb-9"}})
    runner = AutoCurateRunner(gate, store, executor)
    await runner.run_cycle([_arc("a1")])

    # Second cycle: gate dedups arc so executor is never called again.
    report2 = await runner.run_cycle([_arc("a1")])
    assert report2.approved_count == 0
    assert executor.calls == ["a1"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_rejected_arcs_never_invoke_executor(store):
    gate = CurateGate(store, CurateGatePolicy(min_significance=0.8))
    executor = _make_executor({})
    runner = AutoCurateRunner(gate, store, executor)
    report = await runner.run_cycle(
        [_arc("low", significance=0.3), _arc("high", significance=0.95)]
    )

    assert executor.calls == ["high"]  # type: ignore[attr-defined]
    assert report.approved_count == 1
    assert len(report.gate_result.rejected) == 1


@pytest.mark.asyncio
async def test_convenience_wrapper_matches_runner(store, gate):
    executor = _make_executor({"a1": {"notebook_id": "nb-w"}})
    report = await run_auto_curate_cycle(gate, store, executor, [_arc("a1")])
    assert isinstance(report, RunReport)
    assert report.succeeded_count == 1


@pytest.mark.asyncio
async def test_derives_from_edge_wired(store, gate):
    executor = _make_executor({"a1": {"notebook_id": "nb-edge"}})
    runner = AutoCurateRunner(gate, store, executor)
    await runner.run_cycle([_arc("a1")])

    edges = store.get_neighbors("notebook:nb-edge")
    assert any(
        e.source_id == "notebook:nb-edge"
        and e.target_id == "arc:a1"
        and e.edge_type == "derives_from"
        for e in edges
    )


@pytest.mark.asyncio
async def test_empty_candidates_yields_empty_report(store, gate):
    executor = _make_executor({})
    runner = AutoCurateRunner(gate, store, executor)
    report = await runner.run_cycle([])
    assert report.approved_count == 0
    assert report.outcomes == []


@pytest.mark.asyncio
async def test_run_outcome_succeeded_requires_notebook_and_no_error():
    o = RunOutcome(arc_id="x")
    assert not o.succeeded
    o.notebook_id = "nb"
    assert o.succeeded
    o.error = "late failure"
    assert not o.succeeded
