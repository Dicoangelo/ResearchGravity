"""E7 tests: coherence-gated auto-curate policy + lineage wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from delegation.auto_curate_gate import (
    AUTO_CURATE_ROOT_ID,
    CurateGate,
    CurateGatePolicy,
)
from delegation.lineage_store import LineageStore


@pytest.fixture
def store(tmp_path: Path) -> LineageStore:
    return LineageStore(db_path=tmp_path / "lineage.db")


def _arc(
    arc_id: str = "a1",
    topic: str = "convergence",
    significance: float = 0.9,
    platforms: int = 3,
    moments: int = 5,
    last_curated_at=None,
) -> dict:
    return {
        "arc_id": arc_id,
        "topic": topic,
        "significance": significance,
        "platforms": platforms,
        "moments": moments,
        "last_curated_at": last_curated_at,
    }


def test_approves_high_significance_arc_and_records_lineage(store):
    gate = CurateGate(store)
    result = gate.evaluate([_arc()])

    assert len(result.approved) == 1
    assert result.approved[0].arc_id == "a1"
    node = store.get_node("arc:a1")
    assert node is not None
    assert node.parent_id == AUTO_CURATE_ROOT_ID
    assert node.node_type == "concept"
    assert node.metadata["source"] == "auto_curate_gate"
    assert node.metadata["topic"] == "convergence"


def test_rejects_below_significance_floor(store):
    gate = CurateGate(store, CurateGatePolicy(min_significance=0.8))
    result = gate.evaluate([_arc(significance=0.5)])

    assert result.approved == []
    assert len(result.rejected) == 1
    assert "below_significance_floor" in result.rejected[0].reason


def test_rejects_below_platform_floor(store):
    gate = CurateGate(store, CurateGatePolicy(min_platforms=3))
    result = gate.evaluate([_arc(platforms=2)])

    assert result.approved == []
    assert "below_platform_floor" in result.rejected[0].reason


def test_rejects_below_moment_floor(store):
    gate = CurateGate(store, CurateGatePolicy(min_moments=10))
    result = gate.evaluate([_arc(moments=5)])

    assert result.approved == []
    assert "below_moment_floor" in result.rejected[0].reason


def test_cooldown_skips_recently_curated(store):
    gate = CurateGate(store, CurateGatePolicy(cooldown_hours=24))
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    result = gate.evaluate([_arc(last_curated_at=recent)])

    assert result.approved == []
    assert "in_cooldown" in result.rejected[0].reason


def test_cooldown_expired_allows_curation(store):
    gate = CurateGate(store, CurateGatePolicy(cooldown_hours=24))
    stale = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    result = gate.evaluate([_arc(last_curated_at=stale)])

    assert len(result.approved) == 1


def test_dedup_via_lineage_store(store):
    gate = CurateGate(store)
    gate.evaluate([_arc()])
    result = gate.evaluate([_arc()])

    assert result.approved == []
    assert result.rejected[0].reason == "already_curated"


def test_budget_caps_approvals(store):
    gate = CurateGate(store, CurateGatePolicy(budget_per_cycle=2))
    arcs = [
        _arc(arc_id=f"a{i}", significance=0.7 + i * 0.05, moments=5)
        for i in range(5)
    ]
    result = gate.evaluate(arcs)

    assert len(result.approved) == 2
    assert {d.reason for d in result.rejected} == {"over_budget"}
    # highest-significance arcs win (a4, a3)
    assert {d.arc_id for d in result.approved} == {"a4", "a3"}


def test_ranking_prefers_high_significance_then_moments(store):
    gate = CurateGate(store, CurateGatePolicy(budget_per_cycle=1))
    arcs = [
        _arc(arc_id="tie1", significance=0.9, moments=4),
        _arc(arc_id="tie2", significance=0.9, moments=10),
    ]
    result = gate.evaluate(arcs)

    assert [d.arc_id for d in result.approved] == ["tie2"]


def test_auto_curate_root_is_created_once(store):
    CurateGate(store)
    CurateGate(store)  # second instantiation must not raise
    root = store.get_node(AUTO_CURATE_ROOT_ID)
    assert root is not None
    assert root.node_type == "session"


def test_empty_candidate_list_yields_empty_result(store):
    gate = CurateGate(store)
    result = gate.evaluate([])
    assert result.approved == []
    assert result.rejected == []


def test_mixed_batch_splits_correctly(store):
    gate = CurateGate(store, CurateGatePolicy(budget_per_cycle=10))
    arcs = [
        _arc(arc_id="pass", significance=0.9),
        _arc(arc_id="lowsig", significance=0.3),
        _arc(arc_id="fewmoments", moments=1),
        _arc(arc_id="singleplat", platforms=1),
    ]
    result = gate.evaluate(arcs)

    assert [d.arc_id for d in result.approved] == ["pass"]
    reasons = {d.arc_id: d.reason for d in result.rejected}
    assert "below_significance_floor" in reasons["lowsig"]
    assert "below_moment_floor" in reasons["fewmoments"]
    assert "below_platform_floor" in reasons["singleplat"]


def test_datetime_object_for_last_curated_works(store):
    gate = CurateGate(store, CurateGatePolicy(cooldown_hours=24))
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    result = gate.evaluate([_arc(last_curated_at=recent)])
    assert result.approved == []


def test_malformed_last_curated_is_ignored(store):
    gate = CurateGate(store)
    result = gate.evaluate([_arc(last_curated_at="not-a-date")])
    assert len(result.approved) == 1


def test_approved_arc_ids_helper(store):
    gate = CurateGate(store, CurateGatePolicy(budget_per_cycle=2))
    arcs = [_arc(arc_id="x"), _arc(arc_id="y"), _arc(arc_id="z")]
    result = gate.evaluate(arcs)
    assert set(result.approved_arc_ids) == {d.arc_id for d in result.approved}
    assert len(result.approved_arc_ids) == 2
