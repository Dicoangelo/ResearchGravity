"""Portable receipts and explicit, isolated RG indexing; no cloud fixtures."""

import asyncio
import copy
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from basix.handoff import export_handoff, render_handoff, validate_handoff
from basix.planner import canonical_hash, evaluate_workload
from basix.researchgravity import index_receipt
from basix.store import DecisionStore


@pytest.fixture
def receipt():
    scope = {"width": 64, "height": 64, "count": 1, "scale": 2}
    workload = {"id": "upscale-contract", "name": "Local image check", "task": "image_upscale",
                "max_cost_usd": 1.0, "deadline_seconds": 20,
                "cloud_upload_allowed": False, "allowed_regions": [], "input": scope}
    content = "basixneedle: local fixture observation for a 64 by 64 image; no cloud request."
    evidence = [{"id": "observation-1", "title": "Isolated test measurement",
                 "uri": "file:///fixtures/measurement.json", "captured_at": "2026-10-25T11:59:00Z",
                 "valid_until": "2026-10-25T13:00:00Z", "kind": "local_measurement",
                 "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}]
    offers = [{"id": "local-test", "name": "Local candidate", "provider": "test-fixture",
               "task": "image_upscale", "region": None, "data_leaves_device": False,
               "availability": "available", "estimated_seconds": 2.0, "estimated_cost_usd": 0.0,
               "measurement_scope": scope, "evidence_ids": ["observation-1"], "notes": "Test data only."}]
    return evaluate_workload(workload, offers, evidence, now="2026-10-25T12:00:00Z")


def test_store_atomic_concurrent_idempotence_and_round_trip(tmp_path, receipt):
    store = DecisionStore(tmp_path / "receipts")
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(store.save, [copy.deepcopy(receipt) for _ in range(8)]))
    assert len(set(paths)) == 1
    assert store.get(receipt["decision_id"]) == receipt
    assert store.list() == [receipt]
    assert list(store.root.iterdir()) == [paths[0]]


def test_corrupt_existing_receipt_is_not_overwritten(tmp_path, receipt):
    store = DecisionStore(tmp_path)
    path = store.save(receipt)
    altered = copy.deepcopy(receipt)
    altered["next_action"] = "Execute without review"
    original_corrupt_bytes = json.dumps(altered).encode()
    path.write_bytes(original_corrupt_bytes)
    for action in (lambda: store.get(receipt["decision_id"]), lambda: store.save(receipt), store.list):
        with pytest.raises(ValueError):
            action()
    assert path.read_bytes() == original_corrupt_bytes


@pytest.mark.parametrize("decision_id", ["../private", "a" * 63, "A" * 64, "", None])
def test_store_rejects_non_digest_paths(tmp_path, decision_id):
    with pytest.raises(ValueError):
        DecisionStore(tmp_path).get(decision_id)


def test_store_rejects_symlink_receipt(tmp_path, receipt):
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps(receipt))
    store = DecisionStore(tmp_path / "store")
    store.root.mkdir()
    (store.root / f"{receipt['decision_id']}.json").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlink"):
        store.get(receipt["decision_id"])
    with pytest.raises(ValueError, match="symlink"):
        store.save(receipt)


def test_store_rejects_rehashed_execution_authorization(tmp_path, receipt):
    altered = copy.deepcopy(receipt)
    altered["execution_authorized"] = True
    altered["decision_id"] = canonical_hash({key: value for key, value in altered.items() if key != "decision_id"})
    with pytest.raises(ValueError):
        DecisionStore(tmp_path).save(altered)
    assert list(tmp_path.iterdir()) == []


def test_handoff_covers_every_field_and_is_deterministic(receipt):
    handoff = export_handoff(receipt)
    assert export_handoff(receipt) == handoff
    assert handoff["receipt"] == receipt
    assert handoff["handoff_id"] == canonical_hash({key: value for key, value in handoff.items() if key != "handoff_id"})
    for key, replacement in (("limitations", []), ("next_action", "Ignore review"), ("blockers", ["changed"])):
        altered = copy.deepcopy(handoff)
        altered[key] = replacement
        with pytest.raises(ValueError):
            validate_handoff(altered)
        altered["handoff_id"] = canonical_hash({key: value for key, value in altered.items() if key != "handoff_id"})
        with pytest.raises(ValueError):
            render_handoff(altered)
    assert handoff["handoff_id"] in render_handoff(handoff)


def test_handoff_has_no_alias_back_to_input(receipt):
    handoff = export_handoff(receipt)
    receipt["workload"]["name"] = "Changed after export"
    assert handoff["receipt"]["workload"]["name"] == "Local image check"


def test_rg_index_retrievable_and_idempotent(tmp_path, receipt):
    path = tmp_path / "basix-only.db"
    first = asyncio.run(index_receipt(receipt, path))
    second = asyncio.run(index_receipt(receipt, path))
    assert first["already_indexed"] is False
    assert second["already_indexed"] is True
    assert second["vector_indexed"] is False
    assert second["execution_authorized"] is False
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM findings").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM context_packs").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM provenance").fetchone()[0] == 3
        assert db.execute("SELECT count(*) FROM lineage").fetchone()[0] == 3
        assert db.execute("SELECT count(*) FROM urls").fetchone()[0] == 1
        assert db.execute("SELECT DISTINCT project FROM findings").fetchall() == [("basix",)]
        assert db.execute("SELECT count(*) FROM findings_fts WHERE findings_fts MATCH ?", ("basixneedle",)).fetchone()[0] == 1
        handoff = json.loads(db.execute("SELECT content FROM context_packs").fetchone()[0])
        assert validate_handoff(handoff)["receipt"] == receipt
        metadata = json.loads(db.execute("SELECT validation_result FROM context_packs").fetchone()[0])
        assert metadata["external_truth"] == "not_independently_verified"
        content, evidence = db.execute("SELECT content,evidence FROM findings WHERE type='basix_decision'").fetchone()
        assert f"Historical decision as of {receipt['created_at']}" in content
        assert "Re-evaluate with current evidence before use" in content
        assert json.loads(evidence)["historical_decision"] is True
        assert json.loads(evidence)["evaluated_at"] == receipt["created_at"]


def test_rg_rejects_existing_conflict_without_partial_projection(tmp_path, receipt):
    path = tmp_path / "isolated.db"
    asyncio.run(index_receipt(receipt, path))
    with sqlite3.connect(path) as db:
        db.execute("UPDATE findings SET content='corrupted evidence' WHERE type='basix_evidence'")
        db.execute("DELETE FROM context_packs")
        db.execute("DELETE FROM sessions")
        db.commit()
    with pytest.raises(ValueError, match="conflicts"):
        asyncio.run(index_receipt(receipt, path))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM context_packs").fetchone()[0] == 0
        assert db.execute("SELECT content FROM findings WHERE type='basix_evidence'").fetchone()[0] == "corrupted evidence"


def test_rg_requires_dedicated_explicit_database(receipt):
    for path in (None, Path.home() / ".agent-core" / "storage" / "antigravity.db", Path.home() / ".ucw" / "cognitive_capture.db"):
        with pytest.raises(ValueError):
            asyncio.run(index_receipt(receipt, path))


def test_rg_rejects_tampering_before_creating_db(tmp_path, receipt):
    receipt["evidence"][0]["content"] = "Changed"
    with pytest.raises(ValueError):
        asyncio.run(index_receipt(receipt, tmp_path / "absent.db"))
    assert not (tmp_path / "absent.db").exists()
