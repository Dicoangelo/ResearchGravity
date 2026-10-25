"""Constraint, provenance and tamper checks for the offline BASIX planner."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math

import pytest

from basix.planner import canonical_hash, evaluate_workload, validate_receipt

NOW = "2026-10-25T12:00:00Z"


def sample():
    scope = {"width": 64, "height": 64, "count": 1, "scale": 4}
    workload = {
        "id": "image-1", "task": "image_upscale", "name": "Local image",
        "max_cost_usd": 1.0, "deadline_seconds": 20,
        "cloud_upload_allowed": False, "allowed_regions": [], "input": scope,
    }
    content = "Measured one 64x64 image at 4x; time 2s; marginal provider charge $0."
    evidence = [{
        "id": "local-run", "title": "Local measurement", "uri": "urn:basix:local-run",
        "captured_at": "2026-10-25T11:00:00Z", "valid_until": "2026-10-26T11:00:00Z",
        "kind": "local_measurement", "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }]
    offers = [{
        "id": "mac", "name": "Local", "provider": "local", "task": "image_upscale",
        "region": None, "data_leaves_device": False, "availability": "available",
        "estimated_seconds": 2, "estimated_cost_usd": 0,
        "measurement_scope": deepcopy(scope), "evidence_ids": ["local-run"], "notes": "",
    }]
    return workload, offers, evidence


def run(data):
    return evaluate_workload(*data, now=NOW)


def status(data):
    return run(data)["evaluations"][0]["status"]


def test_known_local_constraints_and_hash_are_deterministic():
    data = sample()
    receipt = run(data)
    assert receipt == run(data)
    assert receipt == validate_receipt(receipt)
    assert receipt["recommended_offer_id"] == "mac"
    assert receipt["execution_authorized"] is False
    assert receipt["evaluations"][0]["status"] == "feasible"
    assert "not independent truth" in " ".join(receipt["evaluations"][0]["reasons"])
    assert "approval" in receipt["next_action"]
    assert receipt["decision_id"] == canonical_hash({k: v for k, v in receipt.items() if k != "decision_id"})


def test_receipt_is_a_deep_snapshot():
    data = sample()
    receipt = run(data)
    data[0]["input"]["width"] = 256
    data[1][0]["notes"] = "changed"
    data[2][0]["title"] = "changed"
    assert receipt["workload"]["input"]["width"] == 64
    assert receipt["offers"][0]["notes"] == ""
    assert receipt["evidence"][0]["title"] == "Local measurement"


@pytest.mark.parametrize("change", [
    lambda w, o, e: o[0].update(availability="unknown"),
    lambda w, o, e: o[0].update(estimated_seconds=None),
    lambda w, o, e: o[0].update(estimated_cost_usd=None),
    lambda w, o, e: o[0].update(measurement_scope=None),
    lambda w, o, e: o[0]["measurement_scope"].update(count=2),
    lambda w, o, e: o[0].update(evidence_ids=[]),
    lambda w, o, e: e[0].update(kind="synthetic_fixture"),
    lambda w, o, e: e[0].update(kind="historical_record"),
    lambda w, o, e: e[0].update(kind="provider_documentation"),
    lambda w, o, e: e[0].update(valid_until=NOW),
    lambda w, o, e: w.update(allowed_regions=["us-east1"]),
])
def test_unknown_evidence_or_constraints_never_recommended(change):
    data = sample()
    change(*data)
    assert status(data) == "unknown"
    assert run(data)["recommended_offer_id"] is None


@pytest.mark.parametrize("change", [
    lambda w, o, e: o[0].update(data_leaves_device=True),
    lambda w, o, e: o[0].update(availability="unavailable"),
    lambda w, o, e: o[0].update(task="image_generate"),
    lambda w, o, e: o[0].update(estimated_seconds=21),
    lambda w, o, e: o[0].update(estimated_cost_usd=1.01),
])
def test_hard_constraint_failure_blocks(change):
    data = sample()
    change(*data)
    assert status(data) == "blocked"
    assert run(data)["recommended_offer_id"] is None


def test_region_mismatch_blocks_and_missing_location_is_unknown():
    data = sample()
    data[0]["allowed_regions"] = ["ca-central"]
    assert status(data) == "unknown"
    data[1][0]["region"] = "us-east"
    assert status(data) == "blocked"
    data[1][0]["region"] = "ca-central"
    assert status(data) == "feasible"


def test_cloud_documented_estimates_are_eligible_but_never_approved():
    w, offers, evidence = sample()
    w["cloud_upload_allowed"] = True
    offers[0].update(data_leaves_device=True, measurement_scope=None)
    evidence[0]["kind"] = "provider_documentation"
    receipt = run((w, offers, evidence))
    assert receipt["evaluations"][0]["status"] == "feasible"
    assert receipt["execution_authorized"] is False
    assert "not measurements" in " ".join(receipt["evaluations"][0]["reasons"])


def test_cloud_measurements_for_other_input_cannot_be_extrapolated():
    data = sample()
    data[0]["cloud_upload_allowed"] = True
    data[1][0]["data_leaves_device"] = True
    data[1][0]["measurement_scope"]["width"] = 128
    assert status(data) == "unknown"


def test_boundary_budget_and_deadline_are_accepted():
    data = sample()
    data[1][0].update(estimated_seconds=20, estimated_cost_usd=1.0)
    assert status(data) == "feasible"


def test_hard_failure_retains_unknown_evidence_reasons():
    data = sample()
    data[1][0].update(estimated_cost_usd=100)
    data[2][0]["kind"] = "synthetic_fixture"
    evaluation = run(data)["evaluations"][0]
    assert evaluation["status"] == "blocked"
    assert any("synthetic" in reason for reason in evaluation["reasons"])


@pytest.mark.parametrize("value", [True, False, -1, float("nan"), float("inf"), "0"])
@pytest.mark.parametrize("field", ["estimated_cost_usd", "estimated_seconds"])
def test_invalid_offer_numbers_are_rejected(field, value):
    data = sample()
    data[1][0][field] = value
    with pytest.raises(ValueError):
        run(data)


@pytest.mark.parametrize("field", ["max_cost_usd", "deadline_seconds"])
@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "10"])
def test_invalid_workload_numbers_are_rejected(field, value):
    data = sample()
    data[0][field] = value
    with pytest.raises(ValueError):
        run(data)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_privacy_fields_require_actual_booleans(value):
    for target, field in [(0, "cloud_upload_allowed"), (1, "data_leaves_device")]:
        data = sample()
        obj = data[target] if target == 0 else data[target][0]
        obj[field] = value
        with pytest.raises(ValueError):
            run(data)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "64"])
def test_dimensions_are_positive_integers(value):
    data = sample()
    data[0]["input"]["width"] = value
    with pytest.raises(ValueError):
        run(data)


@pytest.mark.parametrize("change", [
    lambda w, o, e: e[0].update(content="tampered"),
    lambda w, o, e: e[0].update(sha256="0" * 64),
    lambda w, o, e: e[0].update(captured_at="2026-10-25T12:00:01Z"),
    lambda w, o, e: e[0].update(captured_at="2026-10-25T11:00:00"),
    lambda w, o, e: e[0].update(valid_until="2026-10-24T12:00:00Z"),
    lambda w, o, e: e[0].update(kind="invention"),
    lambda w, o, e: e[0].update(kind=[]),
    lambda w, o, e: o[0].update(availability=[]),
    lambda w, o, e: e[0].update(uri="javascript:alert(1)"),
    lambda w, o, e: e[0].update(uri="https://user:secret@example.com/"),
    lambda w, o, e: o[0].update(evidence_ids=["missing"]),
    lambda w, o, e: o[0].update(evidence_ids=["local-run", "local-run"]),
    lambda w, o, e: o.append(deepcopy(o[0])),
    lambda w, o, e: e.append(deepcopy(e[0])),
    lambda w, o, e: w.update(extra="ambiguous"),
    lambda w, o, e: o[0].pop("availability"),
])
def test_invalid_provenance_and_ambiguous_inputs_are_rejected(change):
    data = sample()
    change(*data)
    with pytest.raises(ValueError):
        run(data)


def test_a_synthetic_source_cannot_be_laundered_by_a_real_source():
    data = sample()
    fake = deepcopy(data[2][0])
    fake.update(id="fixture", kind="synthetic_fixture")
    data[2].append(fake)
    data[1][0]["evidence_ids"].append("fixture")
    assert status(data) == "unknown"


def test_stale_cloud_source_cannot_be_laundered_by_a_current_source():
    data = sample()
    old = deepcopy(data[2][0])
    old.update(id="old", valid_until=NOW)
    data[2].append(old)
    data[1][0]["evidence_ids"].append("old")
    assert status(data) == "unknown"


def test_recommendation_has_stable_cost_time_id_tie_break():
    data = sample()
    a = deepcopy(data[1][0])
    a["id"] = "a"
    data[1].append(a)
    assert run(data)["recommended_offer_id"] == "a"
    a["estimated_seconds"] = 3
    assert run(data)["recommended_offer_id"] == "mac"
    data[1][0]["estimated_cost_usd"] = 0.1
    assert run(data)["recommended_offer_id"] == "a"


def test_empty_catalog_abstains():
    data = sample()
    receipt = evaluate_workload(data[0], [], [], now=NOW)
    assert receipt["recommended_offer_id"] is None
    assert receipt["evaluations"] == []


def test_timezones_compare_instants_and_receipt_is_utc():
    data = sample()
    data[2][0]["captured_at"] = "2026-10-25T07:00:00-04:00"
    first = run(data)
    second = evaluate_workload(*data, now=datetime(2026, 10, 25, 12, tzinfo=timezone.utc))
    assert first == second
    assert first["created_at"] == NOW


def test_rehashed_forged_decision_still_fails_semantic_validation():
    receipt = run(sample())
    receipt["recommended_offer_id"] = "some-other-offer"
    receipt["decision_id"] = canonical_hash({k: v for k, v in receipt.items() if k != "decision_id"})
    with pytest.raises(ValueError, match="deterministic"):
        validate_receipt(receipt)


def test_receipt_hash_detects_edit_without_rehash():
    receipt = run(sample())
    receipt["next_action"] = "Run now"
    with pytest.raises(ValueError, match="hash"):
        validate_receipt(receipt)


def test_execution_approval_cannot_be_smuggled_into_a_receipt():
    receipt = run(sample())
    receipt["execution_authorized"] = True
    receipt["decision_id"] = canonical_hash({k: v for k, v in receipt.items() if k != "decision_id"})
    with pytest.raises(ValueError, match="never authorize"):
        validate_receipt(receipt)


def test_nonfinite_values_cannot_be_hashed():
    with pytest.raises(ValueError):
        canonical_hash({"cost": math.nan})
