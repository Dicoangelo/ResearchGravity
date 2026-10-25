"""Deterministic, local-only BASIX feasibility decisions.

Provenance validation establishes byte integrity and freshness, not the truth of
provider claims. This module never calls providers, runs jobs, or authorizes
execution. It deliberately uses hard constraints rather than a synthetic score.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = "basix.decision.v1"
PROVENANCE_CAVEAT = (
    "Source links and hashes establish provenance, not independent truth or guaranteed capacity."
)
WORKLOAD_FIELDS = {
    "id", "task", "name", "max_cost_usd", "deadline_seconds",
    "cloud_upload_allowed", "allowed_regions", "input",
}
OFFER_FIELDS = {
    "id", "name", "provider", "task", "region", "data_leaves_device",
    "availability", "estimated_seconds", "estimated_cost_usd",
    "measurement_scope", "evidence_ids", "notes",
}
EVIDENCE_FIELDS = {
    "id", "title", "uri", "captured_at", "valid_until", "kind", "content", "sha256",
}
INPUT_FIELDS = {"width", "height", "count", "scale"}
RECEIPT_FIELDS = {
    "schema_version", "decision_id", "created_at", "workload", "offers", "evidence",
    "evaluations", "recommended_offer_id", "next_action", "execution_authorized",
}


def canonical_hash(payload: Any) -> str:
    """Hash canonical JSON. NaN/Infinity and non-JSON values are rejected."""
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Payload must be finite UTF-8 JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _copy_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Inputs must contain only finite JSON values") from exc


def _object(value: Any, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _text(value: Any, label: str, *, limit: int = 512, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f"{label} must be {'possibly empty' if empty else 'nonempty'} text <= {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8") from exc
    return value


def _number(value: Any, label: str, *, positive: bool = False, nullable: bool = False) -> Any:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (value <= 0 if positive else value < 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label} must be an aware ISO timestamp") from exc
    else:
        raise ValueError(f"{label} must be an aware ISO timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _input(value: Any, label: str) -> dict:
    value = _object(value, INPUT_FIELDS, label)
    for key in INPUT_FIELDS:
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")
    return value


def _unique_strings(value: Any, label: str, *, maximum: int = 1000) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{label} must be a list of at most {maximum} strings")
    for item in value:
        _text(item, label)
    if len(set(value)) != len(value):
        raise ValueError(f"{label} cannot contain duplicates")
    return value


def _validated_inputs(workload: Any, offers: Any, evidence: Any, now: datetime):
    w = _object(_copy_json(workload), WORKLOAD_FIELDS, "workload")
    for key in ("id", "name"):
        _text(w[key], f"workload.{key}")
    if w["task"] != "image_upscale":
        raise ValueError("workload.task must be image_upscale")
    _number(w["max_cost_usd"], "workload.max_cost_usd")
    _number(w["deadline_seconds"], "workload.deadline_seconds", positive=True)
    _boolean(w["cloud_upload_allowed"], "workload.cloud_upload_allowed")
    _unique_strings(w["allowed_regions"], "workload.allowed_regions")
    _input(w["input"], "workload.input")
    if not isinstance(offers, list) or len(offers) > 1000:
        raise ValueError("offers must be a list of at most 1000 offers")
    if not isinstance(evidence, list) or len(evidence) > 1000:
        raise ValueError("evidence must be a list of at most 1000 records")
    records, evidence_map = [], {}
    for raw in evidence:
        e = _object(_copy_json(raw), EVIDENCE_FIELDS, "evidence record")
        for key in ("id", "title", "uri"):
            _text(e[key], f"evidence.{key}", limit=4096 if key == "uri" else 512)
        if e["id"] in evidence_map:
            raise ValueError("Duplicate evidence ID")
        try:
            uri = urlsplit(e["uri"])
        except ValueError as exc:
            raise ValueError("Invalid evidence URI") from exc
        if uri.scheme not in {"http", "https", "file", "urn"}:
            raise ValueError("Evidence URI must use http, https, file, or urn")
        if uri.scheme in {"http", "https"} and (not uri.netloc or uri.username or uri.password):
            raise ValueError("Evidence HTTP URI must have a host and no embedded credentials")
        if not isinstance(e["captured_at"], str) or not isinstance(e["valid_until"], str):
            raise ValueError("Evidence dates must be ISO strings")
        captured = _timestamp(e["captured_at"], "evidence.captured_at")
        until = _timestamp(e["valid_until"], "evidence.valid_until")
        if captured > now:
            raise ValueError("Evidence capture time cannot be in the future")
        if until < captured:
            raise ValueError("Evidence expiry cannot precede capture")
        if not isinstance(e["kind"], str) or e["kind"] not in {"local_measurement", "provider_documentation", "historical_record", "synthetic_fixture"}:
            raise ValueError("Unsupported evidence kind")
        _text(e["content"], "evidence.content", limit=200000)
        digest = hashlib.sha256(e["content"].encode("utf-8")).hexdigest()
        if not isinstance(e["sha256"], str) or e["sha256"] != digest:
            raise ValueError("Evidence content hash mismatch")
        evidence_map[e["id"]] = e
        records.append(e)
    normalized_offers, seen_offers = [], set()
    for raw in offers:
        o = _object(_copy_json(raw), OFFER_FIELDS, "offer")
        for key in ("id", "name", "provider", "task"):
            _text(o[key], f"offer.{key}")
        if o["id"] in seen_offers:
            raise ValueError("Duplicate offer ID")
        seen_offers.add(o["id"])
        if o["region"] is not None:
            _text(o["region"], "offer.region")
        _boolean(o["data_leaves_device"], "offer.data_leaves_device")
        if not isinstance(o["availability"], str) or o["availability"] not in {"available", "unavailable", "unknown"}:
            raise ValueError("Unsupported availability")
        _number(o["estimated_seconds"], "offer.estimated_seconds", nullable=True)
        _number(o["estimated_cost_usd"], "offer.estimated_cost_usd", nullable=True)
        if o["measurement_scope"] is not None:
            _input(o["measurement_scope"], "offer.measurement_scope")
        _unique_strings(o["evidence_ids"], "offer.evidence_ids")
        for eid in o["evidence_ids"]:
            if eid not in evidence_map:
                raise ValueError(f"Offer references missing evidence: {eid}")
        _text(o["notes"], "offer.notes", limit=10000, empty=True)
        normalized_offers.append(o)
    return w, normalized_offers, records, evidence_map


def evaluate_workload(workload: Any, offers: Any, evidence: Any, *, now: Any = None) -> dict:
    """Return a deterministic, content-addressed receipt; never execute anything."""
    instant = datetime.now(timezone.utc) if now is None else _timestamp(now, "now")
    w, offers, evidence, sources = _validated_inputs(workload, offers, evidence, instant)
    evaluations = []
    for offer in offers:
        blocked, unknown = [], []
        if offer["task"] != w["task"]:
            blocked.append("Offer does not support the requested task.")
        if offer["data_leaves_device"] and not w["cloud_upload_allowed"]:
            blocked.append("Input may not leave this device; cloud upload is disallowed.")
        if w["allowed_regions"]:
            if offer["region"] is None:
                unknown.append("Offer location is unknown; cannot verify the region constraint.")
            elif offer["region"] not in w["allowed_regions"]:
                blocked.append("Offer region is outside the allowed regions.")
        if offer["availability"] == "unavailable":
            blocked.append("Offer is unavailable.")
        elif offer["availability"] == "unknown":
            unknown.append("Current offer availability is unknown.")
        seconds, cost = offer["estimated_seconds"], offer["estimated_cost_usd"]
        if seconds is None:
            unknown.append("Completion time is unknown.")
        elif seconds > w["deadline_seconds"]:
            blocked.append("Estimated completion time exceeds the deadline.")
        if cost is None:
            unknown.append("Total estimated cost is unknown.")
        elif cost > w["max_cost_usd"]:
            blocked.append("Estimated cost exceeds the budget.")
        source_records = [sources[eid] for eid in offer["evidence_ids"]]
        if not source_records:
            unknown.append("No evidence supports this offer.")
        for record in source_records:
            if _timestamp(record["valid_until"], "valid_until") <= instant:
                unknown.append(f"Evidence {record['id']} is stale.")
            if record["kind"] == "synthetic_fixture":
                unknown.append(f"Evidence {record['id']} is synthetic and cannot establish feasibility.")
        if source_records and not any(e["kind"] in {"local_measurement", "provider_documentation"} for e in source_records):
            unknown.append("Historical records alone do not establish current feasibility.")
        scope = offer["measurement_scope"]
        if not offer["data_leaves_device"]:
            if not any(e["kind"] == "local_measurement" for e in source_records):
                unknown.append("An on-device offer requires a local measurement.")
            if scope != w["input"]:
                unknown.append("Local measurement scope does not exactly match the input; no extrapolation is permitted.")
        elif scope is not None and scope != w["input"]:
            unknown.append("Measurement scope differs from the input; no extrapolation is permitted.")
        status = "blocked" if blocked else "unknown" if unknown else "feasible"
        reasons = blocked + unknown
        if status == "feasible":
            reasons.append("Supplied evidence and estimates satisfy the declared constraints.")
        if offer["data_leaves_device"] and scope is None:
            reasons.append("Time and cost are supplied estimates, not measurements of this exact input.")
        reasons.extend([PROVENANCE_CAVEAT, "This decision does not authorize execution or spending."])
        evaluations.append({
            "offer_id": offer["id"], "status": status, "reasons": reasons,
            "estimated_seconds": seconds, "estimated_cost_usd": cost,
        })
    feasible = sorted(
        (item for item in evaluations if item["status"] == "feasible"),
        key=lambda item: (item["estimated_cost_usd"], item["estimated_seconds"], item["offer_id"]),
    )
    recommended = feasible[0]["offer_id"] if feasible else None
    next_action = (
        f"Review the evidence and estimates for {recommended}; obtain explicit execution approval before running a job."
        if recommended else
        "Resolve blocked constraints or collect fresh, matching evidence and estimates; no offer is currently feasible."
    )
    receipt = {
        "schema_version": SCHEMA_VERSION, "created_at": _iso(instant),
        "workload": w, "offers": offers, "evidence": evidence,
        "evaluations": evaluations, "recommended_offer_id": recommended,
        "next_action": next_action, "execution_authorized": False,
    }
    receipt["decision_id"] = canonical_hash(receipt)
    return receipt


def validate_receipt(receipt: Any) -> dict:
    """Validate immutable bytes and their deterministic decision semantics."""
    r = _object(_copy_json(receipt), RECEIPT_FIELDS, "receipt")
    if r["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported receipt schema")
    if r["execution_authorized"] is not False:
        raise ValueError("BASIX v1 receipts never authorize execution")
    if not isinstance(r["decision_id"], str) or re.fullmatch(r"[a-f0-9]{64}", r["decision_id"]) is None:
        raise ValueError("Invalid decision ID")
    payload = {key: value for key, value in r.items() if key != "decision_id"}
    if canonical_hash(payload) != r["decision_id"]:
        raise ValueError("Receipt hash mismatch")
    expected = evaluate_workload(r["workload"], r["offers"], r["evidence"], now=r["created_at"])
    if expected != r:
        raise ValueError("Receipt does not match deterministic decision evaluation")
    return r
