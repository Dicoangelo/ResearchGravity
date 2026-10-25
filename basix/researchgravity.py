"""Explicit, local-only projection of BASIX receipts into ResearchGravity."""

import json
from pathlib import Path

from .handoff import export_handoff
from .planner import canonical_hash, validate_receipt


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


async def index_receipt(receipt: dict, db_path: Path) -> dict:
    """Index one validated receipt atomically and idempotently in a chosen DB.

    No global storage engine, model, vector service or active session is opened.
    The existing RG schema/connection pool is reused; one SQLite transaction
    makes the projection complete or absent. JSON receipts remain authoritative.
    """
    checked = validate_receipt(receipt)
    if db_path is None:
        raise ValueError("An explicit project database path is required")
    target = Path(db_path).expanduser().resolve()
    private_roots = (Path.home() / ".agent-core", Path.home() / ".ucw")
    if any(target.is_relative_to(root.resolve()) for root in private_roots):
        raise ValueError("Use a dedicated BASIX database, not the personal corpus")

    # Lazy import keeps storage dependencies out of the standalone planner/CLI.
    from storage.sqlite_db import SQLiteDB

    decision_id = checked["decision_id"]
    session_id = f"basix-session-{decision_id}"
    pack_id = f"basix-decision-{decision_id}"
    source_type = "basix_receipt"
    handoff = export_handoff(checked)
    findings = []
    for evidence in checked["evidence"]:
        findings.append({
            "id": f"basix-evidence-{decision_id}-{canonical_hash(evidence['id'])}",
            "type": "basix_evidence", "content": evidence["content"],
            "evidence": {"decision_id": decision_id, "source": evidence},
        })
    findings.append({
        "id": f"basix-summary-{decision_id}", "type": "basix_decision",
        "content": f"Historical decision as of {checked['created_at']}: {checked['workload']['name']}. "
                   f"Recorded recommendation: {checked['recommended_offer_id'] or 'none established'}. "
                   "Re-evaluate with current evidence before use. "
                   f"Recorded next action: {checked['next_action']}",
        "evidence": {"decision_id": decision_id, "evaluations": checked["evaluations"],
                     "source_ids": [item["id"] for item in checked["evidence"]],
                     "evaluated_at": checked["created_at"], "historical_decision": True,
                     "execution_authorized": False},
    })
    database = SQLiteDB(db_path=target, pool_size=1)
    await database.initialize()
    try:
        async with database.connection() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute("SELECT content FROM context_packs WHERE id = ?", (pack_id,))
                existing = await cursor.fetchone()
                if existing is not None and json.loads(existing["content"]) != handoff:
                    raise ValueError("Existing RG projection conflicts with this immutable receipt")
                await connection.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(id,topic,status,project,started_at,finding_count,url_count,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (session_id, checked["workload"]["name"], "archived", "basix",
                     checked["created_at"], len(findings), len({item['uri'] for item in checked['evidence']}),
                     _json({"decision_id": decision_id, "schema_version": checked["schema_version"]})),
                )
                for finding in findings:
                    cursor = await connection.execute(
                        "SELECT content,evidence,type,session_id,project FROM findings WHERE id = ?", (finding["id"],))
                    previous = await cursor.fetchone()
                    expected = (finding["content"], _json(finding["evidence"]), finding["type"], session_id, "basix")
                    if previous is not None and tuple(previous) != expected:
                        raise ValueError("Existing RG finding conflicts with this immutable receipt")
                    await connection.execute(
                        "INSERT OR IGNORE INTO findings (id,session_id,content,type,evidence,project) VALUES (?,?,?,?,?,?)",
                        (finding["id"], session_id, finding["content"], finding["type"], _json(finding["evidence"]), "basix"),
                    )
                    await connection.execute(
                        "INSERT OR IGNORE INTO provenance (entity_type,entity_id,source_type,source_id,source_metadata) VALUES (?,?,?,?,?)",
                        ("finding", finding["id"], source_type, decision_id, _json({"receipt_hash": decision_id})),
                    )
                    await connection.execute(
                        "INSERT OR IGNORE INTO lineage (source_type,source_id,target_type,target_id,relation,metadata,valid_at) VALUES (?,?,?,?,?,?,?)",
                        ("pack", pack_id, "finding", finding["id"], "contains",
                         _json({"decision_id": decision_id}), checked["created_at"]),
                    )
                for evidence in checked["evidence"]:
                    await connection.execute(
                        "INSERT OR IGNORE INTO urls (session_id,url,category,source,context,captured_at) VALUES (?,?,?,?,?,?)",
                        (session_id, evidence["uri"], evidence["kind"], source_type, evidence["title"], evidence["captured_at"]),
                    )
                await connection.execute(
                    "INSERT OR IGNORE INTO context_packs (id,name,type,content,source,source_id,validated,validation_result) VALUES (?,?,?,?,?,?,?,?)",
                    (pack_id, checked["workload"]["name"], "project", _json(handoff), source_type, decision_id, 1,
                     _json({"receipt_integrity": "valid", "external_truth": "not_independently_verified"})),
                )
                await connection.execute(
                    "INSERT OR IGNORE INTO provenance (entity_type,entity_id,source_type,source_id,source_metadata) VALUES (?,?,?,?,?)",
                    ("pack", pack_id, source_type, decision_id, _json({"receipt_hash": decision_id, "handoff_hash": handoff["handoff_id"]})),
                )
                await connection.execute(
                    "INSERT OR IGNORE INTO lineage (source_type,source_id,target_type,target_id,relation,metadata,valid_at) VALUES (?,?,?,?,?,?,?)",
                    ("session", session_id, "pack", pack_id, "contains", _json({"decision_id": decision_id}), checked["created_at"]),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
    finally:
        await database.close()
    return {"decision_id": decision_id, "project": "basix", "session_id": session_id,
            "pack_id": pack_id, "finding_count": len(findings), "db_path": str(target),
            "already_indexed": existing is not None, "vector_indexed": False,
            "execution_authorized": False}
