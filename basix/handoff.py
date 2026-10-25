"""Portable selected-decision handoff, independent of personal conversation logs."""

import html
import json
import re

from .planner import canonical_hash, validate_receipt


def export_handoff(receipt: dict) -> dict:
    checked = validate_receipt(receipt)
    handoff = {
        "schema_version": "basix.handoff.v1",
        "created_at": checked["created_at"],
        "decision_id": checked["decision_id"],
        "receipt": checked,
        "limitations": [
            "Planning evidence only; execution is not authorized.",
            "Capacity, prices and timings may change after the recorded observation.",
            "Provider links establish provenance, not independent verification.",
            "Measurements apply only to the exact recorded workload scope.",
            "This handoff contains only the selected decision, not personal conversation history.",
            "Hashes detect changed content; they are not signatures or proof of source truth.",
        ],
        "blockers": [
            {"offer_id": result["offer_id"], "status": result["status"],
             "reasons": result["reasons"]}
            for result in checked["evaluations"] if result["status"] != "feasible"
        ],
        "next_action": checked["next_action"],
    }
    handoff["handoff_id"] = canonical_hash(handoff)
    return handoff


def validate_handoff(handoff: dict) -> dict:
    if not isinstance(handoff, dict) or "receipt" not in handoff:
        raise ValueError("A complete BASIX handoff is required")
    # Rebuilding checks all fields, including limitations and next action, even
    # if someone recalculated the outer hash after modifying the handoff.
    expected = export_handoff(handoff["receipt"])
    if handoff != expected:
        raise ValueError("Handoff content or covering hash is invalid")
    return expected


def render_handoff(handoff: dict) -> str:
    checked = validate_handoff(handoff)
    receipt = checked["receipt"]
    escape = lambda value: html.escape(str(value), quote=False)
    lines = [
        "# BASIX decision handoff", "",
        f"Decision: `{checked['decision_id']}`",
        f"Handoff: `{checked['handoff_id']}`",
        f"Recorded: {escape(checked['created_at'])}", "",
        f"Workload: {escape(receipt['workload']['name'])}",
        f"Recommendation: {escape(receipt['recommended_offer_id'] or 'None established')}",
        "Execution authorization: **not granted**", "",
        "## Next action", "", escape(checked["next_action"]), "",
        "## Blockers and unknowns", "",
    ]
    for blocker in checked["blockers"]:
        reasons = "; ".join(str(reason) for reason in blocker["reasons"])
        lines.append(f"- {escape(blocker['offer_id'])} ({escape(blocker['status'])}): {escape(reasons)}")
    if not checked["blockers"]:
        lines.append("No blockers were identified for the recorded evidence and workload.")
    lines += ["", "## Limitations", ""]
    lines += [f"- {escape(item)}" for item in checked["limitations"]]
    lines += ["", "## Complete portable handoff", ""]
    payload = json.dumps(checked, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    # Source text may contain Markdown fences. Use a longer delimiter so the
    # complete JSON stays data instead of escaping into rendered instructions.
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", payload)), default=0))
    lines += [f"{fence}json", payload, fence, ""]
    return "\n".join(lines)
