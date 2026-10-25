# BASIX × ResearchGravity v1 contract

October 25, 2026. Local, single-operator workload planning and evidence handoff.
The source checkout is isolated from existing ResearchGravity work. The first
workload is image upscaling, using Sovereign Media as a measured local lane.
Investment-index research, portfolio weights and frozen evaluations are separate.

## Python interfaces

`basix.planner.evaluate_workload(workload, offers, evidence, *, now=None) -> dict`
accepts JSON values and returns a complete receipt. Validation errors raise
`ValueError`. `now` is an aware datetime or ISO UTC string; default is current UTC.

Workload fields: `id`, `task` (`image_upscale`), `name`, `max_cost_usd`,
`deadline_seconds`, `cloud_upload_allowed` (boolean), `allowed_regions` (list;
empty means no geographic restriction), `input` (`width`, `height`, `count`,
`scale`, all positive integers). No credential or local private file is required.

Offer fields: `id`, `name`, `provider`, `task`, `region` (string or null),
`data_leaves_device` (boolean), `availability` (`available`, `unavailable`,
`unknown`), `estimated_seconds` (number or null), `estimated_cost_usd` (number
or null), `measurement_scope` (the exact input object or null), `evidence_ids`
(list of source IDs), `notes` (text). Time/cost are estimates unless measurement
scope matches; no extrapolation is implicit. Unknown inputs must remain unknown.

Evidence fields: `id`, `title`, `uri`, `captured_at`, `valid_until` (ISO timestamps),
`kind` (`local_measurement`, `provider_documentation`, `historical_record`,
`synthetic_fixture`), `content` (bounded text), `sha256` (SHA256 UTF-8 content).
Dates, hashes and referenced IDs are checked. Source links establish provenance,
not independent truth or guaranteed capacity. Synthetic or stale evidence cannot
establish a currently feasible option. Local measurement scope must match input.

Receipt fields: `schema_version: "basix.decision.v1"`, `decision_id` (content hash),
`created_at`, `workload`, `offers`, `evidence`, `evaluations` (each with `offer_id`,
`status: feasible|blocked|unknown`, `reasons`, `estimated_seconds`,
`estimated_cost_usd`), `recommended_offer_id` (null allowed), `next_action`,
`execution_authorized: false`. Hash covers all fields except `decision_id` using
sorted compact JSON, UTF-8, `ensure_ascii=False`, `allow_nan=False`.

`basix.store.DecisionStore(root: Path)` supplies `save(receipt) -> Path`,
`get(decision_id) -> dict`, `list() -> list[dict]`. Immutable content addressed
JSON, exclusive creation, same receipt idempotent, altered/corrupt data rejected.

`basix.handoff.export_handoff(receipt) -> dict` returns a portable selected-decision
handoff with complete receipt, limitations, blockers, next action and a covering
hash; no ambient personal corpus. `render_handoff(handoff) -> str` outputs Markdown.

`basix.researchgravity.index_receipt(receipt, db_path: Path) -> dict` is async and
uses existing `storage.sqlite_db.SQLiteDB` only with an explicit path, storing
the receipt as a context pack and project-scoped findings/provenance/lineage.
No global active session, vector API, wallet sweep or credential discovery.

## First acceptance boundary

CLI and loopback UI evaluate, persist, reload and export a decision. MCP exposes
the same read/evaluate/handoff functions. One real small local measurement feeds
the demo; historical cloud observations are visibly stale and non-executable.
No cloud provisioning, uploads, trading, billing changes or model API spend.
Tests must cover unknown/stale evidence, privacy, budget, deadline, source
tampering, exact workload scope, immutable receipts and isolated RG retrieval.
