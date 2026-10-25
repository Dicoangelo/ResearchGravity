# BASIX × ResearchGravity: first product

Working synthesis, October 25, 2026. This plan compounds existing code and
research; it does not replace the broader BASIX thesis or alter frozen studies.

## Product decision

**BASIX explains where a specific workload can run under declared constraints.
ResearchGravity preserves the evidence, decision, and unfinished work so another
session can continue it.** The first workload is a small image-upscaling job.

This media-workload planner is the new bounded implementation wedge selected for
this build, not a claim that earlier BASIX plans had only this purpose. A recovered
September 15, 2025 user-submitted execution specification already grouped the
operational direction into **Index, Broker, Rails and Governance**, with
inference-first routing, provider adapters, policy constraints, metering, an
evidence chain and observability. Its source predates this implementation; the
origin of the text before it was submitted is not established, and the recovered
record is partial. Its proposed twelve-week schedule and service targets are
historical planning, not current commitments or measured results.

Recovered history also explores a Sovereign OS / AI service layer, AI frameworks
and hardware, a ledger, and cognitive-equity infrastructure for AI finance. Those
directions remain separately recorded hypotheses and plans; they are not silently
collapsed into this planner or treated as already approved product requirements.

| Historical operational layer | What v1 compounds | What remains separate or deferred |
|---|---|---|
| Index | Dated evidence and comparable workload observations | A broad live index, certification and generalized ratings |
| Broker | Constraint-based option evaluation and a reviewable choice | Provider allocation, autonomous routing and external execution |
| Rails | A portable receipt linking inputs, evidence and decisions | Payments, settlement, asset issuance and financial integration |
| Governance | Explicit policy inputs, abstention and execution disabled by default | Compliance certification, enterprise controls and legal claims |

The original inference-first scope remains an intended next workload direction.
The image-upscaling slice supplies a reproducible local measurement now; its
runtime or quality cannot be extrapolated to LLM inference. Generalization must
add a separate inference contract and evaluation before claiming that part of
the earlier plan is implemented.

The first user is a single operator choosing between an on-device media tool and
an external GPU lane. The immediate value is a defensible decision with an honest
"unknown" when evidence is insufficient. It is not automatic provisioning.

## Keep three kinds of work distinct

| Work | Purpose | Boundary |
|---|---|---|
| Operational BASIX | Compare feasible compute choices for a workload | Actual constraints, dated evidence, estimates and observed results |
| BASIX research map | Investigate compute, verification, coordination and economic infrastructure | Hypotheses and source-backed research; no validated rankings implied |
| Investment research | Evaluate economic exposure and prospective rules | Separate frozen protocols, holdings decisions and outcome quarantine |

The historical investment rubric is not a workload score. A useful provider is
not automatically an attractive investment. No portfolio, token, index exposure,
trading connection, settlement system or ownership claim belongs in this v1.

## One complete loop

1. Describe the task, input dimensions/count/scale, deadline, budget, cloud-upload
   permission and allowed regions.
2. Read explicitly supplied offers and evidence. Verify content hashes, source
   references, timestamps, types and exact measurement scope.
3. Apply hard constraints. Reject known violations; abstain on missing, stale,
   synthetic or mismatched evidence. Rank feasible choices by stated cost, time,
   then stable identifier. Do not invent a composite rating.
4. Persist an immutable decision receipt. Expose why alternatives failed, which
   figures are estimates and what remains unverified.
5. Export a portable handoff containing the selected decision and its evidence.
6. Index that selected record in an explicitly chosen ResearchGravity SQLite
   database, including findings, provenance and lineage. Retrieve it in the next
   session without scanning unrelated personal history.

Every receipt has `execution_authorized: false`. Eligibility is not consent to
spend, upload, provision, or run an external job. Links and hashes establish
provenance; they do not independently verify provider claims or capacity.

## v1 implementation boundary

The build includes a deterministic planner, content-addressed receipt store,
portable handoff, project-specific ResearchGravity SQLite adapter, CLI, loopback
web interface, and MCP interface using the same core functions. A small local
upscale has been observed: one 64×64 input produced a 256×256 output in 0.97s.
Dimensions and hashes were checked; this is one sample and output quality is
unmeasured. The release owner records the full measurement and integrated
acceptance separately; this observation is not general throughput evidence.

Reuse existing capabilities:

- `storage/sqlite_db.py`: context packs, findings, full-text retrieval, provenance
  and lineage through an explicit path.
- Sovereign Media: real local image-upscale executable and a historical GPU lane
  as a source of workload experience.
- Existing ResearchGravity MCP structure: expose bounded evaluate/read/handoff
  operations without enabling global capture or arbitrary shell execution.
- Existing context-pack principles: preserve the minimal sufficient evidence for
  the next action. Do not require the full multi-agent/vector architecture first.

## Acceptance before calling the first version usable

- A small local measurement is recorded with exact scope, duration, tool/model,
  source digest and observation time; output inspection is reported separately.
- CLI, UI and MCP agree on feasibility for the same supplied data.
- A modified input or stale source cannot silently reuse a feasible decision.
- Receipt persistence is immutable and idempotent; tampering is rejected.
- Export/reload and isolated ResearchGravity retrieval preserve evidence and the
  next action without leaking unrelated history.
- Privacy, deadline, budget, missing-source and invalid-input paths are covered.
- Browser/mobile usability and final test totals are captured in a dated
  validation receipt. A unit-test count alone is not product acceptance.

## Bounded next phases

1. **Repeatability and quality.** Repeat the same small workload; separate cold
   and warm runtime; inspect artifacts and choose an output-quality check. Keep
   calibration data distinct from evaluation inputs.
2. **Current provider data.** Add one read-only capacity/price adapter with dated
   source responses, expiry, unit normalization and unknown-state handling. No
   inferred guarantees from a listing alone.
3. **Harden a cloud executor.** Add explicit budget approval, job identity,
   resumable state, input/output integrity, safe argument handling, verified
   cleanup and cost reconciliation. Test failure paths before a bounded real job.
4. **Generalize workloads.** Add one new workload class only after its own
   requirement and measurement schema exist. Do not extrapolate image timings to
   video or LLM inference.
5. **Maintain the history.** Add an explicit, incremental source-update pipeline:
   approved sources, source revisions, deduplication, supersession, conflict
   review and an exportable project handoff. Validate retrieval on a frozen,
   sanitized set of real project questions with human-reviewed evidence labels.
   Existing synthetic memory experiments inform the method, not a production
   default change. Keep private transcripts out of public repositories.

The next decision is whether repeated local measurements produce a useful,
reliable choice. Cloud-marketplace breadth, energy scoring and autonomous
execution follow evidence from that first loop.

See [source map](SOURCE-MAP.md), [gap map](GAP-MAP.md) and the
[implementation contract](V1-CONTRACT.md).
