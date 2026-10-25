# BASIX × ResearchGravity gap map

As of October 25, 2026. Status labels distinguish code inspection, bounded tests,
historical observation and still-unverified behavior. Final integrated checks
are recorded separately; this document is not a production-readiness claim.

| Capability | Current evidence / status | Gap or next action |
|---|---|---|
| Canonical direction | Historical specs reconciled; operational and investment work are explicitly distinct | Validate the operational decision with real users/workloads |
| Typed workload and offer contract | Implemented in planner; bounded unit tests pass | Add new workload classes only with explicit contracts |
| Source integrity and freshness | Hash/date/reference checks implemented; stale/synthetic evidence cannot establish feasibility | Provenance does not verify truth; review provider assertions and data scope |
| Hard constraints | Privacy, region, deadline, budget and scope evaluated; unknown values cause abstention | Real model-memory requirements and resource units are outside image-upscale v1 |
| Repeatable decision | Deterministic ranking and content-hashed receipt implemented | Compare repeated actual runs with estimates before broader claims |
| Local execution evidence | One 64×64 → 256×256 local upscale observed at 0.97s; dimensions/hash checked, quality unmeasured | Record exact tool/model/input/output, inspect quality, repeat warm/cold runs |
| Immutable persistence | Atomic store, tamper rejection and CLI/UI/MCP readback verified | Extend retention/migration policy before multi-user service |
| Portable project handoff | Selected receipt + limitations + next action round-tripped through CLI, MCP and SQLite | Validate usefulness with a second human/operator workflow |
| ResearchGravity indexing | Real selected receipt indexed twice idempotently; exact handoff readback and FTS verified | Incremental maintenance and vector retrieval remain separate work |
| MCP and loopback UI | Same planner, transport boundaries, browser restore and responsive layouts verified | Hosted authentication and multi-user concurrency are outside v1 |
| Live cloud capacity / prices | No current adapter connected | Implement one read-only provider adapter; capture unit definitions and expiry |
| Cloud execution | Existing external harness is a prototype, excluded from v1 execution | Harden approval, ceilings, retry/idempotency, shell safety, output checks and cleanup |
| Actual total cost | Historical examples and estimates exist | Include setup/download/storage/egress and reconcile against measured charges |
| Energy / carbon / sovereignty scoring | Broad research topics and old rubric exist | Define sourced quantities and limitations before implementation; no invented scores |
| Continuous history compounding | Capture and context-pack infrastructure exists | Explicit allowlist, incremental updates, conflict/supersession handling and selected export |
| Market validation | A first-user operational use case is defined | Three real workload owners; test whether briefs change decisions and save measurable effort |

## Important integration limitations

1. **Global wallet export is not the BASIX handoff.** Existing `export_wallet()`
   defaults can sweep a broad private corpus. Its legacy integrity behavior does
   not cover every metadata/asset/value field and can accept missing hashes.
   BASIX instead exports one selected, fully hashed receipt. No global export
   path is exposed or claimed supported by this build.
2. **Global session logging is not project provenance.** Existing `log_finding`
   behavior can bind to the active global session. BASIX uses an explicitly chosen
   database and project-specific identifiers, findings, provenance and lineage.
3. **Text retrieval is not vector search.** Existing search paths differ: simple
   text matching, SQLite FTS and PostgreSQL-backed hybrid search are not the same
   capability. This v1 uses isolated SQLite storage/retrieval and makes no claim
   that vectors or remote databases are running.
4. **Archived completion claims are not acceptance evidence.** Context-pack and
   UCW documents contain historical metrics and broad novelty claims. They are
   design inputs until reproduced under a defined current test.
5. **Issue tooling is unavailable.** The checkout instructs contributors to use
   `bd`, but no `bd` executable was found in the current environment. Follow-ups
   are recorded here and in this repository's GitHub issues; no alternate local
   tracker or issue CLI was installed.

See the [dated validation](VALIDATION-2026-10-25.md) for exact test scope and
artifact-level results.

## Ordered work after v1 acceptance

| Priority | Bounded deliverable | Completion evidence |
|---|---|---|
| P0 | Repeat local upscale; inspect actual output; capture cold/warm timings | Same workload scope, multiple receipts, disclosed variability and quality assessment |
| P1 | One current read-only capacity/price adapter | Recorded provider response, normalized units, timestamps, expiry and unavailable/unknown cases |
| P1 | Harden existing cloud harness before exposing execution | Failure tests, budget approval, unique job state, verified output and teardown; no open-ended spend |
| P2 | One new workload class | Explicit requirements, separate calibration/evaluation data and no silent extrapolation |
| P2 | Incremental project history updates | Approved source list, deduplication, revisions, conflict review and a portable selected handoff |

The original Food Equity deployment, frozen FrontierAlpha evidence, broad UCW
personal corpus, accounts, billing and provider credentials remain outside this
implementation scope.

## Tracked follow-ups

- [#25 — local quality and repeatability](https://github.com/Dicoangelo/ResearchGravity/issues/25)
- [#26 — read-only provider observations](https://github.com/Dicoangelo/ResearchGravity/issues/26)
- [#27 — bounded cloud executor](https://github.com/Dicoangelo/ResearchGravity/issues/27)
- [#28 — incremental project history](https://github.com/Dicoangelo/ResearchGravity/issues/28)
