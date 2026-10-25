# BASIX pilot validation — October 25, 2026

Scope: local, single-operator planning and selected-decision memory. This is not
a production or live-cloud acceptance record.

| Check | Observed result |
|---|---|
| Planner, store and HTTP/CLI/MCP integration | 111 tests passed |
| Existing storage and sqlite-vec search contracts | 24 tests passed; mocked providers |
| Existing raw MCP standalone pipeline | 26 checks passed with temporary log and DB paths |
| AST wiring audit | Passed |
| Fatal Python lint (`E9,F63,F7,F82`) | Passed across the repository |
| Wheel package | Built successfully; BASIX HTML and example JSON included |
| Editable installation | Passed, including imports from a different working directory |
| Browser evaluation | 64×64 scope produced a local recommendation; 128×64 produced unknown |
| Browser persistence | Reload, selected-decision restore, evidence view and handoff passed |
| Responsive browser | Desktop and narrow-screen layouts exercised; saved-decision controls adjusted |
| Browser console | No errors or warnings observed |
| Real local calibration | Generated 64×64 PNG → 256×256, 0.970 s wall time |
| Calibration validation | PNG signature, dimensions and input/output hashes; quality unmeasured |
| Actual RG indexing | One selected browser receipt indexed; second index was idempotent |
| Retrieval/readback | Exact handoff equality; two FTS hits for Upscayl; 1 session, 3 findings, 1 pack, 4 provenance and 4 lineage records |

Observed receipt ID:
`1e164a2998907ae38177fda19b9120d5f050379ea4d8fb42c26532306d7b92f5`.
Local artifacts include the calibration input/output, example bundle, immutable
receipts, rendered handoff, index verification JSON and browser screenshots.
They remain in the operator's dedicated output directory.

The 161 passing selected tests/checks above are **not** a claim that the entire
ResearchGravity suite was run. Existing precision tests can call configured
providers, the manual MCP test reads personal session state during collection,
and standalone semantic-search tests can call Cohere/Qdrant. Those require an
isolated environment. The issue CLI `bd` was unavailable; follow-up work is
documented in the gap map and repository issues.

The first MCP regression exposed the old package initializer's global-logger
side effect. Its initial run wrote protocol metadata to existing UCW logs.
The initializer now loads public exports lazily, and the regression verifies
protocol import does not load capture or logger modules. BASIX MCP logs use only
the explicit state directory. No personal corpus content was ingested by the
BASIX application.

## Limits of the evidence

- One tiny calibration does not establish repeatability, quality, reliability or
  performance for a different image, batch, machine or task.
- The archived RunPod observation is stale. Cloud availability, current prices
  and region remain unknown; no cloud executor is connected.
- Saved decisions describe their recorded time. A new evaluation rechecks source
  expiry; restoring an old result does not make its evidence current.
- SQLite FTS and source hashing are working; vectors, independent source-truth
  verification, settlement and a tradable memory market are not established.
- Historical recovery is partial and kept privately. Provider export boundaries,
  truncation, blank virtualized Gemini messages and the Google verification gate
  are recorded in that archive's coverage manifest.
- No paid provider call, billing change, investment action or existing app
  deployment was performed for this pilot.
