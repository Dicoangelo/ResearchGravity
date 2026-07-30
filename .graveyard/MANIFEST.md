# Graveyard Manifest

Archived per the dead-code policy: disconnected code with real logic is archived,
not deleted, so the reference implementation stays discoverable. Each entry says
what the file did, why it was disconnected, and how to reconnect it.

Triage evidence (2026-07-18): full reference scan of every script against
repo code, docs, tests, launchd plists, ~/.zshrc, ~/.claude config, and ~/bin.
Everything here had zero live references. Files with any consumer, any recent
commit, or unclear status were KEPT in scripts/.

## scripts/backfill/ — completed one-time migrations

| File | What it did | Why disconnected | Recovery |
|---|---|---|---|
| `migrate_embeddings.py` | Migrated embeddings all-MiniLM-L6-v2 (384d) → nomic-embed-text | Migration completed; current stack is Cohere embed-v4 | Template for the next embedding-model migration |
| `migrate_to_cognitive_db.py` | Migrated RG data → Unified Cognitive Database schema | Migration completed | Reference for schema-migration mechanics |
| `simple_backfill.py` | Direct SQLite+Qdrant backfill writes | Superseded by `scripts/backfill/backfill_vectors.py` | Use backfill_vectors.py instead |
| `rebackfill_phase4.py` | Phase-4 re-backfill after prediction-tracking schema change | Phase 4 completed 2026 Q1 | Pattern for targeted re-backfills |

## scripts/session/ — superseded session utilities

| File | What it did | Why disconnected | Recovery |
|---|---|---|---|
| `checkpoint.py` | Mid-session restore points | Never wired into REPL/archive flow; no consumers | Wire into `repl.py` checkpoint command if session restore is wanted |
| `reinvigorate.py` | Session resume with full context reload (CLI) | Capability re-implemented self-contained in `api/server.py` `/api/reinvigorate/{session_id}`; CLI version unreferenced | Use the API endpoint; this is the CLI-shaped reference |
| `sync_to_ccc.py` | Pushed storage-triad data to Claude Command Center | CCC integration moved to UCW capture path (capture/) | Reference for CCC export shape |

## Misc

| File | What it did | Why disconnected | Recovery |
|---|---|---|---|
| `routing/routing-test-suite.py` | Test harness for autonomous routing | Never referenced by tests/ or CI; routing tested via tests/test_delegation | Mine for routing test cases |
| `sync_qdrant.py` | Full SQLite→Qdrant re-embed rebuild | One-time rebuild; superseded by backfill_vectors.py | Template for full vector rebuilds |
| `verify_extension.py` | UCW Chrome-extension health check | Extension health now surfaced via capture pipeline | Standalone extension debug tool |
| `visual/generate_remaining.py` | One-off batch visual generation run | Run completed; hardcoded batch list | Example of PaperBanana batch invocation |
| `visual/generate_variants.py` | One-off visual variant generation run | Run completed; hardcoded variants | Example of variant generation |
| `rg-semantic.sh` | Semantic search shell wrapper | Provably broken: cd's to `~/researchgravity`, a path that predates the move to `~/projects/apps/researchgravity` | Recreate as alias on `scripts/query_research.sh` if wanted |

## critic/ — disconnected by the base-class refactor

| File | What it did | Why disconnected | Recovery |
|---|---|---|---|
| `critic/oracle_adapter.py` | Oracle multi-stream consensus over a *single* critic: split one critic's issues into accuracy/completeness/relevance streams by `Issue.category`, weighted 0.40/0.35/0.25, and returned a composite approve/reject. Provided `run_oracle_consensus`, `OracleValidator`, `validate_with_oracle`. | Orphaned by the critic base refactor on three axes at once. `BaseCritic`/`CriticResult`/`ValidationIssue`/`IssueSeverity` were renamed to `CriticBase`/`ValidationResult`/`Issue`/`Severity`; `CriticBase.validate` became async taking `target_id` rather than sync taking a content dict; and `IssueCategory` was removed entirely, deleting the field the perspective split depends on. The module could not be imported at all, so both callers (`storage/ucw_ingestion.py`, `scripts/context-packs/build_packs.py`) silently ran with validation disabled — for months, with no log line. Both now call `PackCritic.validate` directly. | The renames are mechanical (verified: the package imports cleanly once applied). Reviving the perspective split additionally needs a **new** issue-code→category mapping over the 27 codes the critics emit — that is fresh design, not recovery, and there is no baseline to validate the resulting weights against. For a genuine 3-critic consensus use `critic.base.OracleConsensus`, which is live and used by `scripts/session/archive_session.py`. Perspective weights worth reusing if rebuilt: accuracy 0.40, completeness 0.35, relevance 0.25. Note `scripts/evidence/evidence_validator.py` has an unrelated `run_oracle_consensus` for *findings* — do not conflate them. |
