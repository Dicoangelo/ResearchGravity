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
