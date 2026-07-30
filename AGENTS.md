# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Quality Gates

```bash
pip install -e . --no-deps              # once — puts the repo root on sys.path
pytest --ignore=tests/test_mcp.py       # suite: tests/ (455) + cpb/tests/ (16)
python3 tests/test_mcp_raw.py           # MCP pipeline; standalone, skipped under pytest
python3 scripts/audit/wiring_audit.py   # phantom guarded imports (--ci to gate)
```

The editable install is not optional. Without it, scripts under `scripts/` can
import `storage/`, `critic/` etc. only when the interpreter happens to start at
the repo root. Note Homebrew Python refuses it under PEP 668, so this needs a
venv locally; CI installs directly.

### Guarded imports degrade silently — prefer failing loudly

Optional integrations here are wrapped in `try/except ImportError` with a
fallback. That is correct for third-party packages and a trap for **repo-local**
imports: rename or move a symbol and the `ImportError` is swallowed, the feature
turns itself off permanently, and nothing fails or logs.

Nine instances were found across two audit passes, in **three** distinct shapes.

*Renamed or missing symbol* — the import names something that no longer exists:

- ReACT synthesis ran 3 of its 4 tools while reporting 4
- The 4Ds delegation gate never reached its LLM path on any run
- Oracle consensus never ran in pack building or UCW ingestion
- A phantom `CriticResult` import disabled the critic system

*`sys.path`* — the import is **correct** and the target **exists**, but the
importing script cannot reach it, because Python puts only the invoked script's
own directory on the path:

- `scripts/prediction/intelligence.py` lost its storage engine, silently
- `scripts/session/auto_capture_v2.py` printed "Run from researchgravity
  directory", which is exactly what fails
- `scripts/session/archive_session.py` never loaded the evidence layer
- The `predict` REPL command reported "not available" on every invocation

`scripts/audit/wiring_audit.py` gates CI against the first shape. It **cannot
see the second** — it resolves modules by name and asks whether the target
defines the symbol, never whether the importer can reach it at runtime. The
second shape is addressed structurally instead, by `pyproject.toml` plus
`pip install -e .`, which puts the repo root on `sys.path` for every process.

When adding a guarded import, ask whether the target is genuinely optional — if
it lives in this repo, let it fail loudly instead of widening the `except`.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

