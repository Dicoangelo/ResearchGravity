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
pytest --ignore=tests/test_mcp.py       # suite: tests/ (455) + cpb/tests/ (16)
python3 tests/test_mcp_raw.py           # MCP pipeline; standalone, skipped under pytest
python3 scripts/audit/wiring_audit.py   # phantom guarded imports (--ci to gate)
```

### Guarded imports degrade silently — prefer failing loudly

Optional integrations here are wrapped in `try/except ImportError` with a
fallback. That is correct for third-party packages and a trap for **repo-local**
imports: rename or move a symbol and the `ImportError` is swallowed, the feature
turns itself off permanently, and nothing fails or logs.

Four instances were found in a single audit pass:

- ReACT synthesis ran 3 of its 4 tools while reporting 4
- The 4Ds delegation gate never reached its LLM path on any run
- Oracle consensus never ran in pack building or UCW ingestion
- A phantom `CriticResult` import disabled the critic system

`scripts/audit/wiring_audit.py` gates CI against new occurrences. When adding a
guarded import, ask whether the target is genuinely optional — if it lives in
this repo, let it fail loudly instead of widening the `except`.

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

