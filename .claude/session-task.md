# Autonomous Session: ResearchGravity

## Mission
Fix all 120 ruff lint errors, fix broken test collection, and expand test coverage. Pure code quality session.

## Current State
- 120 ruff errors (98 auto-fixable)
- 321 tests collected but `tests/test_mcp.py` has a `sys.exit(1)` that crashes pytest
- Python 3.14, ruff for linting

## Task List (execute in order)

### Phase 1: Fix Test Infrastructure
1. Fix `tests/test_mcp.py` — it calls `exit(1)` at module level (line 15), crashing all test collection. Remove the exit or gate it properly.
2. Run `python3 -m pytest --co -q` to verify all 321+ tests collect cleanly
3. Run `python3 -m pytest` to see which tests pass/fail

### Phase 2: Ruff Auto-fix
1. Run `ruff check . --fix` to auto-fix the 98 fixable errors
2. Manually fix the remaining ~22 errors
3. Run `ruff check .` — target: 0 errors
4. Run `ruff format --check .` and fix any formatting issues

### Phase 3: Test Coverage
1. Run `python3 -m pytest` and catalog pass/fail
2. Fix any failing tests (likely import issues or missing deps)
3. Add tests for untested modules, especially:
   - MCP server tools (21 + 37 tools)
   - Coherence engine
   - UCW capture pipeline
4. Target: all collected tests passing

### Phase 4: Type Hints
- Add type hints to any public functions missing them
- Focus on MCP server entry points and core APIs

## Validation
```bash
ruff check .                    # 0 errors
python3 -m pytest -q            # All tests pass
python3 -m pytest --tb=short    # No failures
```

## Rules
- Don't break working tests
- Commit after each phase
- If a test needs external services (PostgreSQL, etc.), mock appropriately or skip with `@pytest.mark.skip`
