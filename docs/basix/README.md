# BASIX × ResearchGravity local pilot

A workload planner with a durable memory. This first version compares an image
upscale against supplied evidence, records why an option is feasible, blocked or
unknown, and exports one selected decision for another session.

Read the [masterplan](MASTERPLAN.md), [source map](SOURCE-MAP.md),
[gap map](GAP-MAP.md), [contract](V1-CONTRACT.md) and
[dated validation](VALIDATION-2026-10-25.md).

## Run locally

Python 3.12+ is required. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m basix --state-dir ./basix-local/decisions serve --port 8097
```

Open `http://127.0.0.1:8097`. The server binds only to loopback. It is a
single-operator pilot, without a hosted service or multi-user authentication.
Do not point its state directory at a personal UCW corpus or a shared directory.

The included example is a **dated observation**, not a fresh benchmark on your
machine. It expires on October 26, 2026, and should then produce an unknown
decision. Never change its timestamp to make it pass. Restoring an old receipt
preserves the decision at its original time; evaluate again for a current result.

## Make a fresh, bounded local observation

If the existing Sovereign Media `upscale-image` executable and model are installed:

```bash
.venv/bin/python scripts/basix/measure_local.py \
  --tool /absolute/path/to/sovereign-media/bin/upscale-image \
  --output-dir ./basix-local/measurement-001
.venv/bin/python -m basix --state-dir ./basix-local/decisions \
  serve --port 8097 --bundle ./basix-local/measurement-001/media-upscale.json
```

Use a new output directory each time. The script creates a 64×64 calibration
image, runs the existing local tool with a 90-second timeout, checks the PNG
signature and 256×256 dimensions, and records hashes and elapsed time.
It does not install models, contact cloud providers, assess perceptual quality,
or establish performance for other input sizes. It invokes the executable
you explicitly supply; inspect that local tool before using a different path.

## CLI and project memory

```bash
.venv/bin/python -m basix --state-dir ./basix-local/decisions \
  evaluate basix/examples/media-upscale.json
.venv/bin/python -m basix --state-dir ./basix-local/decisions list
.venv/bin/python -m basix --state-dir ./basix-local/decisions get DECISION_ID
.venv/bin/python -m basix --state-dir ./basix-local/decisions handoff DECISION_ID
```

Optional ResearchGravity indexing uses its existing SQLite schema. Install
`aiosqlite` if it is not already supplied by the repository's requirements:

```bash
.venv/bin/python -m pip install aiosqlite
.venv/bin/python -m basix --state-dir ./basix-local/decisions \
  index DECISION_ID --db ./basix-local/researchgravity.db
```

Indexing is explicit and idempotent. It stores a session, selected findings,
context pack, provenance and lineage in the chosen database, with SQLite FTS
retrieval. It does not start vector services or import personal chat history.
The immutable JSON receipt remains authoritative.

## MCP

Launch the same installed interpreter as a stdio MCP server:

```bash
.venv/bin/python -m basix --state-dir /absolute/private/basix/decisions mcp
```

Tools: `basix_evaluate_workload`, `basix_get_decision`,
`basix_export_handoff`. The evaluation tool accepts the contract's complete
workload/offer/evidence objects; get/export accept a `decision_id`. Protocol logs
stay under the selected state's `mcp-logs/`. No automatic capture, provider
credentials, arbitrary commands, provisioning or spending tool is exposed.
Configure this command in the chosen client explicitly; the build does not edit
your global MCP configuration.

## What this version can establish

- A supplied candidate satisfies known constraints, or has a stated violation
  or missing evidence. Feasible candidates are ordered by stated provider cost,
  duration, then identifier.
- A source's content matches its recorded digest and was current at evaluation.
  Hashes do not establish external truth or authenticate a provider.
- A saved/exported receipt is complete and internally consistent. Receipts are
  immutable records, not cryptographic signatures or execution permissions.

Current cloud discovery, cloud execution, payment rails, quality scoring and
energy/carbon measurement remain on the [gap map](GAP-MAP.md). Provider charge
excludes hardware and electricity. Private historical transcripts stay outside
the repository and outside exported decision handoffs.

## Focused checks

```bash
.venv/bin/python -m pytest -q tests/test_basix_planner.py \
  tests/test_basix_store.py tests/test_basix_interfaces.py tests/test_basix_measurement.py \
  tests/test_storage_search_contract.py tests/test_sqlite_vec_search_contract.py
ruff check --select E9,F63,F7,F82 basix mcp_raw/__init__.py scripts/basix \
  tests/test_basix_planner.py tests/test_basix_store.py tests/test_basix_interfaces.py
.venv/bin/python scripts/audit/wiring_audit.py --ci
```

Use an isolated CI host for the repository's broader suite: some pre-existing
tests can invoke configured providers or read personal session state.
