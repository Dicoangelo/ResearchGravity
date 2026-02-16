# Session B Brief — Intelligence Layer (Tools + Coherence)

**IMPORTANT: Session A (another terminal) is building the server backbone. DO NOT touch these files:**
- `mcp_raw/server.py`
- `mcp_raw/router.py`
- `mcp_raw/__main__.py`
- `mcp_raw/database.py`
- `unified_cognitive_schema.sql`

**Your job: Build 4 files in `~/researchgravity/mcp_raw/`:**

1. `coherence.py` — Coherence signature generation & temporal alignment
2. `tools/research_tools.py` — Port 8 tools from existing SDK server
3. `tools/ucw_tools.py` — New UCW-specific tools
4. `tools/coherence_tools.py` — Cross-platform coherence queries

---

## Interface Contract

Each tool module MUST export exactly two things:

```python
TOOLS: list[dict]  # MCP tool definitions
async def handle_tool(name: str, args: dict) -> dict  # Returns tool_result_content()
```

The router imports these and dispatches. Use the protocol helpers:

```python
from mcp_raw.protocol import tool_result_content, text_content

# In handle_tool:
return tool_result_content([text_content("your response")])
# For errors:
return tool_result_content([text_content("error message")], is_error=True)
```

---

## File 1: `mcp_raw/coherence.py`

Coherence detection engine. Used by tools and the server.

**Must provide:**
- `CoherenceEngine` class with:
  - `detect_temporal_alignment(events, window_ns)` — Find events close in time across platforms
  - `detect_semantic_similarity(event, candidates, threshold)` — Find similar events by content
  - `detect_synchronicity(events)` — Find the UCW "founding moment" patterns
  - `generate_moment(events, coherence_type, confidence)` — Create a coherence_moment record
- Standalone `coherence_signature()` already exists in `ucw_bridge.py` — reuse it, don't duplicate

**Reference:** UCW PRD section on Coherence Detection — 5-minute time buckets, SHA-256 signatures.

---

## File 2: `mcp_raw/tools/research_tools.py`

Port the 8 tools from `~/researchgravity/mcp_server.py` (the SDK-based server).

**Tools to port (read mcp_server.py for full implementation):**

| Tool | SDK lines | What it does |
|------|-----------|-------------|
| `get_session_context` | 316-356 | Active research session info |
| `search_learnings` | 358-377 | Search archived learnings |
| `get_project_research` | 379-399 | Load project research files |
| `log_finding` | 401-416 | Record finding to session |
| `select_context_packs` | 418-443 | Context pack selection (V2) |
| `get_research_index` | 445-455 | Unified research index |
| `list_projects` | 457-474 | List tracked projects |
| `get_session_stats` | 476-505 | Session/research statistics |

**Key paths (from config.py):**
```python
from mcp_raw.config import Config
# Config.SESSION_TRACKER  = ~/.agent-core/session_tracker.json
# Config.PROJECTS_FILE    = ~/.agent-core/projects.json
# Config.LEARNINGS_FILE   = ~/.agent-core/memory/learnings.md
# Config.RESEARCH_DIR     = ~/.agent-core/research
```

**Pattern:**
```python
TOOLS = [
    {
        "name": "get_session_context",
        "description": "Get active research session context...",
        "inputSchema": {
            "type": "object",
            "properties": {...},
        },
    },
    # ... more tools
]

async def handle_tool(name: str, args: dict) -> dict:
    if name == "get_session_context":
        return await _get_session_context(args)
    elif name == "search_learnings":
        return await _search_learnings(args)
    # ...
```

---

## File 3: `mcp_raw/tools/ucw_tools.py`

New UCW-specific tools (not in the SDK server).

**3 tools to build:**

### `ucw_capture_stats`
- Returns current capture session stats (events, turns, topics, gut signals)
- Reads from `CaptureDB` (import from `mcp_raw.db`)

### `ucw_timeline`
- Unified cross-platform timeline
- Query: `platform`, `since_ns`, `limit`
- Returns events across platforms sorted by time

### `detect_emergence`
- Real-time emergence signal detection
- Scans recent events for high coherence_potential (>0.7), concept clusters, meta-cognitive signals
- Returns emergence report

**These tools need database access.** Import CaptureDB:
```python
from mcp_raw.db import CaptureDB
from mcp_raw.config import Config

# Create a shared DB instance
_db = CaptureDB()
# Note: initialize() must be called before use (the server handles this)
```

---

## File 4: `mcp_raw/tools/coherence_tools.py`

Cross-platform coherence query tools.

**3 tools to build:**

### `find_coherent_events`
- Find events matching a coherence signature across platforms
- Args: `signature` (optional), `time_window_minutes`, `min_confidence`
- Queries cognitive_events by coherence_sig

### `coherence_report`
- Summary of all detected coherence moments
- Group by type (temporal, semantic, synchronicity)
- Include confidence distribution

### `cross_platform_search`
- Search across all platforms by topic/intent/concept
- Args: `query`, `platforms` (optional list), `limit`
- Returns unified results sorted by relevance

---

## Existing Files Reference

These files already exist — read them for context:

- `mcp_raw/config.py` (42 lines) — All paths and settings
- `mcp_raw/protocol.py` (142 lines) — `tool_result_content()`, `text_content()` helpers
- `mcp_raw/ucw_bridge.py` (149 lines) — `extract_layers()`, `coherence_signature()`
- `mcp_raw/capture.py` (205 lines) — `CaptureEngine`, `CaptureEvent`
- `mcp_raw/db.py` (256 lines) — `CaptureDB` SQLite storage
- `mcp_raw/logger.py` (40 lines) — `get_logger()` (stderr only, never stdout)
- `mcp_server.py` (605 lines) — SDK server to port from

---

## Rules

1. **NEVER write to stdout** — only stderr via `get_logger()`
2. **NEVER import the MCP SDK** — we're raw
3. **Use `from mcp_raw.protocol import tool_result_content, text_content`** for all tool responses
4. **Use `from mcp_raw.config import Config`** for all paths
5. **All functions must be async** — even if they don't await (consistency)
6. **Error handling**: catch exceptions, return `tool_result_content([...], is_error=True)`
7. **Don't touch** server.py, router.py, __main__.py, database.py, or unified_cognitive_schema.sql

---

## Quick Start

```bash
cd ~/researchgravity

# Read the existing files first
cat mcp_raw/config.py
cat mcp_raw/protocol.py
cat mcp_raw/ucw_bridge.py
cat mcp_raw/db.py
cat mcp_server.py  # SDK server to port from

# Then build:
# 1. mcp_raw/coherence.py
# 2. mcp_raw/tools/research_tools.py
# 3. mcp_raw/tools/ucw_tools.py
# 4. mcp_raw/tools/coherence_tools.py
```

When done, the server (__main__.py) will auto-discover and register your tools.
