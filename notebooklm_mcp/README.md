# NotebookLM Cognitive Engine

**Version:** 2.2.0
**Status:** Complete (HTTP/RPC + Cognitive Intelligence + Convergence Tools + Upstream-Verified Rename/Revise)

Sovereign knowledge synthesis engine connecting NotebookLM to the Universal Cognitive Wallet. Uses direct HTTP/RPC to Google's `batchexecute` API (no browser required). Every interaction feeds the cognitive database — bidirectional, self-curating, knowledge-compounding.

## Architecture

```
notebooklm_mcp/
├── __init__.py                      # Package init
├── __main__.py                      # Entry point (python -m notebooklm_mcp)
├── server.py                        # MCP server orchestrator
├── config_notebooklm.py             # Configuration (v2.0.0)
├── api/
│   ├── __init__.py                  # Exports NotebookLMAPIClient
│   ├── client.py                    # HTTP/RPC client (88 methods, 34 RPC IDs)
│   ├── constants.py                 # CodeMapper + type codes
│   └── cognitive.py                 # Cognitive Intelligence Layer (GraphRAG, FSRS, coherence)
└── tools/
    └── notebooklm_tools.py          # 47 MCP tools (35 NotebookLM + 9 Cognitive/Convergence + 3 Auth)

UCW Infrastructure (reused):
  mcp_raw/ → transport, protocol, router, capture, database, embeddings
  coherence_engine/ → knowledge graph, hybrid search, FSRS, significance
```

## Authentication

**Cookies only** — CSRF token and session ID are auto-extracted.

### Method 1: Environment Variable

Set `NOTEBOOKLM_COOKIES` in `~/.mcp.json` (see MCP Registration below).

### Method 2: Runtime via MCP Tool

Use `save_auth_tokens` tool from Chrome DevTools:

```
1. Open Chrome DevTools → Network tab
2. Navigate to notebooklm.google.com
3. Find any batchexecute request
4. Copy the Cookie header
5. Use save_auth_tokens(cookies=<cookie_header>)
```

### Method 3: Chrome DevTools MCP (Fast)

```
save_auth_tokens(
    cookies=<cookie_header>,
    request_body=<body>,      # Contains CSRF token
    request_url=<url>         # Contains session ID
)
```

## MCP Registration

Already configured in `~/.mcp.json`:

```json
{
  "notebooklm": {
    "command": "python3",
    "args": ["-m", "notebooklm_mcp"],
    "cwd": "/Users/dicoangelo/projects/apps/researchgravity",
    "env": {
      "UCW_DATABASE_URL": "postgresql://localhost:5432/ucw_cognitive",
      "PYTHONPATH": "/Users/dicoangelo/projects/apps/researchgravity",
      "NOTEBOOKLM_COOKIES": ""
    }
  }
}
```

## Available Tools (47)

### NotebookLM Core (35)

| Tool | Purpose | Confirm? |
|------|---------|----------|
| `notebook_list` | List all notebooks | |
| `notebook_create` | Create new notebook | |
| `notebook_get` | Get notebook details | |
| `notebook_describe` | AI-generated summary with keywords | |
| `notebook_rename` | Rename a notebook | |
| `notebook_delete` | Delete notebook | Yes |
| `chat_configure` | Set chat goal/style/length | |
| `source_add` | Add source (url, text, drive, file) | |
| `source_describe` | AI-generated source summary | |
| `source_get_content` | Raw text content from source | |
| `source_list_drive` | List sources with Drive freshness | |
| `source_sync_drive` | Sync stale Drive sources | Yes |
| `source_delete` | Delete source | Yes |
| `source_rename` | Rename a source in place | |
| `notebook_query` | Ask questions synchronously (AI answers) | |
| `notebook_query_start` | Kick off a query as a background job (returns job_id) | |
| `notebook_query_status` | Poll an async query job; returns answer on completion | |
| `studio_create` | Generate content (audio, video, report, etc.) | Yes |
| `studio_status` | Check artifact generation status | |
| `studio_delete` | Delete studio artifacts | Yes |
| `artifact_rename` | Rename a studio artifact in place | |
| `slide_deck_revise` | Revise slide deck with per-slide instructions (creates NEW artifact) | |
| `download_artifact` | Download artifacts | |
| `export_artifact` | Export to Sheets/Docs | |
| `research_start` | Start web/Drive research | |
| `research_status` | Check research progress | |
| `research_import` | Import research results | |
| `notebook_share_status` | Get sharing settings | |
| `notebook_share_public` | Toggle public link access | |
| `notebook_share_invite` | Invite collaborator by email | |
| `notebook_share_batch` | Invite multiple collaborators in one call | |
| `note_create` | Create a note | |
| `note_list` | List all notes | |
| `note_update` | Update note content/title | |
| `note_delete` | Delete note | Yes |

### Cognitive Intelligence & Convergence (9)

| Tool | Purpose |
|------|---------|
| `cognitive_enrich_query` | GraphRAG-enriched query (knowledge graph + coherence moments + FSRS) |
| `cognitive_search` | Hybrid search across cognitive DB for notebook context |
| `cognitive_insights` | Surface FSRS-due insights related to a topic |
| `research_to_notebook` | Bridge ResearchGravity session into NotebookLM notebook |
| `knowledge_evolution` | Track understanding evolution over repeated queries |
| `cognitive_auto_curate` | Trigger coherence-driven notebook creation |
| `cross_notebook_query` | Run one query across many notebooks, aggregate answers (enriched or plain) |
| `batch_execute` | Chain multiple MCP tool calls in one request (continue_on_error opt-in) |
| `pipeline_research` | End-to-end: create notebook → add sources → query → optional studio artifact |

### Auth (3)

| Tool | Purpose |
|------|---------|
| `auto_auth` | Extract cookies from Chrome keychain automatically |
| `save_auth_tokens` | Save cookies from Chrome DevTools |
| `refresh_auth` | Reload auth tokens |

## Cognitive Intelligence Layer

What makes this 10x beyond a stateless API wrapper:

| Capability | Description |
|-----------|-------------|
| **GraphRAG** | Enriches queries with knowledge graph entities via spreading activation |
| **Bidirectional Capture** | Every NotebookLM result becomes a cognitive event with UCW layers |
| **Coherence Detection** | Cross-platform pattern matching (NotebookLM + Claude + ChatGPT + Grok) |
| **FSRS Resurfacing** | Spaced repetition surfaces insights when they're due |
| **Auto-Curation** | Convergence patterns trigger automatic notebook creation |
| **Research Bridge** | One command turns ResearchGravity sessions into synthesizable notebooks |
| **Knowledge Evolution** | Tracks how understanding crystallizes or fragments over time |
| **Hybrid Search** | BM25 + semantic + Reciprocal Rank Fusion across 140K+ events |

## UCW Integration

All NotebookLM events are captured to UCW `cognitive_events` table:
- **Platform:** `notebooklm`
- **Data Layer:** Content, tokens, bytes
- **Light Layer:** Intent, topic, key concepts
- **Instinct Layer:** Coherence signals, flow state, emergence indicators

```sql
SELECT event_id, method, light_intent, light_topic
FROM cognitive_events
WHERE platform='notebooklm'
ORDER BY timestamp_ns DESC
LIMIT 10;
```

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `NOTEBOOKLM_COOKIES` | For auto-init | Full cookie header from Chrome DevTools |
| `UCW_DATABASE_URL` | For cognitive | PostgreSQL connection string |
| `PYTHONPATH` | Yes | Path to researchgravity root |

## Logs

- Main: `~/.ucw/logs/notebooklm-mcp.log`
- Errors: `~/.ucw/logs/notebooklm-errors.log`
- Capture: `~/.ucw/logs/notebooklm-capture.log`

## Testing

```bash
# Test module imports
python3 -c "import notebooklm_mcp; print('Imports OK')"

# Test tools + handlers
python3 -c "
from notebooklm_mcp.tools.notebooklm_tools import TOOLS, _HANDLERS
print(f'{len(TOOLS)} tools, {len(_HANDLERS)} handlers')
"

# Test API client (requires cookies)
python3 -c "
import os
from notebooklm_mcp.api.client import NotebookLMAPIClient
client = NotebookLMAPIClient(cookies=os.environ['NOTEBOOKLM_COOKIES'])
notebooks = client.list_notebooks()
print(f'Found {len(notebooks)} notebooks')
"

# Server startup (Claude Desktop auto-starts via MCP)
python3 -m notebooklm_mcp
```

## Changelog

### 2.2.0 — Phase 4: Upstream Audit Port

Reverse-engineered RPC IDs + param shapes from `jacob-bd/notebooklm-mcp-cli` (read-only audit clone, nothing installed). Promoted both previously-deferred items:

- `source_rename` — RPC `b7Wfje`, params `[None, [source_id], [[[new_title]]]]`. In-place rename, no re-upload.
- `slide_deck_revise` — RPC `KmcKPe`, params `[[2], artifact_id, [[[slide_idx, instruction], ...]]]`. Creates a NEW slide-deck artifact from per-slide revision instructions; original is untouched. Slide-deck only (other studio types have no revise RPC).

Tool/handler count: 45 → 47. Parity verified.

**Live-verified 2026-04-11:**
- `source_rename` — non-destructive roundtrip on MFTH notebook (`8d73aec5`), source `22b1d49f` renamed and restored cleanly. Returns `{id, title}`.
- `slide_deck_revise` — revised `e7fb53e7` "The Closed Loop Architecture" with slide-0 instruction; original untouched, new `99586736` "The Closed Loop Architecture (2)" (`in_progress`) created. Artifact count 3 → 4.

### 2.1.0 — Convergence (Phases 1–3)

**Phase 1 — Security / correctness**
- Drift fixes vs upstream `jacob-bd/notebooklm-mcp-cli` v0.5.20
- Base URL allowlist + tightened file permissions
- `BaseClient` thread-safety locks for `_source_rpc_version`, `_conversation_cache`, `_query_jobs`
- Dual-RPC fallback `izAoDd → ozz5Z` for source RPC version skew
- gRPC canonical error code mapping (3/5/7/16) surfaced through handler errors
- Research pipeline correctness fixes

**Phase 2 — New capabilities**
- Async query polling: `notebook_query_start` + `notebook_query_status` with 10-min TTL job store
- `artifact_rename` (RPC `rc3d8d`) and `notebook_share_batch` for multi-invite
- Studio video enhancements: `video_style_prompt` with custom-style warning

**Phase 3 — Cognitive convergence**
- `cross_notebook_query` — aggregate a single query across N notebooks, enriched or plain
- `batch_execute` — serial MCP tool dispatch, `continue_on_error` flag, nested-batch guard
- `pipeline_research` — one call: create notebook → add sources → query → optional studio

**Deferred (documented, not shipped):**
- Studio status code 3 remap — speculative, would risk regression
- Audio source type code 10 — no ground truth
- `tag` wrapper — would require new local sidecar schema

(`source_rename` and `slide_deck_revise` shipped in 2.2.0 — see above.)

## References

- [jacob-bd/notebooklm-mcp-cli](https://github.com/jacob-bd/notebooklm-mcp-cli) — HTTP/RPC protocol reference
- [UCW Master Plan](~/projects/apps/researchgravity/PRD_UCW_RAW_MCP.md)
- [Whitepaper](~/WHITEPAPER.md) — Cognitive Equity thesis
