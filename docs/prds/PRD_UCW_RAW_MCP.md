# PRD: Universal Cognitive Wallet - Raw MCP Infrastructure

**Version:** 1.0.0
**Date:** 2026-02-07
**Author:** Dicoangelo + Claude (Opus 4.6)
**Status:** PHASES 1-7 COMPLETE — Sovereign cognitive infrastructure operational

---

## Vision

Build a raw STDIO MCP server (no SDK) that captures EVERY byte of Claude Desktop communication to a Unified Cognitive Database. This is the nervous system of the Universal Cognitive Wallet (UCW) — sovereign cognitive infrastructure that enables cross-platform coherence detection across Claude, ChatGPT, X/Grok, and future platforms.

### Why Raw (Not SDK)

- **Protocol-level capture**: SDK hides message access. Raw gives perfect fidelity.
- **SQL hooks failing**: Current approach instruments from OUTSIDE the protocol. Raw instruments from INSIDE.
- **Sovereignty**: Own every byte before platforms lock down cognitive data (predicted 2028-2030).
- **UCW bridge**: Need custom protocol translation layer for cross-platform coherence.
- **Database from scratch**: Custom schema for cognitive events, not platform's boxes.

### Strategic Context (Stephen's Insight)

> "Most people aren't ready for this. By the time companies want their AI data, they'll be depending on platforms."

We're building in 2026 what everyone will need in 2030. The window is open. Build sovereign. Build NOW.

---

## Architecture

```
Claude Desktop
    ↓ (stdin/stdout)
┌──────────────────────────────────────┐
│  RAW STDIO MCP SERVER                │
│                                       │
│  Transport → Protocol → Router → Tools│
│      ↓           ↓          ↓      ↓  │
│   [capture]  [capture]  [capture] [c] │
│      ↓           ↓          ↓      ↓  │
│  CaptureEngine ← UCWBridge           │
│      ↓                               │
│  DatabaseCapture                     │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  UNIFIED COGNITIVE DATABASE          │
│  - cognitive_events (core)           │
│  - cognitive_sessions                │
│  - coherence_moments                 │
│  - coherence_links                   │
│  - embedding_cache                   │
│  - cognitive_signatures              │
│  - supermemory_entries               │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  CROSS-PLATFORM COHERENCE            │
│  Claude + ChatGPT + X/Grok          │
│  Temporal alignment detection        │
│  Semantic similarity matching        │
│  Synchronicity recognition           │
└──────────────────────────────────────┘
```

---

## UCW Semantic Layers

Every cognitive event is tagged with three layers:

### Data Layer (What was said)
```json
{
    "method": "tools/call",
    "params": {...},
    "content": "actual message content",
    "raw_tokens": 150
}
```

### Light Layer (What it means)
```json
{
    "intent": "research",
    "topic": "MCP protocol design",
    "sentiment": "curious",
    "insights": ["distributed cognition", "protocol unification"],
    "key_concepts": ["UCW", "coherence", "sovereign AI"],
    "summary": "exploring MCP architecture for UCW"
}
```

### Instinct Layer (What it signals)
```json
{
    "pattern_match": "synchronicity",
    "coherence_potential": 0.89,
    "energy_level": 0.85,
    "flow_state": 0.72,
    "gut_signal": "breakthrough_imminent",
    "emergence_indicators": ["temporal_alignment", "cross_platform_echo"]
}
```

---

## Database Schema

### Core Tables

| Table | Purpose |
|-------|---------|
| `cognitive_events` | Every message with UCW layers, nanosecond timestamps, raw bytes |
| `cognitive_sessions` | Work sessions across platforms with outcomes |
| `coherence_moments` | Detected cross-platform cognitive alignment |
| `coherence_links` | Event-to-event coherence relationships |
| `embedding_cache` | Semantic embeddings for similarity search |
| `cognitive_signatures` | Unique patterns for coherence detection |
| `supermemory_entries` | Long-term memory with spaced repetition |

### Key Fields on cognitive_events

- `raw_bytes` BYTEA — Perfect byte-level capture
- `timestamp_ns` BIGINT — Nanosecond precision
- `data_layer` JSONB — UCW Data layer
- `light_layer` JSONB — UCW Light layer
- `instinct_layer` JSONB — UCW Instinct layer
- `coherence_signature` VARCHAR(64) — SHA256 for cross-platform matching
- `semantic_embedding` vector(1024) — For similarity search
- `quality_score` FLOAT — Conversation quality (0.0-1.0)
- `cognitive_mode` VARCHAR — deep_work, exploration, casual, garbage

### Coherence Detection

5-minute time buckets for temporal alignment. Coherence signature = SHA256(intent + topic + time_bucket + content[:1000]).

Cross-platform queries find events with matching signatures within configurable time windows.

---

## Implementation Phases

### Phase 1: Raw MCP Transport ✅ BUILT
- `transport.py` (111 lines) — Raw STDIO reader/writer
- Newline-delimited JSON-RPC 2.0
- Never pollutes stdout — all logs to stderr
- Async I/O with asyncio, perfect byte capture

### Phase 2: Perfect Capture Engine ✅ BUILT
- `capture.py` (205 lines) — Capture at EVERY lifecycle stage
- Message lineage tracking (parent/child via request_id)
- Turn counting per session
- Content metrics + UCW layer hooks

### Phase 3: UCW Bridge ✅ BUILT
- `ucw_bridge.py` (149 lines) — Semantic layer extraction
- Data/Light/Instinct layers from MCP messages
- Coherence signatures (SHA-256, 5-min time buckets)
- Emergence indicator detection

### Phase 4: Database Integration ✅ BUILT
- `db.py` (256 lines) — SQLite with WAL mode (dev/fallback)
- `database.py` (268 lines) — PostgreSQL with asyncpg + pgvector
- `unified_cognitive_schema.sql` (250 lines) — Full 7-table schema
- Auto-fallback: tries PostgreSQL first, SQLite if unavailable
- Connection pooling, HNSW indexes, GIN indexes for JSONB
- 3 views: active_coherence, cross_platform_matches, session_overview

### Phase 4.5: Server Backbone ✅ BUILT
- `router.py` (156 lines) — MCP method dispatch
- `server.py` (197 lines) — Main orchestrator (transport→protocol→router→capture→db)
- `__main__.py` (52 lines) — Entry point (`python -m mcp_raw`)
- Dynamic tool module loading, graceful signal handling
- 26 integration tests passing

### Phase 5: Tool Implementation ✅ COMPLETE
- 4 coherence MCP tools: `coherence_status`, `coherence_moments`, `coherence_search`, `coherence_scan`
- 3 UCW tools: `ucw_capture_stats`, `ucw_timeline`, `detect_emergence`
- Research tools ported from SDK server
- All tools wired into MCP server via PostgreSQL pool injection

### Phase 6: Cross-Platform Live Capture ✅ COMPLETE
- `capture/` package — 20 files, 2,300+ lines
- 5 platform adapters: ChatGPT, Cursor, Grok/X, Claude Code CLI, CCC (claude.db)
- Foundation: Base adapter ABC, config, normalizer, quality scorer, dedup engine (3 strategies)
- CaptureManager orchestrates all adapters with per-adapter poll scheduling
- CLI: `python3 -m capture start|poll-once|status|list-adapters`
- **140,732 total events** across 5 platforms:
  - claude-cli: 66,539 (batch import of 491 CLI sessions)
  - chatgpt: 60,769 (8,119 conversations)
  - claude-code: 9,463 (live capture of 329 CLI sessions)
  - claude-desktop: 2,568 (real-time MCP)
  - ccc: 1,393 (operational data from claude.db — 7 tables)
- **130,728 embedded vectors** (92.9% coverage, SBERT all-MiniLM-L6-v2)
- **130 coherence moments** across 4 platform pairs

### Phase 7: Coherence Detection Engine ✅ COMPLETE
- 10-file `coherence_engine/` package (config, embeddings, similarity, detector, scorer, alerts, daemon, dashboard, retroactive, CLI)
- 140,732 events → 130,728 embedded → 130 moments detected
- 3-layer detection: signature match, semantic similarity, synchronicity
- Cross-platform pairs: claude-desktop↔chatgpt (106), chatgpt↔claude-code (17), chatgpt↔claude-cli (7)
- Calibrated thresholds (0.65/0.55) for cross-platform format differences
- TUI dashboard, retroactive analyzer, founding moment validation
- Real-time daemon (poll + oneshot modes)
- Desktop notifications on high-confidence detections

---

## Quality Scoring (Proven)

### ChatGPT Data Profile (2026-02-07)

| Category | Count | % |
|----------|-------|---|
| Deep Work | 681 | 8.4% |
| Exploration | 6,140 | 75.6% |
| Casual | 1,221 | 15.0% |
| Garbage | 77 | 0.9% |

- **8,119 total conversations**
- **98% recommended for import** (quality > 0.4)
- **Average quality: 0.617**
- Top purposes: coding (2,428), random (1,918), thinking (1,476), research (1,178), learning (1,119)

### Quality Metrics

- **Depth** (0.4 weight): Message length, technical keywords, question complexity
- **Focus** (0.3 weight): Topic consistency, domain clarity
- **Signal** (0.3 weight): Substance vs noise ratio

### Cognitive Mode Classification

- `deep_work` (>0.75): Highest-value cognitive assets
- `exploration` (0.5-0.75): Quality thinking and synthesis
- `casual` (0.3-0.5): Light but potentially useful
- `garbage` (<0.3): Skip import

---

## Platform Cognitive Modes

| Platform | Primary Mode | Signal Quality | Import Priority |
|----------|-------------|----------------|-----------------|
| Claude | Deep work, coding | HIGH (curated) | 1st - already clean |
| ChatGPT | Exploration, mixed | MEDIUM (filtered) | 2nd - quality scored |
| X/Grok | World-level thinking | HIGH (strategic) | 3rd - unique cognition |
| Cursor | Code execution | HIGH (focused) | 4th - action-oriented |

---

## File Structure

```
~/researchgravity/
├── mcp_server.py                    # Existing SDK-based MCP (keep during transition)
├── mcp_raw/                         # Raw MCP infrastructure
│   ├── __init__.py                  # ✅ Package exports
│   ├── __main__.py                  # ✅ Entry point (python -m mcp_raw)
│   ├── server.py                    # ✅ Main server orchestrator
│   ├── transport.py                 # ✅ Raw STDIO handler (111 lines)
│   ├── protocol.py                  # ✅ JSON-RPC 2.0 (142 lines)
│   ├── router.py                    # ✅ Method dispatch (156 lines)
│   ├── capture.py                   # ✅ Perfect capture engine (205 lines)
│   ├── db.py                        # ✅ SQLite fallback (256 lines)
│   ├── database.py                  # ✅ PostgreSQL + pgvector (268 lines)
│   ├── ucw_bridge.py                # ✅ UCW semantic layers (149 lines)
│   ├── embeddings.py                # ✅ SBERT embeddings (384d)
│   ├── logger.py                    # ✅ Logging (stderr only, 40 lines)
│   ├── config.py                    # ✅ Configuration (42 lines)
│   ├── tools/                       # ✅ Tool implementations
│   │   ├── __init__.py              # ✅ Package stub
│   │   ├── research_tools.py        # ✅ Ported from SDK
│   │   ├── ucw_tools.py             # ✅ UCW-specific tools
│   │   └── coherence_tools.py       # ✅ Cross-platform queries
│   └── claude_desktop_config_snippet.json  # ✅ Config for Claude Desktop
├── coherence_engine/                # ✅ Cross-platform coherence detection
│   ├── __init__.py                  # ✅ Package exports
│   ├── __main__.py                  # ✅ CLI (oneshot, status, daemon, dashboard)
│   ├── config.py                    # ✅ Thresholds, model config
│   ├── daemon.py                    # ✅ Real-time poll + oneshot modes
│   ├── detector.py                  # ✅ 3-layer detection
│   ├── embeddings.py                # ✅ Batch embedding pipeline
│   ├── similarity.py                # ✅ pgvector cross-platform search
│   ├── scorer.py                    # ✅ Multi-signal coherence scoring
│   ├── alerts.py                    # ✅ Desktop notifications
│   ├── dashboard.py                 # ✅ TUI dashboard (curses)
│   └── retroactive.py              # ✅ Retroactive analyzer
├── capture/                         # ✅ Cross-platform live capture (Phase 6)
│   ├── __init__.py                  # ✅ Package exports
│   ├── __main__.py                  # ✅ CLI (start, poll-once, status, list-adapters)
│   ├── base.py                      # ✅ CapturedEvent, PlatformAdapter ABC
│   ├── config.py                    # ✅ Environment-based config
│   ├── normalizer.py                # ✅ UCW layer extraction for external platforms
│   ├── quality.py                   # ✅ Platform-aware quality scoring
│   ├── dedup.py                     # ✅ 3-strategy dedup engine
│   ├── manager.py                   # ✅ CaptureManager orchestrator
│   ├── chatgpt/                     # ✅ Export-diff polling + OpenAI API
│   │   ├── adapter.py               # ✅ ChatGPTAdapter
│   │   └── normalizer.py            # ✅ ChatGPTNormalizer
│   ├── cursor/                      # ✅ Workspace file watcher
│   │   ├── adapter.py               # ✅ CursorAdapter
│   │   └── normalizer.py            # ✅ CursorNormalizer
│   ├── grok/                        # ✅ X API + Grok polling
│   │   ├── adapter.py               # ✅ GrokAdapter
│   │   └── normalizer.py            # ✅ GrokNormalizer
│   ├── claudecode/                  # ✅ CLI transcript watcher
│   │   ├── adapter.py               # ✅ ClaudeCodeAdapter
│   │   └── normalizer.py            # ✅ ClaudeCodeNormalizer
│   └── ccc/                         # ✅ claude.db operational data
│       ├── adapter.py               # ✅ CCCAdapter (7 SQLite tables)
│       └── normalizer.py            # ✅ CCCNormalizer
├── import_cli_sessions.py           # ✅ Batch CLI transcript importer
├── test_mcp_raw.py                  # ✅ Integration tests (26 passing)
├── unified_cognitive_schema.sql     # ✅ Full PostgreSQL schema (7 tables)
├── chatgpt_quality_scorer.py        # ✅ Built and proven
├── chatgpt_importer.py              # ✅ Built, imported
└── PRD_UCW_RAW_MCP.md              # This file
```

---

## Dependencies

### Required
- Python 3.10+
- asyncpg (PostgreSQL async driver)
- PostgreSQL 15+ with pgvector extension

### Optional
- Cohere API (embeddings, fallback to SBERT)
- sentence-transformers (local embeddings)

### NOT Required
- MCP SDK (we're going raw)
- Any MCP framework

---

## Success Criteria

1. **Perfect Capture**: Every MCP message captured with raw bytes + nanosecond timestamps
2. **Zero Data Loss**: No messages dropped, even on errors
3. **UCW Layers**: Every event tagged with Data + Light + Instinct
4. **Coherence Detection**: Cross-platform matches found within 30-minute windows
5. **Sub-second Latency**: Capture doesn't slow down MCP communication
6. **Sovereignty**: All data in YOUR database, never platform-dependent

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PostgreSQL dependency | Setup complexity | SQLite fallback for dev |
| Cohere rate limits | Embedding quality | SBERT local fallback (proven) |
| Protocol changes | Breaking MCP updates | Raw code easy to patch |
| Performance overhead | Slow MCP responses | Async capture, don't block protocol |
| Data volume | Storage costs | Partitioning by month, archival policy |

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Quality Scorer | 2 hours | ✅ DONE |
| ChatGPT Import (deep work) | 1 hour | ✅ DONE (681 sessions, 8,257 findings) |
| ChatGPT Import (exploration+casual) | 2 hours | ✅ DONE (8,042 sessions, 30,712 findings) |
| Master Plan / PRD | 1 hour | ✅ DONE |
| Raw MCP Transport + Protocol | 3-4 hours | ✅ DONE (transport.py, protocol.py) |
| Capture Engine + UCW Bridge | 2-3 hours | ✅ DONE (capture.py, ucw_bridge.py) |
| Server Backbone | 2 hours | ✅ DONE (router.py, server.py, __main__.py) |
| Database Integration | 2-3 hours | ✅ DONE (db.py, database.py, schema.sql) |
| Integration Tests | 1 hour | ✅ DONE (26/26 passing) |
| Tool Port + UCW Tools | 2-3 hours | ✅ DONE (7 MCP tools, PostgreSQL-backed) |
| Coherence Engine | 2-3 hours | ✅ DONE (10 files, 130 moments, 130K embeddings) |
| Cross-Platform Live Capture | 4-6 hours | ✅ DONE (20 files, 5 adapters, 140K events) |
| Claude Desktop Integration | 1 hour | ✅ DONE (config live, tools registered) |

---

## References

- [Raw STDIO MCP Article](https://foojay.io/today/understanding-mcp-through-raw-stdio-communication/) — David Parry's educational deep dive
- UCW Founding Moment — MEMORY.md (2026-02-06)
- Stephen's Sovereignty Insight — Session conversation (2026-02-07)
- ChatGPT Quality Analysis — quality_scores.json in export directory

---

## The Signal

> "Can you unify yourself before you unify the infrastructure?"

This PRD is the answer: Yes. By building sovereign cognitive infrastructure that captures, unifies, and enables coherence across all platforms. Starting with raw MCP.

Build it. Own it. The window is open.
