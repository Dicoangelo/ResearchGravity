# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ResearchGravity is a Python research session tracking framework with auto-capture, lineage tracking, and multi-tier source management. It serves as the backend for the **Antigravity Chief of Staff** system.

## RAG Stack (Elite Configuration)

| Component | Implementation | Rating |
|-----------|----------------|--------|
| **Vector DB** | Qdrant (primary) + sqlite-vec (fallback) | ⭐⭐⭐⭐⭐ |
| **Embeddings** | Cohere embed-english-v3.0 (1024d) + SBERT offline fallback | ⭐⭐⭐⭐⭐ |
| **Reranking** | Cohere rerank-v3.5 / Hybrid BM25+cosine | ⭐⭐⭐⭐⭐ |
| **Storage** | SQLite + dual-write (Qdrant + sqlite-vec) | ⭐⭐⭐⭐⭐ |

### V2 Storage Modes

```
Priority: Qdrant → sqlite-vec → FTS fallback
- Qdrant: Full semantic search (requires server)
- sqlite-vec: Single-file vectors (offline capable)
- FTS: Full-text search fallback (always available)

Embeddings: Cohere → sentence-transformers fallback
- Cohere: embed-english-v3.0 (1024d, requires API)
- SBERT: all-MiniLM-L6-v2 (384d → padded to 1024d, fully offline)
```

### V6.1 Security & Reliability (API v2.1.0)

| Feature | Implementation |
|---------|----------------|
| **Authentication** | JWT tokens + API key (`X-API-Key` header) |
| **Rate Limiting** | slowapi (10/min search, 30/min write, 60/min default) |
| **Input Validation** | Regex-based session/project ID sanitization |
| **Dead-Letter Queue** | Failed writes queued for retry with exponential backoff |
| **Structured Logging** | JSON/console formats with request context |
| **Async Cohere** | Non-blocking embedding via `asyncio.to_thread` |

```bash
# Authentication
curl -X POST http://localhost:3847/api/auth/token \
  -d '{"client_id": "my-app", "scope": "write"}'
curl -H "Authorization: Bearer <token>" http://localhost:3847/api/auth/me

# Environment variables
export RG_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export RG_API_KEY="your-service-api-key"
export RG_LOG_LEVEL="INFO"    # DEBUG, INFO, WARNING, ERROR
export RG_LOG_JSON="true"     # JSON format for production
```

## Commands

```bash
# ═══════════════════════════════════════════════════════════
# API SERVER
# ═══════════════════════════════════════════════════════════
source .venv/bin/activate
python3 -m api.server --port 3847            # Start Chief of Staff API

# ═══════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════
python3 status.py                              # Check session state (always run first)
python3 init_session.py "topic"                # Start new research session
python3 init_session.py "topic" --impl-project os-app  # Pre-link to implementation project

# URL logging
python3 log_url.py <url> --tier 1 --category research --relevance 5 --used
python3 log_url.py <url> --tier 2 --category industry --relevance 4

# Session lifecycle
python3 archive_session.py                     # Archive completed session
python3 session_tracker.py status              # Check auto-capture status
python3 session_tracker.py link <session-id> <project>  # Link session to project

# ═══════════════════════════════════════════════════════════
# CONTEXT & PREFETCH
# ═══════════════════════════════════════════════════════════
python3 project_context.py                     # Auto-detect from current directory
python3 project_context.py --list              # List all projects
python3 project_context.py --index             # View unified index

python3 prefetch.py                            # Auto-detect project, inject context
python3 prefetch.py --project os-app --papers  # Specific project with papers
python3 prefetch.py --topic multi-agent        # Filter by topic
python3 prefetch.py --clipboard                # Copy to clipboard
python3 prefetch.py --inject                   # Inject into ~/CLAUDE.md

# ═══════════════════════════════════════════════════════════
# STORAGE & MIGRATION
# ═══════════════════════════════════════════════════════════
python3 -m storage.migrate                     # Migrate JSON → SQLite + Qdrant
python3 -m storage.migrate --dry-run           # Preview migration

# ═══════════════════════════════════════════════════════════
# BACKFILL & LEARNINGS
# ═══════════════════════════════════════════════════════════
python3 auto_capture.py scan --hours 48        # Scan recent history
python3 auto_capture.py backfill <path> --topic "..."  # Recover from old session
python3 backfill_learnings.py                  # Regenerate learnings.md from all sessions
python3 backfill_learnings.py --since 7        # Last 7 days only

# ═══════════════════════════════════════════════════════════
# CPB PRECISION MODE v2.5 (Hardened)
# ═══════════════════════════════════════════════════════════
python3 -m cpb.precision_cli "query"           # Run precision mode (95%+ DQ target)
python3 -m cpb.precision_cli "query" --verbose # With detailed output
python3 -m cpb.precision_cli --status          # System status (deps, providers, cache)
python3 -m cpb.precision_cli "query" --dry-run # Show execution plan without running
python3 -m cpb.precision_cli --interactive     # Interactive REPL mode
python3 -m cpb.precision_cli --agents          # List 7 agent personas

# v2.4+ Features: Pioneer Mode, Trust Context, Deep Research
python3 -m cpb.precision_cli "cutting-edge query" --pioneer        # Adjusted DQ weights
python3 -m cpb.precision_cli "query" --deep-research               # Gemini/Perplexity search
python3 -m cpb.precision_cli "query" --context @file.md --trust-context  # Tier 1 user context
python3 -m cpb.precision_cli "query" --pioneer --deep-research --verbose # Full stack

# ═══════════════════════════════════════════════════════════
# CPB FEEDBACK & GROUND TRUTH LEARNING
# ═══════════════════════════════════════════════════════════
python3 -m cpb feedback --stats                # Show feedback statistics
python3 -m cpb feedback --list                 # List recent feedback
python3 -m cpb feedback --interactive          # Interactive feedback collection
python3 -m cpb feedback --query "Q" --output "A" --rating 4  # Record feedback
python3 -m cpb feedback --export feedback.json # Export feedback data

# ═══════════════════════════════════════════════════════════
# V2: INTERACTIVE REPL
# ═══════════════════════════════════════════════════════════
python3 repl.py                                # Start interactive REPL
python3 repl.py --resume SESSION               # Resume existing session
python3 repl.py --status                       # Show status only

# REPL Commands:
#   start <topic>     - Initialize session
#   url <URL>         - Log URL (auto-classify)
#   finding <text>    - Capture insight
#   thesis/gap/direction - Set synthesis fields
#   status            - Show session progress
#   search <query>    - Semantic search past sessions
#   predict           - Quality predictions
#   checkpoint        - Save intermediate state
#   archive           - Finalize session
#   quit              - Exit REPL

# ═══════════════════════════════════════════════════════════
# V2: AUTO-CAPTURE (Enhanced)
# ═══════════════════════════════════════════════════════════
python3 auto_capture_v2.py scan                # Scan last 24 hours
python3 auto_capture_v2.py scan --hours 48     # Scan last 48 hours
python3 auto_capture_v2.py watch               # Watch mode (daemon)
python3 auto_capture_v2.py sync                # Sync to storage engine
python3 auto_capture_v2.py status              # Show capture stats

# ═══════════════════════════════════════════════════════════
# V2: INTELLIGENCE LAYER
# ═══════════════════════════════════════════════════════════
python3 intelligence.py predict "task"         # Session quality prediction
python3 intelligence.py optimal-time           # Best hour for tasks
python3 intelligence.py errors "context"       # Likely errors
python3 intelligence.py research "query"       # Related papers
python3 intelligence.py patterns               # Session patterns
python3 intelligence.py calibrate              # Run calibration loop
python3 intelligence.py status                 # System status

# ═══════════════════════════════════════════════════════════
# V2: FILE WATCHER (Implicit Sessions)
# ═══════════════════════════════════════════════════════════
python3 watcher.py start                       # Start in foreground
python3 watcher.py daemon                      # Start as background daemon
python3 watcher.py stop                        # Stop daemon
python3 watcher.py status                      # Show daemon status

# ═══════════════════════════════════════════════════════════
# V2: SQLITE-VEC MIGRATION
# ═══════════════════════════════════════════════════════════
python3 -m storage.migrate_to_vec --dry-run    # Preview migration
python3 -m storage.migrate_to_vec              # Run migration
python3 -m storage.migrate_to_vec --validate   # Validate migration
python3 -m storage.migrate_to_vec --status     # Show migration status
```

## Architecture

```
researchgravity/               # Scripts (this repo)
├── api/
│   ├── server.py              # FastAPI server @ :3847 (v2.1.0)
│   └── security.py            # JWT auth, rate limiting, validation (v6.1)
├── storage/
│   ├── qdrant_db.py           # Cohere embeddings + Qdrant vectors (1024d)
│   ├── sqlite_db.py           # SQLite relational storage (semaphore pool)
│   ├── sqlite_vec.py          # sqlite-vec vector storage (v6.0)
│   ├── engine.py              # Unified storage engine (dual-write + DLQ)
│   ├── dead_letter_queue.py   # Failed write recovery (v6.1)
│   ├── logging_config.py      # Structured logging (v6.1)
│   └── migrate.py             # JSON → SQLite + Qdrant migration
├── critic/
│   ├── base.py                # Writer-Critic base class
│   ├── archive_critic.py      # Archive validation
│   ├── evidence_critic.py     # Evidence validation
│   └── pack_critic.py         # Pack validation
├── cpb/                       # Cognitive Precision Bridge v2.5
│   ├── precision_orchestrator.py  # 7-agent cascade with ground truth
│   ├── search_layer.py        # Tiered search (arXiv, GitHub, internal)
│   ├── deep_research.py       # Gemini/Perplexity integration (v2.5 hardened)
│   ├── ground_truth.py        # Ground truth validation system
│   ├── critic_verifier.py     # DQ scoring with ground truth weight
│   ├── query_enhancer.py      # Query expansion with pioneer detection
│   ├── run_logger.py          # Run documentation with cost tracking
│   ├── feedback_cli.py        # Human feedback collection
│   ├── llm_client.py          # Multi-provider LLM client
│   ├── precision_cli.py       # Precision mode CLI (--status, --dry-run)
│   └── tests/                 # Test suite (17 tests)
├── prefetch.py                # Context prefetcher for Claude sessions
├── backfill_learnings.py      # Extract learnings from archived sessions
├── init_session.py            # Session initialization
├── session_tracker.py         # Auto-capture engine
├── auto_capture.py            # Backfill historical sessions
├── project_context.py         # Project context loader
├── log_url.py                 # URL logging
├── status.py                  # Cold start checker
├── archive_session.py         # Session archival
└── evidence_extractor.py      # Extract evidence from transcripts

~/.agent-core/                 # Data (single source of truth)
├── config.json                # API keys (cohere, youtube, etc.)
├── storage/
│   └── antigravity.db         # SQLite database
├── sessions/                  # Archived sessions (114 indexed)
├── context-packs/             # Context packs
├── memory/
│   └── learnings.md           # Extracted learnings archive
└── INTEGRATION_STATUS.md      # Full system documentation
```

## Cohere Configuration

API key stored in `~/.agent-core/config.json`:
```json
{
  "cohere": {
    "api_key": "your-key-here"
  }
}
```

Or via environment: `export COHERE_API_KEY='your-key-here'`

**Models:**
- Embeddings: `embed-english-v3.0` (1024 dimensions)
- Reranking: `rerank-v3.5`

**Free tier limits:**
- Embed: 1M tokens/month
- Rerank: 1,000 calls/month (~33 searches/day)

## API Endpoints

Base URL: `http://localhost:3847`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/sessions` | List sessions |
| POST | `/api/search/semantic` | Semantic search (Cohere + reranking) |
| GET | `/api/v2/stats` | Storage stats (Cohere model info) |
| POST | `/api/v2/findings/batch` | Batch store findings |

## Cold Start Protocol

Always run `status.py` first when starting a session. It shows:
- Active session state
- URLs logged, findings count, thesis status
- Recent archived sessions

## Source Hierarchy

**Tier 1 (Primary):** arXiv, HuggingFace Papers, OpenAI, Anthropic, Google AI, Meta AI, DeepMind, TechCrunch, The Verge

**Tier 2 (Amplifiers):** GitHub Trending, METR, ARC Prize, LMSYS, X/Twitter key accounts, HN, Reddit ML

**Tier 3 (Context):** Import AI, The Batch, Latent Space, LessWrong, Alignment Forum

## Research Workflow

1. **Signal Capture (30 min):** Scan Tier 1 sources, log all URLs via `log_url.py`
2. **Synthesis (20 min):** Group by theme, identify gaps, draft thesis
3. **Editorial Frame (10 min):** Write summary, link findings with rationale

## Lineage Tracking

Link research sessions to implementation projects for traceability:
```bash
python3 init_session.py "topic" --impl-project os-app
python3 session_tracker.py link <session-id> <project>
```

## Integration

Works across:
- **CLI (Claude Code):** Planning, parallel sessions, synthesis
- **Antigravity (VSCode):** Coding, preview, browser research
- **Web (claude.ai):** Handoff, visual review
- **OS-App:** Agent Core SDK (React hooks, TypeScript client)

## Testing Cohere Integration

```bash
source .venv/bin/activate
python3 -c "
import asyncio
from storage.qdrant_db import get_qdrant

async def test():
    q = await get_qdrant()
    results = await q.search_findings('multi-agent orchestration', limit=3)
    for r in results:
        score = r.get('relevance_score', r.get('score', 0))
        print(f'[{score:.3f}] {r[\"content\"][:60]}...')
    await q.close()

asyncio.run(test())
"
```
