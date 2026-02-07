# PRD: Cross-Platform Live Capture Pipelines

**Version:** 1.0.0
**Date:** 2026-02-07
**Author:** Dicoangelo + Claude (Opus 4.6)
**Status:** Ready for Session B
**Depends on:** UCW Raw MCP Server (COMPLETE)

---

## Vision

Build live capture pipelines for every cognitive platform — ChatGPT, Grok/X, Cursor, and future platforms. Each pipeline intercepts, enriches, and stores cognitive events in the Unified Cognitive Database with full UCW semantic layers (Data + Light + Instinct).

This is Phase 6 of the UCW master plan: turning static imports into real-time cognitive capture across all platforms simultaneously.

### Why This Matters

- **ChatGPT import is done** (8,042 sessions, 30,712 findings) — but it's historical. We need LIVE.
- **Claude capture is done** (raw MCP server captures everything) — other platforms need parity.
- **Coherence detection needs real-time data** from multiple platforms to find cross-platform alignment.
- **Sovereignty window** is open (2026). Platforms will restrict export/API access by 2028-2030.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PLATFORM ADAPTERS                         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │  ChatGPT    │  │  Grok/X     │  │  Cursor          │    │
│  │  Adapter    │  │  Adapter    │  │  Adapter          │    │
│  │             │  │             │  │                    │    │
│  │ - API proxy │  │ - X API v2  │  │ - File watcher    │    │
│  │ - Export    │  │ - Streaming │  │ - Session parser   │    │
│  │   watcher   │  │ - Bookmarks │  │ - Git integration  │    │
│  └──────┬──────┘  └──────┬──────┘  └────────┬───────────┘    │
│         │                │                    │               │
│         └────────┬───────┴──────────┬────────┘               │
│                  │                  │                         │
│         ┌────────▼──────────────────▼────────┐               │
│         │      Platform Normalizer            │               │
│         │  - Normalize to CognitiveEvent      │               │
│         │  - Platform-specific UCW extraction  │               │
│         │  - Quality scoring per platform      │               │
│         └────────────────┬───────────────────┘               │
│                          │                                    │
│                  ┌───────▼───────┐                           │
│                  │  UCW Bridge   │                           │
│                  │  (enrichment) │                           │
│                  └───────┬───────┘                           │
│                          │                                    │
│               ┌──────────▼──────────┐                        │
│               │  Cognitive Database  │                        │
│               │  (PostgreSQL/SQLite) │                        │
│               └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Platform Adapter Interface

Every adapter MUST implement this interface:

```python
class PlatformAdapter:
    """Base class for all platform capture adapters."""

    PLATFORM: str                          # "chatgpt", "grok", "cursor"
    COGNITIVE_MODE: str                    # Default mode for this platform

    async def start(self) -> None:
        """Start capturing. May spawn background tasks."""

    async def stop(self) -> None:
        """Stop capturing gracefully."""

    async def poll(self) -> list[CognitiveEvent]:
        """Poll for new events since last check."""

    async def stream(self) -> AsyncIterator[CognitiveEvent]:
        """Stream events in real-time (if supported)."""

    def health_check(self) -> dict:
        """Return adapter health status."""
```

And every adapter produces `CognitiveEvent` objects:

```python
@dataclass
class CognitiveEvent:
    event_id: str
    timestamp_ns: int
    platform: str                          # chatgpt, grok, cursor, claude-desktop
    direction: str                         # in/out
    content: str                           # Raw content
    role: str                              # user/assistant/system
    conversation_id: str                   # Platform conversation ID
    message_id: str                        # Platform message ID
    data_layer: dict                       # UCW Data
    light_layer: dict                      # UCW Light
    instinct_layer: dict                   # UCW Instinct
    coherence_signature: str               # For cross-platform matching
    quality_score: float                   # 0.0 - 1.0
    cognitive_mode: str                    # deep_work/exploration/casual
    metadata: dict                         # Platform-specific extras
```

---

## Adapter 1: ChatGPT Live Capture

### Capture Methods (3 modes)

#### Mode A: Export Watcher (Passive)
- Watch `~/Downloads/` or configured dir for ChatGPT export ZIPs
- Parse conversations.json when new export detected
- Score with proven quality scorer
- Import new conversations (skip already-imported by conversation_id)
- **Trigger:** File system watcher (watchdog or inotify)

#### Mode B: OpenAI API Proxy (Active — future)
- MITM proxy between user's browser and api.openai.com
- Capture request/response pairs in real-time
- Requires: mitmproxy or custom HTTPS proxy
- **Complexity:** HIGH — requires certificate trust, browser config
- **Priority:** Phase 2 (after export watcher is proven)

#### Mode C: Browser Extension Bridge (Active — future)
- Chrome/Firefox extension that captures ChatGPT DOM
- Sends events to local UCW server via WebSocket
- **Complexity:** MEDIUM
- **Priority:** Phase 3

### Implementation Plan (Mode A — build this)

```
~/researchgravity/
├── capture/
│   ├── __init__.py
│   ├── base.py                    # PlatformAdapter base class + CognitiveEvent
│   ├── chatgpt_adapter.py         # ChatGPT export watcher
│   ├── chatgpt_normalizer.py      # Normalize ChatGPT JSON → CognitiveEvent
│   ├── grok_adapter.py            # Grok/X capture
│   ├── grok_normalizer.py         # Normalize Grok → CognitiveEvent
│   ├── cursor_adapter.py          # Cursor capture
│   ├── cursor_normalizer.py       # Normalize Cursor → CognitiveEvent
│   ├── manager.py                 # Multi-platform capture manager
│   └── config.py                  # Capture configuration
```

### ChatGPT Normalizer Details

**Input:** ChatGPT conversations.json format
```json
{
  "title": "conversation title",
  "create_time": 1707350400,
  "update_time": 1707354000,
  "mapping": {
    "msg-id-1": {
      "message": {
        "author": {"role": "user"},
        "content": {"parts": ["message text"]},
        "create_time": 1707350400
      }
    }
  }
}
```

**Output:** CognitiveEvent for each message in each conversation

**Quality scoring:** Reuse `chatgpt_quality_scorer.py` (proven at 98% accuracy)

**Deduplication:** Track imported conversation_ids in cognitive_sessions table. Skip if `platform='chatgpt' AND metadata->>'conversation_id' = X` exists.

### ChatGPT-Specific UCW Extraction

| UCW Layer | ChatGPT-Specific Fields |
|-----------|------------------------|
| Data | `conversation_title`, `model_slug`, `message_parts`, `plugin_used` |
| Light | `purpose` (from quality scorer), `topic_thread` (conversation context) |
| Instinct | `conversation_depth` (turn count → deeper = higher coherence), `time_span` |

---

## Adapter 2: Grok/X Capture

### Capture Methods

#### Mode A: X API v2 (Bookmarks + Posts)
- Use X API v2 to capture bookmarked posts (Grok interactions often bookmarked)
- Capture user's own posts/replies that reference AI topics
- Rate limits: 300 requests/15 min (Basic), 900 (Pro)
- **Auth:** OAuth 2.0 Bearer Token

#### Mode B: Grok API Direct (when available)
- xAI is building a Grok API
- When released: capture Grok conversations directly
- **Priority:** Monitor xAI announcements, build adapter shell now

### Implementation (Mode A)

**X API Fields to Capture:**
```json
{
  "tweet_id": "...",
  "text": "...",
  "created_at": "2026-02-07T...",
  "author_id": "...",
  "conversation_id": "...",
  "referenced_tweets": [...],
  "context_annotations": [...],
  "entities": {...}
}
```

**Grok-Specific UCW Extraction:**

| UCW Layer | Grok/X-Specific Fields |
|-----------|------------------------|
| Data | `tweet_text`, `thread_context`, `referenced_tweets`, `entities` |
| Light | `topic_from_annotations`, `intent_from_text`, `public_vs_private` |
| Instinct | `engagement_signal` (likes/retweets → community resonance), `thread_depth` |

### X API Configuration

```python
# Environment variables
X_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN")
X_API_BASE = "https://api.x.com/2"

# Endpoints
BOOKMARKS = f"{X_API_BASE}/users/{{user_id}}/bookmarks"
USER_TWEETS = f"{X_API_BASE}/users/{{user_id}}/tweets"
TWEET_SEARCH = f"{X_API_BASE}/tweets/search/recent"
```

### Grok Cognitive Mode Mapping

Stephen's insight: Grok captures "world-level thinking" — strategic, cross-domain.

| Signal | Cognitive Mode | Quality Weight |
|--------|---------------|----------------|
| Grok conversation | strategic | HIGH (0.8 base) |
| Bookmarked AI post | research | MEDIUM (0.6 base) |
| AI-related reply | exploration | MEDIUM (0.5 base) |
| General bookmark | casual | LOW (0.3 base) |

---

## Adapter 3: Cursor Capture

### Capture Methods

#### Mode A: File System Watcher
- Watch `~/.cursor/` directory for session changes
- Parse Cursor's internal session format
- Extract: prompts, completions, file context, accepted/rejected diffs
- **Trigger:** File modification events via watchdog

#### Mode B: Cursor Extension (future)
- Custom VS Code extension that hooks into Cursor's AI calls
- **Complexity:** MEDIUM — requires extension API understanding
- **Priority:** Phase 2

### Implementation (Mode A)

**Cursor Session Data Locations:**
```
~/.cursor/
├── User/
│   ├── workspaceStorage/     # Per-workspace session data
│   └── globalStorage/        # Global Cursor state
├── logs/                     # Session logs
└── extensions/               # Extension data
```

**Cursor-Specific UCW Extraction:**

| UCW Layer | Cursor-Specific Fields |
|-----------|----------------------|
| Data | `prompt`, `completion`, `file_context`, `language`, `diff_accepted` |
| Light | `intent=coding` (always), `topic` from file path, `complexity` from diff size |
| Instinct | `acceptance_rate` (accepted/total), `iteration_count`, `code_quality_signal` |

---

## Capture Manager

Central orchestrator that runs all adapters concurrently.

```python
class CaptureManager:
    """Multi-platform capture orchestrator."""

    def __init__(self, db, adapters: list[PlatformAdapter]):
        self.db = db
        self.adapters = adapters
        self._running = False

    async def start(self):
        """Start all adapters concurrently."""
        self._running = True
        tasks = [self._run_adapter(a) for a in self.adapters]
        await asyncio.gather(*tasks)

    async def _run_adapter(self, adapter):
        """Run a single adapter with error recovery."""
        while self._running:
            try:
                async for event in adapter.stream():
                    enriched = self._enrich(event)
                    await self.db.store_event(enriched)
                    await self._check_coherence(enriched)
            except Exception as e:
                log.error(f"{adapter.PLATFORM} error: {e}")
                await asyncio.sleep(30)  # Back off and retry

    async def _check_coherence(self, event):
        """Check if this event creates cross-platform coherence."""
        # Find events with same coherence signature from OTHER platforms
        matches = await self.db.find_coherent_events(
            event.coherence_signature,
            time_window_ns=30 * 60 * 1_000_000_000,  # 30 min
        )
        cross = [m for m in matches if m['platform'] != event.platform]
        if cross:
            log.info(f"COHERENCE DETECTED: {event.platform} ↔ {cross[0]['platform']}")
            # TODO: Create coherence_moment record
```

### Manager CLI

```bash
# Start all adapters
python3 -m capture start

# Start specific adapter
python3 -m capture start --platform chatgpt

# Status
python3 -m capture status

# One-shot import (ChatGPT export)
python3 -m capture import chatgpt ~/Downloads/chatgpt-export.zip

# Daemon mode (background)
python3 -m capture daemon
```

---

## Configuration

```python
# ~/researchgravity/capture/config.py

CAPTURE_CONFIG = {
    "chatgpt": {
        "enabled": True,
        "mode": "export_watcher",
        "watch_dirs": ["~/Downloads"],
        "export_pattern": "*.zip",
        "poll_interval_s": 60,
        "quality_threshold": 0.4,
        "dedup_by": "conversation_id",
    },
    "grok": {
        "enabled": False,       # Enable when X API token configured
        "mode": "api",
        "poll_interval_s": 300, # 5 min (rate limit friendly)
        "bookmarks": True,
        "user_tweets": True,
        "search_queries": ["AI", "AGI", "cognitive", "sovereign"],
    },
    "cursor": {
        "enabled": False,       # Enable when Cursor detected
        "mode": "file_watcher",
        "cursor_dir": "~/.cursor",
        "poll_interval_s": 30,
    },
}
```

---

## File Structure

```
~/researchgravity/
├── capture/                        # NEW: Cross-platform capture
│   ├── __init__.py
│   ├── base.py                     # PlatformAdapter, CognitiveEvent
│   ├── config.py                   # Capture configuration
│   ├── manager.py                  # Multi-platform capture manager
│   ├── normalizer.py               # Shared normalization utilities
│   │
│   ├── chatgpt_adapter.py          # ChatGPT export watcher
│   ├── chatgpt_normalizer.py       # ChatGPT → CognitiveEvent
│   │
│   ├── grok_adapter.py             # Grok/X API capture
│   ├── grok_normalizer.py          # X posts → CognitiveEvent
│   │
│   ├── cursor_adapter.py           # Cursor file watcher
│   ├── cursor_normalizer.py        # Cursor sessions → CognitiveEvent
│   │
│   └── __main__.py                 # CLI entry point
```

---

## Implementation Priority

| Priority | What | Complexity | Dependencies |
|----------|------|-----------|-------------|
| **P0** | `base.py` — Adapter interface + CognitiveEvent | LOW | None |
| **P0** | `normalizer.py` — Shared UCW extraction | LOW | base.py |
| **P0** | `chatgpt_adapter.py` — Export watcher | MEDIUM | base.py, quality scorer |
| **P0** | `chatgpt_normalizer.py` — ChatGPT → events | MEDIUM | normalizer.py |
| **P0** | `manager.py` — Multi-platform orchestrator | MEDIUM | base.py |
| **P0** | `config.py` + `__init__.py` + `__main__.py` | LOW | — |
| **P1** | `grok_adapter.py` + `grok_normalizer.py` | MEDIUM | X API token |
| **P2** | `cursor_adapter.py` + `cursor_normalizer.py` | MEDIUM | Cursor installed |

**Session B should build P0 files (7 files).** P1/P2 can wait until API tokens are configured.

---

## Dependencies

### Required
- `watchdog` — File system watcher (pip install watchdog)

### Optional
- `httpx` — For X API calls (pip install httpx)
- `tweepy` — Alternative X API client

### Already Available
- `chatgpt_quality_scorer.py` — Proven quality scoring
- `chatgpt_importer.py` — Existing import logic to reference
- `mcp_raw/ucw_bridge.py` — UCW layer extraction
- `mcp_raw/db.py` / `mcp_raw/database.py` — Storage

---

## Integration Points

### With UCW Raw MCP Server
- Capture adapters write to the SAME cognitive database
- Events get the same UCW layers (Data/Light/Instinct)
- Same coherence_signature algorithm for cross-platform matching
- MCP tools can query across all platforms

### With Coherence Engine
- CaptureManager calls `_check_coherence()` on every event
- Real-time detection of cross-platform alignment
- Creates coherence_moments when matches found

### With Embedding Pipeline
- New events get embedded after capture
- Stored in embedding_cache for similarity search
- Enables semantic coherence detection (not just signature matching)

---

## Success Criteria

1. **ChatGPT export watcher** detects and imports new exports within 60 seconds
2. **Quality scoring** matches proven 98% accuracy on new imports
3. **Deduplication** prevents duplicate conversations (by conversation_id)
4. **UCW layers** populated on every captured event
5. **Coherence signatures** generated for cross-platform matching
6. **Manager** runs all enabled adapters concurrently without crashes
7. **CLI** provides start/stop/status/import commands

---

## Race Condition Prevention

**Session B owns ALL files in `~/researchgravity/capture/`.**
**Session A does NOT touch the `capture/` directory.**

Session A is working on:
- PostgreSQL setup + schema
- Claude Desktop integration
- Embedding pipeline (`mcp_raw/embeddings.py`)
- Data migration scripts

No file overlap between sessions.

---

## Quick Start for Session B

```bash
cd ~/researchgravity

# Read existing code for reference
cat chatgpt_quality_scorer.py    # Quality scoring algorithm
cat chatgpt_importer.py          # Import logic to adapt
cat mcp_raw/ucw_bridge.py        # UCW layer extraction
cat mcp_raw/db.py                # Database interface

# Build in order:
# 1. capture/base.py              — PlatformAdapter + CognitiveEvent
# 2. capture/normalizer.py        — Shared UCW extraction
# 3. capture/config.py            — Configuration
# 4. capture/chatgpt_normalizer.py — ChatGPT → CognitiveEvent
# 5. capture/chatgpt_adapter.py   — Export watcher
# 6. capture/manager.py           — Multi-platform orchestrator
# 7. capture/__init__.py + __main__.py — Package + CLI
```
