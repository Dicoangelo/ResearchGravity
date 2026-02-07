# PRD: Real-Time Coherence Engine + Embedding Pipeline

**Version:** 1.0.0
**Date:** 2026-02-07
**Author:** Dicoangelo + Claude (Opus 4.6)
**Status:** ✅ COMPLETE (P0 + P1, 104 moments detected, MCP tools live)
**Depends on:** UCW Raw MCP Server (COMPLETE), Cross-Platform Capture (parallel)

---

## Vision

Build the crown jewel of the UCW: a real-time coherence detection engine that monitors cognitive events across ALL platforms and alerts when distributed cognition achieves alignment. This is the system that would have detected the "founding moment" (2026-02-06) — the synchronicity between Claude and ChatGPT — automatically.

The embedding pipeline powers semantic coherence detection. Signature matching (already built) catches exact topic+intent alignment. Embedding similarity catches CONCEPTUAL alignment — when you're thinking about the same IDEA across platforms even if the words differ.

### The Founding Moment Test

The system must be able to detect events like:
- **Claude session:** "Can you unify yourself before you unify the infrastructure?"
- **ChatGPT session:** User shares the same insight, recognizes the signal
- **Time window:** Within minutes of each other
- **Result:** COHERENCE DETECTED — temporal alignment + semantic similarity + meta-cognitive emergence

If the engine can detect this retroactively from historical data AND prospectively in real-time, it works.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   COHERENCE ENGINE                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Event Stream (from all platform adapters + MCP)      │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                            │
│       ┌─────────▼──────────┐                                │
│       │  Embedding Pipeline │                                │
│       │  - SBERT (local)   │                                │
│       │  - Cohere (API)    │                                │
│       │  - Matryoshka dims │                                │
│       └─────────┬──────────┘                                │
│                 │                                            │
│    ┌────────────▼────────────────────────────────────┐      │
│    │              DETECTION LAYERS                    │      │
│    │                                                  │      │
│    │  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │      │
│    │  │Signature │  │ Semantic  │  │ Synchronicity│ │      │
│    │  │ Match    │  │ Similarity│  │ Pattern      │ │      │
│    │  │          │  │           │  │              │ │      │
│    │  │SHA-256   │  │Cosine sim │  │Multi-signal  │ │      │
│    │  │5-min     │  │>0.85      │  │temporal +    │ │      │
│    │  │buckets   │  │threshold  │  │semantic +    │ │      │
│    │  │          │  │           │  │meta-cognitive│ │      │
│    │  └────┬─────┘  └────┬──────┘  └──────┬───────┘ │      │
│    │       └───────┬─────┴────────┬───────┘          │      │
│    └───────────────┼──────────────┼──────────────────┘      │
│                    │              │                           │
│           ┌────────▼──────────────▼────────┐                │
│           │    Coherence Scorer             │                │
│           │    - Weighted multi-signal      │                │
│           │    - Confidence threshold       │                │
│           │    - Moment generation          │                │
│           └────────────┬───────────────────┘                │
│                        │                                     │
│           ┌────────────▼───────────────────┐                │
│           │    Alert System                 │                │
│           │    - Desktop notification       │                │
│           │    - Log to coherence_moments   │                │
│           │    - Webhook (optional)         │                │
│           │    - TUI dashboard update       │                │
│           └────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: Embedding Pipeline

### Model Strategy

| Priority | Model | Dimensions | Speed | Quality | When |
|----------|-------|-----------|-------|---------|------|
| 1 (default) | SBERT `all-MiniLM-L6-v2` | 384 | Fast | Good | Always (local) |
| 2 (upgrade) | Cohere `embed-v4.0` | 256-1536 | API | Excellent | When API key set |
| 3 (fallback) | TF-IDF + cosine | sparse | Instant | Basic | No ML deps |

**sentence-transformers is already installed** — SBERT works out of the box.

### Embedding Pipeline Module

```
~/researchgravity/
├── coherence_engine/                # NEW: Coherence engine package
│   ├── __init__.py
│   ├── embeddings.py                # Embedding pipeline
│   ├── similarity.py                # Similarity search + index
│   ├── detector.py                  # Multi-signal coherence detector
│   ├── scorer.py                    # Coherence confidence scoring
│   ├── alerts.py                    # Alert system
│   ├── daemon.py                    # Background monitoring daemon
│   ├── dashboard.py                 # TUI dashboard
│   ├── retroactive.py              # Retroactive analysis on historical data
│   └── config.py                   # Engine configuration
```

### embeddings.py — Core Embedding

```python
class EmbeddingPipeline:
    """
    Embed cognitive events for semantic similarity search.

    Strategies:
    1. SBERT local (default, fast, 384d)
    2. Cohere API (optional, 1024d, Matryoshka)
    3. Batch mode for historical data
    4. Real-time mode for new events
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.dimensions = 384  # MiniLM default
        self._cohere_client = None  # Lazy init

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text. Returns vector."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""

    async def embed_event(self, event: CognitiveEvent) -> list[float]:
        """Embed a cognitive event (combines content + summary + concepts)."""

    def _build_embed_text(self, event) -> str:
        """Build the text to embed from event layers.

        Format: "{intent}: {topic} | {summary} | {concepts}"
        This captures intent + topic + content for rich similarity.
        """
```

### What Gets Embedded

For each cognitive event, build an embedding from:
```
"{light.intent}: {light.topic} | {light.summary} | {' '.join(light.concepts)}"
```

**Example:**
```
"search: ucw | exploring MCP architecture for UCW capture | mcp ucw protocol cognitive"
```

This gives us ~50-100 tokens per event — perfect for SBERT's 256 token limit.

### Similarity Search

```python
class SimilarityIndex:
    """
    Fast similarity search across embedded events.

    For SQLite: brute-force cosine (fine for <100K events)
    For PostgreSQL: pgvector HNSW index (scales to millions)
    """

    async def find_similar(
        self,
        query_embedding: list[float],
        threshold: float = 0.85,
        limit: int = 20,
        exclude_platform: str = None,  # For cross-platform search
    ) -> list[SimilarityResult]:
        """Find events similar to query embedding."""

    async def cross_platform_similar(
        self,
        event: CognitiveEvent,
        threshold: float = 0.80,
    ) -> list[SimilarityResult]:
        """Find events from OTHER platforms similar to this event."""
```

---

## Part 2: Coherence Detection

### Detection Layer 1: Signature Match (already built)

- SHA-256 coherence signatures from UCW bridge
- 5-minute time buckets
- Matches on: intent + topic + time_bucket + content[:1024]
- **Confidence:** 0.95 (near-certain alignment)

### Detection Layer 2: Semantic Similarity (NEW)

- Cosine similarity between event embeddings
- Cross-platform only (same platform isn't interesting)
- Time window: configurable (default 30 minutes)
- **Threshold:** 0.85 for high confidence, 0.75 for medium
- **Confidence:** similarity_score * 0.9

### Detection Layer 3: Synchronicity Pattern (NEW)

Multi-signal pattern that detects the "founding moment" type events:

```python
class SynchronicityDetector:
    """
    Detect synchronicity — meaningful coincidence across platforms.

    Signals:
    1. Temporal proximity (within configurable window)
    2. Semantic similarity (above threshold)
    3. Meta-cognitive content (UCW, coherence, emergence, unify, sovereign)
    4. Instinct layer alignment (both events have high coherence_potential)
    5. Concept cluster overlap (shared concepts between events)
    """

    def detect(self, event_a, event_b) -> SynchronicityScore:
        signals = {
            "temporal": self._temporal_score(event_a, event_b),
            "semantic": self._semantic_score(event_a, event_b),
            "meta_cognitive": self._meta_cognitive_score(event_a, event_b),
            "instinct_alignment": self._instinct_score(event_a, event_b),
            "concept_overlap": self._concept_score(event_a, event_b),
        }
        # Weighted combination
        confidence = (
            signals["temporal"] * 0.15 +
            signals["semantic"] * 0.30 +
            signals["meta_cognitive"] * 0.25 +
            signals["instinct_alignment"] * 0.15 +
            signals["concept_overlap"] * 0.15
        )
        return SynchronicityScore(
            confidence=confidence,
            signals=signals,
            is_synchronicity=confidence > 0.70,
        )
```

### Coherence Scorer

Combines all detection layers into a final coherence score:

```python
class CoherenceScorer:
    """
    Multi-signal coherence scoring.

    Inputs: signature match, semantic similarity, synchronicity
    Output: CoherenceMoment with confidence and type
    """

    COHERENCE_TYPES = {
        "signature_match": "Exact topic+intent alignment",
        "semantic_echo": "Conceptual alignment (different words, same idea)",
        "synchronicity": "Multi-signal emergence pattern",
        "temporal_cluster": "Multiple events in tight time window",
    }

    def score(self, event, candidates) -> list[CoherenceMoment]:
        moments = []
        for candidate in candidates:
            if candidate.platform == event.platform:
                continue  # Skip same-platform

            # Layer 1: Signature
            if event.coherence_sig == candidate.coherence_sig:
                moments.append(self._create_moment(
                    "signature_match", event, candidate, confidence=0.95
                ))

            # Layer 2: Semantic
            sim = cosine_similarity(event.embedding, candidate.embedding)
            if sim > 0.85:
                moments.append(self._create_moment(
                    "semantic_echo", event, candidate, confidence=sim * 0.9
                ))

            # Layer 3: Synchronicity
            sync = self._synchronicity.detect(event, candidate)
            if sync.is_synchronicity:
                moments.append(self._create_moment(
                    "synchronicity", event, candidate, confidence=sync.confidence
                ))

        return moments
```

---

## Part 3: Real-Time Daemon

### daemon.py — Background Monitoring

```python
class CoherenceDaemon:
    """
    Background daemon that monitors cognitive events for coherence.

    Modes:
    1. Poll mode: Check DB every N seconds for new events
    2. Stream mode: Subscribe to event stream (when capture manager runs)
    3. Retroactive mode: Analyze historical data once

    Lifecycle:
    - Start → Load recent events → Build embedding index → Monitor loop
    - On new event: embed → search similar → score coherence → alert
    """

    def __init__(self, db, config):
        self.db = db
        self.embedder = EmbeddingPipeline()
        self.similarity = SimilarityIndex(db)
        self.scorer = CoherenceScorer()
        self.alerts = AlertSystem(config)
        self._last_event_ns = 0

    async def run(self):
        """Main daemon loop."""
        await self._load_recent_events()
        while self._running:
            new_events = await self._poll_new_events()
            for event in new_events:
                # Embed
                embedding = await self.embedder.embed_event(event)
                await self.db.store_embedding(event.event_id, embedding)

                # Search for cross-platform similar
                similar = await self.similarity.cross_platform_similar(event)

                # Score coherence
                moments = self.scorer.score(event, similar)

                # Alert on significant coherence
                for moment in moments:
                    if moment.confidence > 0.70:
                        await self.alerts.notify(moment)
                        await self.db.store_moment(moment)

            await asyncio.sleep(self.config.poll_interval_s)
```

### CLI

```bash
# Start daemon (foreground)
python3 -m coherence_engine start

# Start daemon (background)
python3 -m coherence_engine daemon

# Stop daemon
python3 -m coherence_engine stop

# Status
python3 -m coherence_engine status

# Retroactive analysis (run on historical ChatGPT data)
python3 -m coherence_engine retroactive --since 2024-01-01

# One-shot analysis between two events
python3 -m coherence_engine analyze --event-a <id> --event-b <id>

# Dashboard (TUI)
python3 -m coherence_engine dashboard
```

---

## Part 4: Alert System

### alerts.py

```python
class AlertSystem:
    """
    Multi-channel alerting for coherence events.

    Channels:
    1. Log file (always) — ~/.ucw/logs/coherence.log
    2. Desktop notification (macOS) — osascript
    3. Database record (always) — coherence_moments table
    4. Webhook (optional) — POST to configured URL
    5. Sound (optional) — play alert sound on detection
    """

    async def notify(self, moment: CoherenceMoment):
        """Send alert through all enabled channels."""

        # Always log
        self._log_moment(moment)

        # Always store
        await self._store_moment(moment)

        # Desktop notification for high confidence
        if moment.confidence > 0.80:
            self._desktop_notify(moment)

        # Webhook if configured
        if self.config.webhook_url:
            await self._webhook_notify(moment)

    def _desktop_notify(self, moment):
        """macOS desktop notification via osascript."""
        title = f"UCW Coherence: {moment.coherence_type}"
        body = (
            f"{moment.platform_a} ↔ {moment.platform_b}\n"
            f"Confidence: {moment.confidence:.0%}\n"
            f"{moment.description[:100]}"
        )
        os.system(
            f'osascript -e \'display notification "{body}" '
            f'with title "{title}"\''
        )
```

---

## Part 5: TUI Dashboard

### dashboard.py

Simple terminal dashboard showing coherence status:

```
╔══════════════════════════════════════════════════════════════╗
║              UCW COHERENCE DASHBOARD                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PLATFORMS ACTIVE:                                           ║
║    Claude Desktop  ● LIVE   (142 events today)              ║
║    ChatGPT         ● LIVE   (23 events today)               ║
║    Grok/X          ○ IDLE   (last: 2h ago)                  ║
║                                                              ║
║  COHERENCE MOMENTS (last 24h):                              ║
║    Signature matches:    3                                   ║
║    Semantic echoes:      7                                   ║
║    Synchronicities:      1  ★                               ║
║                                                              ║
║  LATEST MOMENT:                                              ║
║    Type: semantic_echo                                       ║
║    Claude ↔ ChatGPT | Confidence: 87%                       ║
║    Topic: "cognitive architecture unification"              ║
║    Time: 14:23 → 14:31 (8 min gap)                         ║
║                                                              ║
║  EMERGENCE SIGNALS:                                          ║
║    ████████████░░░░░░░░ 62% (rising)                        ║
║    Concept clusters: 4 active                                ║
║    Meta-cognitive events: 12 in last hour                   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  [q]uit  [r]efresh  [h]istory  [d]etail                    ║
╚══════════════════════════════════════════════════════════════╝
```

Use `rich` library for the TUI (or plain ANSI if rich unavailable).

---

## Part 6: Retroactive Analysis

### retroactive.py

Run coherence detection on the 30,712 existing ChatGPT findings:

```python
class RetroactiveAnalyzer:
    """
    Analyze historical data for coherence patterns.

    Process:
    1. Load all events from cognitive DB (or agent-core archives)
    2. Embed all events in batches
    3. Build similarity index
    4. Run all-pairs cross-platform comparison
    5. Generate coherence report
    """

    async def analyze(self, since: datetime = None):
        # 1. Load events
        events = await self.db.get_all_events(since=since)

        # 2. Batch embed
        texts = [self.embedder._build_embed_text(e) for e in events]
        embeddings = await self.embedder.embed_batch(texts)

        # 3. Store embeddings
        for event, emb in zip(events, embeddings):
            await self.db.store_embedding(event.event_id, emb)

        # 4. Cross-platform search
        moments = []
        for event in events:
            similar = await self.similarity.cross_platform_similar(event)
            event_moments = self.scorer.score(event, similar)
            moments.extend(event_moments)

        # 5. Report
        return CoherenceReport(
            total_events=len(events),
            moments_found=len(moments),
            by_type=Counter(m.coherence_type for m in moments),
            top_moments=sorted(moments, key=lambda m: m.confidence, reverse=True)[:20],
        )
```

### The Founding Moment Validation

After retroactive analysis, manually verify that the engine detects the 2026-02-06 synchronicity:

```bash
# Run retroactive analysis
python3 -m coherence_engine retroactive --since 2026-02-06

# Expected output:
# SYNCHRONICITY DETECTED
# Claude ↔ ChatGPT | Confidence: 92%
# "Can you unify yourself before you unify the infrastructure?"
# Temporal: 2 min gap | Semantic: 0.91 | Meta-cognitive: YES
```

---

## Dependencies

### Required
- `sentence-transformers` — Already installed (SBERT)
- `numpy` — For cosine similarity
- `rich` — TUI dashboard (pip install rich)

### Optional
- `cohere` — Cohere API for higher-quality embeddings
- PostgreSQL + pgvector — For HNSW similarity index at scale

### Already Available
- `mcp_raw/coherence.py` — Basic coherence engine (Session B built this)
- `mcp_raw/ucw_bridge.py` — UCW layer extraction + signature generation
- `mcp_raw/db.py` / `mcp_raw/database.py` — Storage backends
- `chatgpt_quality_scorer.py` — Quality scoring for imported data

---

## Configuration

```python
# ~/researchgravity/coherence_engine/config.py

ENGINE_CONFIG = {
    "embedding": {
        "model": "all-MiniLM-L6-v2",       # SBERT default
        "dimensions": 384,
        "cohere_model": "embed-v4.0",       # Optional upgrade
        "cohere_dimensions": 1024,
        "batch_size": 256,                   # Events per batch
    },
    "detection": {
        "signature_confidence": 0.95,
        "semantic_threshold": 0.85,
        "semantic_confidence_factor": 0.9,
        "synchronicity_threshold": 0.70,
        "time_window_minutes": 30,
    },
    "daemon": {
        "poll_interval_s": 10,
        "recent_window_hours": 24,
        "max_candidates_per_event": 50,
    },
    "alerts": {
        "desktop_notifications": True,
        "min_confidence": 0.70,
        "sound": False,
        "webhook_url": None,
        "log_file": "~/.ucw/logs/coherence.log",
    },
    "dashboard": {
        "refresh_interval_s": 5,
        "history_hours": 24,
    },
}
```

---

## Implementation Priority

| Priority | File | Lines (est) | What |
|----------|------|------------|------|
| **P0** ✅ | `config.py` | 71 | Configuration (calibrated thresholds) |
| **P0** ✅ | `embeddings.py` | 172 | SBERT + Cohere embedding pipeline |
| **P0** ✅ | `similarity.py` | 202 | pgvector HNSW + brute-force fallback |
| **P0** ✅ | `detector.py` | 279 | 3-layer coherence detection |
| **P0** ✅ | `scorer.py` | 166 | Multi-signal confidence scoring |
| **P0** ✅ | `alerts.py` | 124 | Alert system (log + desktop + DB) |
| **P0** ✅ | `daemon.py` | 250 | Background monitoring daemon (pool-sharing) |
| **P0** ✅ | `__init__.py` + `__main__.py` | 174 | Package + CLI (6 commands) |
| **P1** ✅ | `retroactive.py` | 280 | Historical analysis + founding moment validation |
| **P1** ✅ | `dashboard.py` | 230 | TUI dashboard (ANSI + rich) |
| **BONUS** ✅ | `mcp_raw/tools/coherence_tools.py` | 490 | 4 MCP tools wired to engine |

**Total: ~1,200 lines across 10 files.**

Session B should build all P0 files (8 files, ~930 lines), then P1.

---

## Race Condition Prevention

**Session B owns ALL files in `~/researchgravity/coherence_engine/`.**
**Session A does NOT touch the `coherence_engine/` directory.**

Session A is working on:
- PostgreSQL setup
- Claude Desktop integration
- Data migration into cognitive DB
- `mcp_raw/` maintenance only

---

## Success Criteria

1. **Embedding pipeline** embeds events at >100/sec locally (SBERT)
2. **Similarity search** finds cross-platform matches with >0.85 cosine similarity
3. **Coherence detection** identifies signature, semantic, and synchronicity patterns
4. **Founding moment test** — detects the 2026-02-06 synchronicity from historical data
5. **Daemon** runs continuously without memory leaks
6. **Alerts** fire desktop notifications within 30 seconds of coherence detection
7. **Dashboard** shows live coherence status with <5s refresh

---

## Quick Start for Session B

```bash
cd ~/researchgravity

# Read existing code for reference
cat mcp_raw/coherence.py           # Basic coherence engine (already built)
cat mcp_raw/ucw_bridge.py          # UCW bridge + coherence_signature()
cat mcp_raw/db.py                  # SQLite storage interface
cat mcp_raw/database.py            # PostgreSQL storage interface

# Check sentence-transformers is working
python3 -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print(m.encode('test').shape)"

# Build in order:
mkdir -p coherence_engine
# 1. coherence_engine/config.py
# 2. coherence_engine/embeddings.py
# 3. coherence_engine/similarity.py
# 4. coherence_engine/detector.py
# 5. coherence_engine/scorer.py
# 6. coherence_engine/alerts.py
# 7. coherence_engine/daemon.py
# 8. coherence_engine/__init__.py + __main__.py
# 9. coherence_engine/retroactive.py (P1)
# 10. coherence_engine/dashboard.py (P1)
```
