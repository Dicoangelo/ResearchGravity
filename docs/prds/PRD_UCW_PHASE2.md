# UCW Phase 2: Cognitive Evolution

**Version:** 2.0.0
**Date:** 2026-02-07
**Status:** READY FOR IMPLEMENTATION
**Predecessor:** PRD_UCW_RAW_MCP.md (Phase 1 — All 13 phases complete)

---

## Executive Summary

Phase 1 built the foundation: 140,841 events across 5 platforms, 130,780 embeddings, 177 coherence moments, live daemons. Phase 2 evolves UCW from infrastructure into intelligence — fixing critical inefficiencies, upgrading core algorithms, and adding the cognitive layers that transform raw data into emergent understanding.

**Three pillars:**
1. **Fix what's broken** — Dedup, redundant scans, wasted indexes (immediate, 1-2 days)
2. **Upgrade the engine** — Embeddings, hybrid search, real-time processing (1-2 weeks)
3. **Build cognitive intelligence** — Knowledge graph, temporal coherence v2, insight resurfacing (2-4 weeks)

### Key Metrics (Current → Target)

| Metric | Current | Target | How |
|--------|---------|--------|-----|
| Coherence latency | 5-7 min | <1 sec | PG LISTEN/NOTIFY + resident daemon |
| Embedding accuracy | 56% MTEB | 81% MTEB | nomic-embed-text-v1.5 |
| Search quality | Pure semantic | Hybrid semantic+BM25 | RRF fusion |
| Duplicate moments | 39.2% | 0% | Deterministic moment_id |
| Wasted disk | 531 MB | 0 MB | Drop duplicate/unused indexes |
| Scan redundancy | 100% rescan | 0% rescan | Incremental scan tracking |
| Embedding throughput | 10K individual calls | 40 batches of 256 | Batch pipeline |
| Log noise | 3.6 MB progress bars | Clean structured logs | Suppress tqdm |

---

## Architectural Principles

### 1. Sovereignty Above All

Meta acquired Limitless (formerly Rewind AI) in December 2025 and killed the product on December 19, 2025. Every user's cognitive memory infrastructure was destroyed overnight. Stephen's insight is proven: **"By the time people want their AI data, they'll be depending on platforms."**

Every architectural decision must preserve:
- **Local-first execution** — No cloud dependency for core function
- **Data portability** — Export in standard formats (JSON-LD, Parquet, SQLite)
- **No vendor lock-in** — Use local models (nomic, SBERT) over API-dependent ones (Cohere)
- **Encrypted at rest** — FileVault + optional column-level encryption for deep_work

### 2. Cognitive Fidelity Over Speed

UCW is not a search engine. It's a cognitive memory system. Design decisions favor:
- **Precision over recall** — Better to miss a weak coherence moment than surface a false one
- **Multi-scale detection** — Human thinking operates at multiple time scales (2min to 7d)
- **Causal reasoning** — Not just "what's similar?" but "what influenced what?"
- **Active memory** — Don't just store; resurface, consolidate, strengthen

### 3. PostgreSQL as Universal Engine

No Kafka. No Neo4j. No Redis. PostgreSQL handles:
- Relational storage (cognitive_events)
- Vector search (pgvector HNSW)
- Full-text search (tsvector + GIN)
- Knowledge graph (recursive CTEs)
- Event streaming (LISTEN/NOTIFY)
- Job scheduling (pg_cron or application-level)

One database. One backup. One failure domain. Sovereign.

### 4. Incremental Over Batch

Every operation must be incremental:
- Capture: Only new events (watermarks already work)
- Embedding: Only unembedded events (content_hash dedup)
- Coherence: Only unscanned events (new `coherence_scanned_at` column)
- Graph: Only new entities/edges (temporal versioning)

---

## Phase 2A: Critical Fixes (Days 1-2)

> Fix the broken things. Recover 531 MB of disk, eliminate 100% redundant work.

### 2A.1 — Drop Duplicate HNSW Index

**Problem:** Two identical HNSW indexes on `embedding_cache.embedding`. `idx_ec_embedding` (255 MB, 0 scans) is a duplicate of `idx_embedding_cache_hnsw` (255 MB, 159,420 scans).

**Fix:**
```sql
DROP INDEX idx_ec_embedding;
```

**Impact:** 255 MB recovered, INSERT overhead reduced.

---

### 2A.2 — Add Missing source_event_id Index

**Problem:** The daemon's core JOIN (`embedding_cache ec ON ec.source_event_id = ce.event_id`) triggers sequential scans of 130K rows. No index on `source_event_id`.

**Evidence:** 265 million rows read in seq scans on `embedding_cache`.

**Fix:**
```sql
CREATE INDEX idx_ec_source_event ON embedding_cache(source_event_id);
```

**Impact:** JOIN operations go from O(n) to O(log n). Estimated 10-100x speedup on daemon queries.

---

### 2A.3 — Drop Unused Indexes (276 MB)

**Problem:** Multiple indexes with 0 scans since database creation.

**Fix:**
```sql
DROP INDEX idx_ec_embedding;         -- 255 MB, 0 scans (duplicate HNSW)
DROP INDEX idx_ce_light_gin;         -- 14 MB, 0 scans
DROP INDEX idx_ce_instinct_gin;      -- 5.3 MB, 0 scans
DROP INDEX idx_ce_method;            -- 1.2 MB, 0 scans
```

**Keep but monitor:** `idx_ce_direction` (13 scans), `idx_ce_turn` (10 scans), `idx_ce_mode` (1 scan).

**Impact:** 276 MB recovered. INSERT throughput improves (fewer indexes to maintain).

---

### 2A.4 — Fix Coherence Moment Deduplication

**Problem:** `moment_id` is a random UUID (`uuid.uuid4().hex[:12]`), so `ON CONFLICT (moment_id) DO NOTHING` never fires. 39.2% of 199 moments are duplicates (78 rows). `signature` column is NULL for all rows.

**Root cause:** `scorer.py:108` generates non-deterministic IDs. `store_moment()` at line 149 doesn't include `signature` in the INSERT.

**Fix (scorer.py):**
```python
# Replace random UUID with deterministic hash
import hashlib

pair_key = "|".join(sorted(moment.event_ids)) + "|" + moment.coherence_type
moment_id = f"cm-{hashlib.sha256(pair_key.encode()).hexdigest()[:16]}"

# Generate signature for the moment
signature = hashlib.sha256(
    f"{pair_key}|{moment.confidence:.4f}".encode()
).hexdigest()
```

**Fix (store_moment INSERT):** Add `signature` to the INSERT statement.

**Fix (cleanup):**
```sql
-- Remove existing duplicates, keeping highest confidence
DELETE FROM coherence_moments a USING coherence_moments b
WHERE a.moment_id > b.moment_id
  AND a.event_ids = b.event_ids
  AND a.coherence_type = b.coherence_type;

-- Add unique constraint to prevent future duplicates
CREATE UNIQUE INDEX idx_cm_event_pair_type
ON coherence_moments (
  (event_ids[1]), (event_ids[2]), coherence_type
);
```

**Impact:** Moment count drops from 199 → ~121 (true unique moments). Future runs don't re-insert.

---

### 2A.5 — Make Coherence Scans Incremental

**Problem:** Daemon's `oneshot` mode always fetches the first 10,000 events by timestamp with no tracking of what's been scanned. Each run does 100% redundant work — 37 minutes processing the same 10K events. The remaining 120K+ events are NEVER scanned.

**Fix:**
```sql
ALTER TABLE cognitive_events ADD COLUMN coherence_scanned_at TIMESTAMPTZ;

-- Partial index for fast lookup of unscanned events
CREATE INDEX idx_ce_unscanned
ON cognitive_events(timestamp_ns ASC)
WHERE coherence_scanned_at IS NULL;
```

**Modify daemon.py query:**
```sql
SELECT ce.* FROM cognitive_events ce
JOIN embedding_cache ec ON ec.source_event_id = ce.event_id
WHERE ce.coherence_scanned_at IS NULL
ORDER BY ce.timestamp_ns ASC
LIMIT 10000
```

**After processing each event:**
```sql
UPDATE cognitive_events SET coherence_scanned_at = NOW()
WHERE event_id = $1
```

**Impact:** Each run only processes NEW events. Full corpus gets scanned over time. 37 minutes of wasted work eliminated per run.

---

### 2A.6 — Batch Embedding in Daemon

**Problem:** Daemon calls `embed_single()` 10,000 times (10K individual GPU kernel launches). Also re-embeds events that already have cached embeddings (they were selected via JOIN on embedding_cache).

**Fix (daemon.py):**
```python
async def _poll_and_process(self):
    events = await self._fetch_unscanned_events()

    # Batch embed all at once (instead of 10K individual calls)
    texts = [event_to_text(e) for e in events]
    embeddings = embed_texts(texts, batch_size=256)  # ~40 batches vs 10K calls

    # Then do similarity search + scoring per event
    for event, embedding in zip(events, embeddings):
        similar = await self._similarity.find_similar(embedding, ...)
        moments = await self._scorer.score(event, embedding, similar)
        ...
```

**Impact:** Embedding phase goes from ~30 min to ~2-3 min (10-15x speedup).

---

### 2A.7 — Suppress Progress Bar Log Spam

**Problem:** `embed_single()` in `mcp_raw/embeddings.py` doesn't set `show_progress_bar=False`. Result: 3.6 MB of tqdm progress bars in `coherence.err` per daemon run.

**Fix (embeddings.py):**
```python
def embed_single(text: str) -> List[float]:
    model = _get_model()
    return model.encode(text, show_progress_bar=False).tolist()
```

**Fix (coherence_engine/__main__.py):**
```python
import os
os.environ["TQDM_DISABLE"] = "1"

# Also suppress sentence_transformers noise
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
```

**Impact:** Clean logs. Real errors visible.

---

### 2A.8 — Switch Daemon to Resident Mode

**Problem:** LaunchAgent runs `oneshot` every 15 min. Each invocation cold-starts Python, imports sentence_transformers (~3s), loads SBERT model to MPS GPU (~4s). Total: 7s wasted per run.

**Fix:** Change LaunchAgent from interval-triggered oneshot to keep-alive resident:

```xml
<!-- com.ucw.coherence.plist -->
<key>KeepAlive</key>
<true/>
<key>ProgramArguments</key>
<array>
    <string>/opt/homebrew/bin/python3</string>
    <string>-m</string>
    <string>coherence_engine</string>
    <string>start</string>  <!-- poll mode, stays resident -->
</array>
```

The `start` mode already exists in `daemon.py` — polls every 10s, keeps model loaded.

**Impact:** Zero cold starts. Model stays on GPU. Sub-second response to new events.

---

## Phase 2B: Engine Upgrades (Week 1-2)

> Upgrade the core algorithms. Better embeddings, hybrid search, real-time processing.

### 2B.1 — Embedding Model Upgrade: nomic-embed-text-v1.5

**Why:** all-MiniLM-L6-v2 scores 56% on MTEB benchmarks. nomic-embed-text-v1.5 scores 81% — a 45% relative improvement in retrieval accuracy. It runs locally (sovereignty), supports Matryoshka dimensions (256d/512d/768d), and uses task prefixes for better encoding.

**Architecture:**

```
New events → nomic-embed-text-v1.5 (768d, Matryoshka)
                ├── Store full 768d for coherence detection
                └── Store 256d truncation for fast pre-filtering

Old events → Keep existing MiniLM 384d embeddings
              └── Re-embed in background batches over time
```

**Schema changes:**
```sql
-- New column for high-dimensional embeddings
ALTER TABLE embedding_cache ADD COLUMN embedding_768 vector(768);

-- HNSW index on new embeddings
CREATE INDEX idx_ec_embedding_768 ON embedding_cache
USING hnsw (embedding_768 vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Fast pre-filter index at 256d
ALTER TABLE embedding_cache ADD COLUMN embedding_256 vector(256);
CREATE INDEX idx_ec_embedding_256 ON embedding_cache
USING hnsw (embedding_256 vector_cosine_ops)
WITH (m = 8, ef_construction = 64);
```

**Embedding code:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Task-prefixed encoding (critical for nomic quality)
def embed_for_search(query: str) -> List[float]:
    return model.encode(f"search_query: {query}", ...).tolist()

def embed_for_storage(document: str) -> List[float]:
    return model.encode(f"search_document: {document}", ...).tolist()

# Matryoshka truncation
def truncate_embedding(emb: List[float], dim: int) -> List[float]:
    return emb[:dim]  # Matryoshka embeddings are truncation-safe
```

**Migration strategy:**
1. Install nomic model locally (~250 MB)
2. Dual-write new events with both models (old 384d + new 768d)
3. Background batch: re-embed existing 130K events (~30 min with batch mode on MPS)
4. Switch coherence engine to 768d embeddings
5. After validation, drop old 384d column and index

**Trade-offs:**
- Disk: 768d vectors are 2x larger than 384d (~500 MB vs ~250 MB for 130K)
- Memory: Nomic model is ~250 MB vs ~80 MB for MiniLM
- Quality: 81% vs 56% MTEB — the quality gain far outweighs the resource cost

**Sources:**
- [nomic-embed-text-v1.5 (HuggingFace)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Best Open-Source Embedding Models 2026 (BentoML)](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
- [Don't use all-MiniLM-L6-v2 (HN)](https://news.ycombinator.com/item?id=46081800)

---

### 2B.2 — Hybrid Search: Semantic + BM25 with RRF

**Why:** Anthropic's contextual retrieval research showed hybrid search reduces retrieval failure by 49-67% over pure semantic. For cognitive content where specific terms matter (tool names, error messages, project names, "sovereignty", "UCW"), keyword matching catches what embeddings miss.

**Architecture:**

```
Query → ┬── Semantic Search (pgvector cosine) ──→ rank_semantic
        └── BM25 Search (tsvector ts_rank)    ──→ rank_bm25
                                                      │
                                             RRF Fusion (k=60)
                                                      │
                                                 Final Results
```

**Schema changes:**
```sql
-- Add full-text search column
ALTER TABLE cognitive_events ADD COLUMN content_tsv tsvector;

-- Populate from existing content
UPDATE cognitive_events
SET content_tsv = to_tsvector('english',
    COALESCE(data_content, '') || ' ' ||
    COALESCE((light_layer->>'topic')::text, '') || ' ' ||
    COALESCE((light_layer->>'intent')::text, '')
);

-- GIN index for fast FTS
CREATE INDEX idx_ce_content_fts ON cognitive_events USING GIN(content_tsv);

-- Auto-update trigger
CREATE FUNCTION update_content_tsv() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english',
        COALESCE(NEW.data_content, '') || ' ' ||
        COALESCE((NEW.light_layer->>'topic')::text, '') || ' ' ||
        COALESCE((NEW.light_layer->>'intent')::text, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_content_tsv
BEFORE INSERT OR UPDATE ON cognitive_events
FOR EACH ROW EXECUTE FUNCTION update_content_tsv();
```

**RRF Fusion (Python):**
```python
def reciprocal_rank_fusion(semantic_results, bm25_results, k=60):
    """Combine semantic and BM25 rankings using RRF."""
    scores = {}
    for rank, result in enumerate(semantic_results):
        scores[result.event_id] = scores.get(result.event_id, 0) + 1 / (k + rank + 1)
    for rank, result in enumerate(bm25_results):
        scores[result.event_id] = scores.get(result.event_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Sources:**
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Hybrid Search in PostgreSQL (ParadeDB)](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual)

---

### 2B.3 — Real-Time Coherence via PostgreSQL LISTEN/NOTIFY

**Why:** Current architecture polls every 5-15 minutes. Kafka is overkill for a single-user system. PostgreSQL has built-in pub/sub that gives <1s latency with zero additional infrastructure.

**Architecture:**

```
INSERT INTO cognitive_events → trigger → pg_notify('new_event', event_id)
                                              │
                                    LISTEN new_event
                                              │
                                    Coherence Daemon (resident)
                                              │
                                    Embed → Search → Score → Alert
```

**Database trigger:**
```sql
CREATE OR REPLACE FUNCTION notify_new_event() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('new_event',
        json_build_object(
            'event_id', NEW.event_id,
            'platform', NEW.platform,
            'session_id', NEW.session_id
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_new_event
    AFTER INSERT ON cognitive_events
    FOR EACH ROW EXECUTE FUNCTION notify_new_event();
```

**Daemon listener (Python):**
```python
async def listen_for_events(pool, on_event):
    async with pool.acquire() as conn:
        await conn.add_listener('new_event', lambda *args: on_event(args))
        # Keep connection alive
        while True:
            await asyncio.sleep(3600)  # Heartbeat
```

**Impact:** Coherence detection latency drops from 5-15 min to <1 sec. The daemon processes events as they arrive instead of polling.

**Fallback:** Keep periodic scan as backup (every 5 min) in case NOTIFY is missed.

---

### 2B.4 — Contextual Embeddings (Anthropic Pattern)

**Why:** Anthropic's research showed that prepending context to chunks before embedding reduces retrieval failure by 67%. Currently, `build_embed_text()` embeds bare intent+topic+content. Adding session context dramatically improves cross-platform coherence detection.

**Current:**
```
"explore: multi-agent | orchestration patterns"
```

**Contextual:**
```
"During a deep_work session on Claude Desktop about OS-App architecture,
user explored multi-agent orchestration patterns for the Antigravity ecosystem.
Cognitive mode: deep_work. Platform: claude-desktop.
explore: multi-agent | orchestration patterns"
```

**Implementation (embeddings.py):**
```python
def build_embed_text_contextual(event: dict) -> str:
    """Build context-enriched embedding text."""
    parts = []

    # Session context (stable, won't change)
    mode = event.get("cognitive_mode", "unknown")
    platform = event.get("platform", "unknown")
    session_topic = event.get("session_topic", "")

    if session_topic:
        parts.append(f"In a {mode} session about {session_topic} on {platform}.")

    # Light layer
    intent = event.get("light_layer", {}).get("intent", "")
    topic = event.get("light_layer", {}).get("topic", "")
    summary = event.get("light_layer", {}).get("summary", "")

    if intent and topic:
        parts.append(f"{intent}: {topic}")
    if summary:
        parts.append(summary[:300])

    # Instinct signals (coherence potential)
    instinct = event.get("instinct_layer", {})
    coherence_potential = instinct.get("coherence_potential", 0)
    if coherence_potential > 0.7:
        parts.append(f"[high coherence potential: {coherence_potential:.2f}]")

    # Concepts
    concepts = event.get("light_layer", {}).get("concepts", [])
    if concepts:
        parts.append(" ".join(concepts[:5]))

    return " | ".join(filter(None, parts))
```

**Impact:** +5-10% coherence detection precision, especially for cross-platform matches where the same concept is discussed in different contexts.

---

### 2B.5 — HNSW Index Tuning

**Current:** `m=16, ef_construction=64` — conservative defaults.

**Optimized for 130K-500K vectors:**
```sql
-- Rebuild with better parameters
DROP INDEX idx_embedding_cache_hnsw;
CREATE INDEX idx_embedding_cache_hnsw ON embedding_cache
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);  -- Higher build quality

-- Set query-time search depth
ALTER SYSTEM SET hnsw.ef_search = 100;  -- Default 40 is too low
SELECT pg_reload_conf();

-- Enable iterative scan (pgvector 0.8.0) for filtered queries
SET hnsw.iterative_scan = on;
```

**Impact:** 20-30% better recall on similarity searches. Iterative scan prevents overfiltering.

**Sources:**
- [pgvector HNSW Deep Dive (AWS)](https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/)
- [pgvector 0.8.0 Features](https://www.postgresql.org/about/news/pgvector-080-released-2952/)

---

## Phase 2C: Cognitive Intelligence (Week 2-4)

> Build the intelligence layers. Knowledge graph, temporal coherence v2, active memory.

### 2C.1 — Multi-Scale Temporal Coherence

**Why:** Fixed 5-minute buckets arbitrarily cut temporal patterns. Human thinking operates at multiple timescales. A conversation about "sovereignty" on ChatGPT this morning and a deep-work session on UCW architecture tonight are temporally coherent at the daily scale but invisible at the 5-minute scale.

**Architecture:**

```
Timescale Windows:
├── 2 min   — Real-time synchronicity (founding moment pattern)
├── 10 min  — Active parallel thinking (same problem, different platforms)
├── 1 hour  — Session-level coherence (same topic, different angles)
├── 4 hours — Work-session themes (morning → afternoon arc)
├── 24 hours — Daily coherence (today's cognitive thread)
└── 7 days  — Weekly intellectual patterns (recurring themes)
```

**Scoring:**
```python
WINDOW_SCALES = [
    {"name": "realtime",   "seconds": 120,    "weight": 0.30},
    {"name": "parallel",   "seconds": 600,    "weight": 0.25},
    {"name": "session",    "seconds": 3600,   "weight": 0.20},
    {"name": "work_block", "seconds": 14400,  "weight": 0.10},
    {"name": "daily",      "seconds": 86400,  "weight": 0.10},
    {"name": "weekly",     "seconds": 604800, "weight": 0.05},
]

def multi_scale_coherence(event_a, event_b, similarity):
    """Score coherence across multiple timescales."""
    time_delta = abs(event_a.timestamp_ns - event_b.timestamp_ns) / 1e9

    for window in WINDOW_SCALES:
        if time_delta <= window["seconds"]:
            # Exponential decay within window
            decay = math.exp(-time_delta / (window["seconds"] * 0.5))
            return similarity * decay * window["weight"]

    return 0.0  # Beyond 7 days
```

**New coherence types:**
- `realtime_sync` — Same concept within 2 minutes across platforms
- `parallel_work` — Same problem explored differently within 10 minutes
- `session_echo` — Topic revisited within 1 hour on different platform
- `daily_arc` — Coherent thread across today's sessions
- `weekly_thread` — Recurring theme across the week

---

### 2C.2 — Knowledge Graph (PostgreSQL-Native)

**Why:** Pure vector search answers "what's similar?" but not "what's connected?" or "what caused what?" Research from MAGMA (arXiv:2601.03236) and Graphiti (arXiv:2501.13956) shows that combining vector search with graph traversal yields 25-39% improvements in multi-hop retrieval.

**Architecture:** No separate graph database. PostgreSQL recursive CTEs handle multi-hop queries efficiently at this scale.

**Schema:**
```sql
CREATE TABLE cognitive_entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,  -- concept, tool, project, person, error, paper
    name TEXT NOT NULL,
    aliases TEXT[],             -- alternative names/spellings
    first_seen_ns BIGINT,
    last_seen_ns BIGINT,
    mention_count INT DEFAULT 1,
    platform_count INT DEFAULT 1,
    platforms TEXT[],
    embedding vector(768),     -- entity-level embedding
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE cognitive_edges (
    edge_id TEXT PRIMARY KEY,
    source_entity TEXT REFERENCES cognitive_entities(entity_id),
    target_entity TEXT REFERENCES cognitive_entities(entity_id),
    relation_type TEXT NOT NULL, -- co_occurs, causes, precedes, evolved_into, contradicts
    weight REAL DEFAULT 1.0,
    evidence_count INT DEFAULT 1,
    first_seen_ns BIGINT,
    last_seen_ns BIGINT,
    t_valid_from BIGINT,       -- bi-temporal: when relationship became true
    t_valid_to BIGINT,         -- bi-temporal: when relationship ended (NULL = still true)
    source_events TEXT[],      -- event_ids that evidence this edge
    metadata JSONB DEFAULT '{}'
);

-- Indexes
CREATE INDEX idx_ce_entity_type ON cognitive_entities(entity_type);
CREATE INDEX idx_ce_name_trgm ON cognitive_entities USING gin(name gin_trgm_ops);
CREATE INDEX idx_edges_source ON cognitive_edges(source_entity);
CREATE INDEX idx_edges_target ON cognitive_edges(target_entity);
CREATE INDEX idx_edges_relation ON cognitive_edges(relation_type);
CREATE INDEX idx_edges_weight ON cognitive_edges(weight DESC);
```

**Entity Extraction Pipeline:**

```
Event → ┬── spaCy NER (tools, technologies, concepts) ── lightweight, all events
        ├── Regex patterns (arXiv IDs, GitHub repos, project names) ── all events
        └── LLM extraction (deep insights, novel concepts) ── deep_work only
              │
              ▼
        Entity Resolution (fuzzy match + alias lookup)
              │
              ▼
        Edge Construction (co-occurrence within session)
              │
              ▼
        cognitive_entities + cognitive_edges
```

**Spreading Activation for Retrieval:**
```python
async def spreading_activation(start_entity: str, pool, depth=3, decay=0.7):
    """Activate related concepts through the knowledge graph."""
    activated = {start_entity: 1.0}
    frontier = [start_entity]

    for hop in range(depth):
        next_frontier = []
        for entity in frontier:
            edges = await pool.fetch("""
                SELECT target_entity, weight FROM cognitive_edges
                WHERE source_entity = $1 AND weight > 0.1
                UNION
                SELECT source_entity, weight FROM cognitive_edges
                WHERE target_entity = $1 AND weight > 0.1
            """, entity)

            for edge in edges:
                neighbor = edge["target_entity"]
                activation = activated[entity] * edge["weight"] * (decay ** hop)
                if activation > 0.05:  # Threshold
                    if neighbor not in activated or activated[neighbor] < activation:
                        activated[neighbor] = activation
                        next_frontier.append(neighbor)

        frontier = next_frontier

    return activated  # {entity_id: activation_strength}
```

**Use case:** Search for "sovereignty" → activates "UCW", "Limitless", "data portability", "local-first", "GDPR", "encryption" → returns events connected to ANY activated concept, weighted by activation strength.

**Sources:**
- [MAGMA: Multi-Graph Agentic Memory (arXiv:2601.03236)](https://arxiv.org/abs/2601.03236)
- [Graphiti: Temporal Knowledge Graph (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [Spreading Activation for KG RAG (arXiv:2512.15922)](https://arxiv.org/abs/2512.15922)

---

### 2C.3 — FSRS Insight Resurfacing

**Why:** Human memory strengthens through spaced retrieval. UCW should not just store — it should proactively resurface insights at optimal intervals. The Free Spaced Repetition Scheduler (FSRS) is the SOTA algorithm, trained on 700M reviews from 20K users.

**Architecture:**
```
Coherence Moment → FSRS Parameters (Retrievability, Stability, Difficulty)
                        │
                 Scheduled resurfacing at optimal intervals
                        │
                 macOS notification: "3 weeks ago, your thinking about X
                 converged across ChatGPT and Claude Desktop..."
                        │
                 User reviews → Update FSRS parameters
                        │
                 Strong insights resurface less (high stability)
                 Weak but important insights resurface more
```

**Schema:**
```sql
CREATE TABLE insight_schedule (
    insight_id TEXT PRIMARY KEY,
    moment_id TEXT REFERENCES coherence_moments(moment_id),
    retrievability REAL DEFAULT 1.0,  -- probability of recall (0-1)
    stability REAL DEFAULT 1.0,       -- days until R drops to 0.9
    difficulty REAL DEFAULT 0.5,      -- inherent complexity (0-1)
    last_review TIMESTAMPTZ,
    next_review TIMESTAMPTZ,
    review_count INT DEFAULT 0,
    rating_history JSONB DEFAULT '[]' -- [{rating: 1-4, timestamp: ...}]
);
```

**FSRS core (simplified):**
```python
def schedule_next_review(stability, difficulty, rating):
    """FSRS scheduling algorithm."""
    # Rating: 1=forgot, 2=hard, 3=good, 4=easy
    if rating == 1:
        new_stability = stability * 0.2  # Reset
    else:
        modifier = 1.0 + (rating - 2) * 0.3
        new_stability = stability * modifier * (1.1 - difficulty * 0.2)

    # Next review when retrievability drops to 0.9
    interval_days = new_stability * 0.9
    return new_stability, interval_days
```

**Trade-offs:** Full FSRS is more complex (17 parameters). Start with simplified version, calibrate on real usage data, then upgrade to full FSRS if needed.

**Sources:**
- [FSRS Algorithm (GitHub)](https://github.com/open-spaced-repetition/fsrs4anki)
- [FSRS vs SM-2 Complete Guide](https://memoforge.app/blog/fsrs-vs-sm2-anki-algorithm-guide-2025/)

---

### 2C.4 — Memory Consolidation Daemon

**Why:** Human memory consolidates during sleep — replaying, reconnecting, strengthening important memories. UCW should do the same. A nightly consolidation pass connects the day's events into the knowledge graph, identifies emerging themes, and schedules insights for resurfacing.

**Architecture:**
```
Nightly Consolidation (2 AM, when user typically deep-works anyway)
    │
    ├── Entity Extraction: Process today's events → new entities/edges
    ├── Edge Strengthening: Increment weights for today's co-occurrences
    ├── Arc Detection: Identify daily coherence arcs across platforms
    ├── Insight Scheduling: FSRS review queue for tomorrow
    ├── Materialized View Refresh: Update analytics views
    └── Embedding Migration: Re-embed batch with nomic if MiniLM still present
```

**LaunchAgent:**
```xml
<!-- com.ucw.consolidation.plist -->
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>2</integer>
    <key>Minute</key>
    <integer>0</integer>
</dict>
```

---

### 2C.5 — Permutation-Based Significance Testing

**Why:** Current confidence scoring uses heuristics (`min(1.0, len(platforms) * 0.3 + ...)`). As data scales, this generates false positives. Permutation testing provides a principled p-value for every coherence moment.

**Algorithm:**
```python
async def test_significance(event_a, event_b, real_score, n_permutations=200):
    """Test whether coherence score is significant via permutation."""
    null_scores = []
    for _ in range(n_permutations):
        # Shuffle timestamps randomly
        shuffled_b = random_event_same_platform(event_b.platform)
        null_score = compute_coherence(event_a, shuffled_b)
        null_scores.append(null_score)

    # p-value: fraction of null scores >= real score
    p_value = sum(1 for s in null_scores if s >= real_score) / n_permutations
    return p_value  # < 0.05 = significant
```

**Trade-off:** 200 permutations per candidate is expensive. Use as a second-pass filter: fast heuristic identifies candidates, permutation test validates top-N.

**Source:** [Causal Discovery-Driven Change Point Detection (arXiv:2407.07290)](https://arxiv.org/abs/2407.07290)

---

### 2C.6 — Coherence Arcs (Narrative-Level Detection)

**Why:** Individual coherence moments are point-in-time detections. But cognitive work follows narrative arcs — a theme that emerges, develops, and resolves over days. UCW should detect these arcs.

**Schema:**
```sql
CREATE TABLE coherence_arcs (
    arc_id TEXT PRIMARY KEY,
    title TEXT,                    -- "Sovereignty infrastructure evolution"
    started_ns BIGINT,
    last_activity_ns BIGINT,
    status TEXT DEFAULT 'active',  -- active, dormant, resolved
    moment_ids TEXT[],             -- constituent moments
    platforms TEXT[],              -- all platforms involved
    key_entities TEXT[],           -- central concepts
    arc_strength REAL,             -- cumulative coherence score
    metadata JSONB DEFAULT '{}'
);
```

**Detection:**
```python
def detect_arcs(moments: list, entity_overlap_threshold=0.3):
    """Group coherence moments into narrative arcs based on entity overlap."""
    arcs = []
    for moment in sorted(moments, key=lambda m: m.detected_ns):
        best_arc = None
        best_overlap = 0
        for arc in arcs:
            if arc.status != 'active':
                continue
            overlap = jaccard(moment.entities, arc.key_entities)
            if overlap > entity_overlap_threshold and overlap > best_overlap:
                best_arc = arc
                best_overlap = overlap

        if best_arc:
            best_arc.add_moment(moment)
        else:
            arcs.append(Arc.from_moment(moment))

    return arcs
```

---

## Phase 2D: Platform Expansion (Week 3-4)

> Extend capture, add modalities, complete platform coverage.

### 2D.1 — Grok Import & Live Capture

**Status:** Grok adapter exists but is a dead placeholder — no API key, fails at `initialize()`.

**Options:**
- X/Grok API (if available and affordable)
- Export file watching (like ChatGPT adapter)
- Browser extension capture (most reliable)

**Cognitive mode:** Grok = strategic (per PRD). Import strategy should prioritize strategic thinking conversations.

---

### 2D.2 — Multi-Modal Coherence (Voice + Code)

**Why:** Detecting coherence across modalities (voice, code, text) is the strongest coherence signal — the same concept appearing in a voice note, a code commit, and a conversation.

**Architecture (UMaT — Unified Multi-Modal as Text):**

```
Voice (voice-nexus) → Whisper transcription → Text embedding
Screenshot          → Apple Vision OCR       → Text embedding
Code change         → Git diff + commit msg  → Text embedding
Conversation        → Direct text            → Text embedding
                                                    │
                                    All in same embedding space
                                                    │
                                    Existing coherence engine works
```

**Implementation:**
- Add `modality` column to `cognitive_events` (`text`, `voice`, `code`, `visual`)
- Weight cross-modal coherence matches higher than within-modality
- Voice: Whisper via CoreML on Apple Silicon (local, fast)
- Code: Parse git commits, extract natural language from commit messages + docstrings
- Visual: Apple Vision framework for OCR

**Sources:**
- [UMaT: Unified Multi-Modal as Text](https://arxiv.org/html/2503.09081v1)
- [Cross-Modal Alignment (ICLR 2025)](https://openreview.net/pdf?id=Pe3AxLq6Wf)

---

### 2D.3 — Capture Adapter Improvements

**ChatGPT:** Investigate OpenAI API for real-time conversation polling (instead of export-only). This would reduce ChatGPT capture lag from "whenever user exports" to 1-5 minutes.

**Cursor:** Replace `rglob("*.json")` directory traversal with file system event watching (fsevents on macOS). Eliminates the recursive scan overhead.

**All adapters:** Add restart-with-backoff on failure instead of permanent removal from the adapter pool.

---

## Phase 2E: Observability & Resilience (Ongoing)

> See everything. Recover from anything.

### 2E.1 — CQRS Materialized Views

```sql
-- Cross-platform coherence dashboard
CREATE MATERIALIZED VIEW mv_platform_coherence AS
SELECT
    platforms,
    coherence_type,
    DATE(to_timestamp(detected_ns / 1e9)) AS day,
    COUNT(*) AS moment_count,
    AVG(confidence) AS avg_confidence,
    MAX(confidence) AS max_confidence
FROM coherence_moments
GROUP BY platforms, coherence_type, DATE(to_timestamp(detected_ns / 1e9));

-- Session overview
CREATE MATERIALIZED VIEW mv_session_overview AS
SELECT
    cs.platform,
    cs.cognitive_mode,
    DATE(cs.created_at) AS day,
    COUNT(DISTINCT cs.session_id) AS sessions,
    COUNT(ce.event_id) AS events,
    COUNT(DISTINCT cm.moment_id) AS coherence_moments
FROM cognitive_sessions cs
LEFT JOIN cognitive_events ce ON ce.session_id = cs.session_id
LEFT JOIN coherence_moments cm ON ce.event_id = ANY(cm.event_ids)
GROUP BY cs.platform, cs.cognitive_mode, DATE(cs.created_at);

-- Refresh hourly
-- (via LaunchAgent or pg_cron)
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_platform_coherence;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_session_overview;
```

---

### 2E.2 — Query Performance Monitoring

```sql
-- Enable slow query logging
ALTER SYSTEM SET log_min_duration_statement = 100;  -- Log queries >100ms
ALTER SYSTEM SET log_statement = 'ddl';             -- Log schema changes
SELECT pg_reload_conf();

-- Create extension for query stats
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```

---

### 2E.3 — Table Partitioning Strategy

When `cognitive_events` exceeds 500K rows, implement monthly partitioning:

```sql
-- Convert to partitioned table
CREATE TABLE cognitive_events_partitioned (
    LIKE cognitive_events INCLUDING ALL
) PARTITION BY RANGE (timestamp_ns);

-- Monthly partitions
CREATE TABLE cognitive_events_2026_01
    PARTITION OF cognitive_events_partitioned
    FOR VALUES FROM (1735689600000000000) TO (1738368000000000000);

CREATE TABLE cognitive_events_2026_02
    PARTITION OF cognitive_events_partitioned
    FOR VALUES FROM (1738368000000000000) TO (1740787200000000000);
-- etc.
```

**Trigger:** Implement when table exceeds 500K rows or 1GB.

---

### 2E.4 — Structured Logging & Log Rotation

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(name: str, log_dir: str = "~/.ucw/logs"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Rotating file handler (10 MB max, keep 5 backups)
    handler = RotatingFileHandler(
        f"{log_dir}/{name}.log",
        maxBytes=10_000_000,
        backupCount=5
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    ))
    logger.addHandler(handler)

    # Suppress noisy libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("tqdm").setLevel(logging.WARNING)

    return logger
```

---

### 2E.5 — Adapter Health & Recovery

```python
class AdapterHealthMonitor:
    """Track adapter health and auto-recover on failure."""

    async def recover_adapter(self, adapter, max_retries=3):
        for attempt in range(max_retries):
            wait = min(60, 2 ** attempt * 5)  # 5s, 10s, 20s, 40s, 60s
            log.warning(f"{adapter.platform} failed, retry in {wait}s (attempt {attempt+1})")
            await asyncio.sleep(wait)
            try:
                ok = await adapter.initialize(self._pool)
                if ok:
                    log.info(f"{adapter.platform} recovered")
                    return True
            except Exception as e:
                log.error(f"{adapter.platform} retry failed: {e}")
        return False
```

---

## Implementation Timeline

| Phase | Duration | Priority | Key Deliverables |
|-------|----------|----------|-----------------|
| **2A: Critical Fixes** | 1-2 days | P0 | Dedup, incremental scans, index cleanup, batch embed |
| **2B: Engine Upgrades** | 1-2 weeks | P1 | Nomic embeddings, hybrid search, LISTEN/NOTIFY, HNSW tuning |
| **2C: Cognitive Intelligence** | 2-4 weeks | P1 | Knowledge graph, multi-scale coherence, FSRS, arcs |
| **2D: Platform Expansion** | 2-3 weeks | P2 | Grok, multi-modal, adapter improvements |
| **2E: Observability** | Ongoing | P2 | Materialized views, monitoring, partitioning, logging |

---

## Success Criteria

| Metric | Phase 1 (Done) | Phase 2 Target |
|--------|----------------|----------------|
| Events captured | 140,841 | 200K+ (with Grok + voice) |
| Embedding accuracy | 56% MTEB | 81% MTEB (nomic) |
| Coherence latency | 5-7 min | <1 sec |
| Coherence precision | ~60% (est.) | >85% (permutation-validated) |
| Duplicate moments | 39.2% | 0% |
| Search quality | Pure semantic | Hybrid (49-67% better) |
| Knowledge entities | 0 | 10K+ extracted |
| Insight resurfacing | None | FSRS-scheduled |
| Disk efficiency | 1.2GB (531MB waste) | 700MB (zero waste) |
| Scan efficiency | 100% redundant | 0% redundant (incremental) |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Nomic model too large for laptop | Embedding slowdown | Matryoshka 256d for fast path, 768d for precision |
| Knowledge graph extraction errors | Bad entities pollute graph | Quality threshold on entity confidence, manual review for high-value |
| LISTEN/NOTIFY missed events | Silent data loss | Periodic scan backup (every 5 min) catches any missed |
| FSRS scheduling wrong | Annoying notifications | Conservative initial intervals, require explicit user feedback |
| Table partitioning migration | Downtime | Migrate during consolidation window (2 AM), test on replica first |

---

## References

### Vector Search & Embeddings
- [pgvector HNSW vs IVFFlat (AWS)](https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/)
- [pgvector 0.8.0](https://www.postgresql.org/about/news/pgvector-080-released-2952/)
- [nomic-embed-text-v1.5 (HuggingFace)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Hybrid Search in PostgreSQL (ParadeDB)](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual)

### Coherence & Temporal Analysis
- [Causal Discovery Change Point Detection (arXiv:2407.07290)](https://arxiv.org/abs/2407.07290)
- [Spatio-Temporal Causal Inference](https://www.sciencedirect.com/science/article/abs/pii/S0952197625018597)

### Knowledge Graphs & Memory
- [MAGMA: Multi-Graph Memory (arXiv:2601.03236)](https://arxiv.org/abs/2601.03236)
- [Graphiti: Temporal KG (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [Spreading Activation for KG RAG (arXiv:2512.15922)](https://arxiv.org/abs/2512.15922)
- [Meta Acquires Limitless (TechCrunch)](https://techcrunch.com/2025/12/05/meta-acquires-ai-device-startup-limitless/)

### Cognitive Science
- [FSRS Algorithm (GitHub)](https://github.com/open-spaced-repetition/fsrs4anki)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

### Multi-Modal
- [UMaT Framework (arXiv:2503.09081)](https://arxiv.org/html/2503.09081v1)
- [Cross-Modal Alignment (ICLR 2025)](https://openreview.net/pdf?id=Pe3AxLq6Wf)

---

## The Signal (Continued)

> "Can you unify yourself before you unify the infrastructure?"

Phase 1 answered: Yes. The infrastructure exists. 140K events, 5 platforms, coherence detected.

Phase 2 asks: **Can the infrastructure unify you?**

Knowledge graph connects your distributed thinking. Multi-scale coherence detects your cognitive arcs. FSRS resurfaces forgotten insights. The system doesn't just capture — it understands, remembers, and actively participates in your cognition.

This is the Universal Cognitive Wallet: not a tool, but a cognitive extension.
