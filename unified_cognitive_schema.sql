-- ============================================================================
-- Unified Cognitive Database Schema
-- UCW Raw MCP Infrastructure — PostgreSQL 15+ with pgvector
--
-- Tables: 7 (as specified in PRD)
-- Version: 1.0.0
-- Date: 2026-02-07
-- ============================================================================

-- Requires: PostgreSQL 15+ and pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 1. cognitive_events — Every MCP message with UCW semantic layers
-- ============================================================================

CREATE TABLE IF NOT EXISTS cognitive_events (
    -- Identity
    event_id            TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,

    -- Temporal
    timestamp_ns        BIGINT NOT NULL,        -- Nanosecond precision
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    -- Protocol
    direction           TEXT NOT NULL,           -- 'in' or 'out'
    stage               TEXT NOT NULL,           -- received/parsed/routed/executed/sent
    method              TEXT,                    -- MCP method (tools/call, initialize, etc.)
    request_id          TEXT,                    -- JSON-RPC request ID
    parent_event_id     TEXT,                    -- Response → request lineage
    turn                INTEGER DEFAULT 0,       -- Conversation turn number

    -- Raw capture (perfect fidelity)
    raw_bytes           BYTEA,                   -- Original bytes from wire
    parsed_json         JSONB,                   -- Parsed JSON-RPC message
    content_length      INTEGER DEFAULT 0,       -- Byte count
    error               TEXT,                    -- Error if any

    -- UCW Data Layer (what was said)
    data_layer          JSONB,                   -- {method, params, result, content, tokens_est}

    -- UCW Light Layer (what it means)
    light_layer         JSONB,                   -- {intent, topic, concepts, summary}

    -- UCW Instinct Layer (what it signals)
    instinct_layer      JSONB,                   -- {coherence_potential, emergence_indicators, gut_signal}

    -- Coherence
    coherence_sig       TEXT,                    -- SHA-256 for cross-platform matching
    semantic_embedding  vector(1024),            -- For similarity search

    -- Classification
    platform            TEXT DEFAULT 'claude-desktop',
    protocol            TEXT DEFAULT 'mcp',
    quality_score       REAL,                    -- 0.0 - 1.0
    cognitive_mode      TEXT                     -- deep_work/exploration/casual/garbage
);

-- Indexes for cognitive_events
CREATE INDEX IF NOT EXISTS idx_ce_timestamp     ON cognitive_events (timestamp_ns);
CREATE INDEX IF NOT EXISTS idx_ce_session       ON cognitive_events (session_id);
CREATE INDEX IF NOT EXISTS idx_ce_method        ON cognitive_events (method);
CREATE INDEX IF NOT EXISTS idx_ce_direction     ON cognitive_events (direction);
CREATE INDEX IF NOT EXISTS idx_ce_turn          ON cognitive_events (turn);
CREATE INDEX IF NOT EXISTS idx_ce_coherence     ON cognitive_events (coherence_sig);
CREATE INDEX IF NOT EXISTS idx_ce_platform      ON cognitive_events (platform);
CREATE INDEX IF NOT EXISTS idx_ce_mode          ON cognitive_events (cognitive_mode);
CREATE INDEX IF NOT EXISTS idx_ce_quality       ON cognitive_events (quality_score);

-- GIN index for JSONB queries
CREATE INDEX IF NOT EXISTS idx_ce_light_gin     ON cognitive_events USING gin (light_layer);
CREATE INDEX IF NOT EXISTS idx_ce_instinct_gin  ON cognitive_events USING gin (instinct_layer);

-- ============================================================================
-- 2. cognitive_sessions — Work sessions across platforms
-- ============================================================================

CREATE TABLE IF NOT EXISTS cognitive_sessions (
    session_id          TEXT PRIMARY KEY,
    started_ns          BIGINT NOT NULL,
    ended_ns            BIGINT,
    platform            TEXT DEFAULT 'claude-desktop',
    status              TEXT DEFAULT 'active',    -- active/completed/abandoned
    event_count         INTEGER DEFAULT 0,
    turn_count          INTEGER DEFAULT 0,
    topics              JSONB,                    -- Detected topic distribution
    summary             TEXT,                     -- Session summary
    cognitive_mode      TEXT,                     -- Dominant mode
    quality_score       REAL,                     -- Overall quality
    metadata            JSONB,                    -- Extensible metadata
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cs_platform      ON cognitive_sessions (platform);
CREATE INDEX IF NOT EXISTS idx_cs_status        ON cognitive_sessions (status);
CREATE INDEX IF NOT EXISTS idx_cs_started       ON cognitive_sessions (started_ns);

-- ============================================================================
-- 3. coherence_moments — Detected cross-platform cognitive alignment
-- ============================================================================

CREATE TABLE IF NOT EXISTS coherence_moments (
    moment_id           TEXT PRIMARY KEY,
    detected_ns         BIGINT NOT NULL,          -- When detected
    event_ids           TEXT[] NOT NULL,           -- Participating events
    platforms           TEXT[] NOT NULL,           -- Platforms involved
    coherence_type      TEXT NOT NULL,             -- temporal/semantic/synchronicity
    confidence          REAL NOT NULL,             -- 0.0 - 1.0
    description         TEXT,                      -- Human-readable description
    time_window_s       INTEGER,                   -- Window in seconds
    signature           TEXT,                      -- Shared coherence signature
    metadata            JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cm_detected      ON coherence_moments (detected_ns);
CREATE INDEX IF NOT EXISTS idx_cm_type          ON coherence_moments (coherence_type);
CREATE INDEX IF NOT EXISTS idx_cm_confidence    ON coherence_moments (confidence);
CREATE INDEX IF NOT EXISTS idx_cm_platforms     ON coherence_moments USING gin (platforms);

-- ============================================================================
-- 4. coherence_links — Event-to-event coherence relationships
-- ============================================================================

CREATE TABLE IF NOT EXISTS coherence_links (
    link_id             TEXT PRIMARY KEY,
    source_event_id     TEXT NOT NULL,
    target_event_id     TEXT NOT NULL,
    link_type           TEXT NOT NULL,             -- signature_match/semantic_similar/temporal_align
    confidence          REAL NOT NULL,             -- 0.0 - 1.0
    moment_id           TEXT,                      -- FK to coherence_moments
    metadata            JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (source_event_id, target_event_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_cl_source        ON coherence_links (source_event_id);
CREATE INDEX IF NOT EXISTS idx_cl_target        ON coherence_links (target_event_id);
CREATE INDEX IF NOT EXISTS idx_cl_type          ON coherence_links (link_type);
CREATE INDEX IF NOT EXISTS idx_cl_moment        ON coherence_links (moment_id);

-- ============================================================================
-- 5. embedding_cache — Semantic embeddings for similarity search
-- ============================================================================

CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash        TEXT PRIMARY KEY,          -- SHA-256 of content
    content_preview     TEXT,                      -- First 200 chars
    embedding           vector(1024),              -- Cohere embed-v4 / v3
    model               TEXT NOT NULL,             -- Model used
    dimensions          INTEGER NOT NULL,          -- Vector dimensions
    source_event_id     TEXT,                      -- Optional FK
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_ec_embedding
    ON embedding_cache USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- 6. cognitive_signatures — Unique patterns for coherence detection
-- ============================================================================

CREATE TABLE IF NOT EXISTS cognitive_signatures (
    signature_id        TEXT PRIMARY KEY,
    signature_hash      TEXT NOT NULL UNIQUE,      -- The coherence signature
    intent              TEXT,
    topic               TEXT,
    concepts            JSONB,                     -- Concept array
    occurrence_count    INTEGER DEFAULT 1,
    first_seen_ns       BIGINT,
    last_seen_ns        BIGINT,
    platforms           TEXT[],                    -- Which platforms seen on
    representative_event TEXT,                     -- Best example event ID
    metadata            JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sig_hash         ON cognitive_signatures (signature_hash);
CREATE INDEX IF NOT EXISTS idx_sig_intent       ON cognitive_signatures (intent);
CREATE INDEX IF NOT EXISTS idx_sig_topic        ON cognitive_signatures (topic);
CREATE INDEX IF NOT EXISTS idx_sig_platforms    ON cognitive_signatures USING gin (platforms);
CREATE INDEX IF NOT EXISTS idx_sig_count        ON cognitive_signatures (occurrence_count DESC);

-- ============================================================================
-- 7. supermemory_entries — Long-term memory with spaced repetition
-- ============================================================================

CREATE TABLE IF NOT EXISTS supermemory_entries (
    entry_id            TEXT PRIMARY KEY,
    content             TEXT NOT NULL,
    entry_type          TEXT NOT NULL,             -- fact/decision/pattern/insight/error
    source_session      TEXT,
    source_platform     TEXT,
    importance          REAL DEFAULT 0.5,          -- 0.0 - 1.0
    review_count        INTEGER DEFAULT 0,
    next_review_at      TIMESTAMPTZ,
    last_reviewed_at    TIMESTAMPTZ,
    ease_factor         REAL DEFAULT 2.5,          -- SM-2 ease factor
    interval_days       INTEGER DEFAULT 1,         -- Current interval
    embedding           vector(1024),
    tags                TEXT[],
    metadata            JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sm_type          ON supermemory_entries (entry_type);
CREATE INDEX IF NOT EXISTS idx_sm_importance    ON supermemory_entries (importance DESC);
CREATE INDEX IF NOT EXISTS idx_sm_review        ON supermemory_entries (next_review_at);
CREATE INDEX IF NOT EXISTS idx_sm_platform      ON supermemory_entries (source_platform);
CREATE INDEX IF NOT EXISTS idx_sm_tags          ON supermemory_entries USING gin (tags);

-- HNSW index for memory similarity search
CREATE INDEX IF NOT EXISTS idx_sm_embedding
    ON supermemory_entries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- Views
-- ============================================================================

-- Active coherence: recent events with high coherence potential
CREATE OR REPLACE VIEW active_coherence AS
SELECT
    e.event_id,
    e.session_id,
    e.timestamp_ns,
    e.platform,
    e.light_layer->>'intent' AS intent,
    e.light_layer->>'topic' AS topic,
    (e.instinct_layer->>'coherence_potential')::REAL AS coherence_potential,
    e.instinct_layer->>'gut_signal' AS gut_signal,
    e.coherence_sig
FROM cognitive_events e
WHERE (e.instinct_layer->>'coherence_potential')::REAL > 0.5
ORDER BY e.timestamp_ns DESC
LIMIT 1000;

-- Cross-platform matches: events sharing coherence signatures across platforms
CREATE OR REPLACE VIEW cross_platform_matches AS
SELECT
    a.coherence_sig,
    a.platform AS platform_a,
    b.platform AS platform_b,
    a.event_id AS event_a,
    b.event_id AS event_b,
    ABS(a.timestamp_ns - b.timestamp_ns) / 1000000000 AS time_diff_seconds,
    a.light_layer->>'topic' AS topic
FROM cognitive_events a
JOIN cognitive_events b
    ON a.coherence_sig = b.coherence_sig
    AND a.platform != b.platform
    AND a.event_id < b.event_id
WHERE a.coherence_sig IS NOT NULL
ORDER BY time_diff_seconds ASC;

-- Session overview
CREATE OR REPLACE VIEW session_overview AS
SELECT
    s.session_id,
    s.platform,
    s.status,
    s.event_count,
    s.turn_count,
    s.quality_score,
    s.cognitive_mode,
    to_timestamp(s.started_ns::DOUBLE PRECISION / 1000000000) AS started_at,
    to_timestamp(s.ended_ns::DOUBLE PRECISION / 1000000000) AS ended_at,
    s.summary
FROM cognitive_sessions s
ORDER BY s.started_ns DESC;
