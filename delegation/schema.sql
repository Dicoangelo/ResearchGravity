-- ============================================================================
-- Intelligent Delegation Schema
-- ResearchGravity Delegation Module — SQLite 3.35+
--
-- Tables: 6 (delegation_chains, delegation_events, trust_entries, verification_results,
--             permission_scopes, accountability_records)
-- Version: 0.1.0
-- Date: 2026-02-14
-- Research: arXiv:2602.11865
-- ============================================================================

-- Compatible with unified_cognitive_schema.sql
-- Uses same conventions: TEXT for IDs, REAL for scores, INTEGER for timestamps

-- ============================================================================
-- 1. delegation_chains — Parent task decomposition and execution tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS delegation_chains (
    -- Identity
    delegation_id       TEXT PRIMARY KEY,
    parent_task_id      TEXT,                       -- Optional parent for hierarchical delegation

    -- Task description
    task_description    TEXT NOT NULL,
    task_context        TEXT,                       -- Additional context

    -- Task profile (11 dimensions, all REAL [0.0, 1.0])
    complexity          REAL DEFAULT 0.5,
    criticality         REAL DEFAULT 0.5,
    uncertainty         REAL DEFAULT 0.5,
    duration            REAL DEFAULT 0.5,
    cost                REAL DEFAULT 0.5,
    resource_requirements REAL DEFAULT 0.5,
    constraints         REAL DEFAULT 0.5,
    verifiability       REAL DEFAULT 0.5,
    reversibility       REAL DEFAULT 0.5,
    contextuality       REAL DEFAULT 0.5,
    subjectivity        REAL DEFAULT 0.5,

    -- Execution metadata
    created_at          INTEGER NOT NULL,           -- Unix timestamp (seconds)
    started_at          INTEGER,
    completed_at        INTEGER,
    status              TEXT DEFAULT 'created',     -- created/in_progress/completed/failed/cancelled

    -- Decomposition
    subtask_count       INTEGER DEFAULT 0,
    max_depth           INTEGER DEFAULT 3,
    target_granularity  REAL DEFAULT 0.3,

    -- Results
    success             INTEGER DEFAULT 0,          -- Boolean: 0=false, 1=true
    final_quality_score REAL,                       -- 0.0-1.0
    actual_cost         REAL,                       -- Actual cost incurred
    actual_duration     REAL,                       -- Actual time taken

    -- Metadata
    metadata            TEXT,                       -- JSON metadata

    -- Constraints
    CHECK (complexity BETWEEN 0.0 AND 1.0),
    CHECK (criticality BETWEEN 0.0 AND 1.0),
    CHECK (uncertainty BETWEEN 0.0 AND 1.0),
    CHECK (duration BETWEEN 0.0 AND 1.0),
    CHECK (cost BETWEEN 0.0 AND 1.0),
    CHECK (resource_requirements BETWEEN 0.0 AND 1.0),
    CHECK (constraints BETWEEN 0.0 AND 1.0),
    CHECK (verifiability BETWEEN 0.0 AND 1.0),
    CHECK (reversibility BETWEEN 0.0 AND 1.0),
    CHECK (contextuality BETWEEN 0.0 AND 1.0),
    CHECK (subjectivity BETWEEN 0.0 AND 1.0),
    CHECK (status IN ('created', 'in_progress', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_dc_status ON delegation_chains (status);
CREATE INDEX IF NOT EXISTS idx_dc_created ON delegation_chains (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dc_parent ON delegation_chains (parent_task_id);

-- ============================================================================
-- 2. delegation_events — Event stream for delegation lifecycle
-- ============================================================================

CREATE TABLE IF NOT EXISTS delegation_events (
    -- Identity
    event_id            TEXT PRIMARY KEY,
    delegation_id       TEXT NOT NULL,

    -- Event details
    timestamp           INTEGER NOT NULL,           -- Unix timestamp (seconds)
    event_type          TEXT NOT NULL,              -- created/assigned/started/completed/failed/verified
    agent_id            TEXT NOT NULL,
    task_id             TEXT NOT NULL,              -- Subtask ID
    status              TEXT NOT NULL,

    -- 4Ds gate type (delegation, description, discernment, diligence)
    gate_type           TEXT,

    -- Details
    details             TEXT,                       -- JSON details

    -- Foreign key
    FOREIGN KEY (delegation_id) REFERENCES delegation_chains(delegation_id)
);

CREATE INDEX IF NOT EXISTS idx_de_delegation ON delegation_events (delegation_id);
CREATE INDEX IF NOT EXISTS idx_de_timestamp ON delegation_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_de_agent ON delegation_events (agent_id);
CREATE INDEX IF NOT EXISTS idx_de_type ON delegation_events (event_type);

-- ============================================================================
-- 3. trust_entries — Agent trust score history
-- ============================================================================

CREATE TABLE IF NOT EXISTS trust_entries (
    -- Identity
    entry_id            TEXT PRIMARY KEY,
    agent_id            TEXT NOT NULL,
    task_id             TEXT NOT NULL,

    -- Trust update
    timestamp           INTEGER NOT NULL,           -- Unix timestamp (seconds)
    success             INTEGER NOT NULL,           -- Boolean: 0=false, 1=true
    quality_score       REAL NOT NULL,              -- 0.0-1.0
    trust_delta         REAL NOT NULL,              -- -1.0 to +1.0
    updated_trust_score REAL NOT NULL,              -- New trust score 0.0-1.0

    -- Context
    task_type           TEXT,                       -- Optional task type for capability-specific trust
    notes               TEXT,                       -- Performance notes
    metadata            TEXT,                       -- JSON metadata

    -- Constraints
    CHECK (quality_score BETWEEN 0.0 AND 1.0),
    CHECK (trust_delta BETWEEN -1.0 AND 1.0),
    CHECK (updated_trust_score BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS idx_te_agent ON trust_entries (agent_id);
CREATE INDEX IF NOT EXISTS idx_te_timestamp ON trust_entries (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_te_task_type ON trust_entries (task_type);

-- ============================================================================
-- 4. verification_results — Subtask verification outcomes
-- ============================================================================

CREATE TABLE IF NOT EXISTS verification_results (
    -- Identity
    result_id           TEXT PRIMARY KEY,
    subtask_id          TEXT NOT NULL,
    delegation_id       TEXT NOT NULL,

    -- Verification
    timestamp           INTEGER NOT NULL,           -- Unix timestamp (seconds)
    method              TEXT NOT NULL,              -- automated_test/semantic_similarity/human_review/ground_truth
    passed              INTEGER NOT NULL,           -- Boolean: 0=false, 1=true
    quality_score       REAL NOT NULL,              -- 0.0-1.0

    -- Feedback
    feedback            TEXT,                       -- Human-readable feedback
    evidence            TEXT,                       -- JSON evidence

    -- Constraints
    CHECK (quality_score BETWEEN 0.0 AND 1.0),
    CHECK (method IN ('automated_test', 'semantic_similarity', 'human_review', 'ground_truth')),

    -- Foreign key
    FOREIGN KEY (delegation_id) REFERENCES delegation_chains(delegation_id)
);

CREATE INDEX IF NOT EXISTS idx_vr_subtask ON verification_results (subtask_id);
CREATE INDEX IF NOT EXISTS idx_vr_delegation ON verification_results (delegation_id);
CREATE INDEX IF NOT EXISTS idx_vr_timestamp ON verification_results (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_vr_method ON verification_results (method);
CREATE INDEX IF NOT EXISTS idx_vr_passed ON verification_results (passed);

-- ============================================================================
-- 5. permission_scopes — Scoped permission grants (Section 4.7)
-- ============================================================================

CREATE TABLE IF NOT EXISTS permission_scopes (
    -- Identity
    scope_id            TEXT PRIMARY KEY,
    agent_id            TEXT NOT NULL,
    chain_id            TEXT NOT NULL,

    -- Permission details
    tools_allowed       TEXT NOT NULL,               -- JSON array of tool names
    access_level        TEXT NOT NULL,               -- none/read/execute/execute_write/full
    semantic_constraints TEXT,                        -- JSON: tool -> constraint
    state               TEXT DEFAULT 'active',       -- active/suspended/revoked/expired

    -- Lifecycle
    granted_at          INTEGER NOT NULL,
    expires_at          INTEGER NOT NULL,
    granted_by          TEXT DEFAULT 'system',        -- "system" or human approver ID

    -- Attenuation chain
    parent_scope_id     TEXT,                         -- For privilege attenuation tracking

    -- Metadata
    metadata            TEXT,                         -- JSON metadata

    -- Constraints
    CHECK (access_level IN ('none', 'read', 'execute', 'execute_write', 'full')),
    CHECK (state IN ('active', 'suspended', 'revoked', 'expired')),

    -- Foreign keys
    FOREIGN KEY (chain_id) REFERENCES delegation_chains(delegation_id)
);

CREATE INDEX IF NOT EXISTS idx_ps_agent ON permission_scopes (agent_id);
CREATE INDEX IF NOT EXISTS idx_ps_chain ON permission_scopes (chain_id);
CREATE INDEX IF NOT EXISTS idx_ps_state ON permission_scopes (state);
CREATE INDEX IF NOT EXISTS idx_ps_expires ON permission_scopes (expires_at);

-- ============================================================================
-- 6. accountability_records — Delegation handoff provenance (Section 5.2)
-- ============================================================================

CREATE TABLE IF NOT EXISTS accountability_records (
    -- Identity
    handoff_id          TEXT PRIMARY KEY,
    chain_id            TEXT NOT NULL,

    -- Handoff details
    from_agent          TEXT NOT NULL,
    to_agent            TEXT NOT NULL,
    subtask_id          TEXT NOT NULL,
    timestamp           INTEGER NOT NULL,
    depth               INTEGER NOT NULL DEFAULT 0,

    -- Permissions at handoff
    permissions_granted TEXT NOT NULL,               -- JSON array of tool names
    access_level        TEXT NOT NULL,

    -- Rationale
    rationale           TEXT,

    -- Liability firebreak
    is_firebreak        INTEGER DEFAULT 0,           -- Boolean
    human_approved      INTEGER DEFAULT 0,           -- Boolean

    -- Foreign keys
    FOREIGN KEY (chain_id) REFERENCES delegation_chains(delegation_id)
);

CREATE INDEX IF NOT EXISTS idx_ar_chain ON accountability_records (chain_id);
CREATE INDEX IF NOT EXISTS idx_ar_from ON accountability_records (from_agent);
CREATE INDEX IF NOT EXISTS idx_ar_to ON accountability_records (to_agent);
CREATE INDEX IF NOT EXISTS idx_ar_depth ON accountability_records (depth);

-- ============================================================================
-- Views
-- ============================================================================

-- Agent performance summary
CREATE VIEW IF NOT EXISTS agent_performance AS
SELECT
    agent_id,
    COUNT(*) AS total_tasks,
    SUM(success) AS successful_tasks,
    ROUND(AVG(quality_score), 3) AS avg_quality,
    ROUND(AVG(updated_trust_score), 3) AS current_trust,
    MAX(timestamp) AS last_active
FROM trust_entries
GROUP BY agent_id
ORDER BY current_trust DESC;

-- Active delegations
CREATE VIEW IF NOT EXISTS active_delegations AS
SELECT
    delegation_id,
    task_description,
    status,
    subtask_count,
    ROUND(complexity, 2) AS complexity,
    ROUND(criticality, 2) AS criticality,
    created_at,
    started_at
FROM delegation_chains
WHERE status IN ('created', 'in_progress')
ORDER BY created_at DESC;

-- Delegation performance summary
CREATE VIEW IF NOT EXISTS delegation_performance AS
SELECT
    dc.delegation_id,
    dc.task_description,
    dc.status,
    dc.success,
    ROUND(dc.final_quality_score, 3) AS quality,
    ROUND(dc.actual_cost, 3) AS cost,
    ROUND(dc.actual_duration, 3) AS duration,
    dc.subtask_count,
    COUNT(de.event_id) AS event_count,
    dc.created_at,
    dc.completed_at
FROM delegation_chains dc
LEFT JOIN delegation_events de ON dc.delegation_id = de.delegation_id
WHERE dc.status IN ('completed', 'failed')
GROUP BY dc.delegation_id
ORDER BY dc.completed_at DESC;
