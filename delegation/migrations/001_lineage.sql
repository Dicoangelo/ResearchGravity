-- E5: Lineage-aware delegation (migration 001)
--
-- Adds a persistent lineage store that survives process restarts and bridges
-- delegation_events, delegation_chains (aspirational), and graph/lineage.py's
-- in-memory LineageTracker.
--
-- Applied against: ~/.agent-core/storage/delegation_events.db
--
-- Design choices:
--   * Additive: does not alter existing tables.
--   * Self-contained: does not depend on the full delegation/schema.sql being
--     loaded (it currently is not — only delegation_events exists).
--   * Materialized path ("/root/a/b/c") + (root_id, depth) so subtree queries
--     are O(log n) via prefix match on the path column.
--   * node_type covers delegation, research session, finding, paper, concept —
--     mirrors graph/lineage.py NodeType enum so the LineageTracker can be
--     rehydrated from this table.

CREATE TABLE IF NOT EXISTS delegation_lineage (
    node_id       TEXT PRIMARY KEY,
    parent_id     TEXT,
    root_id       TEXT NOT NULL,
    depth         INTEGER NOT NULL DEFAULT 0,
    path          TEXT NOT NULL,
    node_type     TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    expired_at    TEXT,
    metadata_json TEXT,
    FOREIGN KEY (parent_id) REFERENCES delegation_lineage(node_id)
);

CREATE INDEX IF NOT EXISTS idx_lineage_parent    ON delegation_lineage (parent_id);
CREATE INDEX IF NOT EXISTS idx_lineage_root      ON delegation_lineage (root_id);
CREATE INDEX IF NOT EXISTS idx_lineage_depth     ON delegation_lineage (depth);
CREATE INDEX IF NOT EXISTS idx_lineage_path      ON delegation_lineage (path);
CREATE INDEX IF NOT EXISTS idx_lineage_node_type ON delegation_lineage (node_type);
CREATE INDEX IF NOT EXISTS idx_lineage_active    ON delegation_lineage (expired_at)
    WHERE expired_at IS NULL;

CREATE TABLE IF NOT EXISTS delegation_lineage_edges (
    edge_id    TEXT PRIMARY KEY,
    source_id  TEXT NOT NULL,
    target_id  TEXT NOT NULL,
    edge_type  TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    valid_at   TEXT,
    expired_at TEXT,
    metadata_json TEXT,
    FOREIGN KEY (source_id) REFERENCES delegation_lineage(node_id),
    FOREIGN KEY (target_id) REFERENCES delegation_lineage(node_id)
);

CREATE INDEX IF NOT EXISTS idx_ledges_source  ON delegation_lineage_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_ledges_target  ON delegation_lineage_edges (target_id);
CREATE INDEX IF NOT EXISTS idx_ledges_type    ON delegation_lineage_edges (edge_type);
CREATE INDEX IF NOT EXISTS idx_ledges_active  ON delegation_lineage_edges (expired_at)
    WHERE expired_at IS NULL;

-- Subtree view: all active descendants of any root, ordered by depth.
CREATE VIEW IF NOT EXISTS lineage_active_subtree AS
SELECT
    root_id,
    node_id,
    parent_id,
    depth,
    path,
    node_type,
    created_at
FROM delegation_lineage
WHERE expired_at IS NULL
ORDER BY root_id, depth, created_at;
