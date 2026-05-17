"""
Lineage Store — persistent backing for delegation/research lineage (E5).

Bridges three previously-disconnected systems:

1. `delegation.four_ds._log_event` writes delegation_events but has no concept of
   a parent → child relationship across chains.
2. `graph/lineage.py::LineageTracker` maintains ancestors/descendants fully
   in-memory — it is lost on process restart and cannot be queried from SQL.
3. `delegation/schema.sql::delegation_chains.parent_task_id` was defined but
   never applied (the full 6-table schema is aspirational — only
   `delegation_events` exists in production).

This module creates a dedicated `delegation_lineage` table in
`~/.agent-core/storage/delegation_events.db` (the live delegation DB) and
exposes a thin, sync-safe API for nodes and edges.

Design:
    * Additive. Does not touch `delegation_events` or `trust_entries`.
    * Materialized path: O(log n) subtree queries via prefix match.
    * (root_id, depth) cached on every row so we never recurse at read-time.
    * Idempotent migration: `ensure_schema()` is safe to call on every import.
    * Sync, WAL mode. Safe to call from async code via `asyncio.to_thread`.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_DB_PATH = Path.home() / ".agent-core" / "storage" / "delegation_events.db"
MIGRATION_PATH = Path(__file__).parent / "migrations" / "001_lineage.sql"

# Node types mirror graph.lineage.NodeType; kept as strings here to avoid a
# hard import dependency between the delegation and graph packages.
VALID_NODE_TYPES = {
    "delegation",
    "session",
    "finding",
    "paper",
    "url",
    "concept",
    "project",
}

VALID_EDGE_TYPES = {
    "contains",
    "cites",
    "derives_from",
    "enables",
    "informs",
    "belongs_to",
    "related",
    "spawned",  # delegation-specific: parent chain spawned child chain
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LineageNode:
    node_id: str
    parent_id: Optional[str]
    root_id: str
    depth: int
    path: str
    node_type: str
    created_at: str
    expired_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LineageNode":
        meta = row["metadata_json"]
        return cls(
            node_id=row["node_id"],
            parent_id=row["parent_id"],
            root_id=row["root_id"],
            depth=row["depth"],
            path=row["path"],
            node_type=row["node_type"],
            created_at=row["created_at"],
            expired_at=row["expired_at"],
            metadata=json.loads(meta) if meta else {},
        )


@dataclass
class LineageEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    valid_at: Optional[str] = None
    expired_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LineageEdge":
        meta = row["metadata_json"]
        return cls(
            edge_id=row["edge_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=row["edge_type"],
            weight=row["weight"],
            valid_at=row["valid_at"],
            expired_at=row["expired_at"],
            metadata=json.loads(meta) if meta else {},
        )


class LineageStoreError(RuntimeError):
    pass


class LineageStore:
    """
    Sync, thread-safe SQLite-backed lineage store.

    Usage:
        store = LineageStore()
        root = store.add_node("root-1", node_type="session")
        child = store.add_node("child-1", parent_id="root-1", node_type="delegation")
        descendants = store.get_descendants("root-1")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    # ------------------------------------------------------------------ schema

    def ensure_schema(self) -> None:
        """Apply migration 001_lineage.sql idempotently."""
        if not MIGRATION_PATH.exists():
            raise LineageStoreError(f"Migration file missing: {MIGRATION_PATH}")
        sql = MIGRATION_PATH.read_text()
        with self._connect() as conn:
            conn.executescript("PRAGMA journal_mode=WAL;")
            conn.executescript(sql)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    # ------------------------------------------------------------------- nodes

    def add_node(
        self,
        node_id: str,
        *,
        parent_id: Optional[str] = None,
        node_type: str = "delegation",
        metadata: Optional[dict[str, Any]] = None,
    ) -> LineageNode:
        """
        Insert a lineage node. If parent_id is given the node inherits the
        parent's root_id and depth+1. If parent_id is None the node becomes its
        own root at depth 0.
        """
        if node_type not in VALID_NODE_TYPES:
            raise LineageStoreError(
                f"Unknown node_type '{node_type}' (valid: {sorted(VALID_NODE_TYPES)})"
            )

        created_at = _utc_now()
        parent: Optional[LineageNode] = None
        if parent_id is not None:
            parent = self.get_node(parent_id)
            if parent is None:
                raise LineageStoreError(f"parent_id '{parent_id}' not found")

        if parent is None:
            root_id = node_id
            depth = 0
            path = f"/{node_id}"
        else:
            root_id = parent.root_id
            depth = parent.depth + 1
            path = f"{parent.path}/{node_id}"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO delegation_lineage
                    (node_id, parent_id, root_id, depth, path,
                     node_type, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    parent_id,
                    root_id,
                    depth,
                    path,
                    node_type,
                    created_at,
                    json.dumps(metadata or {}),
                ),
            )

        return LineageNode(
            node_id=node_id,
            parent_id=parent_id,
            root_id=root_id,
            depth=depth,
            path=path,
            node_type=node_type,
            created_at=created_at,
            metadata=metadata or {},
        )

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM delegation_lineage WHERE node_id = ?",
                (node_id,),
            ).fetchone()
        return LineageNode.from_row(row) if row else None

    def expire_node(self, node_id: str, expired_at: Optional[str] = None) -> bool:
        ts = expired_at or _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE delegation_lineage SET expired_at = ? "
                "WHERE node_id = ? AND expired_at IS NULL",
                (ts, node_id),
            )
            return cur.rowcount > 0

    def get_ancestors(self, node_id: str) -> list[LineageNode]:
        """Return all ancestors from root down to (but excluding) node_id."""
        node = self.get_node(node_id)
        if node is None:
            return []
        # path = "/root/a/b/node_id" → ancestors = ["root", "a", "b"]
        parts = [p for p in node.path.split("/") if p and p != node_id]
        if not parts:
            return []
        with self._connect() as conn:
            placeholders = ",".join("?" * len(parts))
            rows = conn.execute(
                f"SELECT * FROM delegation_lineage WHERE node_id IN ({placeholders}) "
                "ORDER BY depth ASC",
                parts,
            ).fetchall()
        return [LineageNode.from_row(r) for r in rows]

    def get_descendants(
        self, node_id: str, *, include_expired: bool = False
    ) -> list[LineageNode]:
        """Return all descendants (strict: excludes node itself)."""
        node = self.get_node(node_id)
        if node is None:
            return []
        prefix = f"{node.path}/"
        clause = "" if include_expired else " AND expired_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM delegation_lineage WHERE path LIKE ? {clause} "
                "ORDER BY depth ASC, created_at ASC",
                (prefix + "%",),
            ).fetchall()
        return [LineageNode.from_row(r) for r in rows]

    def get_subtree(
        self, root_id: str, *, include_expired: bool = False
    ) -> list[LineageNode]:
        """Return the root and all descendants under root_id."""
        clause = "" if include_expired else " AND expired_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM delegation_lineage WHERE root_id = ? {clause} "
                "ORDER BY depth ASC, created_at ASC",
                (root_id,),
            ).fetchall()
        return [LineageNode.from_row(r) for r in rows]

    def get_roots(self, *, include_expired: bool = False) -> list[LineageNode]:
        clause = "" if include_expired else " AND expired_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM delegation_lineage WHERE depth = 0 {clause} "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [LineageNode.from_row(r) for r in rows]

    # ------------------------------------------------------------------- edges

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_type: str = "related",
        weight: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
        valid_at: Optional[str] = None,
    ) -> LineageEdge:
        if edge_type not in VALID_EDGE_TYPES:
            raise LineageStoreError(
                f"Unknown edge_type '{edge_type}' (valid: {sorted(VALID_EDGE_TYPES)})"
            )
        edge_id = uuid.uuid4().hex
        ts = valid_at or _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO delegation_lineage_edges
                    (edge_id, source_id, target_id, edge_type, weight,
                     valid_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge_id,
                    source_id,
                    target_id,
                    edge_type,
                    weight,
                    ts,
                    json.dumps(metadata or {}),
                ),
            )
        return LineageEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            valid_at=ts,
            metadata=metadata or {},
        )

    def get_neighbors(
        self, node_id: str, *, include_expired: bool = False
    ) -> list[LineageEdge]:
        clause = "" if include_expired else " AND expired_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM delegation_lineage_edges "
                f"WHERE (source_id = ? OR target_id = ?) {clause}",
                (node_id, node_id),
            ).fetchall()
        return [LineageEdge.from_row(r) for r in rows]

    def expire_edge(self, edge_id: str, expired_at: Optional[str] = None) -> bool:
        ts = expired_at or _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE delegation_lineage_edges SET expired_at = ? "
                "WHERE edge_id = ? AND expired_at IS NULL",
                (ts, edge_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            total_nodes = conn.execute(
                "SELECT COUNT(*) FROM delegation_lineage"
            ).fetchone()[0]
            active_nodes = conn.execute(
                "SELECT COUNT(*) FROM delegation_lineage WHERE expired_at IS NULL"
            ).fetchone()[0]
            roots = conn.execute(
                "SELECT COUNT(*) FROM delegation_lineage "
                "WHERE depth = 0 AND expired_at IS NULL"
            ).fetchone()[0]
            max_depth_row = conn.execute(
                "SELECT MAX(depth) FROM delegation_lineage WHERE expired_at IS NULL"
            ).fetchone()[0]
            total_edges = conn.execute(
                "SELECT COUNT(*) FROM delegation_lineage_edges"
            ).fetchone()[0]
            active_edges = conn.execute(
                "SELECT COUNT(*) FROM delegation_lineage_edges "
                "WHERE expired_at IS NULL"
            ).fetchone()[0]
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "roots": roots,
            "max_depth": max_depth_row or 0,
            "total_edges": total_edges,
            "active_edges": active_edges,
            "db_path": str(self.db_path),
        }

    # ------------------------------------------------------------------- bulk

    def bulk_add_nodes(self, nodes: Iterable[tuple[str, Optional[str], str]]) -> int:
        """
        Insert many nodes in order, respecting parent dependencies.
        Input: iterable of (node_id, parent_id, node_type).
        Returns count of inserted rows.
        """
        count = 0
        for node_id, parent_id, node_type in nodes:
            self.add_node(node_id, parent_id=parent_id, node_type=node_type)
            count += 1
        return count


__all__ = [
    "LineageStore",
    "LineageStoreError",
    "LineageNode",
    "LineageEdge",
    "VALID_NODE_TYPES",
    "VALID_EDGE_TYPES",
    "DEFAULT_DB_PATH",
]
