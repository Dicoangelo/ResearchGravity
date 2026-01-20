"""
Concept Graph

Builds and queries the knowledge graph from SQLite storage.
Provides relationship traversal and concept discovery.
"""

import sqlite3
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from .lineage import (
    LineageTracker,
    LineageNode,
    LineageEdge,
    LineageGraph,
    NodeType,
    EdgeType,
)

# Database path
DB_PATH = Path.home() / ".agent-core/storage/antigravity.db"


class ConceptGraph:
    """
    Knowledge graph built from SQLite storage.

    Nodes: sessions, findings, papers, urls
    Edges: contains, cites, derives_from, enables, informs
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._tracker: Optional[LineageTracker] = None
        self._loaded = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def load(self) -> bool:
        """Load graph from database."""
        if self._loaded:
            return True

        self._tracker = LineageTracker()
        conn = self._get_connection()

        try:
            cursor = conn.cursor()

            # Load sessions as nodes
            cursor.execute("""
                SELECT id, topic, project, status, started_at
                FROM sessions
            """)
            for row in cursor.fetchall():
                self._tracker.add_node(LineageNode(
                    id=f"session:{row['id']}",
                    type=NodeType.SESSION,
                    label=row['topic'] or row['id'][:30],
                    metadata={
                        "project": row['project'],
                        "status": row['status'],
                    },
                    created_at=row['started_at'],
                ))

            # Load findings as nodes
            cursor.execute("""
                SELECT id, content, type, session_id, confidence
                FROM findings
            """)
            for row in cursor.fetchall():
                label = (row['content'] or '')[:50] + "..."
                self._tracker.add_node(LineageNode(
                    id=f"finding:{row['id']}",
                    type=NodeType.FINDING,
                    label=label,
                    metadata={
                        "type": row['type'],
                        "confidence": row['confidence'],
                    },
                ))

                # Create session → finding edge
                if row['session_id']:
                    self._tracker.add_edge(LineageEdge(
                        source_id=f"session:{row['session_id']}",
                        target_id=f"finding:{row['id']}",
                        edge_type=EdgeType.CONTAINS,
                    ))

            # Load URLs as nodes
            cursor.execute("""
                SELECT id, url, tier, category, session_id
                FROM urls
            """)
            for row in cursor.fetchall():
                url = row['url'] or ''
                label = url[:40] + "..." if len(url) > 40 else url
                self._tracker.add_node(LineageNode(
                    id=f"url:{row['id']}",
                    type=NodeType.URL,
                    label=label,
                    metadata={
                        "tier": row['tier'],
                        "category": row['category'],
                        "full_url": url,
                    },
                ))

                # Create session → url edge
                if row['session_id']:
                    self._tracker.add_edge(LineageEdge(
                        source_id=f"session:{row['session_id']}",
                        target_id=f"url:{row['id']}",
                        edge_type=EdgeType.CITES,
                        weight=1.0 if row['tier'] == 1 else 0.5,
                    ))

            # Load lineage edges
            cursor.execute("""
                SELECT source_type, source_id, target_type, target_id, relation
                FROM lineage
            """)
            for row in cursor.fetchall():
                edge_type = {
                    'contains': EdgeType.CONTAINS,
                    'cites': EdgeType.CITES,
                    'derives_from': EdgeType.DERIVES_FROM,
                    'enables': EdgeType.ENABLES,
                    'informs': EdgeType.INFORMS,
                    'related': EdgeType.RELATED,
                }.get(row['relation'], EdgeType.RELATED)

                self._tracker.add_edge(LineageEdge(
                    source_id=f"{row['source_type']}:{row['source_id']}",
                    target_id=f"{row['target_type']}:{row['target_id']}",
                    edge_type=edge_type,
                ))

            self._loaded = True
            return True

        except Exception as e:
            print(f"Error loading graph: {e}")
            return False
        finally:
            conn.close()

    async def get_session_graph(self, session_id: str, depth: int = 2) -> LineageGraph:
        """Get the knowledge graph centered on a session."""
        await self.load()

        node_id = f"session:{session_id}"
        return self._tracker.get_neighborhood(node_id, depth=depth)

    async def get_related_sessions(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find sessions related to a given session.

        Uses:
        1. Shared findings (common concepts)
        2. Shared URLs (common sources)
        3. Explicit lineage links
        4. Semantic similarity (via Qdrant if available)
        """
        await self.load()
        conn = self._get_connection()

        try:
            cursor = conn.cursor()

            # Find sessions with shared URL domains
            cursor.execute("""
                WITH session_domains AS (
                    SELECT session_id,
                           SUBSTR(url, INSTR(url, '://') + 3,
                                  CASE WHEN INSTR(SUBSTR(url, INSTR(url, '://') + 3), '/') > 0
                                       THEN INSTR(SUBSTR(url, INSTR(url, '://') + 3), '/') - 1
                                       ELSE LENGTH(url)
                                  END) as domain
                    FROM urls WHERE session_id IS NOT NULL
                ),
                target_domains AS (
                    SELECT domain FROM session_domains WHERE session_id = ?
                )
                SELECT sd.session_id, COUNT(DISTINCT sd.domain) as shared_domains
                FROM session_domains sd
                JOIN target_domains td ON sd.domain = td.domain
                WHERE sd.session_id != ?
                GROUP BY sd.session_id
                ORDER BY shared_domains DESC
                LIMIT ?
            """, (session_id, session_id, limit))

            related = []
            for row in cursor.fetchall():
                # Get session details
                cursor.execute("""
                    SELECT id, topic, project, status
                    FROM sessions WHERE id = ?
                """, (row['session_id'],))
                session = cursor.fetchone()
                if session:
                    related.append({
                        "session_id": session['id'],
                        "topic": session['topic'],
                        "project": session['project'],
                        "status": session['status'],
                        "relation": "shared_sources",
                        "score": row['shared_domains'],
                    })

            return related

        finally:
            conn.close()

    async def get_finding_lineage(self, finding_id: str) -> LineageGraph:
        """Get the lineage chain for a finding (what it derives from, what derives from it)."""
        await self.load()

        node_id = f"finding:{finding_id}"
        ancestors = self._tracker.get_ancestors(node_id, max_depth=3)
        descendants = self._tracker.get_descendants(node_id, max_depth=3)

        # Build subgraph
        nodes = [self._tracker._nodes.get(node_id)] + ancestors + descendants
        nodes = [n for n in nodes if n is not None]

        # Get edges between these nodes
        node_ids = {n.id for n in nodes}
        edges = [
            e for e in self._tracker._edges
            if e.source_id in node_ids and e.target_id in node_ids
        ]

        return LineageGraph(nodes=nodes, edges=edges)

    async def get_concept_clusters(self, min_size: int = 3) -> List[Dict[str, Any]]:
        """
        Find concept clusters (groups of related sessions/findings).

        Uses connected components analysis.
        """
        await self.load()

        # Find connected components
        visited = set()
        clusters = []

        def dfs(node_id: str, cluster: set):
            if node_id in visited:
                return
            visited.add(node_id)
            cluster.add(node_id)

            # Visit neighbors
            for target_id in self._tracker._adjacency.get(node_id, []):
                dfs(target_id, cluster)
            for source_id in self._tracker._reverse_adjacency.get(node_id, []):
                dfs(source_id, cluster)

        for node_id in self._tracker._nodes:
            if node_id not in visited:
                cluster = set()
                dfs(node_id, cluster)
                if len(cluster) >= min_size:
                    # Analyze cluster
                    sessions = [nid for nid in cluster if nid.startswith("session:")]
                    findings = [nid for nid in cluster if nid.startswith("finding:")]
                    urls = [nid for nid in cluster if nid.startswith("url:")]

                    clusters.append({
                        "size": len(cluster),
                        "sessions": len(sessions),
                        "findings": len(findings),
                        "urls": len(urls),
                        "sample_session": sessions[0].replace("session:", "") if sessions else None,
                    })

        return sorted(clusters, key=lambda c: c['size'], reverse=True)

    async def get_research_timeline(
        self,
        project: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get chronological research timeline with lineage links."""
        conn = self._get_connection()

        try:
            cursor = conn.cursor()

            query = """
                SELECT s.id, s.topic, s.project, s.started_at, s.status,
                       COUNT(DISTINCT f.id) as finding_count,
                       COUNT(DISTINCT u.id) as url_count
                FROM sessions s
                LEFT JOIN findings f ON f.session_id = s.id
                LEFT JOIN urls u ON u.session_id = s.id
                WHERE s.started_at IS NOT NULL
            """
            params = []

            if project:
                query += " AND s.project = ?"
                params.append(project)

            query += """
                GROUP BY s.id
                ORDER BY s.started_at DESC
                LIMIT ?
            """
            params.append(limit)

            cursor.execute(query, params)

            timeline = []
            for row in cursor.fetchall():
                timeline.append({
                    "session_id": row['id'],
                    "topic": row['topic'],
                    "project": row['project'],
                    "date": row['started_at'],
                    "status": row['status'],
                    "findings": row['finding_count'],
                    "urls": row['url_count'],
                })

            return timeline

        finally:
            conn.close()

    async def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        await self.load()
        return self._tracker.get_stats()


async def main():
    """CLI for graph queries."""
    import sys
    import json

    graph = ConceptGraph()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python concept_graph.py stats")
        print("  python concept_graph.py session <session-id>")
        print("  python concept_graph.py related <session-id>")
        print("  python concept_graph.py clusters")
        print("  python concept_graph.py timeline [project]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "stats":
        stats = await graph.get_stats()
        print(json.dumps(stats, indent=2))

    elif command == "session":
        session_id = sys.argv[2] if len(sys.argv) > 2 else ""
        subgraph = await graph.get_session_graph(session_id, depth=2)
        print(json.dumps(subgraph.to_d3_format(), indent=2))

    elif command == "related":
        session_id = sys.argv[2] if len(sys.argv) > 2 else ""
        related = await graph.get_related_sessions(session_id)
        print(json.dumps(related, indent=2))

    elif command == "clusters":
        clusters = await graph.get_concept_clusters()
        print(json.dumps(clusters, indent=2))

    elif command == "timeline":
        project = sys.argv[2] if len(sys.argv) > 2 else None
        timeline = await graph.get_research_timeline(project=project)
        print(json.dumps(timeline, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
