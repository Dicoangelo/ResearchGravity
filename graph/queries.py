"""
Graph Query Functions

Convenience functions for common graph queries.
"""

import asyncio
from typing import Dict, Any, List, Optional
from .concept_graph import ConceptGraph

# Global graph instance
_graph: Optional[ConceptGraph] = None


def _get_graph() -> ConceptGraph:
    """Get or create global graph instance."""
    global _graph
    if _graph is None:
        _graph = ConceptGraph()
    return _graph


async def get_related_sessions(
    session_id: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Find sessions related to a given session."""
    graph = _get_graph()
    return await graph.get_related_sessions(session_id, limit=limit)


async def get_related_findings(
    session_id: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Get findings related to a session's findings."""
    graph = _get_graph()
    await graph.load()

    # Get session's findings
    subgraph = await graph.get_session_graph(session_id, depth=1)

    findings = []
    for node in subgraph.nodes:
        if node.type.value == "finding":
            findings.append({
                "id": node.id.replace("finding:", ""),
                "label": node.label,
                "type": node.metadata.get("type"),
                "confidence": node.metadata.get("confidence"),
            })

    return findings[:limit]


async def get_concept_network(
    center_id: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get a concept network centered on a node.

    Returns D3.js compatible format for visualization.
    """
    graph = _get_graph()
    await graph.load()

    # Determine node type from ID format
    if center_id.startswith("session:") or center_id.startswith("finding:"):
        node_id = center_id
    else:
        # Assume it's a session ID
        node_id = f"session:{center_id}"

    subgraph = graph._tracker.get_neighborhood(node_id, depth=depth)
    return subgraph.to_d3_format()


async def get_research_lineage(
    session_id: str,
    include_urls: bool = True,
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Get the complete research lineage for a session.

    Returns:
    - Ancestor sessions (what this research builds on)
    - Descendant sessions (what builds on this research)
    - Key findings and their sources
    """
    graph = _get_graph()
    await graph.load()

    node_id = f"session:{session_id}"

    ancestors = graph._tracker.get_ancestors(node_id, max_depth=max_depth)
    descendants = graph._tracker.get_descendants(node_id, max_depth=max_depth)

    ancestor_sessions = [
        {"id": n.id.replace("session:", ""), "topic": n.label}
        for n in ancestors if n.type.value == "session"
    ]

    descendant_sessions = [
        {"id": n.id.replace("session:", ""), "topic": n.label}
        for n in descendants if n.type.value == "session"
    ]

    # Get session subgraph for findings/urls
    subgraph = await graph.get_session_graph(session_id, depth=1)

    findings = [
        {"id": n.id.replace("finding:", ""), "content": n.label, **n.metadata}
        for n in subgraph.nodes if n.type.value == "finding"
    ]

    urls = []
    if include_urls:
        urls = [
            {"id": n.id.replace("url:", ""), "url": n.metadata.get("full_url", n.label), **n.metadata}
            for n in subgraph.nodes if n.type.value == "url"
        ]

    return {
        "session_id": session_id,
        "ancestors": ancestor_sessions,
        "descendants": descendant_sessions,
        "findings": findings,
        "urls": urls,
        "stats": {
            "ancestor_count": len(ancestor_sessions),
            "descendant_count": len(descendant_sessions),
            "finding_count": len(findings),
            "url_count": len(urls),
        }
    }


async def search_concepts(
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for concepts (sessions, findings) by text.

    Uses SQLite FTS if available, falls back to LIKE.
    """
    from pathlib import Path
    import sqlite3

    db_path = Path.home() / ".agent-core/storage/antigravity.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    results = []

    try:
        cursor = conn.cursor()

        # Try FTS search first
        try:
            cursor.execute("""
                SELECT s.id, s.topic, s.project, 'session' as type
                FROM sessions_fts fts
                JOIN sessions s ON s.id = fts.rowid
                WHERE sessions_fts MATCH ?
                LIMIT ?
            """, (query, limit))
            for row in cursor.fetchall():
                results.append({
                    "id": row['id'],
                    "label": row['topic'],
                    "type": "session",
                    "project": row['project'],
                })
        except sqlite3.OperationalError:
            # FTS not available, use LIKE
            cursor.execute("""
                SELECT id, topic, project, 'session' as type
                FROM sessions
                WHERE topic LIKE ?
                LIMIT ?
            """, (f"%{query}%", limit))
            for row in cursor.fetchall():
                results.append({
                    "id": row['id'],
                    "label": row['topic'],
                    "type": "session",
                    "project": row['project'],
                })

        # Search findings
        remaining = limit - len(results)
        if remaining > 0:
            cursor.execute("""
                SELECT id, content, type as finding_type, session_id
                FROM findings
                WHERE content LIKE ?
                LIMIT ?
            """, (f"%{query}%", remaining))
            for row in cursor.fetchall():
                results.append({
                    "id": row['id'],
                    "label": (row['content'] or '')[:50],
                    "type": "finding",
                    "finding_type": row['finding_type'],
                    "session_id": row['session_id'],
                })

        return results

    finally:
        conn.close()
