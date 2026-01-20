"""
Lineage Tracking

Data structures and utilities for research lineage:
- Sessions → Findings → Papers relationships
- Temporal ordering
- Influence chains
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    SESSION = "session"
    FINDING = "finding"
    PAPER = "paper"
    URL = "url"
    CONCEPT = "concept"
    PROJECT = "project"


class EdgeType(Enum):
    """Types of relationships between nodes."""
    CONTAINS = "contains"       # session → finding
    CITES = "cites"             # finding → paper/url
    DERIVES_FROM = "derives_from"  # finding → finding
    ENABLES = "enables"         # session → session
    INFORMS = "informs"         # paper → finding
    BELONGS_TO = "belongs_to"   # session → project
    RELATED = "related"         # semantic similarity


@dataclass
class LineageNode:
    """A node in the knowledge graph."""
    id: str
    type: NodeType
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class LineageEdge:
    """An edge (relationship) in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class LineageGraph:
    """A subgraph of the knowledge graph."""
    nodes: List[LineageNode] = field(default_factory=list)
    edges: List[LineageEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    def to_d3_format(self) -> Dict[str, Any]:
        """Convert to D3.js force-directed graph format."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "group": n.type.value,
                    "label": n.label,
                    **n.metadata
                }
                for n in self.nodes
            ],
            "links": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "value": e.weight,
                    "type": e.edge_type.value,
                }
                for e in self.edges
            ]
        }


class LineageTracker:
    """
    Tracks and queries research lineage.

    Lineage represents the flow of knowledge:
    - Papers → inform → Findings
    - Sessions → contain → Findings
    - Findings → derive_from → Findings
    - Sessions → enable → Sessions
    """

    def __init__(self):
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        self._adjacency: Dict[str, Set[str]] = {}  # source → targets
        self._reverse_adjacency: Dict[str, Set[str]] = {}  # target → sources

    def add_node(self, node: LineageNode):
        """Add a node to the graph."""
        self._nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = set()
        if node.id not in self._reverse_adjacency:
            self._reverse_adjacency[node.id] = set()

    def add_edge(self, edge: LineageEdge):
        """Add an edge to the graph."""
        self._edges.append(edge)

        # Update adjacency
        if edge.source_id not in self._adjacency:
            self._adjacency[edge.source_id] = set()
        self._adjacency[edge.source_id].add(edge.target_id)

        # Update reverse adjacency
        if edge.target_id not in self._reverse_adjacency:
            self._reverse_adjacency[edge.target_id] = set()
        self._reverse_adjacency[edge.target_id].add(edge.source_id)

    def get_ancestors(self, node_id: str, max_depth: int = 3) -> List[LineageNode]:
        """Get all nodes that lead to this node (sources)."""
        visited = set()
        ancestors = []

        def traverse(nid: str, depth: int):
            if depth > max_depth or nid in visited:
                return
            visited.add(nid)

            for source_id in self._reverse_adjacency.get(nid, []):
                if source_id in self._nodes:
                    ancestors.append(self._nodes[source_id])
                    traverse(source_id, depth + 1)

        traverse(node_id, 0)
        return ancestors

    def get_descendants(self, node_id: str, max_depth: int = 3) -> List[LineageNode]:
        """Get all nodes that derive from this node (targets)."""
        visited = set()
        descendants = []

        def traverse(nid: str, depth: int):
            if depth > max_depth or nid in visited:
                return
            visited.add(nid)

            for target_id in self._adjacency.get(nid, []):
                if target_id in self._nodes:
                    descendants.append(self._nodes[target_id])
                    traverse(target_id, depth + 1)

        traverse(node_id, 0)
        return descendants

    def get_neighborhood(self, node_id: str, depth: int = 1) -> LineageGraph:
        """Get the local neighborhood of a node."""
        visited_nodes = set()
        visited_edges = []

        def traverse(nid: str, current_depth: int):
            if current_depth > depth or nid in visited_nodes:
                return
            visited_nodes.add(nid)

            # Get outgoing edges
            for target_id in self._adjacency.get(nid, []):
                for edge in self._edges:
                    if edge.source_id == nid and edge.target_id == target_id:
                        visited_edges.append(edge)
                        traverse(target_id, current_depth + 1)

            # Get incoming edges
            for source_id in self._reverse_adjacency.get(nid, []):
                for edge in self._edges:
                    if edge.target_id == nid and edge.source_id == source_id:
                        visited_edges.append(edge)
                        traverse(source_id, current_depth + 1)

        traverse(node_id, 0)

        return LineageGraph(
            nodes=[self._nodes[nid] for nid in visited_nodes if nid in self._nodes],
            edges=visited_edges
        )

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[LineageNode]:
        """Find a path between two nodes (BFS)."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return []

        visited = {source_id}
        queue = [(source_id, [self._nodes[source_id]])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for next_id in self._adjacency.get(current_id, []):
                if next_id == target_id:
                    return path + [self._nodes[target_id]]

                if next_id not in visited and next_id in self._nodes:
                    visited.add(next_id)
                    queue.append((next_id, path + [self._nodes[next_id]]))

        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        type_counts = {}
        for node in self._nodes.values():
            type_counts[node.type.value] = type_counts.get(node.type.value, 0) + 1

        edge_type_counts = {}
        for edge in self._edges:
            edge_type_counts[edge.edge_type.value] = edge_type_counts.get(edge.edge_type.value, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
        }
