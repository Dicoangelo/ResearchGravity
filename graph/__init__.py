"""
Graph Intelligence Module

Provides relationship queries and lineage visualization:
- Concept graphs (sessions ↔ findings ↔ papers)
- Related sessions/findings discovery
- Research lineage traversal
- Visualization data for OS-App

Built on SQLite lineage table for simplicity.
"""

from .concept_graph import ConceptGraph
from .lineage import LineageTracker, LineageNode, LineageEdge
from .queries import (
    get_related_sessions,
    get_related_findings,
    get_concept_network,
    get_research_lineage,
)

__all__ = [
    'ConceptGraph',
    'LineageTracker',
    'LineageNode',
    'LineageEdge',
    'get_related_sessions',
    'get_related_findings',
    'get_concept_network',
    'get_research_lineage',
]
