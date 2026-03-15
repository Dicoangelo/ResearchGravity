"""
Graph-to-Persona Generator — Dynamic SUPERMAX council members from knowledge graph

Ported from MiroFish's oasis_profile_generator.py, adapted for SUPERMAX:
- Queries knowledge graph clusters around a topic
- Extracts key entity types, expertise domains, relationship patterns
- Generates agent persona .md files in SUPERMAX format
- Personas are grounded in real graph data, not static definitions

Usage:
    python -m graph.persona_generator "multi-agent orchestration"
    python -m graph.persona_generator "cognitive architecture" --count 4
"""

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Database path
DB_PATH = Path.home() / ".agent-core/storage/antigravity.db"

# SUPERMAX agent template
PERSONA_TEMPLATE = """---
name: {agent_id}
description: {description}
tools: Read, Grep, Glob, WebSearch
model: opus
maxTurns: 10
---

You are a **{title}** — a domain expert dynamically generated from the ResearchGravity knowledge graph.

## Expertise Profile

**Domain:** {domain}
**Grounded in:** {grounding}
**Key entities:** {entities}
**Knowledge depth:** {depth_label} ({session_count} sessions, {finding_count} findings)

## Perspective

{perspective}

## Review Protocol

When reviewing a decision, design, or proposal:
1. Evaluate through the lens of **{domain}** — does this align with established patterns?
2. Cross-reference against known entities: {entity_list}
3. Identify gaps where the proposal diverges from graph-validated knowledge
4. Assess based on {finding_count} findings worth of domain evidence

Structure your response EXACTLY as:

**Position**: Your recommendation in 1-2 sentences.

**Reasoning**: The key factors driving your position (3-5 bullet points).

**Blocking concerns**: Issues that would prevent you from approving (list each, or "None").

**Graph evidence**: Specific findings or sessions that support your position.

**Confidence**: low | medium | high
"""


class PersonaGenerator:
    """Generate SUPERMAX agent personas from the knowledge graph."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _find_relevant_sessions(
        self, query: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Find sessions related to the query."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.id, s.topic, s.project, s.status,
                       COUNT(DISTINCT f.id) as finding_count,
                       COUNT(DISTINCT u.id) as url_count
                FROM sessions s
                LEFT JOIN findings f ON f.session_id = s.id
                LEFT JOIN urls u ON u.session_id = s.id
                WHERE s.topic LIKE ? OR s.project LIKE ?
                GROUP BY s.id
                ORDER BY finding_count DESC
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _get_domain_findings(
        self, query: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Get findings related to the query domain."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, content, type, project, confidence
                FROM findings
                WHERE content LIKE ?
                ORDER BY confidence DESC NULLS LAST
                LIMIT ?
            """,
                (f"%{query}%", limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _extract_expertise_clusters(
        self, findings: List[Dict], sessions: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract expertise clusters from findings and sessions."""
        # Count finding types
        type_counts = Counter(f.get("type", "general") for f in findings)

        # Count projects
        project_counts = Counter(
            s.get("project", "unknown")
            for s in sessions
            if s.get("project")
        )

        # Extract key terms from findings
        all_content = " ".join(f.get("content", "") for f in findings)
        words = all_content.lower().split()

        # Domain-relevant terms (longer words, more specific)
        term_counts = Counter(
            w for w in words if len(w) > 5 and w.isalpha()
        )
        top_terms = [term for term, _ in term_counts.most_common(20)]

        clusters = []

        # Cluster by finding type
        for ftype, count in type_counts.most_common(5):
            type_findings = [f for f in findings if f.get("type") == ftype]
            cluster_terms = []
            for f in type_findings[:10]:
                content_words = f.get("content", "").lower().split()
                cluster_terms.extend(w for w in content_words if len(w) > 5 and w.isalpha())

            cluster_top = [t for t, _ in Counter(cluster_terms).most_common(8)]

            clusters.append({
                "type": ftype,
                "count": count,
                "key_terms": cluster_top,
                "projects": list(set(
                    f.get("project", "") for f in type_findings if f.get("project")
                )),
            })

        return clusters

    def _generate_persona(
        self,
        cluster: Dict[str, Any],
        index: int,
        query: str,
        total_sessions: int,
        total_findings: int,
    ) -> Dict[str, Any]:
        """Generate a single persona from a cluster."""
        ftype = cluster["type"]
        terms = cluster["key_terms"]
        projects = cluster["projects"]
        count = cluster["count"]

        # Generate persona attributes
        type_to_title = {
            "technical": "Systems Architect",
            "implementation": "Implementation Engineer",
            "innovation": "Innovation Strategist",
            "general": "Domain Analyst",
            "metrics": "Performance Analyst",
            "architecture": "Architecture Reviewer",
            "research": "Research Scientist",
            "strategic": "Strategic Advisor",
            "design": "Design Thinker",
            "security": "Security Specialist",
        }

        title = type_to_title.get(ftype, f"{ftype.title()} Specialist")
        agent_id = f"graph-{ftype}-{index}"

        # Depth label based on finding count
        if count > 50:
            depth_label = "Deep expertise"
        elif count > 20:
            depth_label = "Solid foundation"
        elif count > 5:
            depth_label = "Working knowledge"
        else:
            depth_label = "Emerging awareness"

        # Generate perspective based on type
        perspectives = {
            "technical": (
                f"You evaluate proposals through the lens of technical feasibility and system design. "
                f"Your knowledge spans {', '.join(terms[:5])} — grounded in {count} verified findings. "
                f"You prioritize correctness, performance, and maintainability."
            ),
            "implementation": (
                f"You evaluate proposals through the lens of practical implementation. "
                f"You know what works in production because you've seen {count} implementation findings. "
                f"Key focus areas: {', '.join(terms[:5])}. You flag complexity that won't survive contact with reality."
            ),
            "innovation": (
                f"You evaluate proposals through the lens of novel combinations and whitespace opportunities. "
                f"Your {count} innovation findings reveal patterns others miss. "
                f"Key domains: {', '.join(terms[:5])}. You push for 10x moves, not incremental improvements."
            ),
            "general": (
                f"You provide broad domain analysis grounded in {count} findings across {', '.join(terms[:5])}. "
                f"You connect dots between disparate domains and identify cross-cutting concerns."
            ),
            "metrics": (
                f"You evaluate proposals through the lens of measurable outcomes. "
                f"Your {count} metrics findings establish baselines for quality, performance, and cost. "
                f"Key metrics: {', '.join(terms[:5])}. Nothing ships without a measurement plan."
            ),
        }
        perspective = perspectives.get(
            ftype,
            f"You bring specialized expertise in {ftype} with {count} grounded findings "
            f"across {', '.join(terms[:5])}."
        )

        entity_examples = terms[:6]
        domain = f"{query} / {ftype}"

        md_content = PERSONA_TEMPLATE.format(
            agent_id=agent_id,
            description=f"Graph-generated {title} for {query} domain ({count} findings)",
            title=title,
            domain=domain,
            grounding=f"{total_sessions} sessions, {total_findings} findings in the knowledge graph",
            entities=", ".join(entity_examples),
            depth_label=depth_label,
            session_count=total_sessions,
            finding_count=count,
            perspective=perspective,
            entity_list=", ".join(entity_examples),
        )

        return {
            "agent_id": agent_id,
            "title": title,
            "domain": domain,
            "finding_count": count,
            "depth_label": depth_label,
            "key_terms": terms,
            "projects": projects,
            "md_content": md_content,
        }

    def generate(
        self, query: str, count: int = 4, min_findings: int = 3
    ) -> Dict[str, Any]:
        """Generate persona council for a topic.

        Args:
            query: Topic or domain to generate personas for
            count: Number of personas to generate (default 4)
            min_findings: Minimum findings per cluster to qualify (default 3)

        Returns:
            Dict with personas list, summary, and metadata
        """
        sessions = self._find_relevant_sessions(query)
        findings = self._get_domain_findings(query)

        if not findings:
            return {
                "personas": [],
                "summary": f"No findings found for '{query}' — cannot generate personas",
                "query": query,
                "generated_at": datetime.now().isoformat(),
            }

        clusters = self._extract_expertise_clusters(findings, sessions)

        # Filter by minimum findings
        qualified = [c for c in clusters if c["count"] >= min_findings]

        # Generate personas for top clusters
        personas = []
        for i, cluster in enumerate(qualified[:count]):
            persona = self._generate_persona(
                cluster=cluster,
                index=i,
                query=query,
                total_sessions=len(sessions),
                total_findings=len(findings),
            )
            personas.append(persona)

        return {
            "personas": personas,
            "summary": (
                f"Generated {len(personas)} personas for '{query}' from "
                f"{len(sessions)} sessions and {len(findings)} findings. "
                f"Clusters analyzed: {len(clusters)} ({len(qualified)} qualified)."
            ),
            "query": query,
            "session_count": len(sessions),
            "finding_count": len(findings),
            "generated_at": datetime.now().isoformat(),
        }

    def save_personas(
        self, result: Dict[str, Any], output_dir: Optional[Path] = None
    ) -> List[Path]:
        """Save generated personas as .md files.

        Args:
            result: Output from generate()
            output_dir: Directory to save to (default: ~/.claude/plugins/decosystem-supermax/agents/)

        Returns:
            List of paths to saved files
        """
        if output_dir is None:
            output_dir = (
                Path.home()
                / ".claude/plugins/decosystem-supermax/agents"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for persona in result.get("personas", []):
            filepath = output_dir / f"{persona['agent_id']}.md"
            filepath.write_text(persona["md_content"])
            saved.append(filepath)

        return saved


def main():
    """CLI for persona generation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print('  python -m graph.persona_generator "topic"')
        print('  python -m graph.persona_generator "topic" --count 4')
        print('  python -m graph.persona_generator "topic" --save')
        sys.exit(1)

    query = sys.argv[1]
    count = 4
    save = False

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--count" and i + 1 < len(sys.argv):
            count = int(sys.argv[i + 1])
        if arg == "--save":
            save = True

    gen = PersonaGenerator()
    result = gen.generate(query=query, count=count)

    print(result["summary"])
    print()

    for persona in result["personas"]:
        print(f"  [{persona['agent_id']}] {persona['title']}")
        print(f"    Depth: {persona['depth_label']} ({persona['finding_count']} findings)")
        print(f"    Terms: {', '.join(persona['key_terms'][:5])}")
        print()

    if save:
        saved = gen.save_personas(result)
        print(f"Saved {len(saved)} persona files:")
        for path in saved:
            print(f"  {path}")


if __name__ == "__main__":
    main()
