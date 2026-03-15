"""
Ontology Generator — Auto-generate entity type schemas from project knowledge

Ported from MiroFish's ontology_generator.py, adapted for ResearchGravity:
- Analyzes findings, sessions, URLs for a given project
- Extracts entity types from content patterns (no LLM required)
- Outputs JSON ontology with entity_types, edge_types, analysis_summary
- Constraint: exactly 10 types (8 domain-specific + Person/Organization fallbacks)

Usage:
    python -m graph.ontology_generator <project_name>
    python -m graph.ontology_generator --all
"""

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Database path
DB_PATH = Path.home() / ".agent-core/storage/antigravity.db"

# Fallback entity types (always included as last 2)
FALLBACK_TYPES = [
    {
        "name": "Person",
        "description": "Any individual not captured by more specific types",
        "attributes": [
            {"name": "role", "type": "text", "description": "Primary role or title"},
            {"name": "expertise", "type": "text", "description": "Domain expertise areas"},
        ],
        "examples": [],
        "is_fallback": True,
    },
    {
        "name": "Organization",
        "description": "Any organization not captured by more specific types",
        "attributes": [
            {"name": "type", "type": "text", "description": "Organization type"},
            {"name": "domain", "type": "text", "description": "Primary domain"},
        ],
        "examples": [],
        "is_fallback": True,
    },
]

# Entity type detection patterns — maps content patterns to entity types
ENTITY_PATTERNS = {
    "Technology": {
        "patterns": [
            r"\b(?:API|SDK|framework|library|model|algorithm|neural|LLM|GPT|transformer|embedding|vector)\b",
            r"\b(?:React|Vue|Python|TypeScript|Rust|Node\.js|SQLite|PostgreSQL)\b",
        ],
        "description": "Software technology, framework, or technical standard",
        "attributes": [
            {"name": "tech_type", "type": "text", "description": "Category: language, framework, model, protocol"},
            {"name": "version", "type": "text", "description": "Current version if known"},
        ],
    },
    "ResearchPaper": {
        "patterns": [
            r"\b(?:arXiv|paper|publication|benchmark|dataset|SOTA|state.of.the.art)\b",
            r"\b(?:abstract|citation|peer.review|preprint|conference|journal)\b",
        ],
        "description": "Academic paper, preprint, or research publication",
        "attributes": [
            {"name": "arxiv_id", "type": "text", "description": "arXiv identifier"},
            {"name": "domain", "type": "text", "description": "Research domain"},
        ],
    },
    "AIModel": {
        "patterns": [
            r"\b(?:Claude|GPT-4|Gemini|Llama|Mistral|Sonnet|Opus|Haiku|o1|o3)\b",
            r"\b(?:fine.tun|RLHF|instruction.tun|multimodal|reasoning.model)\b",
        ],
        "description": "Specific AI/ML model or model family",
        "attributes": [
            {"name": "provider", "type": "text", "description": "Model provider"},
            {"name": "capability", "type": "text", "description": "Primary capability"},
        ],
    },
    "Platform": {
        "patterns": [
            r"\b(?:GitHub|Twitter|LinkedIn|Vercel|AWS|Azure|GCP|Supabase|Netlify)\b",
            r"\b(?:platform|marketplace|ecosystem|app.store|SaaS)\b",
        ],
        "description": "Software platform, cloud service, or marketplace",
        "attributes": [
            {"name": "platform_type", "type": "text", "description": "Type: cloud, social, marketplace"},
            {"name": "integration", "type": "text", "description": "Integration method"},
        ],
    },
    "Concept": {
        "patterns": [
            r"\b(?:sovereignty|cognitive|coherence|ontology|knowledge.graph|RAG|agent)\b",
            r"\b(?:architecture|pattern|paradigm|protocol|methodology)\b",
        ],
        "description": "Technical concept, design pattern, or methodology",
        "attributes": [
            {"name": "domain", "type": "text", "description": "Knowledge domain"},
            {"name": "maturity", "type": "text", "description": "Maturity: emerging, established, deprecated"},
        ],
    },
    "Company": {
        "patterns": [
            r"\b(?:Anthropic|OpenAI|Google|Meta|Microsoft|Apple|Amazon|Nvidia)\b",
            r"\b(?:startup|company|corporation|enterprise|venture|inc|corp)\b",
        ],
        "description": "Company, startup, or commercial entity",
        "attributes": [
            {"name": "sector", "type": "text", "description": "Industry sector"},
            {"name": "stage", "type": "text", "description": "Stage: startup, growth, enterprise"},
        ],
    },
    "Project": {
        "patterns": [
            r"\b(?:repo|repository|codebase|monorepo|workspace|module|package)\b",
            r"\b(?:OS-App|ResearchGravity|meta-vengine|CareerCoach|MiroFish)\b",
        ],
        "description": "Software project, repository, or product",
        "attributes": [
            {"name": "stack", "type": "text", "description": "Primary tech stack"},
            {"name": "status", "type": "text", "description": "Status: active, archived, planned"},
        ],
    },
    "DataSource": {
        "patterns": [
            r"\b(?:database|SQLite|Qdrant|Postgres|Redis|S3|bucket|storage)\b",
            r"\b(?:dataset|corpus|index|vector.store|embedding.store)\b",
        ],
        "description": "Data storage system, dataset, or data pipeline",
        "attributes": [
            {"name": "format", "type": "text", "description": "Storage format: SQL, vector, document, KV"},
            {"name": "scale", "type": "text", "description": "Scale: local, distributed, cloud"},
        ],
    },
    "Metric": {
        "patterns": [
            r"\b(?:DQ.score|accuracy|precision|recall|F1|latency|throughput)\b",
            r"\b(?:benchmark|performance|quality|cost|token|efficiency)\b",
        ],
        "description": "Performance metric, benchmark, or quality measure",
        "attributes": [
            {"name": "unit", "type": "text", "description": "Measurement unit"},
            {"name": "target", "type": "text", "description": "Target value or threshold"},
        ],
    },
    "Workflow": {
        "patterns": [
            r"\b(?:pipeline|workflow|CI/CD|deployment|migration|ETL|ingestion)\b",
            r"\b(?:orchestrat|coordinat|scheduling|batch|stream|queue)\b",
        ],
        "description": "Process workflow, pipeline, or automation",
        "attributes": [
            {"name": "trigger", "type": "text", "description": "Trigger type: manual, scheduled, event"},
            {"name": "stages", "type": "text", "description": "Pipeline stages"},
        ],
    },
    "Standard": {
        "patterns": [
            r"\b(?:RFC|spec|specification|standard|protocol|MCP|JSON-RPC|REST|GraphQL)\b",
            r"\b(?:compliance|regulation|GDPR|SOC2|ISO)\b",
        ],
        "description": "Technical standard, specification, or protocol",
        "attributes": [
            {"name": "body", "type": "text", "description": "Standards body or author"},
            {"name": "version", "type": "text", "description": "Specification version"},
        ],
    },
    "UseCase": {
        "patterns": [
            r"\b(?:use.case|user.story|requirement|feature|capability|scenario)\b",
            r"\b(?:integration|implementation|deployment|adoption|migration)\b",
        ],
        "description": "Product use case, feature, or capability",
        "attributes": [
            {"name": "priority", "type": "text", "description": "Priority: P0, P1, P2"},
            {"name": "status", "type": "text", "description": "Status: planned, building, shipped"},
        ],
    },
}

# Edge type templates
EDGE_TEMPLATES = [
    {
        "name": "USES",
        "description": "Entity uses or depends on another entity",
        "source_targets": [],
    },
    {
        "name": "PRODUCES",
        "description": "Entity produces, generates, or creates another entity",
        "source_targets": [],
    },
    {
        "name": "INFORMS",
        "description": "Entity provides knowledge or insight to another",
        "source_targets": [],
    },
    {
        "name": "EVOLVES_FROM",
        "description": "Entity is a newer version or evolution of another",
        "source_targets": [],
    },
    {
        "name": "COMPETES_WITH",
        "description": "Entities serve similar purposes or compete in the same space",
        "source_targets": [],
    },
    {
        "name": "MEASURES",
        "description": "A metric or benchmark evaluates an entity",
        "source_targets": [],
    },
]


class OntologyGenerator:
    """Generate entity type ontologies from project knowledge."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_project_content(self, project: Optional[str] = None) -> List[str]:
        """Extract all text content for a project."""
        conn = self._get_connection()
        texts = []

        try:
            cursor = conn.cursor()

            # Get findings
            if project:
                cursor.execute(
                    "SELECT content FROM findings WHERE project = ? AND content IS NOT NULL",
                    (project,),
                )
            else:
                cursor.execute(
                    "SELECT content FROM findings WHERE content IS NOT NULL LIMIT 5000"
                )

            for row in cursor.fetchall():
                texts.append(row["content"])

            # Get session topics
            if project:
                cursor.execute(
                    "SELECT topic FROM sessions WHERE project = ? AND topic IS NOT NULL",
                    (project,),
                )
            else:
                cursor.execute(
                    "SELECT topic FROM sessions WHERE topic IS NOT NULL LIMIT 1000"
                )

            for row in cursor.fetchall():
                texts.append(row["topic"])

            return texts

        finally:
            conn.close()

    def _score_entity_types(self, texts: List[str]) -> List[Tuple[str, int, List[str]]]:
        """Score entity types by pattern match frequency. Returns (name, score, examples)."""
        combined_text = "\n".join(texts)
        scores = []

        for type_name, config in ENTITY_PATTERNS.items():
            match_count = 0
            examples = set()

            for pattern in config["patterns"]:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                match_count += len(matches)
                for m in matches[:5]:
                    examples.add(m)

            if match_count > 0:
                scores.append((type_name, match_count, list(examples)[:5]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def generate(
        self, project: Optional[str] = None, max_domain_types: int = 8
    ) -> Dict[str, Any]:
        """Generate ontology for a project.

        Args:
            project: Project name (None = analyze all data)
            max_domain_types: Max domain-specific types (default 8, + 2 fallbacks = 10)

        Returns:
            Ontology dict with entity_types, edge_types, analysis_summary
        """
        texts = self._get_project_content(project)

        if not texts:
            return {
                "entity_types": FALLBACK_TYPES,
                "edge_types": EDGE_TEMPLATES,
                "analysis_summary": f"No content found for project '{project}'",
                "generated_at": datetime.now().isoformat(),
            }

        # Score entity types
        scored = self._score_entity_types(texts)

        # Take top N domain types
        domain_types = []
        for type_name, score, examples in scored[:max_domain_types]:
            config = ENTITY_PATTERNS[type_name]
            entity_type = {
                "name": type_name,
                "description": config["description"],
                "attributes": config["attributes"],
                "examples": examples,
                "match_score": score,
            }
            domain_types.append(entity_type)

        # Combine: domain types + fallbacks
        entity_types = domain_types + FALLBACK_TYPES

        # Generate edge types based on which entity types are present
        type_names = [t["name"] for t in entity_types]
        edge_types = []
        for template in EDGE_TEMPLATES:
            edge = dict(template)
            # Auto-populate source_targets based on present types
            edge["source_targets"] = [
                {"source": type_names[i], "target": type_names[j]}
                for i in range(min(3, len(type_names)))
                for j in range(min(3, len(type_names)))
                if i != j
            ][:6]
            edge_types.append(edge)

        # Analysis summary
        total_matches = sum(s[1] for s in scored)
        top_types = [f"{name} ({score})" for name, score, _ in scored[:5]]

        return {
            "entity_types": entity_types,
            "edge_types": edge_types,
            "analysis_summary": (
                f"Analyzed {len(texts)} text fragments. "
                f"Total pattern matches: {total_matches}. "
                f"Top entity types: {', '.join(top_types)}. "
                f"Generated {len(domain_types)} domain types + 2 fallbacks."
            ),
            "project": project or "all",
            "text_count": len(texts),
            "generated_at": datetime.now().isoformat(),
        }


async def main():
    """CLI for ontology generation."""
    import sys

    gen = OntologyGenerator()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m graph.ontology_generator <project_name>")
        print("  python -m graph.ontology_generator --all")
        sys.exit(1)

    project = None if sys.argv[1] == "--all" else sys.argv[1]
    ontology = gen.generate(project=project)

    print(json.dumps(ontology, indent=2, default=str))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
