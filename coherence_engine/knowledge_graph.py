"""
Coherence Engine — Knowledge Graph

Entity extraction and graph management for the cognitive knowledge graph.

Extracts entities (concepts, tools, projects, papers, people, errors) from
cognitive events using regex + rule-based patterns (no external NLP deps).
Builds a co-occurrence graph with spreading activation for traversal.

Usage (standalone):
    import asyncio, asyncpg
    from coherence_engine.knowledge_graph import EntityExtractor, GraphManager

    pool = await asyncpg.create_pool("postgresql://localhost:5432/ucw_cognitive")
    extractor = EntityExtractor()
    graph = GraphManager(pool)

    entities = extractor.extract(event_row)
    await graph.ingest_entities(entities, event_row)
    activated = await graph.spreading_activation("ent-concept-ucw", depth=3)
"""

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg

from . import config as cfg

import logging

log = logging.getLogger("coherence.knowledge_graph")


# ── Entity Types ─────────────────────────────────────────────

ENTITY_TYPES = {
    "concept",
    "tool",
    "project",
    "person",
    "error",
    "paper",
    "technology",
    "platform",
}

# ── Regex Patterns ───────────────────────────────────────────

# arXiv paper IDs: 2505.19591, arXiv:2505.19591, arxiv.org/abs/2505.19591
RE_ARXIV = re.compile(
    r"(?:arXiv[:\s]*|arxiv\.org/abs/)?"
    r"(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)

# GitHub repos: only match github.com/owner/repo or owner/repo.git patterns
# The non-URL form requires .git suffix to avoid false positives on file paths
RE_GITHUB_REPO = re.compile(
    r"github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100})"
    r"(?:\.git)?(?:/|(?=\s|$|[)\],;:\"']))"
    r"|"
    r"(?<![/\\])([A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100})\.git",
)

# Error patterns: common error types
RE_ERROR = re.compile(
    r"((?:TypeError|ValueError|KeyError|AttributeError|ImportError"
    r"|RuntimeError|ConnectionError|TimeoutError|FileNotFoundError"
    r"|PermissionError|ModuleNotFoundError|SyntaxError|NameError"
    r"|IndexError|ZeroDivisionError|StopIteration"
    r"|asyncio\.TimeoutError|asyncpg\.\w+Error"
    r"|OSError|IOError|HTTPError|JSONDecodeError"
    r"|IntegrityError|OperationalError)(?::\s*[^\n]{0,120})?)",
    re.IGNORECASE,
)

# ── Known Projects ───────────────────────────────────────────

KNOWN_PROJECTS = {
    "os-app": {"aliases": ["os app", "osapp", "metaventions ai platform"]},
    "careercoach": {"aliases": ["career coach", "careercoachAntigravity", "career governance"]},
    "ucw": {"aliases": ["universal cognitive wallet", "cognitive wallet"]},
    "researchgravity": {"aliases": ["research gravity", "researchgravity"]},
    "meta-vengine": {"aliases": ["meta vengine", "metavengine", "routing engine"]},
    "agent-core": {"aliases": ["agent core", "agentcore"]},
    "clawdbot": {"aliases": ["clawd bot"]},
    "voice-nexus": {"aliases": ["voice nexus"]},
    "antigravity": {"aliases": ["antigravity ecosystem", "d-ecosystem", "decosystem"]},
    "metaventions": {"aliases": ["metaventions ai"]},
    "sovereign-deck": {"aliases": []},
    "enterprise-deck": {"aliases": []},
    "the-decosystem": {"aliases": ["decosystem blueprint"]},
}

# Build reverse lookup: alias -> canonical name
_PROJECT_ALIASES: Dict[str, str] = {}
for canonical, info in KNOWN_PROJECTS.items():
    _PROJECT_ALIASES[canonical.lower()] = canonical
    for alias in info.get("aliases", []):
        _PROJECT_ALIASES[alias.lower()] = canonical

# ── Known Technologies ───────────────────────────────────────

KNOWN_TECHNOLOGIES = {
    # Languages & runtimes
    "python", "typescript", "javascript", "rust", "go", "deno", "node.js",
    # Frameworks
    "react", "next.js", "nextjs", "vite", "express", "fastapi", "flask",
    "svelte", "sveltekit", "nuxt", "astro", "remix",
    # Databases & storage
    "postgresql", "postgres", "sqlite", "qdrant", "pgvector", "supabase",
    "redis", "mongodb",
    # AI/ML
    "openai", "anthropic", "claude", "gpt-4", "gpt-4o", "chatgpt",
    "transformers", "pytorch", "tensorflow", "sbert",
    "cohere", "embeddings", "langchain", "llamaindex",
    # Infrastructure
    "docker", "kubernetes", "vercel", "aws", "gcp", "azure",
    "github actions", "ci/cd",
    # Protocols & standards
    "mcp", "json-rpc", "graphql", "rest", "websocket", "grpc",
    # Libraries
    "asyncpg", "asyncio", "zustand", "tailwind", "vitest",
}

# Build case-insensitive lookup
_TECH_LOWER: Dict[str, str] = {t.lower(): t for t in KNOWN_TECHNOLOGIES}

# ── Known Platforms ──────────────────────────────────────────

KNOWN_PLATFORMS = {
    "claude-code", "claude-cli", "claude-desktop", "chatgpt", "grok", "ccc",
}

# Words to exclude from concept extraction (too generic)
STOP_CONCEPTS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "must",
    "and", "or", "but", "not", "no", "yes", "if", "then", "else",
    "this", "that", "it", "they", "we", "you", "he", "she",
    "for", "from", "with", "at", "by", "to", "in", "on", "of",
    "up", "out", "off", "over", "under", "about", "into", "through",
    "just", "also", "very", "too", "really", "quite", "much",
    "here", "there", "where", "when", "how", "what", "which", "who",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "same", "than", "so", "like", "new",
    "one", "two", "three", "four", "five", "now", "then", "still",
    "use", "used", "using", "get", "got", "make", "made", "see",
    "need", "want", "think", "know", "say", "said", "way", "well",
    "back", "set", "take", "try", "let", "keep", "put", "run",
    "going", "being", "having", "done", "got", "getting",
    "file", "code", "data", "system", "time", "user", "work",
    "type", "text", "value", "key", "name", "id", "list",
    "first", "last", "next", "good", "right", "look", "help",
    "part", "change", "point", "thing", "place", "case", "line",
    "turn", "start", "end", "move", "show", "side", "call", "come",
    "world", "give", "group", "own", "day", "long", "great",
    "even", "after", "before", "high", "low", "old", "big", "small",
    "between", "never", "always", "left", "since", "around",
    "another", "those", "why", "these", "many", "number", "people",
    "hand", "many", "different", "away", "again", "still", "already",
    "while", "something", "nothing", "everything",
    # Common in our domain but too vague alone
    "function", "class", "method", "module", "config", "test",
    "error", "update", "create", "delete", "read", "write",
    "event", "session", "table", "query", "result", "response",
    "request", "process", "log", "message", "content",
}


# ── Dataclasses ──────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    """An entity extracted from a cognitive event."""
    entity_type: str
    name: str
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""  # which extractor found it

    @property
    def entity_id(self) -> str:
        """Deterministic ID from type + normalized name."""
        norm = self.name.lower().strip()
        return f"ent-{self.entity_type}-{hashlib.sha256(norm.encode()).hexdigest()[:12]}"


# ── Entity Extractor ─────────────────────────────────────────

class EntityExtractor:
    """
    Rule-based entity extractor for cognitive events.

    Extracts entities from:
      - data_layer.content (regex patterns)
      - light_layer.concepts (direct)
      - light_layer.topic (mapped to concept)

    No external NLP libraries required.
    """

    def extract(self, event_row: Dict[str, Any]) -> List[ExtractedEntity]:
        """
        Extract all entities from a cognitive event row.

        Args:
            event_row: Dict with data_layer, light_layer, instinct_layer, platform, etc.

        Returns:
            Deduplicated list of ExtractedEntity objects.
        """
        entities: List[ExtractedEntity] = []
        seen: Set[str] = set()  # track entity_id to dedup

        # Get content text
        data_layer = event_row.get("data_layer") or {}
        if isinstance(data_layer, str):
            data_layer = json.loads(data_layer)
        content = data_layer.get("content", "")

        # Get light layer
        light_layer = event_row.get("light_layer") or {}
        if isinstance(light_layer, str):
            light_layer = json.loads(light_layer)

        # 1. Extract from content via regex
        content_entities = self._extract_from_content(content)
        for ent in content_entities:
            if ent.entity_id not in seen:
                seen.add(ent.entity_id)
                entities.append(ent)

        # 2. Extract from light layer
        light_entities = self._extract_from_light_layer(light_layer)
        for ent in light_entities:
            if ent.entity_id not in seen:
                seen.add(ent.entity_id)
                entities.append(ent)

        # 3. Extract platform as entity
        platform = event_row.get("platform", "")
        if platform and platform in KNOWN_PLATFORMS:
            ent = ExtractedEntity(
                entity_type="platform",
                name=platform,
                source="platform_field",
            )
            if ent.entity_id not in seen:
                seen.add(ent.entity_id)
                entities.append(ent)

        return entities

    def _extract_from_content(self, content: str) -> List[ExtractedEntity]:
        """Extract entities from raw content text using regex patterns."""
        if not content or len(content) < 10:
            return []

        entities = []

        # arXiv papers
        for match in RE_ARXIV.finditer(content):
            paper_id = match.group(1)
            entities.append(ExtractedEntity(
                entity_type="paper",
                name=paper_id,
                confidence=0.95,
                source="regex_arxiv",
            ))

        # GitHub repos (two capture groups from alternation: group(1) or group(2))
        for match in RE_GITHUB_REPO.finditer(content):
            repo = match.group(1) or match.group(2)
            if not repo or "/" not in repo:
                continue
            owner, name = repo.split("/", 1)
            if len(owner) < 2 or len(name) < 2:
                continue
            # Skip common false positives
            if owner.lower() in {"e.g", "i.e", "etc", "vs", "http", "https"}:
                continue
            entities.append(ExtractedEntity(
                entity_type="tool",
                name=repo,
                confidence=0.9,
                source="regex_github",
            ))

        # Known projects
        content_lower = content.lower()
        for alias_lower, canonical in _PROJECT_ALIASES.items():
            if alias_lower in content_lower:
                entities.append(ExtractedEntity(
                    entity_type="project",
                    name=canonical,
                    aliases=KNOWN_PROJECTS[canonical].get("aliases", []),
                    confidence=0.85,
                    source="known_project",
                ))

        # Known technologies
        for tech_lower, tech_canonical in _TECH_LOWER.items():
            # Word-boundary check: look for the technology term surrounded by
            # non-alphanumeric characters (or start/end of string)
            if tech_lower in content_lower:
                # Verify it's a word boundary match, not a substring
                pattern = r"(?<![a-zA-Z0-9_-])" + re.escape(tech_lower) + r"(?![a-zA-Z0-9_-])"
                if re.search(pattern, content_lower):
                    entities.append(ExtractedEntity(
                        entity_type="technology",
                        name=tech_canonical,
                        confidence=0.8,
                        source="known_tech",
                    ))

        # Errors
        for match in RE_ERROR.finditer(content):
            error_text = match.group(1).strip()
            # Normalize: just the error type for the entity name
            error_type = error_text.split(":")[0].strip()
            entities.append(ExtractedEntity(
                entity_type="error",
                name=error_type,
                confidence=0.9,
                source="regex_error",
            ))

        return entities

    def _extract_from_light_layer(
        self, light_layer: Dict[str, Any]
    ) -> List[ExtractedEntity]:
        """Extract entities from the light layer (concepts, topic)."""
        entities = []

        # Concepts from light_layer.concepts
        concepts = light_layer.get("concepts", [])
        if isinstance(concepts, list):
            for concept in concepts:
                if not isinstance(concept, str):
                    continue
                concept_clean = concept.strip().lower()
                if len(concept_clean) < 3:
                    continue
                if concept_clean in STOP_CONCEPTS:
                    continue

                entities.append(ExtractedEntity(
                    entity_type="concept",
                    name=concept_clean,
                    confidence=0.7,
                    source="light_layer_concept",
                ))

        # Topic as concept
        topic = light_layer.get("topic", "")
        if isinstance(topic, str) and len(topic) >= 3:
            topic_clean = topic.strip().lower()
            if topic_clean not in STOP_CONCEPTS:
                entities.append(ExtractedEntity(
                    entity_type="concept",
                    name=topic_clean,
                    confidence=0.75,
                    source="light_layer_topic",
                ))

        return entities


# ── Graph Manager ────────────────────────────────────────────

class GraphManager:
    """
    Manages the cognitive knowledge graph in PostgreSQL.

    Handles:
      - Entity upsert with dedup (on name + entity_type)
      - Co-occurrence edge creation
      - Mention count and platform tracking
      - Spreading activation traversal
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def ingest_entities(
        self,
        entities: List[ExtractedEntity],
        event_row: Dict[str, Any],
    ) -> Tuple[int, int]:
        """
        Ingest extracted entities into the graph.

        Upserts entities, updates counts, and creates co-occurrence edges
        for all entity pairs found in the same event.

        Args:
            entities: Extracted entities from EntityExtractor.extract()
            event_row: The source event row (for timestamp and platform)

        Returns:
            Tuple of (entities_upserted, edges_created)
        """
        if not entities:
            return 0, 0

        timestamp_ns = event_row.get("timestamp_ns", time.time_ns())
        platform = event_row.get("platform", "unknown")
        event_id = event_row.get("event_id", "")

        # 1. Upsert all entities
        entity_ids = []
        async with self._pool.acquire() as conn:
            for ent in entities:
                eid = ent.entity_id
                try:
                    await conn.execute(
                        """INSERT INTO cognitive_entities
                               (entity_id, entity_type, name, aliases,
                                first_seen_ns, last_seen_ns,
                                mention_count, platform_count, platforms, metadata)
                           VALUES ($1, $2, $3, $4, $5, $5, 1, 1, ARRAY[$6]::text[], $7)
                           ON CONFLICT (entity_id) DO UPDATE SET
                               last_seen_ns = GREATEST(
                                   cognitive_entities.last_seen_ns, EXCLUDED.last_seen_ns
                               ),
                               mention_count = cognitive_entities.mention_count + 1,
                               platforms = (
                                   SELECT ARRAY(SELECT DISTINCT unnest(
                                       cognitive_entities.platforms || EXCLUDED.platforms
                                   ))
                               ),
                               platform_count = (
                                   SELECT COUNT(DISTINCT unnest) FROM unnest(
                                       cognitive_entities.platforms || EXCLUDED.platforms
                                   )
                               ),
                               aliases = (
                                   SELECT ARRAY(SELECT DISTINCT unnest(
                                       cognitive_entities.aliases || EXCLUDED.aliases
                                   ))
                               )""",
                        eid,
                        ent.entity_type,
                        ent.name,
                        ent.aliases or [],
                        timestamp_ns,
                        platform,
                        json.dumps({
                            "confidence": ent.confidence,
                            "source": ent.source,
                        }),
                    )
                    entity_ids.append(eid)
                except Exception as e:
                    log.warning(f"Entity upsert failed for {ent.name}: {e}")

        # 2. Create co-occurrence edges for all pairs
        edges_created = 0
        if len(entity_ids) >= 2:
            edges_created = await self._create_cooccurrence_edges(
                entity_ids, event_id, timestamp_ns
            )

        return len(entity_ids), edges_created

    async def _create_cooccurrence_edges(
        self,
        entity_ids: List[str],
        event_id: str,
        timestamp_ns: int,
    ) -> int:
        """Create co-occurrence edges between all entity pairs in an event."""
        created = 0

        async with self._pool.acquire() as conn:
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    src, tgt = sorted([entity_ids[i], entity_ids[j]])
                    edge_id = self._edge_id(src, tgt, "co_occurs")

                    try:
                        await conn.execute(
                            """INSERT INTO cognitive_edges
                                   (edge_id, source_entity, target_entity,
                                    relation_type, weight, evidence_count,
                                    first_seen_ns, last_seen_ns,
                                    t_valid_from, source_events)
                               VALUES ($1, $2, $3, 'co_occurs', 1.0, 1,
                                       $4, $4, $4, ARRAY[$5]::text[])
                               ON CONFLICT (edge_id) DO UPDATE SET
                                   weight = cognitive_edges.weight + 0.1,
                                   evidence_count = cognitive_edges.evidence_count + 1,
                                   last_seen_ns = GREATEST(
                                       cognitive_edges.last_seen_ns, EXCLUDED.last_seen_ns
                                   ),
                                   source_events = (
                                       SELECT ARRAY(
                                           SELECT DISTINCT unnest(
                                               cognitive_edges.source_events || EXCLUDED.source_events
                                           )
                                       )
                                   )[1:50]""",  # Keep max 50 evidence events
                            edge_id, src, tgt, timestamp_ns, event_id,
                        )
                        created += 1
                    except Exception as e:
                        log.warning(f"Edge upsert failed {src} -> {tgt}: {e}")

        return created

    @staticmethod
    def _edge_id(source: str, target: str, relation: str) -> str:
        """Deterministic edge ID from sorted entity pair + relation."""
        key = f"{source}|{target}|{relation}"
        return f"edge-{hashlib.sha256(key.encode()).hexdigest()[:16]}"

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single entity by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM cognitive_entities WHERE entity_id = $1",
                entity_id,
            )
            return dict(row) if row else None

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Fuzzy search entities by name using trigram similarity.

        Args:
            query: Search term
            entity_type: Optional filter by entity type
            limit: Max results

        Returns:
            List of entity dicts sorted by similarity score
        """
        async with self._pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch(
                    """SELECT *, similarity(name, $1) AS sim
                       FROM cognitive_entities
                       WHERE entity_type = $2
                         AND (name % $1 OR name ILIKE '%' || $1 || '%')
                       ORDER BY sim DESC
                       LIMIT $3""",
                    query, entity_type, limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT *, similarity(name, $1) AS sim
                       FROM cognitive_entities
                       WHERE name % $1 OR name ILIKE '%' || $1 || '%'
                       ORDER BY sim DESC
                       LIMIT $2""",
                    query, limit,
                )
            return [dict(r) for r in rows]

    async def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        min_weight: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring entities connected by edges.

        Returns entities with edge metadata (relation, weight, evidence_count).
        """
        async with self._pool.acquire() as conn:
            if relation_type:
                rows = await conn.fetch(
                    """SELECT ce.*, e.relation_type, e.weight, e.evidence_count,
                              e.edge_id
                       FROM cognitive_edges e
                       JOIN cognitive_entities ce ON (
                           CASE WHEN e.source_entity = $1
                                THEN e.target_entity
                                ELSE e.source_entity
                           END = ce.entity_id
                       )
                       WHERE (e.source_entity = $1 OR e.target_entity = $1)
                         AND e.relation_type = $2
                         AND e.weight >= $3
                       ORDER BY e.weight DESC
                       LIMIT $4""",
                    entity_id, relation_type, min_weight, limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT ce.*, e.relation_type, e.weight, e.evidence_count,
                              e.edge_id
                       FROM cognitive_edges e
                       JOIN cognitive_entities ce ON (
                           CASE WHEN e.source_entity = $1
                                THEN e.target_entity
                                ELSE e.source_entity
                           END = ce.entity_id
                       )
                       WHERE (e.source_entity = $1 OR e.target_entity = $1)
                         AND e.weight >= $2
                       ORDER BY e.weight DESC
                       LIMIT $3""",
                    entity_id, min_weight, limit,
                )
            return [dict(r) for r in rows]

    async def graph_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the knowledge graph."""
        async with self._pool.acquire() as conn:
            entity_count = await conn.fetchval(
                "SELECT COUNT(*) FROM cognitive_entities"
            )
            edge_count = await conn.fetchval(
                "SELECT COUNT(*) FROM cognitive_edges"
            )
            type_counts = await conn.fetch(
                """SELECT entity_type, COUNT(*) as cnt
                   FROM cognitive_entities
                   GROUP BY entity_type
                   ORDER BY cnt DESC"""
            )
            top_entities = await conn.fetch(
                """SELECT name, entity_type, mention_count, platform_count
                   FROM cognitive_entities
                   ORDER BY mention_count DESC
                   LIMIT 10"""
            )
            relation_counts = await conn.fetch(
                """SELECT relation_type, COUNT(*) as cnt
                   FROM cognitive_edges
                   GROUP BY relation_type
                   ORDER BY cnt DESC"""
            )

        return {
            "entity_count": entity_count,
            "edge_count": edge_count,
            "entities_by_type": {r["entity_type"]: r["cnt"] for r in type_counts},
            "top_entities": [dict(r) for r in top_entities],
            "edges_by_relation": {r["relation_type"]: r["cnt"] for r in relation_counts},
        }


# ── Spreading Activation ─────────────────────────────────────

async def spreading_activation(
    pool: asyncpg.Pool,
    start_entity_id: str,
    depth: int = 3,
    decay: float = 0.6,
    min_activation: float = 0.05,
    min_edge_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Traverse the knowledge graph using spreading activation.

    Starting from a seed entity, activation spreads along edges with
    exponential decay. Returns a dict of {entity_id: activation_score}
    for all reached entities.

    Args:
        pool: asyncpg connection pool
        start_entity_id: Seed entity to start from
        depth: Max traversal depth (hops)
        decay: Multiplicative decay per hop (0-1)
        min_activation: Stop propagating below this threshold
        min_edge_weight: Only traverse edges with weight >= this

    Returns:
        Dict mapping entity_id to activation score (0-1)
    """
    activations: Dict[str, float] = {start_entity_id: 1.0}
    frontier: Set[str] = {start_entity_id}
    visited: Set[str] = set()

    for hop in range(depth):
        if not frontier:
            break

        next_frontier: Set[str] = set()
        current_decay = decay ** (hop + 1)

        async with pool.acquire() as conn:
            for entity_id in frontier:
                if entity_id in visited:
                    continue
                visited.add(entity_id)

                parent_activation = activations.get(entity_id, 0.0)
                if parent_activation < min_activation:
                    continue

                # Get all edges from this entity
                rows = await conn.fetch(
                    """SELECT
                           CASE WHEN source_entity = $1
                                THEN target_entity
                                ELSE source_entity
                           END AS neighbor_id,
                           weight
                       FROM cognitive_edges
                       WHERE (source_entity = $1 OR target_entity = $1)
                         AND weight >= $2""",
                    entity_id, min_edge_weight,
                )

                for row in rows:
                    neighbor = row["neighbor_id"]
                    edge_weight = row["weight"]

                    # Activation = parent_activation * decay * normalized_weight
                    # Normalize weight: cap at 10.0 for scoring
                    norm_weight = min(edge_weight, 10.0) / 10.0
                    child_activation = parent_activation * current_decay * norm_weight

                    if child_activation < min_activation:
                        continue

                    # Keep highest activation if visited from multiple paths
                    if neighbor not in activations or child_activation > activations[neighbor]:
                        activations[neighbor] = child_activation

                    if neighbor not in visited:
                        next_frontier.add(neighbor)

        frontier = next_frontier

    return activations


# ── Batch Processing ─────────────────────────────────────────

async def extract_and_ingest_batch(
    pool: asyncpg.Pool,
    limit: int = 1000,
    offset: int = 0,
) -> Dict[str, int]:
    """
    Batch-extract entities from cognitive events and ingest into the graph.

    Processes events that haven't been entity-extracted yet.
    Uses a simple approach: process in chronological order.

    Args:
        pool: asyncpg connection pool
        limit: Max events to process in this batch
        offset: Skip first N events

    Returns:
        Dict with counts: events_processed, entities_created, edges_created
    """
    extractor = EntityExtractor()
    graph = GraphManager(pool)

    t0 = time.time()
    total_entities = 0
    total_edges = 0

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT event_id, data_layer, light_layer, instinct_layer,
                      platform, timestamp_ns
               FROM cognitive_events
               WHERE light_layer IS NOT NULL
               ORDER BY timestamp_ns ASC
               LIMIT $1 OFFSET $2""",
            limit, offset,
        )

    if not rows:
        log.info("No events to process for entity extraction")
        return {"events_processed": 0, "entities_created": 0, "edges_created": 0}

    log.info(f"Extracting entities from {len(rows)} events...")

    for i, row in enumerate(rows):
        event = dict(row)
        # Parse JSON fields
        for fld in ("data_layer", "light_layer", "instinct_layer"):
            if isinstance(event.get(fld), str):
                event[fld] = json.loads(event[fld])

        entities = extractor.extract(event)
        if entities:
            n_ent, n_edge = await graph.ingest_entities(entities, event)
            total_entities += n_ent
            total_edges += n_edge

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            log.info(
                f"  Processed {i + 1}/{len(rows)} events "
                f"({rate:.0f}/sec) | "
                f"entities={total_entities}, edges={total_edges}"
            )

    elapsed = time.time() - t0
    rate = len(rows) / elapsed if elapsed > 0 else 0
    log.info(
        f"Entity extraction complete: {len(rows)} events in {elapsed:.1f}s "
        f"({rate:.0f}/sec) | "
        f"entities={total_entities}, edges={total_edges}"
    )

    return {
        "events_processed": len(rows),
        "entities_created": total_entities,
        "edges_created": total_edges,
    }
