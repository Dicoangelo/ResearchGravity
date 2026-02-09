"""
Intelligence Tools — Phase 3 MCP tools for advanced cognitive capabilities

Tools:
  hybrid_search      — Semantic + BM25 search with RRF fusion
  knowledge_graph    — Entity search, neighbors, spreading activation, stats
  insight_due        — FSRS-scheduled insights ready for review
  insight_review     — Record review rating and reschedule
  coherence_arcs     — Narrative arc detection and listing
  dashboard_snapshot — Full system status in one call
"""

import json
import logging
from typing import Any, Dict, List

from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger
from mcp_raw.tools.validation import clamp_int, clamp_float

log = get_logger("tools.intelligence")

# Shared DB pool — injected by server via set_db()
_pool = None


def set_db(db):
    """Called by server to inject shared database instance."""
    global _pool
    if hasattr(db, "_pool") and db._pool is not None:
        _pool = db._pool
    log.info("Intelligence tools: DB injected")


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "hybrid_search",
        "description": (
            "Advanced search combining semantic similarity (vector embeddings) with "
            "keyword matching (BM25 full-text) using Reciprocal Rank Fusion. "
            "Better than pure semantic search — captures both meaning AND exact terms. "
            "Returns ranked results with scoring breakdown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum results (default: 20)",
                    "default": 20,
                },
                "semantic_weight": {
                    "type": "number",
                    "description": "Weight for semantic ranking 0-1 (default: 0.6)",
                    "default": 0.6,
                },
                "bm25_weight": {
                    "type": "number",
                    "description": "Weight for keyword ranking 0-1 (default: 0.4)",
                    "default": 0.4,
                },
                "platform": {
                    "type": "string",
                    "description": "Filter to specific platform (e.g., 'chatgpt'). Omit for all.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "knowledge_graph",
        "description": (
            "Query the cognitive knowledge graph — entities (concepts, tools, projects, "
            "papers, technologies) and their connections extracted from 160K+ cognitive events. "
            "Actions: 'search' (find entities), 'neighbors' (explore connections), "
            "'activate' (spreading activation traversal), 'stats' (graph overview)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "neighbors", "activate", "stats"],
                    "description": "Action to perform on the knowledge graph",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'search') or entity_id (for 'neighbors'/'activate')",
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["concept", "tool", "project", "paper", "technology", "error", "platform", "person"],
                    "description": "Filter by entity type (for 'search')",
                },
                "limit": {
                    "type": "number",
                    "description": "Max results (default: 20)",
                    "default": 20,
                },
                "depth": {
                    "type": "number",
                    "description": "Traversal depth for 'activate' (default: 3)",
                    "default": 3,
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "insight_due",
        "description": (
            "Get cognitive insights that are due for review via FSRS spaced repetition. "
            "Shows coherence moments scheduled for resurfacing at optimal intervals "
            "based on the forgetting curve. Review them to strengthen long-term retention."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "number",
                    "description": "Maximum insights to return (default: 10)",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "insight_review",
        "description": (
            "Record your review of a cognitive insight. The FSRS algorithm reschedules "
            "the insight based on your rating: 1=forgot (re-show soon), 2=hard (slow growth), "
            "3=good (normal growth), 4=easy (fast growth)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "insight_id": {
                    "type": "string",
                    "description": "The insight ID to review (from insight_due results)",
                },
                "rating": {
                    "type": "number",
                    "enum": [1, 2, 3, 4],
                    "description": "1=forgot, 2=hard, 3=good, 4=easy",
                },
            },
            "required": ["insight_id", "rating"],
        },
    },
    {
        "name": "coherence_arcs",
        "description": (
            "Detect and list coherence arcs — narrative threads that span multiple "
            "cross-platform moments over time. Arcs group related coherence moments "
            "by entity overlap, showing intellectual trajectories and recurring themes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["all", "active", "dormant"],
                    "description": "Filter arcs by status (default: all)",
                    "default": "all",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum arcs to return (default: 20)",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "dashboard_snapshot",
        "description": (
            "Get a comprehensive UCW system dashboard in one call: events, embeddings, "
            "coherence moments, arcs, knowledge graph stats, FSRS insights, platform "
            "breakdown, and background job status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Dispatcher ───────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    handlers = {
        "hybrid_search": _hybrid_search,
        "knowledge_graph": _knowledge_graph,
        "insight_due": _insight_due,
        "insight_review": _insight_review,
        "coherence_arcs": _coherence_arcs,
        "dashboard_snapshot": _dashboard_snapshot,
    }

    handler = handlers.get(name)
    if not handler:
        return tool_result_content(
            [text_content(f"Unknown intelligence tool: {name}")], is_error=True
        )

    try:
        return await handler(args)
    except Exception as exc:
        log.error(f"Tool {name} failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Error in {name}: {exc}")], is_error=True
        )


# ── hybrid_search ────────────────────────────────────────────────────────────

async def _hybrid_search(args: Dict) -> Dict:
    """Run hybrid semantic + BM25 search with RRF fusion."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    from coherence_engine.hybrid_search import HybridSearch

    query = args["query"]
    limit = clamp_int(args.get("limit"), 20, 1, 100)
    sem_w = clamp_float(args.get("semantic_weight"), 0.6, 0.0, 1.0)
    bm25_w = clamp_float(args.get("bm25_weight"), 0.4, 0.0, 1.0)
    platform = args.get("platform")

    search = HybridSearch(_pool)

    if platform:
        results = await search.search_cross_platform(
            query, exclude_platform=None, limit=limit,
            semantic_weight=sem_w, bm25_weight=bm25_w,
        )
        # Post-filter by platform
        results = [r for r in results if r.platform == platform][:limit]
    else:
        results = await search.search(
            query, limit=limit, semantic_weight=sem_w, bm25_weight=bm25_w,
        )

    if not results:
        return tool_result_content([text_content(
            f"No results for: '{query}'"
        )])

    out = f"# Hybrid Search: '{query}'\n\n"
    out += f"**Results:** {len(results)} | Weights: semantic={sem_w}, bm25={bm25_w}\n\n"

    for i, r in enumerate(results, 1):
        sem_tag = f"sem:{r.semantic_rank}" if r.semantic_rank else "sem:-"
        bm25_tag = f"bm25:{r.bm25_rank}" if r.bm25_rank else "bm25:-"
        mode = r.cognitive_mode or "-"
        preview = (r.preview or "")[:200]

        out += f"**#{i}** [{r.rrf_score:.4f}] **{r.platform}** mode={mode} {sem_tag} {bm25_tag}\n"
        out += f"> {preview}\n\n"

    return tool_result_content([text_content(out)])


# ── knowledge_graph ──────────────────────────────────────────────────────────

async def _knowledge_graph(args: Dict) -> Dict:
    """Query the cognitive knowledge graph."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    from coherence_engine.knowledge_graph import GraphManager, spreading_activation

    action = args["action"]
    graph = GraphManager(_pool)

    if action == "stats":
        stats = await graph.graph_stats()
        out = "# Knowledge Graph Stats\n\n"
        out += f"| Metric | Value |\n|--------|-------|\n"
        out += f"| Entities | {stats['entity_count']:,} |\n"
        out += f"| Edges | {stats['edge_count']:,} |\n\n"

        if stats.get("entities_by_type"):
            out += "## Entities by Type\n\n"
            for etype, cnt in stats["entities_by_type"].items():
                out += f"- **{etype}**: {cnt}\n"

        if stats.get("top_entities"):
            out += "\n## Top Entities (by mentions)\n\n"
            for e in stats["top_entities"]:
                out += (
                    f"- **{e['name']}** ({e['entity_type']}) — "
                    f"{e['mention_count']:,} mentions, {e['platform_count']} platforms\n"
                )

        if stats.get("edges_by_relation"):
            out += "\n## Edges by Relation\n\n"
            for rel, cnt in stats["edges_by_relation"].items():
                out += f"- **{rel}**: {cnt:,}\n"

        return tool_result_content([text_content(out)])

    elif action == "search":
        query = args.get("query", "")
        if not query:
            return tool_result_content(
                [text_content("'query' is required for search action.")], is_error=True
            )

        entity_type = args.get("entity_type")
        limit = clamp_int(args.get("limit"), 20, 1, 100)

        results = await graph.search_entities(query, entity_type=entity_type, limit=limit)

        if not results:
            return tool_result_content([text_content(
                f"No entities matching '{query}'" +
                (f" (type: {entity_type})" if entity_type else "")
            )])

        out = f"# Entity Search: '{query}'\n\n"
        for r in results:
            platforms = ", ".join(r.get("platforms", []))
            out += (
                f"- **{r['name']}** ({r['entity_type']}) | "
                f"mentions: {r['mention_count']} | platforms: {platforms} | "
                f"id: `{r['entity_id']}`\n"
            )

        return tool_result_content([text_content(out)])

    elif action == "neighbors":
        entity_id = args.get("query", "")
        if not entity_id:
            return tool_result_content(
                [text_content("'query' (entity_id) is required for neighbors action.")],
                is_error=True,
            )

        limit = clamp_int(args.get("limit"), 20, 1, 100)
        neighbors = await graph.get_neighbors(entity_id, limit=limit)

        if not neighbors:
            return tool_result_content([text_content(
                f"No neighbors found for entity: {entity_id}"
            )])

        # Get the source entity name
        entity = await graph.get_entity(entity_id)
        entity_name = entity["name"] if entity else entity_id

        out = f"# Neighbors of '{entity_name}'\n\n"
        for n in neighbors:
            out += (
                f"- **{n['name']}** ({n['entity_type']}) — "
                f"weight: {n['weight']:.1f}, evidence: {n['evidence_count']}, "
                f"relation: {n['relation_type']}\n"
            )

        return tool_result_content([text_content(out)])

    elif action == "activate":
        entity_id = args.get("query", "")
        if not entity_id:
            return tool_result_content(
                [text_content("'query' (entity_id) is required for activate action.")],
                is_error=True,
            )

        depth = clamp_int(args.get("depth"), 3, 1, 5)
        activations = await spreading_activation(
            _pool, entity_id, depth=depth, decay=0.6, min_activation=0.05,
        )

        if not activations:
            return tool_result_content([text_content(
                f"No activations from entity: {entity_id}"
            )])

        # Resolve entity names
        sorted_acts = sorted(activations.items(), key=lambda x: x[1], reverse=True)

        out = f"# Spreading Activation from '{entity_id}' (depth={depth})\n\n"
        out += f"**Entities reached:** {len(sorted_acts)}\n\n"

        async with _pool.acquire() as conn:
            for eid, score in sorted_acts[:30]:
                row = await conn.fetchrow(
                    "SELECT name, entity_type FROM cognitive_entities WHERE entity_id = $1",
                    eid,
                )
                if row:
                    out += f"- **{score:.3f}** {row['name']} ({row['entity_type']})\n"
                else:
                    out += f"- **{score:.3f}** {eid}\n"

        return tool_result_content([text_content(out)])

    return tool_result_content(
        [text_content(f"Unknown action: {action}")], is_error=True
    )


# ── insight_due ──────────────────────────────────────────────────────────────

async def _insight_due(args: Dict) -> Dict:
    """Get insights due for FSRS review."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    from coherence_engine.fsrs import InsightScheduler

    scheduler = InsightScheduler(pool=_pool)
    limit = clamp_int(args.get("limit"), 10, 1, 50)
    due = await scheduler.get_due_insights(limit=limit)

    if not due:
        # Check total scheduled
        async with _pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM insight_schedule")
            next_due = await conn.fetchval(
                "SELECT MIN(next_review) FROM insight_schedule WHERE next_review > NOW()"
            )

        msg = f"No insights due for review right now.\n\n"
        msg += f"**Total scheduled:** {total}\n"
        if next_due:
            msg += f"**Next due:** {next_due.strftime('%Y-%m-%d %H:%M')}\n"
        return tool_result_content([text_content(msg)])

    out = f"# Insights Due for Review ({len(due)})\n\n"
    out += "*Rate each: 1=forgot, 2=hard, 3=good, 4=easy*\n\n"

    for i, ins in enumerate(due, 1):
        desc = (ins.get("description") or "")[:200]
        platforms = ins.get("platforms", [])
        ctype = ins.get("coherence_type", "-")
        moment_conf = ins.get("moment_confidence", 0)
        reviews = ins.get("review_count", 0)
        stability = ins.get("stability", 0)

        out += f"## #{i} — {ins['insight_id']}\n\n"
        out += f"**Coherence moment:** {ctype} ({moment_conf:.0%} confidence)\n"
        out += f"**Platforms:** {', '.join(platforms) if platforms else '-'}\n"
        out += f"**Reviews:** {reviews} | **Stability:** {stability:.1f}d\n"
        out += f"> {desc}\n\n"

    return tool_result_content([text_content(out)])


# ── insight_review ───────────────────────────────────────────────────────────

async def _insight_review(args: Dict) -> Dict:
    """Record a review rating for an insight."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    from coherence_engine.fsrs import InsightScheduler

    insight_id = args["insight_id"]
    rating = clamp_int(args.get("rating"), 0, 1, 4)

    if rating < 1:
        return tool_result_content(
            [text_content("Rating must be 1 (forgot), 2 (hard), 3 (good), or 4 (easy).")],
            is_error=True,
        )

    scheduler = InsightScheduler(pool=_pool)
    await scheduler.record_review(insight_id, rating)

    # Get updated state
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT stability, next_review, review_count FROM insight_schedule WHERE insight_id = $1",
            insight_id,
        )

    rating_labels = {1: "forgot", 2: "hard", 3: "good", 4: "easy"}
    label = rating_labels.get(rating, str(rating))

    out = f"Review recorded for `{insight_id}`\n\n"
    out += f"**Rating:** {rating} ({label})\n"
    if row:
        out += f"**New stability:** {row['stability']:.1f} days\n"
        out += f"**Next review:** {row['next_review'].strftime('%Y-%m-%d %H:%M')}\n"
        out += f"**Total reviews:** {row['review_count']}\n"

    return tool_result_content([text_content(out)])


# ── coherence_arcs ───────────────────────────────────────────────────────────

async def _coherence_arcs(args: Dict) -> Dict:
    """Detect and list coherence arcs."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    from coherence_engine.significance import ArcDetector

    status_filter = args.get("status", "all")
    limit = clamp_int(args.get("limit"), 20, 1, 50)

    detector = ArcDetector(pool=_pool)
    arcs = await detector.detect_arcs(limit=500)

    if status_filter != "all":
        arcs = [a for a in arcs if a.status == status_filter]

    arcs = arcs[:limit]

    if not arcs:
        return tool_result_content([text_content(
            "No coherence arcs detected." +
            (f" (filter: {status_filter})" if status_filter != "all" else "")
        )])

    # Count by status
    active = sum(1 for a in arcs if a.status == "active")
    dormant = sum(1 for a in arcs if a.status == "dormant")
    total_moments = sum(a.moment_count for a in arcs)

    out = f"# Coherence Arcs ({len(arcs)} arcs, {total_moments} moments)\n\n"
    out += f"**Active:** {active} | **Dormant:** {dormant}\n\n"

    for i, arc in enumerate(arcs, 1):
        status_icon = {"active": "🟢", "dormant": "🟡", "resolved": "✅"}.get(arc.status, "⚪")
        platforms = ", ".join(arc.platforms) if arc.platforms else "-"
        entities = ", ".join(arc.key_entities[:5]) if arc.key_entities else "-"

        out += f"## #{i} {status_icon} {arc.title}\n\n"
        out += f"**Status:** {arc.status} | **Strength:** {arc.arc_strength:.2f} | "
        out += f"**Moments:** {arc.moment_count}\n"
        out += f"**Platforms:** {platforms}\n"
        out += f"**Key entities:** {entities}\n"
        out += f"**Arc ID:** `{arc.arc_id}`\n\n"

    return tool_result_content([text_content(out)])


# ── dashboard_snapshot ───────────────────────────────────────────────────────

async def _dashboard_snapshot(args: Dict) -> Dict:
    """Full system dashboard in one call."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    async with _pool.acquire() as conn:
        # Core metrics
        total_events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        total_embedded = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
        embedded_768 = await conn.fetchval(
            "SELECT COUNT(*) FROM embedding_cache WHERE embedding_768 IS NOT NULL"
        )
        remaining_768 = await conn.fetchval(
            "SELECT COUNT(*) FROM embedding_cache WHERE embedding_768 IS NULL AND embedding IS NOT NULL"
        )
        total_moments = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")

        # Platform breakdown
        platforms = await conn.fetch(
            "SELECT platform, COUNT(*) AS cnt FROM cognitive_events GROUP BY platform ORDER BY cnt DESC"
        )

        # Knowledge graph
        kg_entities = await conn.fetchval("SELECT COUNT(*) FROM cognitive_entities")
        kg_edges = await conn.fetchval("SELECT COUNT(*) FROM cognitive_edges")
        kg_types = await conn.fetch(
            "SELECT entity_type, COUNT(*) AS cnt FROM cognitive_entities GROUP BY entity_type ORDER BY cnt DESC"
        )

        # FSRS insights
        total_insights = await conn.fetchval("SELECT COUNT(*) FROM insight_schedule")
        due_insights = await conn.fetchval(
            "SELECT COUNT(*) FROM insight_schedule WHERE next_review <= NOW()"
        )
        avg_stability = await conn.fetchval(
            "SELECT AVG(stability) FROM insight_schedule WHERE review_count > 0"
        )

        # Arcs
        total_arcs = await conn.fetchval("SELECT COUNT(*) FROM coherence_arcs")
        active_arcs = await conn.fetchval(
            "SELECT COUNT(*) FROM coherence_arcs WHERE status = 'active'"
        )

        # Confidence tiers
        high_conf = await conn.fetchval(
            "SELECT COUNT(*) FROM coherence_moments WHERE confidence >= 0.80"
        )
        med_conf = await conn.fetchval(
            "SELECT COUNT(*) FROM coherence_moments WHERE confidence >= 0.70 AND confidence < 0.80"
        )

        # HNSW index status
        has_hnsw = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname='idx_embedding_cache_hnsw' "
            "AND indexdef LIKE '%embedding_768%')"
        )

    out = "# UCW Dashboard\n\n"

    out += "## Core Metrics\n\n"
    out += f"| Metric | Value |\n|--------|-------|\n"
    out += f"| Events | {total_events:,} |\n"
    out += f"| Embeddings | {total_embedded:,} |\n"
    out += f"| 768d Migrated | {embedded_768:,} / {embedded_768 + remaining_768:,} ({embedded_768 * 100 // max(1, embedded_768 + remaining_768)}%) |\n"
    out += f"| Coherence Moments | {total_moments} (high: {high_conf}, medium: {med_conf}) |\n"
    out += f"| HNSW Index (768d) | {'✅ Built' if has_hnsw else '⏳ Pending'} |\n\n"

    out += "## Platforms\n\n"
    for r in platforms:
        out += f"- **{r['platform']}**: {r['cnt']:,}\n"

    out += "\n## Knowledge Graph\n\n"
    out += f"- **Entities:** {kg_entities:,}\n"
    out += f"- **Edges:** {kg_edges:,}\n"
    if kg_types:
        type_strs = [f"{r['entity_type']}({r['cnt']})" for r in kg_types]
        out += f"- **By type:** {', '.join(type_strs)}\n"

    out += "\n## FSRS Insights\n\n"
    out += f"- **Scheduled:** {total_insights}\n"
    out += f"- **Due now:** {due_insights}\n"
    if avg_stability:
        out += f"- **Avg stability:** {avg_stability:.1f} days\n"

    out += "\n## Coherence Arcs\n\n"
    out += f"- **Total:** {total_arcs} ({active_arcs} active)\n"

    # Background job status (check lock files)
    import os
    mig_lock = os.path.expanduser("~/.ucw/locks/migration.lock")
    kg_lock = os.path.expanduser("~/.ucw/locks/kg_extraction.lock")
    mig_running = False
    kg_running = False
    if os.path.exists(mig_lock):
        try:
            pid = int(open(mig_lock).read().strip())
            os.kill(pid, 0)
            mig_running = True
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    if os.path.exists(kg_lock):
        try:
            pid = int(open(kg_lock).read().strip())
            os.kill(pid, 0)
            kg_running = True
        except (ValueError, ProcessLookupError, PermissionError):
            pass

    out += "\n## Background Jobs\n\n"
    out += f"- **768d Migration:** {'🟢 Running' if mig_running else ('✅ Complete' if remaining_768 == 0 else '🔴 Stopped')}\n"
    out += f"- **KG Extraction:** {'🟢 Running' if kg_running else '⚪ Idle'}\n"

    return tool_result_content([text_content(out)])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json(value) -> dict:
    """Safely parse a JSON value that might be a string or already a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}
