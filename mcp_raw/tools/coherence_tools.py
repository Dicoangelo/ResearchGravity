"""
Coherence Tools — MCP tools wired to the coherence_engine package

Tools:
  coherence_status     — Quick stats: events, embedded, moments, platforms
  coherence_moments    — List detected cross-platform coherence moments
  coherence_search     — Semantic similarity search across platforms
  coherence_scan       — Run fresh oneshot detection scan
"""

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger
from mcp_raw.tools.validation import clamp_int, clamp_float

log = get_logger("tools.coherence")

# Shared DB — injected by server via set_db()
_db = None       # CognitiveDatabase (PostgreSQL)
_pool = None     # asyncpg.Pool (extracted from _db for direct queries)


def set_db(db):
    """Called by server to inject shared database instance."""
    global _db, _pool
    _db = db
    # Extract the pool if it's a PostgreSQL backend
    if hasattr(db, "_pool") and db._pool is not None:
        _pool = db._pool
    log.info("Coherence tools: DB injected")


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "coherence_status",
        "description": (
            "Get UCW Coherence Engine status: total events, embedded count, "
            "detected moments, platform breakdown, and top coherence themes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "coherence_moments",
        "description": (
            "List detected cross-platform coherence moments with full event content. "
            "Shows what the engine discovered: aligned topics, summaries, emergence signals."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold 0-1 (default: 0.70)",
                    "default": 0.70,
                },
                "topic": {
                    "type": "string",
                    "description": "Filter moments by topic (e.g., 'ucw', 'ai_agents', 'sovereignty')",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum moments to return (default: 20)",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "coherence_search",
        "description": (
            "Semantic similarity search across all platforms using vector embeddings. "
            "Finds events from Claude, ChatGPT, and other platforms that are semantically "
            "similar to a natural language query. Returns ranked results with similarity scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (e.g., 'sovereign AI infrastructure')",
                },
                "platform": {
                    "type": "string",
                    "description": "Filter to specific platform (e.g., 'chatgpt', 'claude-desktop'). Omit for all.",
                },
                "cross_platform_only": {
                    "type": "boolean",
                    "description": "Only show results from platforms OTHER than the current one (default: false)",
                    "default": False,
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Minimum cosine similarity threshold 0-1 (default: 0.45)",
                    "default": 0.45,
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum results to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "coherence_scan",
        "description": (
            "Run a fresh oneshot coherence detection scan. Processes recent unscored events, "
            "computes embeddings, and detects new cross-platform alignment moments. "
            "Returns a summary of newly detected moments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "number",
                    "description": "Only scan events from the last N hours (default: 24)",
                    "default": 24,
                },
            },
        },
    },
]


# ── Dispatcher ───────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    handlers = {
        "coherence_status": _coherence_status,
        "coherence_moments": _coherence_moments,
        "coherence_search": _coherence_search,
        "coherence_scan": _coherence_scan,
    }

    handler = handlers.get(name)
    if not handler:
        return tool_result_content(
            [text_content(f"Unknown coherence tool: {name}")], is_error=True
        )

    try:
        return await handler(args)
    except Exception as exc:
        log.error(f"Tool {name} failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Error in {name}: {exc}")], is_error=True
        )


# ── coherence_status ─────────────────────────────────────────────────────────

async def _coherence_status(args: Dict) -> Dict:
    """Quick engine status."""
    if not _pool:
        return tool_result_content(
            [text_content("Coherence engine not available — PostgreSQL not connected.")],
            is_error=True,
        )

    async with _pool.acquire() as conn:
        total_events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        embedded = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
        moments = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")

        platforms = await conn.fetch(
            "SELECT platform, COUNT(*) AS cnt FROM cognitive_events GROUP BY platform ORDER BY cnt DESC"
        )

        top_topics = await conn.fetch("""
            SELECT coherence_type, AVG(confidence) AS avg_conf, COUNT(*) AS cnt
            FROM coherence_moments
            GROUP BY coherence_type
            ORDER BY cnt DESC
        """)

        # Confidence distribution
        high = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments WHERE confidence >= 0.75")
        med = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments WHERE confidence >= 0.70 AND confidence < 0.75")

    out = "# UCW Coherence Engine Status\n\n"
    out += f"| Metric | Value |\n|--------|-------|\n"
    out += f"| Events | {total_events:,} |\n"
    out += f"| Embedded | {embedded:,} |\n"
    out += f"| Coherence Moments | {moments} |\n"
    out += f"| High confidence (>=75%) | {high} |\n"
    out += f"| Medium confidence (70-74%) | {med} |\n\n"

    out += "## Platforms\n\n"
    for r in platforms:
        out += f"- **{r['platform']}**: {r['cnt']:,} events\n"

    if top_topics:
        out += "\n## Detection Types\n\n"
        for r in top_topics:
            out += f"- **{r['coherence_type']}**: {r['cnt']} moments (avg {r['avg_conf']:.0%})\n"

    return tool_result_content([text_content(out)])


# ── coherence_moments ────────────────────────────────────────────────────────

async def _coherence_moments(args: Dict) -> Dict:
    """List detected moments with full event content."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    min_conf = clamp_float(args.get("min_confidence"), 0.70, 0.0, 1.0)
    topic_filter = args.get("topic")
    limit = clamp_int(args.get("limit"), 20, 1, 100)

    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT moment_id, confidence, coherence_type, platforms, description,
                   event_ids, metadata, time_window_s
            FROM coherence_moments
            WHERE confidence >= $1
            ORDER BY confidence DESC
            LIMIT $2
        """, min_conf, limit * 3)  # over-fetch for topic filtering

    # Deduplicate by event pair
    seen_pairs = set()
    unique = []
    for r in rows:
        pair = tuple(sorted(r["event_ids"]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique.append(r)

    # Resolve linked events and optionally filter by topic
    results = []
    for r in unique:
        if len(results) >= limit:
            break

        events_data = []
        async with _pool.acquire() as conn:
            for eid in list(r["event_ids"])[:4]:
                e = await conn.fetchrow(
                    """SELECT event_id, platform, cognitive_mode, quality_score,
                              light_layer, instinct_layer
                       FROM cognitive_events WHERE event_id = $1""",
                    eid,
                )
                if not e:
                    continue

                ll = _parse_json(e["light_layer"])
                il = _parse_json(e["instinct_layer"])

                events_data.append({
                    "platform": e["platform"],
                    "mode": e["cognitive_mode"] or "-",
                    "quality": e["quality_score"],
                    "topic": ll.get("topic", "-"),
                    "intent": ll.get("intent", "-"),
                    "summary": ll.get("summary", "-"),
                    "concepts": ll.get("concepts", []),
                    "gut": il.get("gut_signal", "-"),
                    "emergence": il.get("emergence_indicators", []),
                })

        # Topic filter
        if topic_filter:
            topics = {e["topic"] for e in events_data}
            if topic_filter.lower() not in {t.lower() for t in topics}:
                continue

        results.append({"moment": r, "events": events_data})

    if not results:
        return tool_result_content([text_content(
            f"No coherence moments found above {min_conf:.0%} confidence"
            + (f" for topic '{topic_filter}'" if topic_filter else "")
            + "."
        )])

    out = f"# Coherence Moments ({len(results)} shown)\n\n"

    for i, item in enumerate(results, 1):
        r = item["moment"]
        conf = r["confidence"]
        ctype = r["coherence_type"]

        out += f"## #{i} — {conf:.0%} confidence — {ctype}\n\n"

        for e in item["events"]:
            q = f"{e['quality']:.2f}" if e["quality"] else "-"
            summary = (e["summary"] or "-")[:250]
            out += f"**{e['platform']}** [{e['mode']}] topic=`{e['topic']}` intent=`{e['intent']}` q={q}\n"
            out += f"> {summary}\n\n"

            if e["emergence"]:
                out += f"*Emergence signals: {e['emergence']}* | gut: {e['gut']}\n\n"

        out += "---\n\n"

    return tool_result_content([text_content(out)])


# ── coherence_search ─────────────────────────────────────────────────────────

async def _coherence_search(args: Dict) -> Dict:
    """Semantic similarity search using embeddings."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    query = args["query"]
    platform_filter = args.get("platform")
    cross_only = args.get("cross_platform_only", False)
    min_sim = clamp_float(args.get("min_similarity"), 0.45, 0.0, 1.0)
    limit = clamp_int(args.get("limit"), 20, 1, 100)

    # Embed the query
    try:
        from mcp_raw.embeddings import embed_single
        query_emb = embed_single(query)
    except Exception as exc:
        return tool_result_content(
            [text_content(f"Embedding failed: {exc}")], is_error=True
        )

    vec_str = "[" + ",".join(str(x) for x in query_emb) + "]"
    dist_threshold = 1.0 - min_sim

    # Build query based on filters
    try:
        async with _pool.acquire() as conn:
            if platform_filter:
                rows = await conn.fetch("""
                    SELECT ec.source_event_id, ec.content_preview,
                           (ec.embedding <=> $1::vector) AS distance,
                           ce.platform, ce.cognitive_mode, ce.quality_score,
                           ce.light_layer, ce.instinct_layer
                    FROM embedding_cache ec
                    JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                    WHERE ce.platform = $2
                      AND (ec.embedding <=> $1::vector) < $3
                    ORDER BY distance
                    LIMIT $4
                """, vec_str, platform_filter, dist_threshold, limit)
            elif cross_only:
                # Exclude claude-desktop (current platform)
                rows = await conn.fetch("""
                    SELECT ec.source_event_id, ec.content_preview,
                           (ec.embedding <=> $1::vector) AS distance,
                           ce.platform, ce.cognitive_mode, ce.quality_score,
                           ce.light_layer, ce.instinct_layer
                    FROM embedding_cache ec
                    JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                    WHERE ce.platform != 'claude-desktop'
                      AND (ec.embedding <=> $1::vector) < $2
                    ORDER BY distance
                    LIMIT $3
                """, vec_str, dist_threshold, limit)
            else:
                rows = await conn.fetch("""
                    SELECT ec.source_event_id, ec.content_preview,
                           (ec.embedding <=> $1::vector) AS distance,
                           ce.platform, ce.cognitive_mode, ce.quality_score,
                           ce.light_layer, ce.instinct_layer
                    FROM embedding_cache ec
                    JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                    WHERE (ec.embedding <=> $1::vector) < $2
                    ORDER BY distance
                    LIMIT $3
                """, vec_str, dist_threshold, limit)
    except Exception as exc:
        return tool_result_content(
            [text_content(f"Search failed: {exc}")], is_error=True
        )

    if not rows:
        return tool_result_content([text_content(
            f"No results above {min_sim:.0%} similarity for: '{query}'"
        )])

    # Group by platform for header
    plat_counts = Counter(r["platform"] for r in rows)

    out = f"# Semantic Search: '{query}'\n\n"
    out += f"**Results:** {len(rows)} | **Threshold:** {min_sim:.0%}\n"
    out += f"**Platforms:** {', '.join(f'{p} ({c})' for p, c in plat_counts.most_common())}\n\n"

    for r in rows:
        sim = 1.0 - r["distance"]
        ll = _parse_json(r["light_layer"])
        il = _parse_json(r["instinct_layer"])

        topic = ll.get("topic", "-")
        intent = ll.get("intent", "-")
        summary = (ll.get("summary", "") or "")[:200]
        mode = r["cognitive_mode"] or "-"
        gut = il.get("gut_signal", "-")
        preview = (r["content_preview"] or "")[:120]

        out += f"**{sim:.0%}** [{r['platform']}] mode={mode} topic=`{topic}` intent=`{intent}` gut={gut}\n"
        if summary:
            out += f"> {summary}\n\n"
        else:
            out += f"> {preview}\n\n"

    return tool_result_content([text_content(out)])


# ── coherence_scan ───────────────────────────────────────────────────────────

async def _coherence_scan(args: Dict) -> Dict:
    """Run a fresh oneshot coherence detection scan."""
    if not _pool:
        return tool_result_content(
            [text_content("PostgreSQL not connected.")], is_error=True
        )

    hours = clamp_int(args.get("hours"), 24, 1, 168)

    # Get moment count before scan
    async with _pool.acquire() as conn:
        before_count = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")

    # Run the coherence engine oneshot scan
    try:
        from coherence_engine.daemon import CoherenceDaemon

        daemon = CoherenceDaemon(pool=_pool)
        await daemon.initialize()
        processed = await daemon.oneshot()

    except Exception as exc:
        return tool_result_content(
            [text_content(f"Scan failed: {exc}")], is_error=True
        )

    # Get moment count after scan
    async with _pool.acquire() as conn:
        after_count = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")
        new_moments = after_count - before_count

        # Get the new moments (cap at 2000)
        new_rows = []
        if new_moments > 0:
            new_rows = await conn.fetch("""
                SELECT moment_id, confidence, coherence_type, platforms, description
                FROM coherence_moments
                ORDER BY created_at DESC
                LIMIT $1
            """, min(new_moments, 2000))

    out = f"# Coherence Scan Complete\n\n"
    out += f"**Events processed:** {processed}\n"
    out += f"**New moments detected:** {new_moments}\n"
    out += f"**Total moments:** {after_count}\n\n"

    if new_rows:
        out += "## Newly Detected\n\n"
        for r in new_rows:
            out += f"- **{r['confidence']:.0%}** {r['coherence_type']} — {r['platforms']} — {(r['description'] or '')[:150]}\n"

    if new_moments == 0:
        out += "*No new coherence moments detected. All recent events have been scored.*\n"

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
