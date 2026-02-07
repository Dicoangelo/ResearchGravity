"""
Coherence Tools — Cross-platform coherence query tools

Tools:
  find_coherent_events   — Find events matching a coherence signature
  coherence_report       — Summary of all detected coherence moments
  cross_platform_search  — Search across platforms by topic/intent/concept
"""

import json
from typing import Any, Dict, List

from mcp_raw.config import Config
from mcp_raw.db import CaptureDB
from mcp_raw.coherence import CoherenceEngine
from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger

log = get_logger("tools.coherence")

# Shared instances — injected by server via set_db() after initialization
_db = None
_engine = CoherenceEngine()


def set_db(db):
    """Called by server to inject shared database instance."""
    global _db
    _db = db

# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "find_coherent_events",
        "description": "Find cognitive events matching a coherence signature or within a time window across platforms",
        "inputSchema": {
            "type": "object",
            "properties": {
                "signature": {
                    "type": "string",
                    "description": "Coherence signature (SHA-256 hex) to match. Omit to find all cross-platform coherent events.",
                },
                "time_window_minutes": {
                    "type": "number",
                    "description": "Time window in minutes for temporal alignment (default: 5)",
                    "default": 5,
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold 0-1 (default: 0.3)",
                    "default": 0.3,
                },
            },
        },
    },
    {
        "name": "coherence_report",
        "description": "Get a summary of all detected coherence moments grouped by type with confidence distribution",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cross_platform_search",
        "description": "Search cognitive events across all platforms by topic, intent, or concept keywords",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (matched against topics, intents, concepts, and summaries)",
                },
                "platforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific platforms (e.g., ['claude-desktop', 'chatgpt']). Omit for all.",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum results to return (default: 30)",
                    "default": 30,
                },
            },
            "required": ["query"],
        },
    },
]


# ── Dispatcher ───────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    handlers = {
        "find_coherent_events": _find_coherent_events,
        "coherence_report": _coherence_report,
        "cross_platform_search": _cross_platform_search,
    }

    handler = handlers.get(name)
    if not handler:
        return tool_result_content([text_content(f"Unknown coherence tool: {name}")], is_error=True)

    try:
        return await handler(args)
    except Exception as exc:
        log.error(f"Tool {name} failed: {exc}", exc_info=True)
        return tool_result_content([text_content(f"Error in {name}: {exc}")], is_error=True)


# ── Implementations ──────────────────────────────────────────────────────────

async def _find_coherent_events(args: Dict) -> Dict:
    """Find events by coherence signature or temporal alignment."""
    signature = args.get("signature")
    window_minutes = int(args.get("time_window_minutes", 5))
    min_confidence = float(args.get("min_confidence", 0.3))

    if not _db:
        return tool_result_content([text_content("Database not initialized.")], is_error=True)

    conn = _db._conn
    if not conn:
        return tool_result_content([text_content("Database not initialized.")], is_error=True)

    if signature:
        # Direct signature match
        cur = conn.execute(
            """SELECT event_id, timestamp_ns, platform, direction, method,
                      light_topic, light_intent, light_summary,
                      instinct_coherence, coherence_sig
               FROM cognitive_events
               WHERE coherence_sig = ?
               ORDER BY timestamp_ns""",
            (signature,),
        )
        rows = cur.fetchall()

        if not rows:
            return tool_result_content([text_content(
                f"No events found with coherence signature: {signature[:16]}..."
            )])

        output = f"# Coherent Events (signature match)\n\n"
        output += f"**Signature:** {signature[:16]}...\n"
        output += f"**Events Found:** {len(rows)}\n\n"

        for row in rows:
            (eid, ts, plat, direction, method, topic, intent,
             summary, coherence, sig) = row
            output += (
                f"- **{method or 'response'}** ({plat}, {direction})\n"
                f"  Topic: {topic} | Intent: {intent} | Coherence: {coherence or 0:.2f}\n"
                f"  {(summary or '')[:150]}\n\n"
            )

        return tool_result_content([text_content(output)])

    else:
        # Find cross-platform temporal alignments
        window_ns = window_minutes * 60 * 1_000_000_000

        cur = conn.execute(
            """SELECT event_id, timestamp_ns, platform, method,
                      light_topic, light_intent, light_concepts,
                      instinct_coherence, instinct_indicators
               FROM cognitive_events
               ORDER BY timestamp_ns DESC LIMIT 500""",
        )
        rows = cur.fetchall()

        if not rows:
            return tool_result_content([text_content("No events in database.")])

        events = [
            {
                "event_id": r[0],
                "timestamp_ns": r[1],
                "platform": r[2],
                "method": r[3],
                "light_topic": r[4],
                "light_intent": r[5],
                "light_concepts": r[6],
                "instinct_coherence": r[7],
                "instinct_indicators": r[8],
            }
            for r in rows
        ]

        alignments = await _engine.detect_temporal_alignment(events, window_ns=window_ns)
        filtered = [a for a in alignments if a["confidence"] >= min_confidence]

        if not filtered:
            return tool_result_content([text_content(
                f"No cross-platform coherence found above confidence {min_confidence} "
                f"in {window_minutes}-minute windows."
            )])

        output = f"# Cross-Platform Coherent Events\n\n"
        output += f"**Window:** {window_minutes} minutes | **Min Confidence:** {min_confidence}\n"
        output += f"**Alignments Found:** {len(filtered)}\n\n"

        for i, alignment in enumerate(filtered[:10], 1):
            output += (
                f"## Alignment {i}\n"
                f"**Platforms:** {', '.join(alignment['platforms'])}\n"
                f"**Events:** {alignment['event_count']}\n"
                f"**Confidence:** {alignment['confidence']:.3f}\n\n"
            )

        return tool_result_content([text_content(output)])


async def _coherence_report(args: Dict) -> Dict:
    """Summary of all detected coherence moments."""
    moments = _engine.moments

    conn = _db._conn if _db else None
    if conn:
        # Also scan DB for synchronicity patterns
        cur = conn.execute(
            """SELECT event_id, timestamp_ns, platform, method,
                      light_topic, light_concepts, light_intent,
                      instinct_coherence, instinct_indicators
               FROM cognitive_events
               ORDER BY timestamp_ns DESC LIMIT 200""",
        )
        rows = cur.fetchall()

        if rows:
            events = [
                {
                    "event_id": r[0],
                    "timestamp_ns": r[1],
                    "platform": r[2],
                    "method": r[3],
                    "light_topic": r[4],
                    "light_concepts": r[5],
                    "light_intent": r[6],
                    "instinct_coherence": r[7],
                    "instinct_indicators": r[8],
                }
                for r in rows
            ]

            # Detect all coherence types
            temporal = await _engine.detect_temporal_alignment(events)
            synchronicity = await _engine.detect_synchronicity(events)
        else:
            temporal = []
            synchronicity = []
    else:
        temporal = []
        synchronicity = []

    output = "# Coherence Report\n\n"

    # Recorded moments
    output += f"## Recorded Moments ({len(moments)})\n\n"
    if moments:
        by_type: Dict[str, List] = {}
        for m in moments:
            by_type.setdefault(m["type"], []).append(m)

        for ctype, items in by_type.items():
            avg_conf = sum(m["confidence"] for m in items) / len(items)
            output += f"### {ctype.title()} ({len(items)} moments, avg confidence {avg_conf:.3f})\n"
            for m in items[:5]:
                output += (
                    f"- {m['moment_id']}: {len(m['event_ids'])} events, "
                    f"confidence={m['confidence']:.3f}, "
                    f"platforms={m['platforms']}\n"
                )
            output += "\n"
    else:
        output += "*No moments recorded yet.*\n\n"

    # Live temporal alignments
    output += f"## Live Temporal Alignments ({len(temporal)})\n\n"
    if temporal:
        for a in temporal[:5]:
            output += (
                f"- Platforms: {a['platforms']} | Events: {a['event_count']} | "
                f"Confidence: {a['confidence']:.3f}\n"
            )
        output += "\n"
    else:
        output += "*No cross-platform temporal alignments detected.*\n\n"

    # Synchronicity signals
    output += f"## Synchronicity Signals ({len(synchronicity)})\n\n"
    if synchronicity:
        for s in synchronicity[:5]:
            output += (
                f"- Platforms: {s['platforms']} | Events: {s['event_count']} | "
                f"Confidence: {s['confidence']:.3f} | "
                f"Meta: {s['has_meta_cognitive']} | "
                f"High coherence: {s['has_high_coherence']}\n"
            )
        output += "\n"
    else:
        output += "*No synchronicity patterns detected. These require cross-platform meta-cognitive alignment.*\n\n"

    # Confidence distribution
    all_confidences = (
        [m["confidence"] for m in moments] +
        [a["confidence"] for a in temporal] +
        [s["confidence"] for s in synchronicity]
    )

    if all_confidences:
        output += "## Confidence Distribution\n\n"
        buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
        for c in all_confidences:
            if c < 0.3:
                buckets["0.0-0.3"] += 1
            elif c < 0.5:
                buckets["0.3-0.5"] += 1
            elif c < 0.7:
                buckets["0.5-0.7"] += 1
            elif c < 0.9:
                buckets["0.7-0.9"] += 1
            else:
                buckets["0.9-1.0"] += 1

        for bucket, count in buckets.items():
            bar = "#" * count
            output += f"  {bucket}: {bar} ({count})\n"
        output += "\n"

    return tool_result_content([text_content(output)])


async def _cross_platform_search(args: Dict) -> Dict:
    """Search across all platforms by topic, intent, or concept."""
    query = args["query"]
    platforms = args.get("platforms")
    limit = int(args.get("limit", 30))

    if not _db:
        return tool_result_content([text_content("Database not initialized.")], is_error=True)

    conn = _db._conn
    if not conn:
        return tool_result_content([text_content("Database not initialized.")], is_error=True)

    query_lower = query.lower()

    # Search across multiple columns
    sql = """
        SELECT event_id, timestamp_ns, platform, direction, method,
               light_topic, light_intent, light_concepts, light_summary,
               instinct_coherence, instinct_gut_signal
        FROM cognitive_events
        WHERE (
            LOWER(light_topic) LIKE ? OR
            LOWER(light_intent) LIKE ? OR
            LOWER(light_concepts) LIKE ? OR
            LOWER(light_summary) LIKE ? OR
            LOWER(data_content) LIKE ?
        )
    """
    params: list = [f"%{query_lower}%"] * 5

    if platforms:
        placeholders = ",".join("?" for _ in platforms)
        sql += f" AND platform IN ({placeholders})"
        params.extend(platforms)

    sql += " ORDER BY timestamp_ns DESC LIMIT ?"
    params.append(limit)

    cur = conn.execute(sql, params)
    rows = cur.fetchall()

    if not rows:
        plat_str = f" on {', '.join(platforms)}" if platforms else ""
        return tool_result_content([text_content(
            f"No events found matching '{query}'{plat_str}."
        )])

    # Group by platform for the header
    platform_counts: Dict[str, int] = {}
    for row in rows:
        plat = row[2] or "unknown"
        platform_counts[plat] = platform_counts.get(plat, 0) + 1

    output = f"# Cross-Platform Search: '{query}'\n\n"
    output += f"**Results:** {len(rows)}\n"
    output += f"**Platforms:** {', '.join(f'{p} ({c})' for p, c in platform_counts.items())}\n\n"

    for row in rows:
        (eid, ts, plat, direction, method, topic, intent,
         concepts_json, summary, coherence, gut) = row

        arrow = "->" if direction == "out" else "<-"
        coherence_str = f" coherence={coherence:.2f}" if coherence else ""

        output += (
            f"**{arrow} {method or 'response'}** [{plat}]\n"
            f"  Topic: {topic} | Intent: {intent} | Gut: {gut}{coherence_str}\n"
            f"  {(summary or '')[:150]}\n\n"
        )

    return tool_result_content([text_content(output)])
