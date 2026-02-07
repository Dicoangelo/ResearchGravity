"""
UCW Tools — New cognitive capture tools (not in SDK server)

Tools:
  ucw_capture_stats  — Current capture session statistics
  ucw_timeline       — Unified cross-platform event timeline
  detect_emergence   — Real-time emergence signal detection
"""

import json
from typing import Any, Dict, List

from mcp_raw.config import Config
from mcp_raw.db import CaptureDB
from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger

log = get_logger("tools.ucw")

# Shared DB instance — injected by server via set_db() after initialization
_db = None


def set_db(db):
    """Called by server to inject shared database instance."""
    global _db
    _db = db

# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "ucw_capture_stats",
        "description": "Get current UCW capture session statistics: events, turns, topics, gut signals, and total capture metrics",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "ucw_timeline",
        "description": "Get a unified cross-platform cognitive event timeline sorted by time",
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "description": "Filter by platform (e.g., 'claude-desktop', 'chatgpt'). Omit for all platforms.",
                },
                "since_ns": {
                    "type": "number",
                    "description": "Only return events after this nanosecond timestamp. Omit for all events.",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum events to return (default: 50)",
                    "default": 50,
                },
            },
        },
    },
    {
        "name": "detect_emergence",
        "description": "Scan recent cognitive events for emergence signals: high coherence potential, concept clusters, and meta-cognitive patterns",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "number",
                    "description": "Number of recent events to scan (default: 100)",
                    "default": 100,
                },
            },
        },
    },
]


# ── Dispatcher ───────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    handlers = {
        "ucw_capture_stats": _ucw_capture_stats,
        "ucw_timeline": _ucw_timeline,
        "detect_emergence": _detect_emergence,
    }

    handler = handlers.get(name)
    if not handler:
        return tool_result_content([text_content(f"Unknown UCW tool: {name}")], is_error=True)

    try:
        return await handler(args)
    except Exception as exc:
        log.error(f"Tool {name} failed: {exc}", exc_info=True)
        return tool_result_content([text_content(f"Error in {name}: {exc}")], is_error=True)


# ── Implementations ──────────────────────────────────────────────────────────

async def _ucw_capture_stats(args: Dict) -> Dict:
    """Return current + all-time capture statistics."""
    if not _db:
        return tool_result_content([text_content(
            "No capture data available. Database not injected — server may not have started."
        )])

    session_stats = await _db.get_session_stats()
    all_stats = await _db.get_all_stats()

    if not session_stats and not all_stats:
        return tool_result_content([text_content(
            "No capture data available. Database may not be initialized."
        )])

    output = "# UCW Capture Statistics\n\n"

    if session_stats:
        output += "## Current Session\n\n"
        output += f"**Session ID:** {session_stats.get('session_id', 'unknown')}\n"
        output += f"**Events Captured:** {session_stats.get('event_count', 0)}\n"
        output += f"**Turns:** {session_stats.get('turn_count', 0)}\n\n"

        topics = session_stats.get("topics", {})
        if topics:
            output += "### Topics\n"
            for topic, count in topics.items():
                output += f"- {topic}: {count}\n"
            output += "\n"

        signals = session_stats.get("gut_signals", {})
        if signals:
            output += "### Gut Signals\n"
            for signal, count in signals.items():
                output += f"- {signal}: {count}\n"
            output += "\n"

    if all_stats:
        output += "## All-Time\n\n"
        output += f"**Total Events:** {all_stats.get('total_events', 0)}\n"
        output += f"**Total Sessions:** {all_stats.get('total_sessions', 0)}\n"
        output += f"**Bytes Captured:** {all_stats.get('total_bytes_captured', 0):,}\n\n"

        all_signals = all_stats.get("gut_signals", {})
        if all_signals:
            output += "### Gut Signal Distribution\n"
            for signal, count in all_signals.items():
                output += f"- {signal}: {count}\n"
            output += "\n"

    return tool_result_content([text_content(output)])


async def _ucw_timeline(args: Dict) -> Dict:
    """Query cognitive events as a unified timeline."""
    platform = args.get("platform")
    since_ns = args.get("since_ns")
    limit = int(args.get("limit", 50))

    if not _db:
        return tool_result_content([text_content("Database not initialized.")], is_error=True)

    conn = _db._conn
    if not conn:
        return tool_result_content([text_content("Database not initialized.")], is_error=True)

    # Build query
    query = "SELECT event_id, timestamp_ns, direction, method, platform, light_topic, light_intent, light_summary, instinct_gut_signal, instinct_coherence FROM cognitive_events"
    conditions = []
    params: list = []

    if platform:
        conditions.append("platform = ?")
        params.append(platform)
    if since_ns:
        conditions.append("timestamp_ns > ?")
        params.append(int(since_ns))

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp_ns DESC LIMIT ?"
    params.append(limit)

    cur = conn.execute(query, params)
    rows = cur.fetchall()

    if not rows:
        return tool_result_content([text_content("No events found matching criteria.")])

    output = f"# Cognitive Event Timeline ({len(rows)} events)\n\n"

    for row in reversed(rows):  # Chronological order (oldest first)
        event_id, ts, direction, method, plat, topic, intent, summary, gut, coherence = row
        arrow = "->" if direction == "out" else "<-"
        coherence_str = f" [coherence={coherence:.2f}]" if coherence else ""
        output += (
            f"**{arrow} {method or 'response'}** ({plat})\n"
            f"  Topic: {topic} | Intent: {intent} | Gut: {gut}{coherence_str}\n"
            f"  {(summary or '')[:150]}\n\n"
        )

    return tool_result_content([text_content(output)])


async def _detect_emergence(args: Dict) -> Dict:
    """Scan recent events for emergence signals."""
    limit = int(args.get("limit", 100))

    if not _db:
        return tool_result_content([text_content("Database not initialized.")], is_error=True)

    conn = _db._conn
    if not conn:
        return tool_result_content([text_content("Database not initialized.")], is_error=True)

    cur = conn.execute(
        """SELECT event_id, timestamp_ns, method, platform,
                  light_topic, light_concepts, light_intent,
                  instinct_coherence, instinct_indicators, instinct_gut_signal
           FROM cognitive_events
           ORDER BY timestamp_ns DESC LIMIT ?""",
        (limit,),
    )
    rows = cur.fetchall()

    if not rows:
        return tool_result_content([text_content("No events to analyze.")])

    # Analyze for emergence patterns
    high_coherence = []
    concept_clusters = []
    meta_cognitive = []
    breakthrough_signals = []

    for row in rows:
        (event_id, ts, method, platform, topic, concepts_json,
         intent, coherence, indicators_json, gut) = row

        indicators = _safe_json_list(indicators_json)
        concepts = _safe_json_list(concepts_json)

        if coherence and coherence > 0.7:
            high_coherence.append({
                "event_id": event_id,
                "coherence": coherence,
                "topic": topic,
                "method": method,
            })

        if len(concepts) >= 3:
            concept_clusters.append({
                "event_id": event_id,
                "concepts": concepts,
                "topic": topic,
            })

        if "meta_cognitive" in indicators:
            meta_cognitive.append({
                "event_id": event_id,
                "topic": topic,
                "indicators": indicators,
            })

        if gut == "breakthrough_potential":
            breakthrough_signals.append({
                "event_id": event_id,
                "topic": topic,
                "coherence": coherence,
            })

    # Build report
    total_scanned = len(rows)
    emergence_score = min(1.0, (
        len(high_coherence) * 0.15 +
        len(concept_clusters) * 0.1 +
        len(meta_cognitive) * 0.25 +
        len(breakthrough_signals) * 0.3
    ))

    output = f"# Emergence Detection Report\n\n"
    output += f"**Events Scanned:** {total_scanned}\n"
    output += f"**Emergence Score:** {emergence_score:.3f}\n\n"

    if breakthrough_signals:
        output += f"## Breakthrough Signals ({len(breakthrough_signals)})\n"
        for s in breakthrough_signals[:5]:
            output += f"- Event {s['event_id']}: topic={s['topic']} coherence={s.get('coherence', 0):.2f}\n"
        output += "\n"

    if meta_cognitive:
        output += f"## Meta-Cognitive Events ({len(meta_cognitive)})\n"
        for m in meta_cognitive[:5]:
            output += f"- Event {m['event_id']}: topic={m['topic']} indicators={m['indicators']}\n"
        output += "\n"

    if high_coherence:
        output += f"## High Coherence Events ({len(high_coherence)})\n"
        for h in high_coherence[:5]:
            output += f"- Event {h['event_id']}: coherence={h['coherence']:.3f} topic={h['topic']}\n"
        output += "\n"

    if concept_clusters:
        output += f"## Concept Clusters ({len(concept_clusters)})\n"
        for c in concept_clusters[:5]:
            output += f"- Event {c['event_id']}: {c['concepts']}\n"
        output += "\n"

    if emergence_score < 0.1:
        output += "\n*No significant emergence signals detected. Continue working — patterns emerge over time.*\n"
    elif emergence_score > 0.5:
        output += "\n*Strong emergence signals detected. Consider capturing this moment as a coherence event.*\n"

    return tool_result_content([text_content(output)])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_json_list(value) -> List[str]:
    """Safely parse a JSON array string."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []
