"""
Webhook Tools — MCP tools for webhook receiver management.

Tools:
  webhook_status     — Server health, provider status, event counts
  webhook_list       — Recent webhook deliveries from audit trail
  webhook_test       — Send a test webhook to verify a provider
"""

import json
import logging
from typing import Any, Dict, List, Optional

from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger

log = get_logger("tools.webhook")

# Shared DB — injected by server via set_db()
_pool = None


def set_db(db):
    """Called by server to inject shared database instance."""
    global _pool
    if hasattr(db, "_pool") and db._pool is not None:
        _pool = db._pool
    log.info("Webhook tools: DB injected")


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "webhook_status",
        "description": (
            "Get UCW Webhook Receiver status: server health, registered providers, "
            "total events received/stored/rejected, and uptime."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "webhook_list",
        "description": (
            "List recent webhook deliveries from the audit trail. "
            "Shows provider, event type, status, and processing time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Filter by provider (e.g., 'github', 'slack'). Omit for all.",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum deliveries to return (default: 20)",
                },
            },
        },
    },
    {
        "name": "webhook_test",
        "description": (
            "Send a test webhook event to verify a provider handler is working. "
            "Returns parsed event count and sample content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Provider to test (e.g., 'github', 'slack', 'generic')",
                },
            },
            "required": ["provider"],
        },
    },
]


# ── Handler ──────────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: dict) -> dict:
    """Dispatch webhook tool calls."""
    try:
        if name == "webhook_status":
            return await _webhook_status()
        elif name == "webhook_list":
            return await _webhook_list(
                provider=args.get("provider"),
                limit=int(args.get("limit", 20)),
            )
        elif name == "webhook_test":
            return await _webhook_test(provider=args["provider"])
        else:
            return tool_result_content(
                [text_content(f"Unknown webhook tool: {name}")], is_error=True
            )
    except Exception as exc:
        log.error(f"Webhook tool error ({name}): {exc}")
        return tool_result_content(
            [text_content(f"Error: {exc}")], is_error=True
        )


async def _webhook_status() -> dict:
    """Get webhook receiver status."""
    if not _pool:
        return tool_result_content([text_content(
            "Webhook tools: no database connection. "
            "Ensure the webhook server is running (python3 -m webhook start)."
        )])

    async with _pool.acquire() as conn:
        # Check if audit table exists
        exists = await conn.fetchval(
            """SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'webhook_events'
            )"""
        )

        if not exists:
            return tool_result_content([text_content(
                "Webhook audit table not found. "
                "Start the webhook server: python3 -m webhook start"
            )])

        total = await conn.fetchval("SELECT COUNT(*) FROM webhook_events")
        stored = await conn.fetchval(
            "SELECT COUNT(*) FROM webhook_events WHERE status = 'stored'"
        )
        rejected = await conn.fetchval(
            "SELECT COUNT(*) FROM webhook_events WHERE status = 'rejected'"
        )
        cognitive = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE platform LIKE '%-webhook'"
        )
        platforms = await conn.fetch(
            """SELECT platform, COUNT(*) as cnt
               FROM cognitive_events
               WHERE platform LIKE '%-webhook'
               GROUP BY platform ORDER BY cnt DESC"""
        )

    lines = [
        "UCW Webhook Receiver Status",
        "=" * 40,
        f"Audit entries:    {total}",
        f"  Stored:         {stored}",
        f"  Rejected:       {rejected}",
        f"Cognitive events: {cognitive} (from webhooks)",
        "",
        "Webhook platforms in cognitive_events:",
    ]
    for r in platforms:
        lines.append(f"  {r['platform']:20s} {r['cnt']} events")

    return tool_result_content([text_content("\n".join(lines))])


async def _webhook_list(provider: Optional[str] = None, limit: int = 20) -> dict:
    """List recent webhook deliveries."""
    if not _pool:
        return tool_result_content([text_content("No database connection.")])

    async with _pool.acquire() as conn:
        if provider:
            rows = await conn.fetch(
                """SELECT provider, event_type, status, events_parsed,
                          events_stored, processing_time_ms, created_at
                   FROM webhook_events
                   WHERE provider = $1
                   ORDER BY created_at DESC LIMIT $2""",
                provider, limit,
            )
        else:
            rows = await conn.fetch(
                """SELECT provider, event_type, status, events_parsed,
                          events_stored, processing_time_ms, created_at
                   FROM webhook_events
                   ORDER BY created_at DESC LIMIT $1""",
                limit,
            )

    if not rows:
        return tool_result_content([text_content("No webhook deliveries found.")])

    lines = [f"Recent Webhook Deliveries (limit={limit})", "=" * 70]
    for r in rows:
        ms = f"{r['processing_time_ms']}ms" if r['processing_time_ms'] else "—"
        lines.append(
            f"[{r['created_at']:%Y-%m-%d %H:%M:%S}] "
            f"{r['provider']:10s} {r['event_type']:18s} "
            f"-> {r['status']:8s} "
            f"(parsed={r['events_parsed']}, stored={r['events_stored']}, {ms})"
        )

    return tool_result_content([text_content("\n".join(lines))])


async def _webhook_test(provider: str) -> dict:
    """Send a test webhook via HTTP to the running server."""
    try:
        import httpx
    except ImportError:
        return tool_result_content([text_content(
            "httpx not installed. Test manually: "
            "curl -X POST http://localhost:3848/webhook/test/github"
        )])

    from webhook.config import WEBHOOK_HOST, WEBHOOK_PORT
    base_url = f"http://{WEBHOOK_HOST}:{WEBHOOK_PORT}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{base_url}/webhook/test/{provider}")
            data = resp.json()
    except Exception as exc:
        return tool_result_content([text_content(
            f"Failed to reach webhook server at {base_url}: {exc}\n"
            "Start it with: python3 -m webhook start"
        )])

    lines = [
        f"Test result for {provider}:",
        f"  Events parsed: {data.get('events_parsed', 0)}",
        f"  Events stored: {data.get('events_stored', 0)}",
    ]
    sample = data.get("sample_content", "")
    if sample:
        lines.append(f"  Sample: {sample[:200]}")

    return tool_result_content([text_content("\n".join(lines))])
