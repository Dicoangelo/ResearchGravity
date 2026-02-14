"""
Delegation Tools — MCP tools for intelligent delegation system

Tools:
  delegate_research       — Submit a research task for intelligent delegation
  delegation_status       — Check status of active delegation chain
  get_agent_trust         — Query agent trust scores
  delegation_history      — View past delegations with outcomes
  delegation_insights     — Meta-insights: top agents, failure patterns, methodology evolution
"""

import asyncio
import json
from typing import Any, Dict, List

from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger

log = get_logger("tools.delegation")

# Shared DB instance — injected by server via set_db() after initialization
_db = None

# Module-level coordinator — persists across MCP calls within same server process
_coordinator = None


def set_db(db):
    """Called by server to inject shared database instance."""
    global _db
    _db = db
    log.info("Delegation tools: DB injected")


async def _get_coordinator():
    """Get or create the shared DelegationCoordinator instance."""
    global _coordinator
    if _coordinator is None:
        from delegation import DelegationCoordinator
        _coordinator = DelegationCoordinator()
        await _coordinator.__aenter__()
    return _coordinator


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "delegate_research",
        "description": (
            "Submit a research task for intelligent delegation. Returns chain_id and "
            "initial decomposition. The system will classify, decompose, and route the "
            "task to appropriate agents based on trust scores and capabilities."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Natural language task description (e.g., 'Research multi-agent orchestration patterns')",
                },
                "context": {
                    "type": "object",
                    "description": "Optional context dict with keys: is_critical, deadline, expected_output, sources, domain",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "delegation_status",
        "description": (
            "Check status of an active delegation chain. Returns progress (0.0-1.0), "
            "per-subtask status, timing information, and agent assignments. Shows which "
            "subtasks are pending, running, completed, or failed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chain_id": {
                    "type": "string",
                    "description": "Delegation chain ID returned from delegate_research",
                },
            },
            "required": ["chain_id"],
        },
    },
    {
        "name": "get_agent_trust",
        "description": (
            "Query agent trust scores. If no agent_id specified, returns all agents "
            "sorted by trust score. Can filter by task_type for domain-specific trust. "
            "Shows success rate, average quality, and failure count."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Optional: Specific agent ID to query. Omit for all agents.",
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional: Filter by task type (e.g., 'research', 'code', 'analysis')",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum agents to return (default: 20)",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "delegation_history",
        "description": (
            "View past delegations with outcomes, timing, and trust score changes. "
            "Shows what was delegated, to which agents, verification results, and "
            "how trust scores evolved. Useful for debugging and learning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "number",
                    "description": "Maximum delegations to return (default: 10)",
                    "default": 10,
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional: Filter by task type",
                },
            },
        },
    },
    {
        "name": "delegation_insights",
        "description": (
            "Meta-insights: top performing agents, failure patterns, methodology evolution, "
            "and routing weight changes. Analyzes delegation data from the last N days to "
            "identify trends, bottlenecks, and improvement opportunities."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "number",
                    "description": "Number of days to analyze (default: 7)",
                    "default": 7,
                },
            },
        },
    },
    {
        "name": "sync_x_trust",
        "description": (
            "Sync X/Twitter author trust scores into the delegation trust ledger. "
            "Both systems use Bayesian Beta distribution, so trust maps exactly. "
            "Pass the JSON output from top_authors or author_trust. Creates delegation "
            "agent entries prefixed 'x-author:' that the router can use for research tasks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "authors_json": {
                    "type": "string",
                    "description": "JSON string from X top_authors or author_trust output",
                },
            },
            "required": ["authors_json"],
        },
    },
]


# ── Dispatcher ───────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    handlers = {
        "delegate_research": _delegate_research,
        "delegation_status": _delegation_status,
        "get_agent_trust": _get_agent_trust,
        "delegation_history": _delegation_history,
        "delegation_insights": _delegation_insights,
        "sync_x_trust": _sync_x_trust,
    }

    handler = handlers.get(name)
    if not handler:
        return tool_result_content([text_content(f"Unknown delegation tool: {name}")], is_error=True)

    try:
        return await handler(args)
    except Exception as exc:
        log.error(f"Tool {name} failed: {exc}", exc_info=True)
        return tool_result_content([text_content(f"Error in {name}: {exc}")], is_error=True)


# ── Implementations ──────────────────────────────────────────────────────────

async def _delegate_research(args: Dict) -> Dict:
    """Submit a research task for intelligent delegation."""
    task = args["task"]
    context = args.get("context")

    try:
        coordinator = await _get_coordinator()
        chain_id = await coordinator.submit_chain(task, context=context)

        # Get initial status
        status = await coordinator.get_chain_status(chain_id)

        # Format response
        output = "# Delegation Chain Submitted\n\n"
        output += f"**Chain ID:** `{chain_id}`\n"
        output += f"**Task:** {task}\n"
        output += f"**Status:** {status['status']}\n"
        output += f"**Progress:** {status['progress']:.0%}\n\n"

        # Show decomposition
        if status["subtask_statuses"]:
            output += f"## Decomposition ({len(status['subtask_statuses'])} subtasks)\n\n"
            for subtask_id, subtask in status["subtask_statuses"].items():
                agent = subtask.get("agent_id", "unassigned")
                desc = subtask.get("description", "")[:80]
                output += f"- **{subtask['status']}** [{agent}]: {desc}\n"

        return tool_result_content([text_content(output)])

    except Exception as exc:
        log.error(f"Delegation submission failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Delegation failed: {exc}")],
            is_error=True,
        )


async def _delegation_status(args: Dict) -> Dict:
    """Check status of active delegation chain."""
    chain_id = args["chain_id"]

    try:
        coordinator = await _get_coordinator()
        status = await coordinator.get_chain_status(chain_id)

        # Format response
        output = f"# Delegation Chain: {chain_id}\n\n"
        output += f"**Status:** {status['status']}\n"
        output += f"**Progress:** {status['progress']:.0%}\n"
        output += f"**Created:** {_format_timestamp(status['created_at'])}\n"
        output += f"**Updated:** {_format_timestamp(status['updated_at'])}\n\n"

        # Show subtasks
        if status["subtask_statuses"]:
            output += f"## Subtasks ({len(status['subtask_statuses'])})\n\n"
            for subtask_id, subtask in status["subtask_statuses"].items():
                st = subtask["status"]
                agent = subtask.get("agent_id", "unassigned")
                desc = subtask.get("description", "")
                started = subtask.get("started_at")
                completed = subtask.get("completed_at")

                output += f"### {subtask_id[:8]}... — {st}\n"
                output += f"**Agent:** {agent}\n"
                output += f"**Description:** {desc[:150]}\n"

                if started:
                    output += f"**Started:** {_format_timestamp(started)}\n"
                if completed:
                    output += f"**Completed:** {_format_timestamp(completed)}\n"

                # Show verification if available
                verification = subtask.get("verification")
                if verification:
                    output += f"**Verification:** Passed={verification.get('passed')} Quality={verification.get('quality_score', 0):.2f}\n"

                output += "\n"

        # Show triggers if any
        triggers = status.get("triggers", [])
        if triggers:
            output += f"## Triggers ({len(triggers)})\n\n"
            for trigger in triggers:
                ttype = trigger.get("type", "unknown")
                tsub = trigger.get("subtask_id", "?")[:8]
                tts = trigger.get("timestamp", 0)
                output += f"- **{ttype}** on {tsub}... at {_format_timestamp(tts)}\n"
            output += "\n"

        # Show recent events
        events = status.get("events", [])
        if events:
            output += f"## Recent Events ({min(5, len(events))})\n\n"
            for event in events[-5:]:
                etype = event.get("event_type", event.get("type", "unknown"))
                output += f"- {etype}\n"
            output += "\n"

        return tool_result_content([text_content(output)])

    except ValueError as exc:
        # Chain not found
        return tool_result_content(
            [text_content(f"Chain not found: {chain_id}")],
            is_error=True,
        )
    except Exception as exc:
        log.error(f"Status query failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Status query failed: {exc}")],
            is_error=True,
        )


async def _get_agent_trust(args: Dict) -> Dict:
    """Query agent trust scores."""
    agent_id = args.get("agent_id")
    task_type = args.get("task_type")
    limit = int(args.get("limit", 20))

    try:
        from delegation.trust_ledger import TrustLedger

        async with TrustLedger() as ledger:
            if agent_id:
                # Get specific agent trust
                trust_score = await ledger.get_trust_score(agent_id, task_type)

                output = f"# Agent Trust: {agent_id}\n\n"
                output += f"**Trust Score:** {trust_score:.3f}\n"

                if task_type:
                    output += f"**Task Type:** {task_type}\n"

                return tool_result_content([text_content(output)])
            else:
                # Get top agents
                agents = await ledger.get_top_agents(task_type, limit)

                if not agents:
                    return tool_result_content([text_content(
                        "No agents found" + (f" for task type '{task_type}'" if task_type else "")
                    )])

                output = f"# Agent Trust Scores ({len(agents)} agents)\n\n"
                if task_type:
                    output += f"**Filtered by task type:** {task_type}\n\n"

                output += "| Agent | Trust | Successes | Failures | Avg Quality | Avg Duration |\n"
                output += "|-------|-------|-----------|----------|-------------|-------------|\n"

                for agent in agents:
                    aid = agent["agent_id"][:20] + "..." if len(agent["agent_id"]) > 20 else agent["agent_id"]
                    trust = agent["trust_score"]
                    succ = agent["success_count"]
                    fail = agent["failure_count"]
                    qual = agent["avg_quality"] or 0
                    dur = agent["avg_duration"] or 0

                    output += f"| {aid} | {trust:.3f} | {succ} | {fail} | {qual:.2f} | {dur:.1f}s |\n"

                return tool_result_content([text_content(output)])

    except Exception as exc:
        log.error(f"Trust query failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Trust query failed: {exc}")],
            is_error=True,
        )


async def _delegation_history(args: Dict) -> Dict:
    """View past delegations with outcomes."""
    limit = int(args.get("limit", 10))
    task_type = args.get("task_type")

    # For now, return a placeholder since delegation_events table is in four_ds.py
    # In the future, this should query the delegation_events database
    try:
        from delegation.four_ds import FourDsGate

        gate = FourDsGate()

        # Query delegation events (this is a simplified implementation)
        # Full implementation would query delegation_events table
        output = "# Delegation History\n\n"
        output += f"**Showing:** Last {limit} delegations\n"
        if task_type:
            output += f"**Task Type:** {task_type}\n"
        output += "\n"

        output += "*Delegation history tracking coming soon. This will show:*\n"
        output += "- Task descriptions and decompositions\n"
        output += "- Agent assignments and routing decisions\n"
        output += "- Verification outcomes and quality scores\n"
        output += "- Trust score changes over time\n"
        output += "- Failure patterns and recovery actions\n"

        return tool_result_content([text_content(output)])

    except Exception as exc:
        log.error(f"History query failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"History query failed: {exc}")],
            is_error=True,
        )


async def _delegation_insights(args: Dict) -> Dict:
    """Meta-insights: top agents, failure patterns, methodology evolution."""
    days = int(args.get("days", 7))

    try:
        from delegation.trust_ledger import TrustLedger

        async with TrustLedger() as ledger:
            # Get top agents across all task types
            top_agents = await ledger.get_top_agents(task_type=None, limit=10)

            output = f"# Delegation Insights (Last {days} days)\n\n"

            # Top performing agents
            if top_agents:
                output += "## Top Performing Agents\n\n"
                output += "| Agent | Trust | Success Rate | Quality |\n"
                output += "|-------|-------|--------------|--------|\n"

                for agent in top_agents[:5]:
                    aid = agent["agent_id"][:30] + "..." if len(agent["agent_id"]) > 30 else agent["agent_id"]
                    trust = agent["trust_score"]
                    total = agent["success_count"] + agent["failure_count"]
                    success_rate = agent["success_count"] / total if total > 0 else 0
                    qual = agent["avg_quality"] or 0

                    output += f"| {aid} | {trust:.3f} | {success_rate:.0%} | {qual:.2f} |\n"

                output += "\n"

            # Failure patterns (placeholder)
            output += "## Failure Patterns\n\n"
            output += "*Analyzing failure patterns across delegations...*\n\n"
            output += "Common failure modes:\n"
            output += "- API timeouts (will be tracked in next iteration)\n"
            output += "- Quality below threshold (will be tracked in next iteration)\n"
            output += "- Resource unavailability (will be tracked in next iteration)\n\n"

            # Methodology evolution (placeholder)
            output += "## Methodology Evolution\n\n"
            output += "*Tracking routing weight changes and decomposition patterns...*\n\n"
            output += "Insights:\n"
            output += "- Trust scores are evolving based on outcomes\n"
            output += "- Bayesian updates ensure smooth convergence\n"
            output += "- Fallback chains provide resilience\n\n"

            # Recommendations
            output += "## Recommendations\n\n"
            if top_agents:
                most_trusted = top_agents[0]
                output += f"- Most trusted agent: {most_trusted['agent_id'][:40]}... (trust={most_trusted['trust_score']:.3f})\n"
                output += "- Consider using this agent for high-criticality tasks\n"

            return tool_result_content([text_content(output)])

    except Exception as exc:
        log.error(f"Insights generation failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Insights generation failed: {exc}")],
            is_error=True,
        )


async def _sync_x_trust(args: Dict) -> Dict:
    """Sync X/Twitter author trust scores into delegation trust ledger."""
    authors_json = args["authors_json"]

    try:
        from delegation.x_trust_bridge import XTrustBridge

        async with XTrustBridge() as bridge:
            results = await bridge.sync_top_authors(authors_json)

        output = f"# X → Delegation Trust Sync\n\n"
        output += f"**Synced:** {len(results)} authors\n\n"

        if results:
            output += "| Author | X Trust | Delegation Trust | Agent ID |\n"
            output += "|--------|---------|------------------|----------|\n"

            for r in results:
                output += (
                    f"| @{r['username']} | {r['x_trust']:.3f} | "
                    f"{r['delegation_trust']:.3f} | {r['agent_id']} |\n"
                )

            output += f"\n*These authors are now available as delegation agents "
            output += f"with prefix `x-author:` for research task routing.*\n"

        return tool_result_content([text_content(output)])

    except Exception as exc:
        log.error(f"X trust sync failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"X trust sync failed: {exc}")],
            is_error=True,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_timestamp(ts: float) -> str:
    """Format Unix timestamp as readable string."""
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
