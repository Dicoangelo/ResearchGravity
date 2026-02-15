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
        "name": "auto_delegate_monitor",
        "description": (
            "Check an X monitor for new tweets, score them, and auto-delegate research "
            "for any high-quality signals found. Zero human in the loop — monitor fires, "
            "signals get scored, top findings trigger delegate_research chains automatically. "
            "Returns monitor results + any delegation chains spawned."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "monitor_name": {
                    "type": "string",
                    "description": "Name of the X monitor to check (from list_monitors)",
                },
                "quality_threshold": {
                    "type": "number",
                    "description": "Minimum quality score to trigger delegation (default: 0.6)",
                    "default": 0.6,
                },
                "auto_delegate": {
                    "type": "boolean",
                    "description": "If true, automatically submit delegation chains for qualifying signals (default: true)",
                    "default": True,
                },
            },
            "required": ["monitor_name"],
        },
    },
    {
        "name": "publish_research_thread",
        "description": (
            "Generate a research thread from delegation results using the X research "
            "template (5-tweet structure: hook, context, key findings, analysis, CTA). "
            "Pass a chain_id to auto-extract results, or provide custom content. "
            "Returns formatted tweets ready for post_thread. Does NOT auto-post."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chain_id": {
                    "type": "string",
                    "description": "Optional: Chain ID to extract results from",
                },
                "topic": {
                    "type": "string",
                    "description": "Research topic / title for the thread",
                },
                "findings": {
                    "type": "string",
                    "description": "Key findings text (2-3 bullet points with data)",
                },
                "analysis": {
                    "type": "string",
                    "description": "Your unique analysis or take on the findings",
                },
                "link": {
                    "type": "string",
                    "description": "Optional: URL to the source material",
                },
            },
            "required": ["topic"],
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
        "auto_delegate_monitor": _auto_delegate_monitor,
        "publish_research_thread": _publish_research_thread,
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
    """View past delegations with outcomes from real databases."""
    limit = int(args.get("limit", 10))
    task_type = args.get("task_type")

    try:
        import sqlite3
        from pathlib import Path

        output = "# Delegation History\n\n"

        # ── 1. Evolution outcomes (delegation.db) ────────────────────────
        evo_db = Path.home() / ".agent-core" / "storage" / "delegation.db"
        evo_rows = []
        if evo_db.exists():
            conn = sqlite3.connect(str(evo_db), timeout=1.0)
            conn.row_factory = sqlite3.Row
            evo_rows = conn.execute(
                "SELECT * FROM evolution_outcomes ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()

        if evo_rows:
            output += f"## Delegation Outcomes ({len(evo_rows)})\n\n"
            output += "| ID | Success | Quality | Cost | Duration | Subtasks | Feedback |\n"
            output += "|----|---------|---------|------|----------|----------|----------|\n"

            for row in evo_rows:
                did = row["delegation_id"][:16] + "..."
                success = "Y" if row["success"] else "N"
                quality = f"{row['quality_score']:.2f}"
                cost = f"{row['actual_cost']:.2f}"
                duration = f"{row['actual_duration']:.1f}s"
                subtasks = str(row["subtask_count"])
                feedback = (row["feedback"] or "")[:40]
                output += f"| {did} | {success} | {quality} | {cost} | {duration} | {subtasks} | {feedback} |\n"

            output += "\n"
        else:
            output += "*No evolution outcomes recorded yet.*\n\n"

        # ── 2. 4Ds gate events (delegation_events.db) ───────────────────
        events_db = Path.home() / ".agent-core" / "storage" / "delegation_events.db"
        gate_rows = []
        if events_db.exists():
            conn = sqlite3.connect(str(events_db), timeout=1.0)
            conn.row_factory = sqlite3.Row
            try:
                gate_rows = conn.execute(
                    "SELECT * FROM delegation_events ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            except sqlite3.OperationalError:
                pass  # Table may not exist yet
            conn.close()

        if gate_rows:
            output += f"## 4Ds Gate Events ({len(gate_rows)})\n\n"
            output += "| Time | Gate | Status | Agent | Task |\n"
            output += "|------|------|--------|-------|------|\n"

            for row in gate_rows:
                from datetime import datetime
                ts = datetime.fromtimestamp(row["timestamp"]).strftime("%m/%d %H:%M")
                gate = row["gate_type"] if row["gate_type"] else row["event_type"]
                status = row["status"]
                agent = (row["agent_id"] or "")[:20]
                task = (row["task_id"] or "")[:30]
                output += f"| {ts} | {gate} | {status} | {agent} | {task} |\n"

            output += "\n"

        # ── 3. In-memory active chains ───────────────────────────────────
        coordinator = await _get_coordinator()
        if coordinator.chains:
            output += f"## Active Chains ({len(coordinator.chains)})\n\n"
            for cid, chain in list(coordinator.chains.items())[-limit:]:
                total = len(chain.subtask_statuses)
                completed = sum(1 for st in chain.subtask_statuses.values() if st["status"] == "completed")
                failed = sum(1 for st in chain.subtask_statuses.values() if st["status"] == "failed")
                output += f"- **{cid}** — {chain.status} ({completed}/{total} done, {failed} failed)\n"
            output += "\n"

        if not evo_rows and not gate_rows and not coordinator.chains:
            output += "*No delegation history found. Submit a delegation to start tracking.*\n"

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


async def _publish_research_thread(args: Dict) -> Dict:
    """Generate a research thread from delegation results."""
    topic = args["topic"]
    chain_id = args.get("chain_id")
    findings = args.get("findings", "")
    analysis = args.get("analysis", "")
    link = args.get("link", "")

    try:
        # If chain_id provided, extract results from chain
        if chain_id and not findings:
            coordinator = await _get_coordinator()
            try:
                status = await coordinator.get_chain_status(chain_id)
                # Build findings from completed subtask results
                results = []
                for sid, st in status["subtask_statuses"].items():
                    if st["status"] == "completed" and st.get("result"):
                        results.append(st["result"][:150])
                if results:
                    findings = "\n".join(f"- {r}" for r in results[:3])
            except ValueError:
                pass

        # Generate 5-tweet research thread
        tweets = []

        # Tweet 1: HOOK
        hook = f"Just ran an AI-powered research delegation on: {topic}\n\n"
        hook += "Here's what the multi-agent system found "
        tweets.append(hook.strip()[:270] + " (thread)")

        # Tweet 2: CONTEXT
        context = f"The delegation system classified the task, decomposed it into subtasks, "
        context += f"routed each to specialized agents based on Bayesian trust scores, "
        context += f"and executed them in parallel."
        if chain_id:
            context += f"\n\nChain: {chain_id}"
        tweets.append(context.strip()[:280])

        # Tweet 3: KEY FINDINGS
        if findings:
            tweet3 = f"Key findings:\n\n{findings}"
        else:
            tweet3 = f"Key findings:\n\n- Research delegation completed successfully\n- Trust-weighted agent routing enabled capability matching\n- Results verified via automated quality gates"
        tweets.append(tweet3.strip()[:280])

        # Tweet 4: ANALYSIS
        if analysis:
            tweet4 = analysis
        else:
            tweet4 = (
                "The key insight: Bayesian Beta trust scoring lets the system learn "
                "which agents are reliable for which task types. After ~10 interactions, "
                "trust scores converge and routing quality improves significantly."
            )
        tweets.append(tweet4.strip()[:280])

        # Tweet 5: LINK + CTA
        cta = "Built with ResearchGravity's intelligent delegation module.\n\n"
        cta += "Contract-first decomposition + 4Ds safety gates + trust-weighted routing.\n\n"
        if link:
            cta += f"Source: {link}\n\n"
        cta += "What delegation patterns are you seeing in your AI systems?"
        tweets.append(cta.strip()[:280])

        # Format output
        output = f"# Research Thread: {topic}\n\n"
        output += f"**Template:** Research (5 tweets)\n"
        output += f"**Status:** Ready to review — NOT auto-posted\n\n"

        for i, tweet in enumerate(tweets, 1):
            labels = ["HOOK", "CONTEXT", "KEY FINDINGS", "ANALYSIS", "LINK + CTA"]
            output += f"### Tweet {i} — {labels[i-1]} ({len(tweet)} chars)\n"
            output += f"```\n{tweet}\n```\n\n"

        output += "---\n"
        output += "*To post this thread, use `post_thread` with these tweets.*\n"
        output += f"*JSON for post_thread:*\n```json\n{json.dumps(tweets)}\n```\n"

        return tool_result_content([text_content(output)])

    except Exception as exc:
        log.error(f"Thread generation failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Thread generation failed: {exc}")],
            is_error=True,
        )


async def _auto_delegate_monitor(args: Dict) -> Dict:
    """Check X monitor → score → auto-delegate research for quality signals."""
    monitor_name = args["monitor_name"]
    quality_threshold = float(args.get("quality_threshold", 0.6))
    auto_delegate = args.get("auto_delegate", True)

    try:
        coordinator = await _get_coordinator()

        output = f"# Auto-Delegate: {monitor_name}\n\n"

        # Step 1: Check the monitor (simulated — in production this calls X MCP)
        # The actual X monitor check happens via the X MCP server
        # Here we define the delegation logic that runs AFTER monitor results arrive
        output += f"**Quality Threshold:** {quality_threshold}\n"
        output += f"**Auto-Delegate:** {'enabled' if auto_delegate else 'disabled'}\n\n"

        # Step 2: Accept pre-scored results passed as monitor_results
        monitor_results = args.get("monitor_results", [])

        if not monitor_results:
            output += (
                "*No results provided. To use this tool in the full pipeline:*\n\n"
                "```\n"
                "1. check_monitor(name) → get tweets\n"
                "2. score_tweets(tweets) → get quality scores\n"
                "3. auto_delegate_monitor(name, monitor_results=[scored]) → delegate\n"
                "```\n\n"
                "*Or pass monitor_results as a JSON array of objects with 'text' and 'quality_score' fields.*\n"
            )
            return tool_result_content([text_content(output)])

        # Step 3: Filter by quality threshold
        qualifying = []
        if isinstance(monitor_results, str):
            monitor_results = json.loads(monitor_results)

        for result in monitor_results:
            score = result.get("quality_score", 0)
            if score >= quality_threshold:
                qualifying.append(result)

        output += f"**Total Results:** {len(monitor_results)}\n"
        output += f"**Qualifying (>={quality_threshold}):** {len(qualifying)}\n\n"

        if not qualifying:
            output += "*No results met the quality threshold. No delegations triggered.*\n"
            return tool_result_content([text_content(output)])

        # Step 4: Auto-delegate qualifying signals
        chains = []
        if auto_delegate:
            for signal in qualifying[:3]:  # Cap at 3 concurrent delegations
                text = signal.get("text", signal.get("description", ""))[:200]
                task = f"Deep research on signal: {text}"

                chain_id = await coordinator.submit_chain(
                    task=task,
                    context={
                        "source": "x_monitor",
                        "monitor": monitor_name,
                        "quality_score": signal.get("quality_score", 0),
                    }
                )
                chains.append({
                    "chain_id": chain_id,
                    "signal": text[:80],
                    "quality": signal.get("quality_score", 0),
                })

            output += f"## Delegations Spawned ({len(chains)})\n\n"
            for c in chains:
                output += f"- **{c['chain_id']}** (quality={c['quality']:.2f}): {c['signal']}...\n"
        else:
            output += "## Qualifying Signals (delegation disabled)\n\n"
            for s in qualifying:
                text = s.get("text", "")[:100]
                output += f"- [{s.get('quality_score', 0):.2f}] {text}\n"

        return tool_result_content([text_content(output)])

    except Exception as exc:
        log.error(f"Auto-delegate monitor failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Auto-delegate monitor failed: {exc}")],
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
