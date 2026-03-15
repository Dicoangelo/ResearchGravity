"""
Oracle Tools — Query coordinator agents from MCP

Exposes the Interview-as-Oracle bridge as MCP tools:
  oracle_status  — List queryable agents and recent tasks
  oracle_ask     — Query agent outputs with natural language
  vibe_config    — Generate coordination config from natural language
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger

log = get_logger("tools.oracle")

# Add coordinator to path
COORDINATOR_DIR = Path.home() / ".claude" / "coordinator"
if str(COORDINATOR_DIR) not in sys.path:
    sys.path.insert(0, str(COORDINATOR_DIR))

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "oracle_status",
        "description": (
            "Get status of coordinator agents — running agents, completed agents, "
            "recent coordination tasks. Use to see what agents are queryable."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "oracle_ask",
        "description": (
            "Query coordinator agent outputs with natural language. "
            "Searches across active agents, completed agent outcomes, and coordination history. "
            "Returns ranked results by relevance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question to ask the oracle",
                },
                "task_id": {
                    "type": "string",
                    "description": "Optional: specific coordination task ID to query",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "vibe_config",
        "description": (
            "Generate a coordination config from natural language. "
            "Auto-detects strategy (research/implement/review/council/team), "
            "model selection, agent count, file locks, and optionally generates "
            "dynamic SUPERMAX personas from the knowledge graph. Dry-run only — "
            "does not execute."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Natural language task description",
                },
                "use_personas": {
                    "type": "boolean",
                    "description": "Generate dynamic personas from knowledge graph (default: false)",
                    "default": False,
                },
            },
            "required": ["task"],
        },
    },
]


# ── Handler ───────────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Route MCP tool calls."""
    handlers = {
        "oracle_status": _oracle_status,
        "oracle_ask": _oracle_ask,
        "vibe_config": _vibe_config,
    }

    handler = handlers.get(name)
    if not handler:
        return tool_result_content(
            [text_content(f"Unknown oracle tool: {name}")], is_error=True
        )

    try:
        return await handler(args)
    except Exception as exc:
        log.error(f"Oracle tool {name} failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Error in {name}: {exc}")], is_error=True
        )


async def _oracle_status(args: Dict) -> Dict:
    from oracle import AgentOracle
    oracle = AgentOracle()
    status = oracle.get_status()

    output = "# Coordinator Oracle Status\n\n"
    output += f"**Running agents:** {status['running_agents']}\n"
    output += f"**Completed agents:** {status['completed_agents']}\n"
    output += f"**Total outcomes:** {status['total_outcomes']}\n"
    output += f"**Recent coordinations:** {status['recent_coordinations']}\n\n"

    if status["agents"]["running"]:
        output += "## Running Agents\n"
        for a in status["agents"]["running"]:
            output += f"- `{a['agent_id']}`: {a['subtask']} ({a['model']})\n"
        output += "\n"

    if status["recent_tasks"]:
        output += "## Recent Tasks\n"
        for t in status["recent_tasks"]:
            output += f"- `{t['task_id']}`: {t['task']} [{t['strategy']}] → {t['status']}\n"

    return tool_result_content([text_content(output)])


async def _oracle_ask(args: Dict) -> Dict:
    from oracle import AgentOracle
    oracle = AgentOracle()

    question = args["question"]
    task_id = args.get("task_id")

    result = oracle.ask(question, task_id=task_id, source="mcp")

    output = f"# Oracle Query: {result.question}\n\n"
    output += f"**Results:** {len(result.results)}\n\n"

    for i, r in enumerate(result.results, 1):
        output += f"## {i}. [{r['source']}] (relevance: {r['relevance']:.0%})\n"
        for k, v in r.items():
            if k not in ("source", "relevance"):
                output += f"- **{k}:** {str(v)[:300]}\n"
        output += "\n"

    if not result.results:
        output += "No matching results found.\n"

    return tool_result_content([text_content(output)])


async def _vibe_config(args: Dict) -> Dict:
    from vibe_coordinator import VibeCoordinator
    vibe = VibeCoordinator()

    task = args["task"]
    use_personas = args.get("use_personas", False)

    config = vibe.generate(task, use_personas=use_personas)

    output = f"# Vibe Config: {config.strategy}\n\n"
    output += f"**Strategy:** {config.strategy} ({config.strategy_confidence:.0%} confidence)\n"
    output += f"**Description:** {config.strategy_description}\n"
    output += f"**Model:** {config.model}\n"
    output += f"**Agents:** {config.agent_count}\n"
    output += f"**Parallel:** {config.parallelizable}\n"
    output += f"**Est. cost:** ${config.estimated_cost_usd:.4f}\n"

    if config.files_detected:
        output += f"**Files:** {', '.join(config.files_detected)}\n"

    output += f"\n## Subtasks\n"
    for i, st in enumerate(config.subtasks, 1):
        output += f"{i}. {st}\n"

    if config.personas:
        output += f"\n## Graph Personas\n"
        for p in config.personas:
            output += f"- **{p['title']}** ({p['finding_count']} findings) — {p['domain']}\n"

    output += f"\n## Reasoning\n{config.reasoning}\n"

    # Include execution command
    output += f"\n## Execute\n```bash\ncoord {config.strategy} \"{task}\"\n```\n"

    return tool_result_content([text_content(output)])
