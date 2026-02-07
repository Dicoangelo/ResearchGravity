"""
Research Tools — Ported from SDK mcp_server.py to raw MCP

8 tools providing ResearchGravity data access:
  get_session_context, search_learnings, get_project_research,
  log_finding, select_context_packs, get_research_index,
  list_projects, get_session_stats
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp_raw.config import Config
from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger

log = get_logger("tools.research")

# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_session_context",
        "description": "Get active research session context including topic, URLs, findings, and status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Optional: Specific session ID. If not provided, returns active session.",
                },
            },
        },
    },
    {
        "name": "search_learnings",
        "description": "Search archived learnings from past research sessions. Returns relevant concepts, findings, and papers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords or phrases)",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum results to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_project_research",
        "description": "Load research files for a specific project (OS-App, CareerCoach, Metaventions, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Project name (e.g., 'os-app', 'careercoach', 'metaventions')",
                },
            },
            "required": ["project_name"],
        },
    },
    {
        "name": "log_finding",
        "description": "Record a finding or insight to the active research session",
        "inputSchema": {
            "type": "object",
            "properties": {
                "finding": {
                    "type": "string",
                    "description": "The finding text to record",
                },
                "type": {
                    "type": "string",
                    "description": "Finding type: general, implementation, metrics, innovation, etc.",
                    "default": "general",
                },
            },
            "required": ["finding"],
        },
    },
    {
        "name": "select_context_packs",
        "description": "Select relevant context packs using Context Packs V2 (7-layer system with semantic embeddings)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query for context pack selection",
                },
                "budget": {
                    "type": "number",
                    "description": "Token budget (default: 50000)",
                    "default": 50000,
                },
                "use_v1": {
                    "type": "boolean",
                    "description": "Force V1 engine (2 layers) instead of V2 (7 layers)",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_research_index",
        "description": "Get the unified research index with cross-project paper references and concepts",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_projects",
        "description": "List all tracked projects with their metadata, tech stack, and status",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_session_stats",
        "description": "Get statistics about research sessions, papers, concepts, and cognitive wallet value",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Dispatcher ───────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Route tool calls to implementations."""
    handlers = {
        "get_session_context": _get_session_context,
        "search_learnings": _search_learnings,
        "get_project_research": _get_project_research,
        "log_finding": _log_finding,
        "select_context_packs": _select_context_packs,
        "get_research_index": _get_research_index,
        "list_projects": _list_projects,
        "get_session_stats": _get_session_stats,
    }

    handler = handlers.get(name)
    if not handler:
        return tool_result_content([text_content(f"Unknown research tool: {name}")], is_error=True)

    try:
        return await handler(args)
    except Exception as exc:
        log.error(f"Tool {name} failed: {exc}", exc_info=True)
        return tool_result_content([text_content(f"Error in {name}: {exc}")], is_error=True)


# ── Implementations ──────────────────────────────────────────────────────────

async def _get_session_context(args: Dict) -> Dict:
    session_id = args.get("session_id")

    if session_id:
        session = _get_session_by_id(session_id)
        if not session:
            return tool_result_content([text_content(f"Session not found: {session_id}")], is_error=True)
    else:
        session = _get_active_session()
        if not session:
            return tool_result_content([text_content(
                'No active session. Start a session with: python3 init_session.py "topic"'
            )])

    output = f"""# Session: {session.get('topic', 'Unknown')}

**ID:** {session.get('session_id', 'unknown')}
**Status:** {session.get('status', 'unknown')}
**Started:** {session.get('started', 'unknown')}
**Working Directory:** {session.get('working_directory', 'unknown')}

## URLs Captured
{len(session.get('urls_captured', []))} URLs logged

## Findings
{len(session.get('findings_captured', []))} findings recorded
"""

    findings = session.get("findings_captured", [])
    if findings:
        output += "\n### Recent Findings:\n"
        for f in findings[-5:]:
            output += f"\n- **{f.get('type', 'general')}**: {f.get('text', '')[:200]}...\n"

    return tool_result_content([text_content(output)])


async def _search_learnings(args: Dict) -> Dict:
    query = args["query"]
    limit = int(args.get("limit", 10))

    results = _search_learnings_text(query, limit)

    if not results:
        return tool_result_content([text_content(f"No learnings found for query: {query}")])

    output = f"# Search Results for: {query}\n\nFound {len(results)} relevant sections:\n\n"
    for i, result in enumerate(results, 1):
        output += f"## {i}. {result['title']}\n\n"
        output += f"{result['preview']}\n\n"
        output += f"**Relevance Score:** {result['relevance']}\n\n---\n\n"

    return tool_result_content([text_content(output)])


async def _get_project_research(args: Dict) -> Dict:
    project_name = args["project_name"].lower()
    files = _get_project_research_files(project_name)

    if not files:
        return tool_result_content([text_content(f"No research files found for project: {project_name}")])

    output = f"# Research Files for: {project_name}\n\nFound {len(files)} files:\n\n"
    for filename, content in files.items():
        preview = content[:1000] + "..." if len(content) > 1000 else content
        output += f"## {filename}\n\n{preview}\n\n---\n\n"

    return tool_result_content([text_content(output)])


async def _log_finding(args: Dict) -> Dict:
    finding = args["finding"]
    finding_type = args.get("type", "general")

    success = _log_finding_to_session(finding, finding_type)

    if success:
        return tool_result_content([text_content(
            f"Finding logged successfully\n\nType: {finding_type}\nText: {finding}"
        )])
    else:
        return tool_result_content([text_content(
            "Failed to log finding. No active session or permission error."
        )], is_error=True)


async def _select_context_packs(args: Dict) -> Dict:
    query = args["query"]
    budget = int(args.get("budget", 50000))
    use_v1 = args.get("use_v1", False)

    try:
        rg_dir = str(Path(__file__).resolve().parent.parent.parent)
        if rg_dir not in sys.path:
            sys.path.insert(0, rg_dir)

        from select_packs_v2_integrated import PackSelectorV2Integrated, format_output

        selector = PackSelectorV2Integrated(force_v1=use_v1)
        packs, metadata = selector.select_packs(
            context=query,
            token_budget=budget,
            enable_pruning=True,
        )
        output = format_output(packs, metadata, output_format="text")
        return tool_result_content([text_content(output)])

    except Exception as exc:
        log.error(f"Context pack selection failed: {exc}", exc_info=True)
        return tool_result_content([text_content(f"Error selecting context packs: {exc}")], is_error=True)


async def _get_research_index(args: Dict) -> Dict:
    index_file = Config.RESEARCH_DIR / "INDEX.md"
    content = _load_text(index_file)

    if not content:
        return tool_result_content([text_content("Research index not found or empty")])

    return tool_result_content([text_content(content)])


async def _list_projects(args: Dict) -> Dict:
    projects = _load_json(Config.PROJECTS_FILE)

    if not projects:
        return tool_result_content([text_content("No projects found")])

    output = "# Tracked Projects\n\n"
    for proj_name, data in projects.items():
        if proj_name.startswith("_"):
            continue
        output += f"## {proj_name}\n\n"
        output += f"**Status:** {data.get('status', 'unknown')}\n"
        output += f"**Tech Stack:** {data.get('tech_stack', 'unknown')}\n"
        output += f"**Focus:** {', '.join(data.get('focus_areas', []))}\n\n"

    return tool_result_content([text_content(output)])


async def _get_session_stats(args: Dict) -> Dict:
    tracker = _load_json(Config.SESSION_TRACKER)
    projects = _load_json(Config.PROJECTS_FILE)

    total_sessions = len(tracker.get("sessions", {}))
    archived = sum(
        1 for s in tracker.get("sessions", {}).values()
        if s.get("status") == "archived"
    )

    stats = projects.get("_stats", {})
    total_concepts = stats.get("concepts", 0)
    total_papers = stats.get("papers", 0)
    total_urls = stats.get("urls", 0)
    wallet_value = stats.get("cognitive_wallet", 0)

    output = f"""# ResearchGravity Statistics

**Total Sessions:** {total_sessions}
**Archived Sessions:** {archived}
**Active Sessions:** {total_sessions - archived}

**Concepts Tracked:** {total_concepts}
**Papers Indexed:** {total_papers}
**URLs Logged:** {total_urls}

**Cognitive Wallet Value:** ${wallet_value:.2f}

**Projects:** {len([k for k in projects.keys() if not k.startswith('_')])}
"""

    return tool_result_content([text_content(output)])


# ── File helpers (no SDK dependency) ─────────────────────────────────────────

def _load_json(path: Path) -> Dict:
    try:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        log.error(f"Error loading {path}: {exc}")
        return {}


def _load_text(path: Path) -> str:
    try:
        if not path.exists():
            return ""
        with open(path, "r") as f:
            return f.read()
    except Exception as exc:
        log.error(f"Error loading {path}: {exc}")
        return ""


def _get_active_session() -> Optional[Dict]:
    tracker = _load_json(Config.SESSION_TRACKER)
    if not tracker or "active_session" not in tracker:
        return None
    session_id = tracker["active_session"]
    if not session_id or session_id not in tracker.get("sessions", {}):
        return None
    return tracker["sessions"][session_id]


def _get_session_by_id(session_id: str) -> Optional[Dict]:
    tracker = _load_json(Config.SESSION_TRACKER)
    return tracker.get("sessions", {}).get(session_id)


def _search_learnings_text(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    learnings_text = _load_text(Config.LEARNINGS_FILE)
    if not learnings_text:
        return []

    query_lower = query.lower()
    results = []

    sections = learnings_text.split("\n## ")
    for section in sections[1:]:
        if query_lower in section.lower():
            lines = section.split("\n", 1)
            title = lines[0].strip()
            content = lines[1] if len(lines) > 1 else ""
            preview = content[:500] + "..." if len(content) > 500 else content

            results.append({
                "title": title,
                "preview": preview,
                "relevance": section.lower().count(query_lower),
            })
            if len(results) >= limit:
                break

    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results


def _get_project_research_files(project_name: str) -> Dict[str, str]:
    project_dir = Config.RESEARCH_DIR / project_name
    if not project_dir.exists():
        return {}

    files = {}
    for file_path in project_dir.glob("*.md"):
        files[file_path.name] = _load_text(file_path)
    return files


def _log_finding_to_session(finding: str, finding_type: str = "general") -> bool:
    tracker = _load_json(Config.SESSION_TRACKER)
    if not tracker or "active_session" not in tracker:
        return False

    session_id = tracker["active_session"]
    if not session_id or session_id not in tracker.get("sessions", {}):
        return False

    session = tracker["sessions"][session_id]
    if "findings_captured" not in session:
        session["findings_captured"] = []

    session["findings_captured"].append({
        "text": finding,
        "type": finding_type,
        "timestamp": datetime.now().isoformat(),
    })

    try:
        with open(Config.SESSION_TRACKER, "w") as f:
            json.dump(tracker, f, indent=2)
        return True
    except Exception as exc:
        log.error(f"Error saving finding: {exc}")
        return False
