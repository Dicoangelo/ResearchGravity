#!/usr/bin/env python3
"""
ResearchGravity MCP Server

Exposes ResearchGravity context and research tools via Model Context Protocol (MCP).
This allows Claude Desktop and other MCP clients to access ResearchGravity data.

Tools Provided:
- get_session_context: Get active session information
- search_learnings: Search archived learnings
- get_project_research: Load project-specific research
- log_finding: Record a finding to active session
- select_context_packs: Select relevant context packs (V2)
- get_research_index: Get unified research index

Resources Provided:
- session://active - Active session data
- session://{id} - Specific session data
- learnings://all - All learnings
- project://{name}/research - Project research files
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        LoggingLevel
    )
except ImportError:
    print("ERROR: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ResearchGravity paths
AGENT_CORE = Path.home() / ".agent-core"
SESSION_TRACKER = AGENT_CORE / "session_tracker.json"
PROJECTS_FILE = AGENT_CORE / "projects.json"
LEARNINGS_FILE = AGENT_CORE / "memory" / "learnings.md"
RESEARCH_DIR = AGENT_CORE / "research"
SESSIONS_DIR = AGENT_CORE / "sessions"
CONTEXT_PACKS_DIR = AGENT_CORE / "context-packs"

# Initialize MCP server
app = Server("researchgravity")


def load_json(path: Path) -> Dict:
    """Load JSON file safely"""
    try:
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        app.server.request_context.session.send_log_message(
            level=LoggingLevel.ERROR,
            data=f"Error loading {path}: {e}"
        )
        return {}


def load_text(path: Path) -> str:
    """Load text file safely"""
    try:
        if not path.exists():
            return ""
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        app.server.request_context.session.send_log_message(
            level=LoggingLevel.ERROR,
            data=f"Error loading {path}: {e}"
        )
        return ""


def get_active_session() -> Optional[Dict]:
    """Get active session data"""
    tracker = load_json(SESSION_TRACKER)
    if not tracker or 'active_session' not in tracker:
        return None

    session_id = tracker['active_session']
    if not session_id or session_id not in tracker.get('sessions', {}):
        return None

    return tracker['sessions'][session_id]


def get_session_by_id(session_id: str) -> Optional[Dict]:
    """Get session by ID"""
    tracker = load_json(SESSION_TRACKER)
    return tracker.get('sessions', {}).get(session_id)


def search_learnings_text(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """Search learnings for query"""
    learnings_text = load_text(LEARNINGS_FILE)
    if not learnings_text:
        return []

    # Simple search: find sections containing query (case-insensitive)
    query_lower = query.lower()
    results = []

    # Split by ## headers
    sections = learnings_text.split('\n## ')
    for section in sections[1:]:  # Skip first (title)
        if query_lower in section.lower():
            lines = section.split('\n', 1)
            title = lines[0].strip()
            content = lines[1] if len(lines) > 1 else ""

            # Get first 500 chars of content
            preview = content[:500] + "..." if len(content) > 500 else content

            results.append({
                'title': title,
                'preview': preview,
                'relevance': section.lower().count(query_lower)
            })

            if len(results) >= limit:
                break

    # Sort by relevance
    results.sort(key=lambda x: x['relevance'], reverse=True)
    return results


def get_project_research_files(project_name: str) -> Dict[str, str]:
    """Get research files for a project"""
    project_dir = RESEARCH_DIR / project_name
    if not project_dir.exists():
        return {}

    files = {}
    for file_path in project_dir.glob("*.md"):
        files[file_path.name] = load_text(file_path)

    return files


def log_finding_to_session(finding: str, finding_type: str = "general") -> bool:
    """Log a finding to the active session"""
    tracker = load_json(SESSION_TRACKER)
    if not tracker or 'active_session' not in tracker:
        return False

    session_id = tracker['active_session']
    if not session_id or session_id not in tracker.get('sessions', {}):
        return False

    # Add finding
    session = tracker['sessions'][session_id]
    if 'findings_captured' not in session:
        session['findings_captured'] = []

    session['findings_captured'].append({
        'text': finding,
        'type': finding_type,
        'timestamp': datetime.now().isoformat()
    })

    # Save
    try:
        with open(SESSION_TRACKER, 'w') as f:
            json.dump(tracker, f, indent=2)
        return True
    except Exception as e:
        app.server.request_context.session.send_log_message(
            level=LoggingLevel.ERROR,
            data=f"Error saving finding: {e}"
        )
        return False


# ============================================================================
# MCP Tool Handlers
# ============================================================================

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available ResearchGravity tools"""
    return [
        Tool(
            name="get_session_context",
            description="Get active research session context including topic, URLs, findings, and status",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Optional: Specific session ID. If not provided, returns active session."
                    }
                }
            }
        ),
        Tool(
            name="search_learnings",
            description="Search archived learnings from past research sessions. Returns relevant concepts, findings, and papers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or phrases)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_project_research",
            description="Load research files for a specific project (OS-App, CareerCoach, Metaventions, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Project name (e.g., 'os-app', 'careercoach', 'metaventions')"
                    }
                },
                "required": ["project_name"]
            }
        ),
        Tool(
            name="log_finding",
            description="Record a finding or insight to the active research session",
            inputSchema={
                "type": "object",
                "properties": {
                    "finding": {
                        "type": "string",
                        "description": "The finding text to record"
                    },
                    "type": {
                        "type": "string",
                        "description": "Finding type: general, implementation, metrics, innovation, etc.",
                        "default": "general"
                    }
                },
                "required": ["finding"]
            }
        ),
        Tool(
            name="select_context_packs",
            description="Select relevant context packs using Context Packs V2 (7-layer system with semantic embeddings)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query for context pack selection"
                    },
                    "budget": {
                        "type": "number",
                        "description": "Token budget (default: 50000)",
                        "default": 50000
                    },
                    "use_v1": {
                        "type": "boolean",
                        "description": "Force V1 engine (2 layers) instead of V2 (7 layers)",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_research_index",
            description="Get the unified research index with cross-project paper references and concepts",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_projects",
            description="List all tracked projects with their metadata, tech stack, and status",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_session_stats",
            description="Get statistics about research sessions, papers, concepts, and cognitive wallet value",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls"""

    if name == "get_session_context":
        session_id = arguments.get("session_id")

        if session_id:
            session = get_session_by_id(session_id)
            if not session:
                return [TextContent(
                    type="text",
                    text=f"Session not found: {session_id}"
                )]
        else:
            session = get_active_session()
            if not session:
                return [TextContent(
                    type="text",
                    text="No active session. Start a session with: python3 init_session.py \"topic\""
                )]

        # Format session info
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

        # Add findings if present
        findings = session.get('findings_captured', [])
        if findings:
            output += "\n### Recent Findings:\n"
            for f in findings[-5:]:  # Last 5
                output += f"\n- **{f.get('type', 'general')}**: {f.get('text', '')[:200]}...\n"

        return [TextContent(type="text", text=output)]

    elif name == "search_learnings":
        query = arguments["query"]
        limit = arguments.get("limit", 10)

        results = search_learnings_text(query, limit)

        if not results:
            return [TextContent(
                type="text",
                text=f"No learnings found for query: {query}"
            )]

        output = f"# Search Results for: {query}\n\nFound {len(results)} relevant sections:\n\n"

        for i, result in enumerate(results, 1):
            output += f"## {i}. {result['title']}\n\n"
            output += f"{result['preview']}\n\n"
            output += f"**Relevance Score:** {result['relevance']}\n\n---\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "get_project_research":
        project_name = arguments["project_name"].lower()

        files = get_project_research_files(project_name)

        if not files:
            return [TextContent(
                type="text",
                text=f"No research files found for project: {project_name}"
            )]

        output = f"# Research Files for: {project_name}\n\n"
        output += f"Found {len(files)} files:\n\n"

        for filename, content in files.items():
            output += f"## {filename}\n\n"
            # Include first 1000 chars of each file
            preview = content[:1000] + "..." if len(content) > 1000 else content
            output += f"{preview}\n\n---\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "log_finding":
        finding = arguments["finding"]
        finding_type = arguments.get("type", "general")

        success = log_finding_to_session(finding, finding_type)

        if success:
            return [TextContent(
                type="text",
                text=f"✅ Finding logged successfully\n\nType: {finding_type}\nText: {finding}"
            )]
        else:
            return [TextContent(
                type="text",
                text="❌ Failed to log finding. No active session or permission error."
            )]

    elif name == "select_context_packs":
        query = arguments["query"]
        budget = arguments.get("budget", 50000)
        use_v1 = arguments.get("use_v1", False)

        # Import and use Context Packs selector
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from select_packs_v2_integrated import PackSelectorV2Integrated, format_output

            selector = PackSelectorV2Integrated(force_v1=use_v1)
            packs, metadata = selector.select_packs(
                context=query,
                token_budget=budget,
                enable_pruning=True
            )

            output = format_output(packs, metadata, output_format='text')

            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Error selecting context packs: {e}"
            )]

    elif name == "get_research_index":
        index_file = RESEARCH_DIR / "INDEX.md"
        content = load_text(index_file)

        if not content:
            return [TextContent(
                type="text",
                text="Research index not found or empty"
            )]

        return [TextContent(type="text", text=content)]

    elif name == "list_projects":
        projects = load_json(PROJECTS_FILE)

        if not projects:
            return [TextContent(
                type="text",
                text="No projects found"
            )]

        output = "# Tracked Projects\n\n"

        for name, data in projects.items():
            output += f"## {name}\n\n"
            output += f"**Status:** {data.get('status', 'unknown')}\n"
            output += f"**Tech Stack:** {data.get('tech_stack', 'unknown')}\n"
            output += f"**Focus:** {', '.join(data.get('focus_areas', []))}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "get_session_stats":
        tracker = load_json(SESSION_TRACKER)
        projects = load_json(PROJECTS_FILE)

        # Count stats
        total_sessions = len(tracker.get('sessions', {}))
        archived = sum(1 for s in tracker.get('sessions', {}).values() if s.get('status') == 'archived')

        # Count from projects.json if available
        total_concepts = projects.get('_stats', {}).get('concepts', 0)
        total_papers = projects.get('_stats', {}).get('papers', 0)
        total_urls = projects.get('_stats', {}).get('urls', 0)
        wallet_value = projects.get('_stats', {}).get('cognitive_wallet', 0)

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

        return [TextContent(type="text", text=output)]

    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


# ============================================================================
# MCP Resource Handlers
# ============================================================================

@app.list_resources()
async def list_resources() -> List[Any]:
    """List available resources"""
    resources = [
        {
            "uri": "session://active",
            "name": "Active Session",
            "description": "Currently active research session data",
            "mimeType": "application/json"
        },
        {
            "uri": "learnings://all",
            "name": "All Learnings",
            "description": "Archived learnings from all sessions",
            "mimeType": "text/markdown"
        },
        {
            "uri": "research://index",
            "name": "Research Index",
            "description": "Unified cross-project research index",
            "mimeType": "text/markdown"
        }
    ]

    # Add project resources
    projects = load_json(PROJECTS_FILE)
    for project_name in projects.keys():
        if not project_name.startswith('_'):
            resources.append({
                "uri": f"project://{project_name}/research",
                "name": f"{project_name} Research",
                "description": f"Research files for {project_name} project",
                "mimeType": "text/markdown"
            })

    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI"""

    if uri == "session://active":
        session = get_active_session()
        if not session:
            return json.dumps({"error": "No active session"})
        return json.dumps(session, indent=2)

    elif uri == "learnings://all":
        return load_text(LEARNINGS_FILE)

    elif uri == "research://index":
        return load_text(RESEARCH_DIR / "INDEX.md")

    elif uri.startswith("project://"):
        # Parse: project://{name}/research
        parts = uri.replace("project://", "").split("/")
        if len(parts) >= 1:
            project_name = parts[0]
            files = get_project_research_files(project_name)

            # Combine all files
            output = f"# Research Files for {project_name}\n\n"
            for filename, content in files.items():
                output += f"## {filename}\n\n{content}\n\n---\n\n"

            return output

    return f"Resource not found: {uri}"


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
