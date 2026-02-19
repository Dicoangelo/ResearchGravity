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
import asyncio as _asyncio

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
        ),
        # ── Visual Intelligence Layer ──
        # PaperBanana 5-agent pipeline (academic diagrams)
        Tool(
            name="visualize_research",
            description="Generate a publication-quality diagram from methodology text using PaperBanana 5-agent pipeline with switchable profiles (max/balanced/fast/budget)",
            inputSchema={
                "type": "object",
                "properties": {
                    "methodology": {
                        "type": "string",
                        "description": "Research methodology or system architecture text to visualize"
                    },
                    "caption": {
                        "type": "string",
                        "description": "Diagram caption (concise, descriptive)"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional: Session ID to associate diagram with"
                    },
                    "profile": {
                        "type": "string",
                        "description": "Quality profile: max, balanced, fast, budget (default: from config)",
                        "enum": ["max", "balanced", "fast", "budget"]
                    }
                },
                "required": ["methodology", "caption"]
            }
        ),
        Tool(
            name="diagram_from_session",
            description="Auto-generate diagrams from a session's findings (extracts methodology-relevant content)",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to generate diagrams for"
                    },
                    "finding_filter": {
                        "type": "string",
                        "description": "Optional: Filter findings by type (innovation, implementation, thesis)"
                    },
                    "profile": {
                        "type": "string",
                        "description": "Quality profile: max, balanced, fast, budget",
                        "enum": ["max", "balanced", "fast", "budget"]
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="illustrate_finding",
            description="Generate a diagram for a specific finding text",
            inputSchema={
                "type": "object",
                "properties": {
                    "finding_text": {
                        "type": "string",
                        "description": "The finding text to illustrate"
                    },
                    "diagram_type": {
                        "type": "string",
                        "description": "Diagram type: methodology or statistical_plot",
                        "enum": ["methodology", "statistical_plot"]
                    },
                    "caption": {
                        "type": "string",
                        "description": "Diagram caption"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional: Session ID"
                    },
                    "profile": {
                        "type": "string",
                        "description": "Quality profile: max, balanced, fast, budget",
                        "enum": ["max", "balanced", "fast", "budget"]
                    }
                },
                "required": ["finding_text", "diagram_type", "caption"]
            }
        ),
        Tool(
            name="evaluate_paper_figures",
            description="Evaluate diagram quality using PaperBanana 4-dimension scoring (faithfulness, readability, conciseness, aesthetics)",
            inputSchema={
                "type": "object",
                "properties": {
                    "generated_path": {
                        "type": "string",
                        "description": "Path to generated diagram"
                    },
                    "reference_path": {
                        "type": "string",
                        "description": "Path to human reference diagram"
                    },
                    "context": {
                        "type": "string",
                        "description": "Methodology text for evaluation context"
                    },
                    "caption": {
                        "type": "string",
                        "description": "Diagram caption"
                    }
                },
                "required": ["generated_path", "reference_path", "context", "caption"]
            }
        ),
        # ── Gemini Native Image Generation (direct google-genai SDK) ──
        Tool(
            name="generate_image",
            description="Generate an image using Google Gemini native image generation (gemini-3-pro-image-preview). Supports 1K-4K resolution, aspect ratios, multi-image editing, and Google Search grounding. The same engine used for logos, brand assets, and scene generation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Image generation prompt — describe what you want to create"
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Output resolution: 1K, 2K, or 4K (default: from quality tier or config)",
                        "enum": ["1K", "2K", "4K"]
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Aspect ratio: 1:1, 3:4, 4:3, 5:4, 9:16, 16:9",
                        "enum": ["1:1", "3:4", "4:3", "5:4", "9:16", "16:9"]
                    },
                    "quality": {
                        "type": "string",
                        "description": "Quality tier: max (4K + quality suffix), high (2K), fast (1K)",
                        "enum": ["max", "high", "fast"]
                    },
                    "input_images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to input images for editing/composition (up to 14)"
                    },
                    "use_search_grounding": {
                        "type": "boolean",
                        "description": "Enable Google Search grounding for factual image content",
                        "default": False
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional: Session ID for UCW capture"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="edit_image",
            description="Edit or compose images using Gemini native — combine, transform, or modify up to 14 input images with a text prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Edit instruction — what to do with the input images"
                    },
                    "input_images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to images to edit/compose (1-14 images, required)"
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Output resolution: 1K, 2K, or 4K",
                        "enum": ["1K", "2K", "4K"]
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Output aspect ratio",
                        "enum": ["1:1", "3:4", "4:3", "5:4", "9:16", "16:9"]
                    },
                    "quality": {
                        "type": "string",
                        "description": "Quality tier: max, high, fast",
                        "enum": ["max", "high", "fast"]
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional: Session ID"
                    }
                },
                "required": ["prompt", "input_images"]
            }
        ),
        # ── Refined Pipeline (PaperBanana-style 5-agent) ──
        Tool(
            name="generate_refined",
            description="Generate a high-quality technical diagram using the PaperBanana-style refined pipeline: Planner → Stylist → [Visualizer → Critic] × T rounds. The critic produces a refined TEXTUAL DESCRIPTION and each round regenerates from scratch — eliminating layout drift from edit-based approaches. Best for architecture diagrams, technical illustrations, and complex visualizations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_context": {
                        "type": "string",
                        "description": "The source material to visualize — architecture description, methodology, system specification, etc."
                    },
                    "caption": {
                        "type": "string",
                        "description": "What the diagram should show (communicative intent)"
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Output resolution: 1K, 2K, or 4K",
                        "enum": ["1K", "2K", "4K"]
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Aspect ratio: 1:1, 3:4, 4:3, 5:4, 9:16, 16:9",
                        "enum": ["1:1", "3:4", "4:3", "5:4", "9:16", "16:9"]
                    },
                    "quality": {
                        "type": "string",
                        "description": "Quality tier: max (4K, 5 iterations), high (2K, 3 iterations), fast (1K, 2 iterations)",
                        "enum": ["max", "high", "fast"]
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Number of Critique→Refine→Regenerate rounds (default: from config/quality tier, recommended: 3)",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional: Session ID for UCW capture"
                    },
                    "skip_planning": {
                        "type": "boolean",
                        "description": "Skip Planner+Stylist stages (use source_context directly as image prompt)",
                        "default": False
                    }
                },
                "required": ["source_context", "caption"]
            }
        ),
        # ── Visual Config & Profiles ──
        Tool(
            name="list_visual_profiles",
            description="List all available visual generation profiles with models, resolutions, costs, and estimated times. Also lists all available VLM and image generation models.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
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

    # ── Visual Intelligence Layer Tools ──

    elif name == "visualize_research":
        try:
            from visual import PaperBananaAdapter, get_visual_config
            config = get_visual_config()
            profile = arguments.get("profile")
            if profile:
                config.apply_profile(profile)
            adapter = PaperBananaAdapter(config=config)
            result = await adapter.generate_diagram(
                methodology=arguments["methodology"],
                caption=arguments["caption"],
                session_id=arguments.get("session_id"),
            )
            if "error" in result:
                return [TextContent(type="text", text=f"❌ {result['error']}")]

            output = f"""✅ Diagram generated

**Asset ID:** {result['asset_id']}
**Path:** {result['png_path']}
**Type:** {result['diagram_type']}
**Profile:** {config.profile}
**Cost:** ${result['metadata']['estimated_cost_usd']:.4f}
**Iterations:** {result['metadata']['iterations']}
**Model:** {result['metadata']['vlm_model']} + {result['metadata']['image_model']}
**Resolution:** {result['metadata']['resolution']}
**Time:** {result['metadata']['elapsed_seconds']}s"""
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Visual generation failed: {e}")]

    elif name == "diagram_from_session":
        try:
            session_id = arguments["session_id"]
            finding_filter = arguments.get("finding_filter")

            session = get_session_by_id(session_id)
            if not session:
                return [TextContent(type="text", text=f"Session not found: {session_id}")]

            findings = session.get("findings_captured", [])
            if finding_filter:
                findings = [f for f in findings if f.get("type") == finding_filter]

            if not findings:
                return [TextContent(type="text", text=f"No findings in session {session_id}")]

            from visual import PaperBananaAdapter, get_visual_config
            config = get_visual_config()
            profile = arguments.get("profile")
            if profile:
                config.apply_profile(profile)
            adapter = PaperBananaAdapter(config=config)

            session_dir = SESSIONS_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            results = await adapter.generate_session_diagrams(
                session_id=session_id,
                session_dir=session_dir,
                findings=findings,
            )

            if not results:
                return [TextContent(type="text", text="No diagrams generated (findings may be too short)")]

            output = f"✅ Generated {len(results)} diagram(s) for session {session_id}\n\n"
            for r in results:
                output += f"- **{r['asset_id']}**: {r['caption'][:60]}... → {r['png_path']}\n"
            output += f"\n**Profile:** {config.profile}\n**Total cost:** ${adapter.get_session_cost():.4f}"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Session diagram generation failed: {e}")]

    elif name == "illustrate_finding":
        try:
            from visual import PaperBananaAdapter, get_visual_config
            config = get_visual_config()
            profile = arguments.get("profile")
            if profile:
                config.apply_profile(profile)
            adapter = PaperBananaAdapter(config=config)

            if arguments["diagram_type"] == "statistical_plot":
                result = await adapter.generate_plot(
                    data_json=arguments["finding_text"],
                    intent=arguments["caption"],
                    session_id=arguments.get("session_id"),
                )
            else:
                result = await adapter.generate_diagram(
                    methodology=arguments["finding_text"],
                    caption=arguments["caption"],
                    session_id=arguments.get("session_id"),
                )

            if "error" in result:
                return [TextContent(type="text", text=f"❌ {result['error']}")]

            output = f"✅ Diagram generated: {result['png_path']}\n**Profile:** {config.profile}\n**Cost:** ${result['metadata']['estimated_cost_usd']:.4f}"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Illustration failed: {e}")]

    elif name == "evaluate_paper_figures":
        try:
            from visual import PaperBananaAdapter
            adapter = PaperBananaAdapter()
            scores = await adapter.evaluate_diagram(
                generated_path=arguments["generated_path"],
                reference_path=arguments["reference_path"],
                context=arguments["context"],
                caption=arguments["caption"],
            )

            if "error" in scores:
                return [TextContent(type="text", text=f"❌ {scores['error']}")]

            output = f"""📊 Diagram Evaluation

| Dimension | Score |
|-----------|-------|
| Faithfulness | {scores.get('faithfulness', 0):.2f} |
| Readability | {scores.get('readability', 0):.2f} |
| Conciseness | {scores.get('conciseness', 0):.2f} |
| Aesthetics | {scores.get('aesthetics', 0):.2f} |
| **Overall** | **{scores.get('overall', 0):.2f}** |"""
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Evaluation failed: {e}")]

    # ── Gemini Native Image Generation Tools ──

    elif name == "generate_image":
        try:
            from visual import GeminiImageGenerator
            gen = GeminiImageGenerator()
            result = await gen.generate(
                prompt=arguments["prompt"],
                resolution=arguments.get("resolution"),
                aspect_ratio=arguments.get("aspect_ratio"),
                quality=arguments.get("quality"),
                input_images=arguments.get("input_images"),
                use_search_grounding=arguments.get("use_search_grounding", False),
                session_id=arguments.get("session_id"),
            )
            if "error" in result:
                return [TextContent(type="text", text=f"❌ {result['error']}")]

            meta = result["metadata"]
            output = f"""✅ Image generated (Gemini Native)

**Asset ID:** {result['asset_id']}
**Path:** {result['png_path']}
**Engine:** gemini_native
**Model:** {meta['model']}
**Resolution:** {meta['resolution']}
**Aspect Ratio:** {meta.get('aspect_ratio') or 'auto'}
**Quality:** {meta['quality']}
**Cost:** ${meta['estimated_cost_usd']:.4f}
**Time:** {meta['elapsed_seconds']}s
**File Size:** {meta['file_size_bytes'] // 1024}KB"""
            if meta.get("input_images", 0) > 0:
                output += f"\n**Input Images:** {meta['input_images']}"
            if result.get("model_text"):
                output += f"\n**Model Response:** {result['model_text'][:200]}"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Native image generation failed: {e}")]

    elif name == "edit_image":
        try:
            from visual import GeminiImageGenerator
            gen = GeminiImageGenerator()
            result = await gen.edit_image(
                prompt=arguments["prompt"],
                input_images=arguments["input_images"],
                resolution=arguments.get("resolution"),
                aspect_ratio=arguments.get("aspect_ratio"),
                quality=arguments.get("quality"),
                session_id=arguments.get("session_id"),
            )
            if "error" in result:
                return [TextContent(type="text", text=f"❌ {result['error']}")]

            meta = result["metadata"]
            output = f"""✅ Image edited (Gemini Native)

**Asset ID:** {result['asset_id']}
**Path:** {result['png_path']}
**Input Images:** {meta['input_images']}
**Resolution:** {meta['resolution']}
**Cost:** ${meta['estimated_cost_usd']:.4f}
**Time:** {meta['elapsed_seconds']}s"""
            if result.get("model_text"):
                output += f"\n**Model Response:** {result['model_text'][:200]}"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Image editing failed: {e}")]

    elif name == "generate_refined":
        try:
            from visual import RefinedPipeline
            pipeline = RefinedPipeline()
            result = await pipeline.generate(
                source_context=arguments["source_context"],
                caption=arguments["caption"],
                resolution=arguments.get("resolution"),
                aspect_ratio=arguments.get("aspect_ratio"),
                quality=arguments.get("quality"),
                iterations=arguments.get("iterations"),
                session_id=arguments.get("session_id"),
                skip_planning=arguments.get("skip_planning", False),
            )
            if "error" in result:
                return [TextContent(type="text", text=f"❌ {result['error']}")]

            meta = result["metadata"]
            output = f"""✅ Refined Pipeline Complete

**Asset ID:** {result['asset_id']}
**Path:** {result['png_path']}
**Engine:** refined_pipeline (Plan → Style → [Generate → Critique] × T)
**VLM Model:** {meta['vlm_model']}
**Image Model:** {meta['model']}
**Resolution:** {meta['resolution']}
**Iterations:** {meta['iterations_run']}/{meta['iterations_planned']}
**Total Cost:** ${meta['estimated_cost_usd']:.4f}
**Total Time:** {meta['elapsed_seconds']}s
**File Size:** {meta['file_size_bytes'] // 1024}KB

## Pipeline Trace"""
            for stage in result.get("pipeline_trace", []):
                s_name = stage.get("stage", "?")
                s_status = stage.get("status", "")
                s_time = stage.get("elapsed_s", "")
                if s_status:
                    output += f"\n- **{s_name}**: {s_status}"
                elif s_time:
                    output += f"\n- **{s_name}**: {s_time}s"

            output += "\n\n## Iteration Details"
            for ir in result.get("iteration_results", []):
                output += f"\n\n### Round {ir['iteration']}"
                output += f"\n- Generation: {ir.get('gen_elapsed_s', '?')}s"
                if ir.get("critic_elapsed_s"):
                    output += f"\n- Critique: {ir['critic_elapsed_s']}s"
                if ir.get("text_errors"):
                    output += f"\n- Text errors found: {', '.join(ir['text_errors'][:5])}"
                if ir.get("hallucinated_content"):
                    output += f"\n- Hallucinations flagged: {', '.join(ir['hallucinated_content'][:5])}"
                if ir.get("duplicated_elements"):
                    output += f"\n- Duplicates: {', '.join(ir['duplicated_elements'][:5])}"
                if ir.get("description_refined"):
                    output += "\n- Description refined for next round"
                if ir.get("early_stop"):
                    output += "\n- **Early stop**: Critic found no issues"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Refined pipeline failed: {e}")]

    elif name == "list_visual_profiles":
        try:
            from visual import list_profiles, list_models, get_visual_config

            config = get_visual_config()
            profiles = list_profiles()
            models = list_models()

            output = f"# Visual Intelligence Layer\n\n"
            output += f"**Active Profile:** {config.profile}\n"
            output += f"**Estimated Cost/Diagram:** ${config.estimate_diagram_cost():.4f}\n\n"

            output += "## Profiles\n\n"
            output += "| Profile | VLM | Image Gen | Resolution | Iterations | Est. Cost | Est. Time |\n"
            output += "|---------|-----|-----------|------------|------------|-----------|----------|\n"
            for name_p, p in profiles.items():
                marker = " **[active]**" if name_p == config.profile else ""
                output += (
                    f"| {name_p}{marker} | {p['vlm_model']} | {p['image_model']} | "
                    f"{p['image_resolution']} | {p['max_iterations']} | "
                    f"${p.get('est_cost_per_diagram', 0):.2f} | {p.get('est_time_seconds', 0)}s |\n"
                )

            output += "\n## VLM Models\n\n"
            output += "| Model | Description | Input $/1M | Output $/1M | Context |\n"
            output += "|-------|-------------|-----------|------------|--------|\n"
            for m_name, m_info in models["vlm_models"].items():
                output += (
                    f"| {m_name} | {m_info['description']} | "
                    f"${m_info['input_cost_per_1m']:.2f} | ${m_info['output_cost_per_1m']:.2f} | "
                    f"{m_info['context_window']:,} |\n"
                )

            output += "\n## Image Generation Models\n\n"
            output += "| Model | Description | 1K | 2K | 4K |\n"
            output += "|-------|-------------|----|----|----|\n"
            for m_name, m_info in models["image_models"].items():
                c1k = f"${m_info['cost_1k']:.3f}" if m_info.get("cost_1k") else "N/A"
                c2k = f"${m_info['cost_2k']:.3f}" if m_info.get("cost_2k") else "N/A"
                c4k = f"${m_info['cost_4k']:.3f}" if m_info.get("cost_4k") else "N/A"
                output += f"| {m_name} | {m_info['description']} | {c1k} | {c2k} | {c4k} |\n"

            output += "\n## Engines\n\n"
            output += "| Engine | Best For | Method | Tool |\n"
            output += "|--------|----------|--------|------|\n"
            output += "| **Refined Pipeline** | Architecture diagrams, technical illustrations | Plan→Style→[Gen→Critique]×T | `generate_refined` |\n"
            output += "| **Gemini Native** | Brand assets, logos, scenes, image editing | Single-shot generation | `generate_image` |\n"
            output += "| **PaperBanana** | Academic diagrams (when package installed) | 5-agent pipeline | `visualize_research` |\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Failed to list profiles: {e}")]

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
