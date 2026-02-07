"""
UCW Raw MCP Tools — ported from SDK server + new capture tools

Modules:
  research_tools   — 8 tools ported from mcp_server.py SDK
  ucw_tools        — 3 UCW-specific capture/emergence tools
  coherence_tools  — 3 cross-platform coherence query tools
"""

from . import research_tools
from . import ucw_tools
from . import coherence_tools

# Aggregate all tools for router discovery
ALL_TOOLS = (
    research_tools.TOOLS +
    ucw_tools.TOOLS +
    coherence_tools.TOOLS
)

# Dispatch map: tool_name -> (module, handler)
_DISPATCH = {}
for tool_def in research_tools.TOOLS:
    _DISPATCH[tool_def["name"]] = research_tools.handle_tool
for tool_def in ucw_tools.TOOLS:
    _DISPATCH[tool_def["name"]] = ucw_tools.handle_tool
for tool_def in coherence_tools.TOOLS:
    _DISPATCH[tool_def["name"]] = coherence_tools.handle_tool


async def handle_tool(name: str, args: dict) -> dict:
    """Unified dispatcher across all tool modules."""
    handler = _DISPATCH.get(name)
    if handler:
        return await handler(name, args)
    from mcp_raw.protocol import tool_result_content, text_content
    return tool_result_content([text_content(f"Unknown tool: {name}")], is_error=True)
