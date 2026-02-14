#!/usr/bin/env python3
"""
Entry point: python -m mcp_raw

Launches the Raw MCP Server with all registered tools.
Tools are loaded dynamically from the tools/ package.
"""

import asyncio
import importlib
import sys
from pathlib import Path

from .logger import get_logger
from .server import RawMCPServer

log = get_logger("main")

# Tool modules to load (Session B builds these)
TOOL_MODULES = [
    "mcp_raw.tools.research_tools",
    "mcp_raw.tools.ucw_tools",
    "mcp_raw.tools.coherence_tools",
    "mcp_raw.tools.intelligence_tools",
    "mcp_raw.tools.delegation_tools",
]


def load_tools(server: RawMCPServer):
    """Dynamically load and register tool modules."""
    for module_path in TOOL_MODULES:
        try:
            mod = importlib.import_module(module_path)
            tools = getattr(mod, "TOOLS", [])
            handler = getattr(mod, "handle_tool", None)
            if tools and handler:
                server.register_tools(tools, handler)
                log.info(f"Loaded {len(tools)} tools from {module_path}")
            else:
                log.warning(f"Module {module_path} missing TOOLS or handle_tool")
        except ImportError as exc:
            log.warning(f"Tool module not available: {module_path} ({exc})")
        except Exception as exc:
            log.error(f"Error loading {module_path}: {exc}")


async def main():
    server = RawMCPServer()
    load_tools(server)
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
