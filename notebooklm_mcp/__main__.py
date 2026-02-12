#!/usr/bin/env python3
"""
Entry point: python -m notebooklm_mcp

Launches the NotebookLM MCP Server with HTTP/RPC API + Cognitive Intelligence.
Integrates with UCW cognitive capture infrastructure.
"""

import asyncio
import importlib
import sys
from pathlib import Path

# Import from parent mcp_raw module
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_raw.logger import get_logger
from .server import NotebookLMMCPServer
from .config_notebooklm import NotebookLMConfig

log = get_logger("notebooklm_main")

# Tool modules to load
TOOL_MODULES = [
    "notebooklm_mcp.tools.notebooklm_tools",
]


def load_tools(server: NotebookLMMCPServer):
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
    # Ensure directories exist
    NotebookLMConfig.ensure_dirs()

    # Create and run server
    server = NotebookLMMCPServer()
    load_tools(server)
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server shutdown requested")
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
