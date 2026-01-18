#!/usr/bin/env python3
"""
Manual test for ResearchGravity MCP Server

Tests server startup and basic functionality without needing MCP client.
"""

import json
import sys
from pathlib import Path

# Add researchgravity to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("ResearchGravity MCP Server - Manual Test")
print("=" * 60)
print()

# Test imports
print("1. Testing imports...")
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    print("   ✅ MCP server imports successful")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test server instantiation
print()
print("2. Testing server instantiation...")
try:
    app = Server("researchgravity-test")
    print("   ✅ Server created successfully")
except Exception as e:
    print(f"   ❌ Server creation error: {e}")
    sys.exit(1)

# Test tool definitions
print()
print("3. Testing tool definitions...")
try:
    # Import our server functions
    import mcp_server

    print("   ✅ Server module loaded")
    print(f"   📁 AGENT_CORE: {mcp_server.AGENT_CORE}")
    print(f"   📁 SESSION_TRACKER: {mcp_server.SESSION_TRACKER}")
    print(f"   📁 LEARNINGS_FILE: {mcp_server.LEARNINGS_FILE}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test helper functions
print()
print("4. Testing helper functions...")
try:
    session = mcp_server.get_active_session()
    if session:
        print(f"   ✅ Active session found: {session.get('topic', 'Unknown')[:50]}...")
    else:
        print("   ⚠️  No active session (this is OK)")

    # Test search
    results = mcp_server.search_learnings_text("multi-agent", limit=3)
    print(f"   ✅ Search learnings works: found {len(results)} results")

    # Test projects
    projects = mcp_server.load_json(mcp_server.PROJECTS_FILE)
    print(f"   ✅ Projects loaded: {len([k for k in projects.keys() if not k.startswith('_')])} projects")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("✅ All tests passed!")
print("=" * 60)
print()
print("Server is ready for use with MCP clients.")
print()
print("Next steps:")
print("1. Configure Claude Desktop:")
print("   Copy claude_desktop_config.json to:")
print("   ~/Library/Application Support/Claude/claude_desktop_config.json")
print()
print("2. Restart Claude Desktop")
print()
print("3. Use tools in Claude Desktop:")
print("   - Get my active session context")
print("   - Search learnings for \"multi-agent\"")
print("   - Select context packs for \"debugging React\"")
print()
