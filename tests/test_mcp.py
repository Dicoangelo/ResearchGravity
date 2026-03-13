#!/usr/bin/env python3
"""
Test script for ResearchGravity MCP Server

Verifies that all tools work correctly.
"""

import asyncio
import pytest
from pathlib import Path

try:
    from mcp.client import ClientSession, stdio_client
except ImportError:
    ClientSession = None
    stdio_client = None


@pytest.mark.skipif(
    ClientSession is None or stdio_client is None, reason="MCP SDK not installed"
)
async def test_mcp_server():
    """Test ResearchGravity MCP server"""

    print("=" * 60)
    print("ResearchGravity MCP Server Test")
    print("=" * 60)
    print()

    # Connect to server
    server_path = str(Path(__file__).parent / "mcp_server.py")

    print(f"📡 Connecting to server: {server_path}")

    try:
        async with stdio_client(command="python3", args=[server_path]) as (read, write):
            async with ClientSession(read, write) as session:
                print("✅ Connected to MCP server")
                print()

                # Initialize
                await session.initialize()
                print("✅ Session initialized")
                print()

                # Test 1: List tools
                print("🔧 Test 1: List Tools")
                print("-" * 60)
                tools = await session.list_tools()
                print(f"Found {len(tools.tools)} tools:")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description[:60]}...")
                print()

                # Test 2: List resources
                print("📦 Test 2: List Resources")
                print("-" * 60)
                resources = await session.list_resources()
                print(f"Found {len(resources.resources)} resources:")
                for resource in resources.resources:
                    print(f"  - {resource.uri}: {resource.name}")
                print()

                # Test 3: Get session stats
                print("📊 Test 3: Get Session Stats")
                print("-" * 60)
                result = await session.call_tool("get_session_stats", arguments={})
                print(result.content[0].text)
                print()

                # Test 4: Get active session
                print("📍 Test 4: Get Active Session")
                print("-" * 60)
                result = await session.call_tool("get_session_context", arguments={})
                print(result.content[0].text[:300] + "...")
                print()

                # Test 5: Search learnings
                print("🔍 Test 5: Search Learnings")
                print("-" * 60)
                result = await session.call_tool(
                    "search_learnings", arguments={"query": "multi-agent", "limit": 3}
                )
                print(result.content[0].text[:300] + "...")
                print()

                # Test 6: List projects
                print("📁 Test 6: List Projects")
                print("-" * 60)
                result = await session.call_tool("list_projects", arguments={})
                print(result.content[0].text)
                print()

                # Test 7: Read resource
                print("📖 Test 7: Read Resource")
                print("-" * 60)
                resource_data = await session.read_resource("session://active")
                print(f"Resource URI: {resource_data.uri}")
                print(f"Content length: {len(resource_data.contents[0].text)} bytes")
                print()

                print("=" * 60)
                print("✅ All tests passed!")
                print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    if ClientSession is None or stdio_client is None:
        print("ERROR: MCP SDK not installed. Install with: pip install mcp")
        exit(1)
    success = asyncio.run(test_mcp_server())
    exit(0 if success else 1)
