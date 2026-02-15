"""
Tests for delegation MCP tools.

Tests all 5 delegation tools exposed via MCP:
- delegate_research
- delegation_status
- get_agent_trust
- delegation_history
- delegation_insights

These are integration tests that verify tool definitions and basic dispatcher routing.
Full functional testing of delegation logic is covered in other test files (test_coordinator.py, etc.)
"""

import pytest
from mcp_raw.tools.delegation_tools import TOOLS, handle_tool

pytestmark = pytest.mark.anyio


# ═══════════════════════════════════════════════════════════════════════════
# Test Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════


class TestToolDefinitions:
    """Test tool definitions are correctly structured."""

    def test_tools_list_structure(self):
        """All 8 tools defined with required fields."""
        assert len(TOOLS) == 8
        tool_names = {t["name"] for t in TOOLS}
        assert tool_names == {
            "delegate_research",
            "delegation_status",
            "get_agent_trust",
            "delegation_history",
            "delegation_insights",
            "sync_x_trust",
            "auto_delegate_monitor",
            "publish_research_thread",
        }

    def test_tools_have_descriptions(self):
        """All tools have non-empty descriptions."""
        for tool in TOOLS:
            assert "description" in tool
            assert len(tool["description"]) > 50  # Meaningful description

    def test_tools_have_input_schemas(self):
        """All tools have valid input schemas."""
        for tool in TOOLS:
            assert "inputSchema" in tool
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_delegate_research_schema(self):
        """delegate_research requires task parameter."""
        tool = next(t for t in TOOLS if t["name"] == "delegate_research")
        schema = tool["inputSchema"]
        assert "task" in schema["properties"]
        assert schema["required"] == ["task"]
        assert "context" in schema["properties"]  # Optional

    def test_delegation_status_schema(self):
        """delegation_status requires chain_id."""
        tool = next(t for t in TOOLS if t["name"] == "delegation_status")
        schema = tool["inputSchema"]
        assert "chain_id" in schema["properties"]
        assert schema["required"] == ["chain_id"]

    def test_get_agent_trust_schema(self):
        """get_agent_trust has optional parameters."""
        tool = next(t for t in TOOLS if t["name"] == "get_agent_trust")
        schema = tool["inputSchema"]
        assert "agent_id" in schema["properties"]
        assert "task_type" in schema["properties"]
        assert "limit" in schema["properties"]
        # All params are optional
        assert schema.get("required", []) == []

    def test_delegation_history_schema(self):
        """delegation_history has optional parameters."""
        tool = next(t for t in TOOLS if t["name"] == "delegation_history")
        schema = tool["inputSchema"]
        assert "limit" in schema["properties"]
        assert "task_type" in schema["properties"]

    def test_delegation_insights_schema(self):
        """delegation_insights has days parameter."""
        tool = next(t for t in TOOLS if t["name"] == "delegation_insights")
        schema = tool["inputSchema"]
        assert "days" in schema["properties"]
        assert schema["properties"]["days"]["default"] == 7


# ═══════════════════════════════════════════════════════════════════════════
# Test Tool Dispatcher
# ═══════════════════════════════════════════════════════════════════════════


class TestToolDispatcher:
    """Test handle_tool dispatcher routing."""

    async def test_unknown_tool_error(self):
        """Unknown tool name returns error."""
        result = await handle_tool("unknown_tool", {})
        # Should contain error indication
        assert "is_error" in result or "Unknown delegation tool" in str(result)

    async def test_delegation_history_placeholder(self):
        """delegation_history returns placeholder (simplest test that doesn't require complex mocking)."""
        result = await handle_tool("delegation_history", {"limit": 5})

        # Should return successful result
        assert "content" in result
        assert len(result["content"]) > 0

        # Content should mention history
        content_text = result["content"][0].get("text", "")
        assert "history" in content_text.lower() or "delegation" in content_text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test Tool Registration and Exports
# ═══════════════════════════════════════════════════════════════════════════


class TestToolRegistration:
    """Test that tools are properly exported and can be registered."""

    def test_tools_constant_is_list(self):
        """TOOLS export is a list."""
        assert isinstance(TOOLS, list)
        assert len(TOOLS) > 0

    def test_handle_tool_is_async_callable(self):
        """handle_tool is an async function."""
        import inspect
        assert inspect.iscoroutinefunction(handle_tool)

    def test_all_tool_names_unique(self):
        """All tool names are unique."""
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names))

    def test_all_schemas_valid_json_schema(self):
        """All input schemas follow JSON Schema spec."""
        for tool in TOOLS:
            schema = tool["inputSchema"]
            # Required JSON Schema fields
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema

            # Properties should be a dict
            assert isinstance(schema["properties"], dict)

            # If required field exists, should be a list
            if "required" in schema:
                assert isinstance(schema["required"], list)
