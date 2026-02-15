"""
Subtask Executor — Dispatches Routed Subtasks to Real MCP Tools

The missing link between routing and verification. Takes subtasks that have
been assigned to agents (MCP tools) and actually executes them.

Execution flow:
1. Parse agent_id → module path + tool name
2. Build tool args from subtask description
3. Call the tool's handle_tool(name, args) async function
4. Extract result text from MCP response format
5. Feed result to verifier
6. Update trust ledger with outcome
7. Feed to evolution engine

Agent ID format: "mcp_raw.tools.research_tools::search_learnings"
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
                  module path                        tool name

For tools that need structured args (not just a query string), the executor
uses an LLM-free heuristic: map the subtask description to the tool's
inputSchema required fields.

Usage:
    from delegation.executor import SubtaskExecutor

    executor = SubtaskExecutor()
    result = await executor.execute(subtask_id, agent_id, description, chain_id)
"""

import asyncio
import importlib
import json
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from .models import VerificationMethod


# Timeout for individual subtask execution (seconds)
EXECUTION_TIMEOUT = 30.0

# Tools that accept a simple query/text arg
QUERY_TOOLS = {
    "search_learnings": "query",
    "get_session_context": "topic",
    "get_project_research": "project",
    "log_finding": "finding",
    "select_context_packs": "query",
    "get_research_index": None,  # no args
    "list_projects": None,
    "get_session_stats": None,
    "hybrid_search": "query",
    "knowledge_graph": "query",
    "insight_due": None,
    "insight_review": "insight_id",
    "coherence_arcs": None,
    "dashboard_snapshot": None,
    "coherence_status": None,
    "coherence_moments": None,
    "coherence_search": "query",
    "coherence_scan": None,
    "ucw_capture_stats": None,
    "ucw_timeline": None,
    "detect_emergence": None,
    "webhook_status": None,
    "webhook_list": None,
    "webhook_test": "provider",
}


class ExecutionResult:
    """Result of a subtask execution."""

    __slots__ = (
        "subtask_id", "agent_id", "success", "output",
        "error", "duration", "timestamp",
    )

    def __init__(
        self,
        subtask_id: str,
        agent_id: str,
        success: bool,
        output: str = "",
        error: str = "",
        duration: float = 0.0,
    ):
        self.subtask_id = subtask_id
        self.agent_id = agent_id
        self.success = success
        self.output = output
        self.error = error
        self.duration = duration
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "agent_id": self.agent_id,
            "success": self.success,
            "output": self.output[:500],  # Truncate for storage
            "error": self.error,
            "duration": self.duration,
            "timestamp": self.timestamp,
        }


class SubtaskExecutor:
    """
    Executes routed subtasks by dispatching to real MCP tool handlers.

    Supports two execution modes:
    1. Direct MCP tool call (for tools in the researchgravity server)
    2. Unified dispatcher (via mcp_raw.tools.handle_tool)
    """

    def __init__(self):
        self._module_cache: Dict[str, Any] = {}

    async def execute(
        self,
        subtask_id: str,
        agent_id: str,
        description: str,
        chain_id: str = "",
    ) -> ExecutionResult:
        """
        Execute a single subtask.

        Args:
            subtask_id: Unique subtask identifier
            agent_id: Agent ID in format "module.path::tool_name"
            description: Subtask description (used to build args)
            chain_id: Parent chain ID (for logging)

        Returns:
            ExecutionResult with success/failure, output, timing
        """
        start = time.time()

        try:
            module_path, tool_name = self._parse_agent_id(agent_id)
            args = self._build_args(tool_name, description)
            handler = self._get_handler(module_path)

            # Execute with timeout
            raw_result = await asyncio.wait_for(
                handler(tool_name, args),
                timeout=EXECUTION_TIMEOUT,
            )

            # Extract text from MCP response format
            output = self._extract_output(raw_result)
            is_error = self._is_error_result(raw_result)

            return ExecutionResult(
                subtask_id=subtask_id,
                agent_id=agent_id,
                success=not is_error,
                output=output,
                error="" if not is_error else output,
                duration=time.time() - start,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                subtask_id=subtask_id,
                agent_id=agent_id,
                success=False,
                error=f"Execution timed out after {EXECUTION_TIMEOUT}s",
                duration=time.time() - start,
            )
        except Exception as exc:
            return ExecutionResult(
                subtask_id=subtask_id,
                agent_id=agent_id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration=time.time() - start,
            )

    async def execute_batch(
        self,
        subtasks: List[Dict[str, str]],
        chain_id: str = "",
        parallel: bool = True,
    ) -> List[ExecutionResult]:
        """
        Execute multiple subtasks, optionally in parallel.

        Args:
            subtasks: List of dicts with keys: subtask_id, agent_id, description
            chain_id: Parent chain ID
            parallel: If True, run parallel-safe subtasks concurrently

        Returns:
            List of ExecutionResult objects
        """
        if parallel:
            tasks = [
                self.execute(
                    st["subtask_id"],
                    st["agent_id"],
                    st["description"],
                    chain_id,
                )
                for st in subtasks
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for st in subtasks:
                result = await self.execute(
                    st["subtask_id"],
                    st["agent_id"],
                    st["description"],
                    chain_id,
                )
                results.append(result)
            return results

    # ── Internal methods ──────────────────────────────────────────────────

    @staticmethod
    def _parse_agent_id(agent_id: str) -> Tuple[str, str]:
        """Parse 'module.path::tool_name' into (module_path, tool_name)."""
        if "::" in agent_id:
            parts = agent_id.split("::", 1)
            return parts[0], parts[1]
        # Fallback: try unified dispatcher
        return "mcp_raw.tools", agent_id

    def _get_handler(self, module_path: str):
        """Import and cache the handle_tool function from a module."""
        if module_path not in self._module_cache:
            mod = importlib.import_module(module_path)
            handler = getattr(mod, "handle_tool", None)
            if handler is None:
                raise ImportError(
                    f"Module {module_path} has no handle_tool function"
                )
            self._module_cache[module_path] = handler
        return self._module_cache[module_path]

    @staticmethod
    def _build_args(tool_name: str, description: str) -> Dict[str, Any]:
        """
        Build tool arguments from subtask description.

        Uses the QUERY_TOOLS mapping to determine the primary arg name.
        For unknown tools, defaults to {"query": description}.
        """
        if tool_name in QUERY_TOOLS:
            arg_name = QUERY_TOOLS[tool_name]
            if arg_name is None:
                return {}
            return {arg_name: description}

        # Default: pass description as query
        return {"query": description}

    @staticmethod
    def _extract_output(result: Any) -> str:
        """Extract text content from MCP tool_result_content format."""
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            # MCP format: {"content": [{"type": "text", "text": "..."}]}
            content = result.get("content", [])
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                if texts:
                    return "\n".join(texts)

            # Fallback: try "result" key
            if "result" in result:
                return str(result["result"])

            return json.dumps(result, default=str)[:1000]

        return str(result)[:1000]

    @staticmethod
    def _is_error_result(result: Any) -> bool:
        """Check if MCP result indicates an error."""
        if isinstance(result, dict):
            return result.get("isError", False)
        return False
