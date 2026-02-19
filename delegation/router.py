"""
Agent Router — Capability Matching with Trust-Weighted Scoring

Implements the routing strategy from arXiv:2602.11865 Section 4.2.

Routes subtasks to agents based on:
- Capability matching (keyword overlap + semantic similarity)
- Trust scores from TrustLedger (historical performance)
- Cost efficiency (estimated cost vs agent cost)
- Complexity floor (tasks < 0.2 complexity execute directly)
- Fallback chain (next-best agents on failure)

Key Features:
- Agent registry: Loads all MCP tool definitions from 3 servers
- Scoring formula: final_score = capability_match * 0.6 + trust_score * 0.3 + cost_efficiency * 0.1
- Batch routing: route_batch() for parallel task assignment
- Logging: All routing decisions logged as DelegationEvent

Usage:
    from delegation.router import route_subtask, load_agent_registry

    # Load agent registry once at startup
    registry = load_agent_registry()

    # Route single subtask
    assignment = route_subtask(
        subtask=subtask,
        available_agents=registry,
        trust_ledger=ledger
    )

    # Route multiple subtasks in batch
    assignments = route_batch(subtasks, registry, ledger)
"""

import asyncio
import importlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from .models import SubTask, Assignment, DelegationEvent, TaskProfile

# LLM client for semantic similarity (optional)
try:
    from cpb.llm_client import get_llm_client
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False

# Default complexity floor for direct execution (no delegation)
MIN_COMPLEXITY_FOR_DELEGATION = 0.2

# Agent scoring weights (must sum to 1.0)
CAPABILITY_WEIGHT = 0.6  # How well agent matches task requirements
TRUST_WEIGHT = 0.3       # Historical performance
COST_WEIGHT = 0.1        # Resource efficiency

# Evolution feedback — load learned agent affinity if available
try:
    from .evolution import EvolutionEngine
    _evolution_engine = EvolutionEngine()
    HAS_EVOLUTION = True
except Exception:
    _evolution_engine = None
    HAS_EVOLUTION = False


def _get_affinity_boost(agent_id: str) -> float:
    """Get learned affinity boost for an agent from evolution engine."""
    if not HAS_EVOLUTION or not _evolution_engine:
        return 0.0
    try:
        strategies = _evolution_engine.evolve_strategies()
        affinity = strategies.get("agent_affinity", {})
        agent_data = affinity.get(agent_id, {})
        # Boost high-performing agents by up to 0.1
        success_rate = agent_data.get("success_rate", 0.5)
        return max(0.0, (success_rate - 0.5) * 0.2)  # -0.1 to +0.1
    except Exception:
        return 0.0


@dataclass
class AgentCapability:
    """
    Agent capability profile extracted from MCP tool definitions.

    Fields:
    - agent_id: Unique identifier (e.g., "mcp_server::get_session_context")
    - name: Tool name
    - description: Tool description
    - keywords: Extracted keywords for capability matching
    - estimated_cost: Estimated resource cost (0.0-1.0)
    - metadata: Additional tool metadata (inputSchema, etc.)
    """
    agent_id: str
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    estimated_cost: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT REGISTRY LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_agent_registry() -> List[AgentCapability]:
    """
    Load all MCP tool definitions from the 3 servers into a unified registry.

    Sources:
    1. mcp_server.py (8 tools)
    2. mcp_raw/tools/*.py (research, ucw, coherence, intelligence, webhook)
    3. notebooklm_mcp/server.py (if available)

    Returns:
        List of AgentCapability objects, one per tool
    """
    registry = []

    # Load from mcp_server.py
    registry.extend(_load_from_mcp_server())

    # Load from mcp_raw/tools
    registry.extend(_load_from_mcp_raw_tools())

    # Load from notebooklm_mcp (if available)
    registry.extend(_load_from_notebooklm())

    return registry


def _load_from_mcp_server() -> List[AgentCapability]:
    """Load tools from mcp_server.py (8 tools)"""
    agents = []
    try:
        # mcp_server.py defines tools inline in @app.list_tools()
        # We'll parse the file directly since it's not importable
        server_path = Path(__file__).parent.parent / "mcp_server.py"
        if not server_path.exists():
            return agents

        with open(server_path, 'r') as f:
            content = f.read()

        # Extract Tool() definitions using regex
        tool_pattern = r'Tool\(\s*name="([^"]+)",\s*description="([^"]+)"'
        matches = re.findall(tool_pattern, content, re.DOTALL)

        for name, description in matches:
            agent_id = f"mcp_server::{name}"
            keywords = _extract_keywords(f"{name} {description}")
            agents.append(AgentCapability(
                agent_id=agent_id,
                name=name,
                description=description,
                keywords=keywords,
                estimated_cost=0.3,  # MCP server tools are lightweight
                metadata={"source": "mcp_server.py"}
            ))
    except Exception as e:
        # Graceful degradation
        pass

    return agents


def _load_from_mcp_raw_tools() -> List[AgentCapability]:
    """Load tools from mcp_raw/tools/*.py"""
    agents = []
    tool_modules = [
        "mcp_raw.tools.research_tools",
        "mcp_raw.tools.ucw_tools",
        "mcp_raw.tools.coherence_tools",
        "mcp_raw.tools.intelligence_tools",
        "mcp_raw.tools.webhook_tools",
    ]

    for module_path in tool_modules:
        try:
            mod = importlib.import_module(module_path)
            tools = getattr(mod, "TOOLS", [])

            for tool in tools:
                name = tool.get("name", "")
                description = tool.get("description", "")
                agent_id = f"{module_path}::{name}"
                keywords = _extract_keywords(f"{name} {description}")

                agents.append(AgentCapability(
                    agent_id=agent_id,
                    name=name,
                    description=description,
                    keywords=keywords,
                    estimated_cost=0.4,  # Raw MCP tools slightly heavier
                    metadata={
                        "source": module_path,
                        "inputSchema": tool.get("inputSchema", {})
                    }
                ))
        except (ImportError, AttributeError):
            # Module not available or missing TOOLS
            pass

    return agents


def _load_from_notebooklm() -> List[AgentCapability]:
    """Load tools from notebooklm_mcp/server.py (if available)"""
    # NotebookLM MCP server may not have tools defined yet
    # Placeholder for future expansion
    return []


def _extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text for capability matching.

    Uses simple heuristic:
    - Convert to lowercase
    - Remove stopwords
    - Extract words >= 4 chars
    - Return unique words
    """
    # Common stopwords
    stopwords = {
        "the", "and", "for", "from", "with", "this", "that",
        "are", "was", "will", "can", "has", "have", "been",
        "get", "set", "list", "find", "search", "load", "create"
    }

    # Tokenize and filter
    words = re.findall(r'\w+', text.lower())
    keywords = [
        w for w in words
        if len(w) >= 4 and w not in stopwords
    ]

    # Return unique keywords
    return list(set(keywords))


# ═══════════════════════════════════════════════════════════════════════════
# CAPABILITY MATCHING
# ═══════════════════════════════════════════════════════════════════════════


def _calculate_capability_match(
    subtask: SubTask,
    agent: AgentCapability,
    use_llm: bool = True
) -> float:
    """
    Calculate how well agent capabilities match subtask requirements.

    Uses two scoring methods:
    1. Keyword overlap (always available)
    2. Semantic similarity via LLM (optional, if available)

    Returns:
        Score in [0.0, 1.0] where 1.0 = perfect match
    """
    # Extract keywords from subtask description
    subtask_keywords = _extract_keywords(subtask.description)

    # Keyword overlap score
    if not subtask_keywords or not agent.keywords:
        keyword_score = 0.0
    else:
        overlap = set(subtask_keywords) & set(agent.keywords)
        keyword_score = len(overlap) / max(len(subtask_keywords), len(agent.keywords))

    # Semantic similarity (if LLM available)
    if use_llm and HAS_LLM_CLIENT:
        try:
            semantic_score = asyncio.run(_semantic_similarity(
                subtask.description,
                agent.description
            ))
            # Blend keyword and semantic scores (60/40 split)
            final_score = keyword_score * 0.4 + semantic_score * 0.6
        except Exception:
            # Fall back to keyword-only on error
            final_score = keyword_score
    else:
        final_score = keyword_score

    return max(0.0, min(1.0, final_score))


async def _semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using LLM.

    Returns:
        Similarity score in [0.0, 1.0]
    """
    if not HAS_LLM_CLIENT:
        return 0.0

    try:
        client = get_llm_client()

        system_prompt = (
            "You are a semantic similarity scorer. "
            "Rate how similar two text descriptions are on a scale of 0.0 to 1.0. "
            "Output ONLY a JSON object: {\"similarity\": <score>}"
        )

        user_prompt = f"""
Text A: {text1}

Text B: {text2}

Rate semantic similarity (0.0 = completely different, 1.0 = identical meaning):
"""

        response = await asyncio.wait_for(
            client.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model="haiku",  # Fast/cheap model for similarity
                temperature=0.0,
            ),
            timeout=2.0  # Quick timeout for batch routing
        )

        # Parse JSON response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            data = json.loads(json_match.group(0))
            similarity = float(data.get("similarity", 0.0))
            return max(0.0, min(1.0, similarity))
    except Exception:
        pass

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING LOGIC
# ═══════════════════════════════════════════════════════════════════════════


def route_subtask(
    subtask: SubTask,
    available_agents: List[AgentCapability],
    trust_ledger: Optional[Any] = None,
    use_llm: bool = True
) -> Assignment:
    """
    Route a subtask to the optimal agent based on capability matching and trust.

    Routing algorithm (arXiv:2602.11865 Section 4.2):
    1. Check complexity floor: if subtask.profile.complexity < 0.2, execute directly
    2. Score each agent: capability_match * 0.6 + trust_score * 0.3 + cost_efficiency * 0.1
    3. Select agent with highest score
    4. Return Assignment with reasoning

    Args:
        subtask: SubTask to route
        available_agents: List of AgentCapability profiles
        trust_ledger: Optional TrustLedger for historical performance scores
        use_llm: Use LLM for semantic matching (default: True)

    Returns:
        Assignment with selected agent and scoring breakdown
    """
    # Complexity floor: tasks below threshold execute directly (no delegation)
    if subtask.profile and subtask.profile.complexity < MIN_COMPLEXITY_FOR_DELEGATION:
        return Assignment(
            subtask_id=subtask.id,
            agent_id="DIRECT_EXECUTION",
            trust_score=1.0,
            capability_match=1.0,
            timestamp=time.time(),
            assignment_reasoning=(
                f"Complexity {subtask.profile.complexity:.2f} below delegation threshold "
                f"{MIN_COMPLEXITY_FOR_DELEGATION} → direct execution"
            ),
            metadata={"delegation_bypassed": True}
        )

    # Score all available agents
    scored_agents = []
    for agent in available_agents:
        # Capability matching
        capability_match = _calculate_capability_match(subtask, agent, use_llm=use_llm)

        # Trust score (from TrustLedger if available)
        if trust_ledger:
            # Use async context manager pattern
            trust_score = asyncio.run(_get_trust_score(trust_ledger, agent.agent_id))
        else:
            trust_score = 0.5  # Neutral trust if ledger unavailable

        # Cost efficiency (inverse of cost difference)
        if subtask.profile:
            cost_diff = abs(subtask.estimated_cost - agent.estimated_cost)
            cost_efficiency = 1.0 - cost_diff
        else:
            cost_efficiency = 0.5

        # Evolution affinity boost (learned from past outcomes)
        affinity_boost = _get_affinity_boost(agent.agent_id)

        # Weighted final score + evolution feedback
        final_score = (
            capability_match * CAPABILITY_WEIGHT +
            trust_score * TRUST_WEIGHT +
            cost_efficiency * COST_WEIGHT +
            affinity_boost
        )
        final_score = max(0.0, min(1.0, final_score))

        scored_agents.append({
            "agent": agent,
            "capability_match": capability_match,
            "trust_score": trust_score,
            "cost_efficiency": cost_efficiency,
            "final_score": final_score,
        })

    # Sort by final score (descending)
    scored_agents.sort(key=lambda x: x["final_score"], reverse=True)

    # Select top agent
    if not scored_agents:
        # No agents available → direct execution fallback
        return Assignment(
            subtask_id=subtask.id,
            agent_id="DIRECT_EXECUTION",
            trust_score=0.5,
            capability_match=0.0,
            timestamp=time.time(),
            assignment_reasoning="No agents available → fallback to direct execution",
            metadata={"no_agents_available": True}
        )

    best = scored_agents[0]
    agent = best["agent"]

    # Build reasoning
    reasoning = (
        f"Selected {agent.name} (score: {best['final_score']:.3f}) | "
        f"Capability: {best['capability_match']:.3f}, "
        f"Trust: {best['trust_score']:.3f}, "
        f"Cost: {best['cost_efficiency']:.3f}"
    )

    # Create assignment
    assignment = Assignment(
        subtask_id=subtask.id,
        agent_id=agent.agent_id,
        trust_score=best["trust_score"],
        capability_match=best["capability_match"],
        timestamp=time.time(),
        assignment_reasoning=reasoning,
        metadata={
            "final_score": best["final_score"],
            "cost_efficiency": best["cost_efficiency"],
            "agent_name": agent.name,
            "agent_description": agent.description,
            "fallback_chain": [s["agent"].agent_id for s in scored_agents[1:4]],  # Top 3 backups
        }
    )

    return assignment


async def _get_trust_score(trust_ledger: Any, agent_id: str) -> float:
    """Get trust score for agent from TrustLedger (async wrapper)"""
    try:
        # TrustLedger uses async context manager pattern
        async with trust_ledger as ledger:
            score = await ledger.get_trust_score(agent_id)
            return score if score is not None else 0.5
    except Exception:
        return 0.5  # Neutral trust on error


def route_batch(
    subtasks: List[SubTask],
    available_agents: List[AgentCapability],
    trust_ledger: Optional[Any] = None,
    use_llm: bool = True
) -> List[Assignment]:
    """
    Route multiple subtasks in batch (parallel routing).

    More efficient than calling route_subtask() in a loop because:
    - Avoids repeated agent registry lookups
    - Can parallelize LLM calls for semantic matching

    Args:
        subtasks: List of SubTasks to route
        available_agents: List of AgentCapability profiles
        trust_ledger: Optional TrustLedger for trust scores
        use_llm: Use LLM for semantic matching

    Returns:
        List of Assignments (one per subtask, in same order)
    """
    assignments = []

    for subtask in subtasks:
        assignment = route_subtask(
            subtask=subtask,
            available_agents=available_agents,
            trust_ledger=trust_ledger,
            use_llm=use_llm
        )
        assignments.append(assignment)

    return assignments
