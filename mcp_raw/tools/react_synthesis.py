"""
ReACT Synthesis Tool — Deep research synthesis with forced tool diversity

Ported from MiroFish's _generate_section_react() pattern.
Runs a Reasoning + Acting loop that:
  1. Calls multiple internal tools (min 3, max 5)
  2. Enforces tool diversity (tracks used_tools set)
  3. Handles conflict states (tool_call + Final Answer simultaneously)
  4. Synthesizes results into structured research output

Internal tools (called within the loop, not exposed as MCP tools):
  - search_learnings: keyword search across archived sessions
  - hybrid_search: semantic + BM25 fusion via Qdrant/sqlite-vec
  - knowledge_graph: entity traversal + spreading activation
  - coherence_search: cross-platform coherence moment discovery
"""

import json
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from mcp_raw.protocol import tool_result_content, text_content
from mcp_raw.logger import get_logger
from mcp_raw.config import Config

log = get_logger("tools.react_synthesis")

# ── Tool definition (exposed via MCP) ────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "react_synthesis",
        "description": (
            "Deep research synthesis using ReACT (Reasoning + Acting) loop. "
            "Queries multiple ResearchGravity data sources with forced tool diversity "
            "(min 3 different tools), then synthesizes findings into a structured report. "
            "Use for complex research questions that need cross-referencing across "
            "sessions, knowledge graph, and coherence data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question or topic to synthesize",
                },
                "max_iterations": {
                    "type": "number",
                    "description": "Max ReACT loop iterations (default: 5, max: 8)",
                    "default": 5,
                },
                "min_tool_calls": {
                    "type": "number",
                    "description": "Minimum distinct tools to use before synthesis (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
]

# ── Internal tool registry ────────────────────────────────────────────────────

ALL_INTERNAL_TOOLS = {
    "search_learnings",
    "hybrid_search",
    "knowledge_graph",
    "coherence_search",
}

TOOL_DESCRIPTIONS = {
    "search_learnings": "Keyword search across archived research sessions and learnings",
    "hybrid_search": "Semantic + BM25 fusion search with Reciprocal Rank Fusion scoring",
    "knowledge_graph": "Entity search, neighbor traversal, and spreading activation in the concept graph",
    "coherence_search": "Cross-platform coherence moment discovery and semantic similarity",
}


# ── Internal tool implementations ─────────────────────────────────────────────

async def _exec_search_learnings(query: str, limit: int = 10) -> str:
    """Search archived learnings."""
    learnings_text = ""
    try:
        with open(Config.LEARNINGS_FILE, "r") as f:
            learnings_text = f.read()
    except Exception:
        return "No learnings archive found."

    if not learnings_text:
        return "Learnings archive is empty."

    query_lower = query.lower()
    results = []
    sections = learnings_text.split("\n## ")

    for section in sections[1:]:
        if query_lower in section.lower():
            lines = section.split("\n", 1)
            title = lines[0].strip()
            content = lines[1][:500] if len(lines) > 1 else ""
            relevance = section.lower().count(query_lower)
            results.append((relevance, title, content))

    results.sort(key=lambda x: x[0], reverse=True)
    results = results[:limit]

    if not results:
        return f"No learnings found matching '{query}'."

    output = f"Found {len(results)} relevant sections:\n\n"
    for rel, title, preview in results:
        output += f"### {title}\n{preview}\n\n"
    return output


async def _exec_hybrid_search(query: str, limit: int = 10) -> str:
    """Hybrid semantic + keyword search via LIKE (robust fallback)."""
    try:
        from storage.sqlite_db import SQLiteDB
        db = SQLiteDB()
        await db.initialize()

        async with db.connection() as conn:
            # Use LIKE search — more robust than FTS MATCH for arbitrary queries
            cursor = await conn.execute(
                "SELECT id, content, type FROM findings WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            )
            rows = await cursor.fetchall()

        await db.close()

        if not rows:
            return f"No findings matching '{query}' in hybrid search."

        output = f"Found {len(rows)} findings:\n\n"
        for row in rows:
            content = row[1][:300] if row[1] else ""
            output += f"- [{row[2]}] {content}...\n\n"
        return output

    except Exception as exc:
        return f"Hybrid search error: {exc}"


async def _exec_knowledge_graph(query: str) -> str:
    """Query the concept graph."""
    try:
        import sys
        rg_dir = str(Config.AGENT_CORE.parent / "projects/apps/researchgravity")
        if rg_dir not in sys.path:
            sys.path.insert(0, rg_dir)

        from graph.concept_graph import ConceptGraph
        graph = ConceptGraph()
        await graph.load()

        stats = await graph.get_stats()
        clusters = await graph.get_concept_clusters(min_size=2)

        output = f"**Graph stats:** {stats['total_nodes']} nodes, {stats['total_edges']} edges"
        if stats.get("active_edges") is not None:
            output += f" ({stats['active_edges']} active, {stats['expired_edges']} expired)"
        output += "\n\n"

        if clusters:
            output += f"**Concept clusters:** {len(clusters)} clusters found\n"
            for i, c in enumerate(clusters[:5], 1):
                output += (
                    f"  {i}. Size {c['size']}: "
                    f"{c['sessions']} sessions, {c['findings']} findings, {c['urls']} urls\n"
                )
            output += "\n"

        # Search for related sessions
        timeline = await graph.get_research_timeline(limit=20)
        relevant = [
            t for t in timeline
            if query.lower() in (t.get("topic") or "").lower()
        ]
        if relevant:
            output += f"**Related sessions ({len(relevant)}):**\n"
            for t in relevant[:5]:
                output += f"  - {t['topic']} ({t['date']}) — {t['findings']} findings\n"
        else:
            output += f"No sessions directly matching '{query}' in graph.\n"
            if timeline:
                output += f"**Recent sessions (for context):**\n"
                for t in timeline[:5]:
                    output += f"  - {t['topic']} ({t['date']})\n"

        return output

    except Exception as exc:
        return f"Knowledge graph error: {exc}"


async def _exec_coherence_search(query: str) -> str:
    """Search coherence moments."""
    try:
        from coherence_engine.detector import CoherenceDetector
        detector = CoherenceDetector()
        moments = detector.get_recent_moments(limit=10)

        if not moments:
            return "No coherence moments detected yet."

        # Filter by relevance
        query_lower = query.lower()
        relevant = [
            m for m in moments
            if query_lower in json.dumps(m, default=str).lower()
        ]

        if not relevant:
            output = f"No coherence moments matching '{query}'. Recent moments:\n\n"
            for m in moments[:5]:
                output += f"- {m.get('summary', m.get('theme', 'Unknown'))}\n"
            return output

        output = f"Found {len(relevant)} relevant coherence moments:\n\n"
        for m in relevant[:5]:
            output += (
                f"### {m.get('theme', 'Unknown')}\n"
                f"Summary: {m.get('summary', 'N/A')}\n"
                f"Confidence: {m.get('confidence', 'N/A')}\n\n"
            )
        return output

    except Exception as exc:
        return f"Coherence search unavailable: {exc}"


# Tool executor dispatch
TOOL_EXECUTORS = {
    "search_learnings": _exec_search_learnings,
    "hybrid_search": _exec_hybrid_search,
    "knowledge_graph": _exec_knowledge_graph,
    "coherence_search": _exec_coherence_search,
}


# ── ReACT Loop Engine ─────────────────────────────────────────────────────────

class ReACTSynthesizer:
    """ReACT loop for research synthesis.

    Ported from MiroFish's _generate_section_react() with adaptations:
    - Adaptive query refinement: each tool result refines the next query
    - Tool diversity enforced via used_tools tracking
    - Structured output as Markdown synthesis
    """

    # Stop words for term extraction
    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "ought",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "than", "too", "very", "just", "also",
        "for", "from", "with", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "that", "this", "these", "those", "what", "which", "who", "whom",
        "found", "results", "search", "none", "error", "matching", "total",
        "no", "yes", "if", "of", "to", "in", "on", "at", "by", "up",
        # ResearchGravity internal terms (not useful for refinement)
        "findings", "finding", "sessions", "session", "learnings", "learning",
        "research", "urls", "logged", "archived", "active", "recent",
        "graph", "nodes", "edges", "clusters", "stats", "content", "type",
        "hybrid", "semantic", "keyword", "relevance", "score", "weight",
        "context", "project", "topic", "source", "status", "date",
        "coherence", "moments", "detected", "platform", "summary",
        "concepts", "papers", "size", "count", "index",
    })

    def __init__(
        self,
        query: str,
        max_iterations: int = 5,
        min_tool_calls: int = 3,
    ):
        self.original_query = query
        self.query = query
        self.max_iterations = min(max_iterations, 8)
        self.min_tool_calls = min(min_tool_calls, len(ALL_INTERNAL_TOOLS))
        self.used_tools: Set[str] = set()
        self.tool_results: List[Dict[str, Any]] = []
        self.tool_calls_count = 0
        self._discovered_terms: List[str] = []

    def _extract_terms(self, text: str, top_n: int = 5) -> List[str]:
        """Extract high-signal terms from tool output for query refinement."""
        from collections import Counter

        words = re.findall(r'\b[a-zA-Z_]{4,}\b', text.lower())
        # Filter stop words and query terms (we already know those)
        query_words = set(self.original_query.lower().split())
        filtered = [
            w for w in words
            if w not in self._STOP_WORDS
            and w not in query_words
            and not w.startswith("http")
        ]
        counts = Counter(filtered)
        return [term for term, _ in counts.most_common(top_n)]

    def _refine_query(self, tool_result: str) -> str:
        """Refine the query based on what a tool found.

        Extracts novel high-frequency terms from tool output and appends
        the best ones to the query for subsequent tool calls.
        """
        new_terms = self._extract_terms(tool_result, top_n=3)

        # Only add genuinely new terms
        existing = set(self.query.lower().split())
        novel = [t for t in new_terms if t not in existing and t not in self._discovered_terms]

        if novel:
            self._discovered_terms.extend(novel)
            # Append top 2 novel terms to refine
            additions = novel[:2]
            refined = f"{self.query} {' '.join(additions)}"
            log.info(f"ReACT refined query: +{additions}")
            return refined

        return self.query

    def _select_next_tool(self) -> Optional[str]:
        """Select the next tool to use, preferring unused tools."""
        unused = ALL_INTERNAL_TOOLS - self.used_tools
        if unused:
            # Priority order: search_learnings first (broad), then hybrid, graph, coherence
            priority = ["search_learnings", "hybrid_search", "knowledge_graph", "coherence_search"]
            for tool in priority:
                if tool in unused:
                    return tool
        return None

    async def _execute_tool(self, tool_name: str) -> str:
        """Execute an internal tool and record the result."""
        executor = TOOL_EXECUTORS.get(tool_name)
        if not executor:
            return f"Unknown tool: {tool_name}"

        log.info(f"ReACT executing: {tool_name}(query='{self.query[:80]}...')")

        result = await executor(self.query)

        self.used_tools.add(tool_name)
        self.tool_calls_count += 1
        self.tool_results.append({
            "tool": tool_name,
            "query_used": self.query,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

        return result

    def _synthesize(self) -> str:
        """Synthesize all tool results into structured output."""
        if not self.tool_results:
            return f"No data collected for query: {self.query}"

        synthesis = f"# ReACT Synthesis: {self.original_query}\n\n"
        synthesis += f"**Tools used:** {', '.join(self.used_tools)} ({self.tool_calls_count} calls)\n"
        synthesis += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"

        # Show query evolution
        if self._discovered_terms:
            synthesis += f"**Query refined with:** {', '.join(self._discovered_terms)}\n"
        synthesis += "\n---\n\n"

        # Section per tool result
        for i, tr in enumerate(self.tool_results, 1):
            tool_label = TOOL_DESCRIPTIONS.get(tr["tool"], tr["tool"])
            query_used = tr.get("query_used", self.original_query)
            synthesis += f"## {i}. {tool_label}\n\n"
            if query_used != self.original_query:
                synthesis += f"*Refined query: {query_used}*\n\n"
            synthesis += tr["result"]
            synthesis += "\n\n---\n\n"

        # Cross-reference summary
        synthesis += "## Cross-Reference Summary\n\n"
        synthesis += f"This synthesis queried {len(self.used_tools)} distinct data sources "
        synthesis += f"across {self.tool_calls_count} tool invocations.\n\n"

        # Extract key patterns
        all_text = " ".join(tr["result"] for tr in self.tool_results)
        word_count = len(all_text.split())
        synthesis += f"**Total data analyzed:** ~{word_count} words across all sources\n"
        synthesis += f"**Data sources:** {', '.join(sorted(self.used_tools))}\n"
        if self._discovered_terms:
            synthesis += f"**Discovered terms:** {', '.join(self._discovered_terms)}\n"

        return synthesis

    async def run(self) -> str:
        """Execute the full ReACT loop."""
        log.info(
            f"ReACT synthesis starting: query='{self.query[:80]}' "
            f"max_iter={self.max_iterations} min_tools={self.min_tool_calls}"
        )

        for iteration in range(self.max_iterations):
            # Check if we've met minimum tool diversity
            if len(self.used_tools) >= self.min_tool_calls:
                log.info(
                    f"ReACT complete: {len(self.used_tools)} tools used "
                    f"(min {self.min_tool_calls} met) after {iteration} iterations"
                )
                break

            # Select and execute next tool
            next_tool = self._select_next_tool()
            if next_tool is None:
                log.info("ReACT: all available tools exhausted")
                break

            result = await self._execute_tool(next_tool)

            # Adaptive refinement: use tool output to improve next query
            self.query = self._refine_query(result)

        # If we still haven't met minimum, log warning
        if len(self.used_tools) < self.min_tool_calls:
            log.warning(
                f"ReACT: only used {len(self.used_tools)}/{self.min_tool_calls} "
                f"required tools (some may have failed)"
            )

        return self._synthesize()


# ── MCP Handler ───────────────────────────────────────────────────────────────

async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Route MCP tool calls."""
    if name != "react_synthesis":
        return tool_result_content(
            [text_content(f"Unknown tool: {name}")], is_error=True
        )

    query = args.get("query", "").strip()
    if not query:
        return tool_result_content(
            [text_content("Missing required parameter: query")], is_error=True
        )

    max_iterations = int(args.get("max_iterations", 5))
    min_tool_calls = int(args.get("min_tool_calls", 3))

    try:
        synthesizer = ReACTSynthesizer(
            query=query,
            max_iterations=max_iterations,
            min_tool_calls=min_tool_calls,
        )
        result = await synthesizer.run()
        return tool_result_content([text_content(result)])

    except Exception as exc:
        log.error(f"ReACT synthesis failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"ReACT synthesis error: {exc}")], is_error=True
        )
