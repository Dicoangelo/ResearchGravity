"""
Frontier Tools — the Firecrawl Trilogy as MCP tools

3 tools closing the research loop at corpus scale:
  frontier_scout     — discovery: citation-neighborhood gap detection
  grounding_audit    — trust: verify inline citations against source text
  paper_code_bridge  — implementation: link papers/methods to prior-art code
"""

import json
from typing import Any, Dict, List

from mcp_raw.logger import get_logger
from mcp_raw.protocol import text_content, tool_result_content

log = get_logger("tools.frontier")

# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "frontier_scout",
        "description": (
            "Scan the citation neighborhood of the research corpus for new papers "
            "not yet logged (seed -> expand -> subtract -> rank). Automates the "
            "Thesis -> Gap -> Direction gap step. Modes: citers=frontier watch, "
            "similar=lateral discovery, references=foundations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Scope seeds + intent to a topic (optional; omit for corpus-wide)",
                },
                "project": {
                    "type": "string",
                    "description": "Scope seeds to a lineage project (optional)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["citers", "similar", "references"],
                    "default": "citers",
                    "description": "Expansion mode",
                },
                "seeds": {
                    "type": "number",
                    "default": 5,
                    "description": "Seed papers to expand",
                },
                "limit": {
                    "type": "number",
                    "default": 10,
                    "description": "Frontier papers to surface",
                },
            },
        },
    },
    {
        "name": "grounding_audit",
        "description": (
            "Audit inline arXiv-cited findings against actual paper text via "
            "read-paper passages. Verdicts: grounded | unresolved (dangling "
            "citation). Appends to the resumable trust ledger; already-audited "
            "findings are skipped. Honest labeling: retrieval confirms presence, "
            "not truth."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "number",
                    "description": "Max findings to audit this run (omit for all remaining)",
                },
            },
        },
    },
    {
        "name": "paper_code_bridge",
        "description": (
            "Link a paper or method to implementation prior-art on GitHub "
            "(merged PRs > issues > readmes, noise-filtered). Pass either a "
            "free-text method/query or an arXiv id from the corpus."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Method or technique to find implementations of",
                },
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv id from the corpus (looks up the paper's topic first)",
                },
                "limit": {
                    "type": "number",
                    "default": 8,
                    "description": "Max results",
                },
            },
        },
    },
]


# ── Dispatcher ───────────────────────────────────────────────────────────────


async def handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Route tool calls to implementations."""
    handlers = {
        "frontier_scout": _frontier_scout,
        "grounding_audit": _grounding_audit,
        "paper_code_bridge": _paper_code_bridge,
    }

    handler = handlers.get(name)
    if not handler:
        return tool_result_content(
            [text_content(f"Unknown frontier tool: {name}")], is_error=True
        )

    try:
        return await handler(args)
    except Exception as exc:
        log.error(f"Tool {name} failed: {exc}", exc_info=True)
        return tool_result_content(
            [text_content(f"Error in {name}: {exc}")], is_error=True
        )


# ── Implementations ──────────────────────────────────────────────────────────


async def _frontier_scout(args: Dict) -> Dict:
    from cpb.frontier_scout import FrontierScout

    scout = FrontierScout()
    out = await scout.scout(
        topic=args.get("topic"),
        project=args.get("project"),
        seeds=int(args.get("seeds", 5)),
        mode=args.get("mode", "citers"),
        limit=int(args.get("limit", 10)),
    )

    frontier = out.get("frontier", [])
    lines = [
        "# Frontier Scout",
        f"mode={out.get('mode')} seeds={len(out.get('seeds') or [])} "
        f"corpus={out.get('corpus_size')} candidates={out.get('candidates_found')}",
        "",
    ]
    if not frontier:
        lines.append(
            "No new frontier papers surfaced. (Keyless Firecrawl mode is "
            "rate-limited; set FIRECRAWL_API_KEY / config.json firecrawl.api_key "
            "if this persists.)"
        )
    for p in frontier:
        if isinstance(p, dict):
            title = p.get("title", "?")
            pid = p.get("primary_id") or p.get("paper_id") or ""
            score = p.get("score")
            score_s = f" [{score:.2f}]" if isinstance(score, (int, float)) else ""
            lines.append(f"- **{title}**{score_s} {pid}")
        else:
            lines.append(f"- {p}")
    return tool_result_content([text_content("\n".join(lines))])


async def _grounding_audit(args: Dict) -> Dict:
    from cpb.grounding_audit import GroundingAudit

    limit = args.get("limit")
    audit = GroundingAudit()
    out = await audit.run(limit=int(limit) if limit else None)

    summary = json.dumps(out, indent=2, default=str)
    return tool_result_content(
        [text_content(f"# Grounding Audit\n\n```json\n{summary}\n```")]
    )


async def _paper_code_bridge(args: Dict) -> Dict:
    from cpb.paper_code_bridge import PaperCodeBridge

    query = args.get("query")
    arxiv_id = args.get("arxiv_id")
    if not query and not arxiv_id:
        return tool_result_content(
            [text_content("Provide either `query` or `arxiv_id`.")], is_error=True
        )

    bridge = PaperCodeBridge()
    limit = int(args.get("limit", 8))
    if arxiv_id:
        out = await bridge.bridge_arxiv(arxiv_id, limit=limit)
    else:
        out = await bridge.bridge(query, limit=limit)

    results = out.get("results", out) if isinstance(out, dict) else out
    lines = ["# Paper -> Code Bridge", ""]
    if isinstance(results, list):
        if not results:
            lines.append("No implementation prior-art found.")
        for r in results:
            if isinstance(r, dict):
                kind = str(r.get("kind", "")).replace("_", " ")
                repo = r.get("repo", "")
                rank = r.get("rank", "")
                lines.append(f"- [{kind}] **{repo}** (rank {rank})")
                if r.get("url"):
                    lines.append(f"  {r['url']}")
                if r.get("snippet"):
                    lines.append(f"  {r['snippet']}")
            else:
                lines.append(f"- {r}")
    else:
        lines.append(f"```json\n{json.dumps(out, indent=2, default=str)}\n```")
    return tool_result_content([text_content("\n".join(lines))])
