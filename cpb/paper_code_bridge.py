#!/usr/bin/env python3
"""
Paper -> Code Bridge — close RG's research->implementation lineage loop.

ResearchGravity already links research sessions to implementation projects.
The missing half was the outside world's engineering record: for any paper or
method, what actually got built, what broke, what was debated. Firecrawl's
GitHub-history search (issues / PRs / discussions / READMEs) supplies it.

Given a query (method / paper title) or an arXiv id already in the corpus,
surface the ranked implementation prior-art: working repos, known bugs, design
discussions — the "does it actually work in code" signal that a paper alone
never carries.

Usage:
    python3 -m cpb.paper_code_bridge "flash attention implementation"
    python3 -m cpb.paper_code_bridge --arxiv 1706.03762   # look up title, then search
    python3 -m cpb.paper_code_bridge "speculative decoding" --limit 8 --json
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

from .search_layer import get_search_layer

DB_PATH = Path.home() / ".agent-core" / "storage" / "antigravity.db"

# Prefer the engineering record that actually carries design signal.
PAGETYPE_WEIGHT = {
    "merged_pr": 1.0,
    "pull_request": 0.9,
    "issue": 0.8,
    "discussion": 0.7,
    "readme": 0.6,
}


def _fusion(cand: dict) -> float:
    m = re.search(r"'fusion':\s*([\d.]+)", str(cand.get("scores", "")))
    return float(m.group(1)) if m else 0.0


def _arxiv_title_from_corpus(arxiv_id: str, db: Path = DB_PATH) -> Optional[str]:
    """Recover a search phrase for an arXiv id from the corpus (session topic)."""
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT COALESCE(s.topic, u.context) FROM urls u "
            "LEFT JOIN sessions s ON s.id = u.session_id "
            "WHERE u.url LIKE ? ORDER BY u.relevance DESC LIMIT 1",
            (f"%{arxiv_id}%",),
        ).fetchone()
        return (row[0].strip() if row and row[0] else None)
    finally:
        conn.close()


class PaperCodeBridge:
    def __init__(self, db: Path = DB_PATH):
        self.db = db
        self.layer = get_search_layer()

    async def bridge(
        self, query: str, limit: int = 8
    ) -> list[dict]:
        raw = await self.layer.related_github(query, k=limit * 2)
        scored = []
        for r in raw:
            weight = PAGETYPE_WEIGHT.get(r.get("pageType", ""), 0.5)
            rank = weight + _fusion(r)
            scored.append(
                {
                    "repo": r.get("repo", ""),
                    "url": r.get("url", ""),
                    "kind": r.get("pageType", ""),
                    "title": (r.get("title") or "").strip()[:90],
                    "snippet": re.sub(r"\s+", " ", r.get("snippet", "")).strip()[:200],
                    "rank": round(rank, 4),
                }
            )
        scored.sort(key=lambda x: x["rank"], reverse=True)
        return scored[:limit]

    async def bridge_arxiv(self, arxiv_id: str, limit: int = 8) -> dict:
        title = _arxiv_title_from_corpus(arxiv_id, self.db)
        query = title or arxiv_id
        results = await self.bridge(f"{query} implementation", limit=limit)
        return {"arxiv_id": arxiv_id, "query": query, "results": results}


def _print(query: str, results: list[dict]) -> None:
    if not results:
        print(f"No implementation prior-art found for '{query}'.")
        return
    print(f"\n=== PAPER -> CODE — implementation record for '{query}' ===\n")
    for i, r in enumerate(results, 1):
        kind = r["kind"].replace("_", " ")
        print(f"{i}. [{kind}] {r['repo']}")
        print(f"   {r['url']}")
        if r["snippet"]:
            print(f"   {r['snippet']}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper -> Code Bridge")
    ap.add_argument("query", nargs="?", help="method / paper title to search")
    ap.add_argument("--arxiv", help="arXiv id in the corpus; look up title then search")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    bridge = PaperCodeBridge()
    if args.arxiv:
        out = asyncio.run(bridge.bridge_arxiv(args.arxiv, limit=args.limit))
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            _print(out["query"], out["results"])
    elif args.query:
        results = asyncio.run(bridge.bridge(args.query, limit=args.limit))
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            _print(args.query, results)
    else:
        ap.error("provide a query or --arxiv <id>")


if __name__ == "__main__":
    main()
