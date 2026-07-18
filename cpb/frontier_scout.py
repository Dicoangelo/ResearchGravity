#!/usr/bin/env python3
"""
Frontier Scout — turn ResearchGravity from a passive logger into an active scout.

Loop:
    1. SEED     — pick the strongest arXiv papers already in the corpus
                  (optionally scoped to a topic/project).
    2. EXPAND   — for each seed, pull its citation neighborhood via Firecrawl
                  related-papers (citers = forward/frontier, similar = lateral).
    3. SUBTRACT — drop anything already in the corpus (the 1.2k+ arXiv ids
                  you've logged). What remains is, by construction, new to you.
    4. RANK     — score each candidate by how many seeds surfaced it and its
                  structural/semantic signal, so densely-connected frontier
                  papers float to the top.
    5. SURFACE  — "NEW in your <topic> neighborhood you haven't seen."

This automates RG's own methodology (Thesis -> Gap -> Innovation Direction):
the Gap step is exactly "what exists in my neighborhood that I haven't logged."

Usage:
    python3 -m cpb.frontier_scout --topic "multi-agent trust" --limit 10
    python3 -m cpb.frontier_scout --project os-app --mode citers --seeds 5
    python3 -m cpb.frontier_scout --json               # machine-readable
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .search_layer import get_search_layer

DB_PATH = Path.home() / ".agent-core" / "storage" / "antigravity.db"
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


@dataclass
class Seed:
    arxiv_id: str
    title: str
    topic: str
    relevance: float


@dataclass
class FrontierPaper:
    arxiv_id: str
    title: str
    abstract: str
    best_score: float = 0.0
    structural: float = 0.0
    seed_hits: int = 0  # how many of my seeds pointed here (density = signal)
    from_seeds: list[str] = field(default_factory=list)

    @property
    def frontier_score(self) -> float:
        # Density across seeds dominates; semantic score breaks ties.
        return self.seed_hits * 1.0 + self.best_score

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class FrontierScout:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.layer = get_search_layer()

    # ---- corpus access -------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def corpus_arxiv_ids(self) -> set[str]:
        """Every arXiv id already logged — the subtract set."""
        ids: set[str] = set()
        with self._connect() as c:
            for (url,) in c.execute(
                "SELECT url FROM urls WHERE url LIKE '%arxiv.org%'"
            ):
                m = ARXIV_RE.search(url or "")
                if m:
                    ids.add(m.group(1))
        return ids

    def select_seeds(
        self,
        topic: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 5,
    ) -> list[Seed]:
        """Strongest arXiv papers in the corpus, optionally topic/project-scoped."""
        q = """
            SELECT u.url AS url, u.relevance AS relevance,
                   COALESCE(s.topic, u.context, '') AS topic
            FROM urls u
            LEFT JOIN sessions s ON s.id = u.session_id
            WHERE u.url LIKE '%arxiv.org%'
        """
        params: list = []
        if topic:
            q += " AND (s.topic LIKE ? OR u.context LIKE ? OR u.category LIKE ?)"
            like = f"%{topic}%"
            params += [like, like, like]
        if project:
            q += " AND s.project = ?"
            params.append(project)
        q += " ORDER BY u.relevance DESC, u.captured_at DESC LIMIT ?"
        params.append(limit * 3)  # over-pull, then dedupe by arxiv id

        seeds: list[Seed] = []
        seen: set[str] = set()
        with self._connect() as c:
            for row in c.execute(q, params):
                m = ARXIV_RE.search(row["url"] or "")
                if not m:
                    continue
                aid = m.group(1)
                if aid in seen:
                    continue
                seen.add(aid)
                seeds.append(
                    Seed(
                        arxiv_id=aid,
                        title=(row["topic"] or "").strip()[:70] or aid,
                        topic=(row["topic"] or "").strip(),
                        relevance=float(row["relevance"] or 0),
                    )
                )
                if len(seeds) >= limit:
                    break
        return seeds

    # ---- the loop ------------------------------------------------------
    async def scout(
        self,
        topic: Optional[str] = None,
        project: Optional[str] = None,
        seeds: int = 5,
        mode: str = "citers",
        limit: int = 10,
        per_seed: int = 20,
    ) -> dict:
        seed_list = self.select_seeds(topic=topic, project=project, limit=seeds)
        if not seed_list:
            return {"seeds": [], "frontier": [], "note": "no arXiv seeds matched"}

        known = self.corpus_arxiv_ids()
        intent = topic or "frontier work extending these papers"

        # EXPAND — one related-papers call per seed, in parallel
        tasks = [
            self.layer.related_papers(
                f"arxiv:{s.arxiv_id}", intent, mode=mode, k=per_seed
            )
            for s in seed_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # SUBTRACT + RANK — aggregate candidates across seeds
        frontier: dict[str, FrontierPaper] = {}
        for seed, res in zip(seed_list, results):
            if isinstance(res, Exception):
                continue
            for cand in res:
                aid = self._cand_arxiv_id(cand)
                if not aid or aid in known or aid == seed.arxiv_id:
                    continue  # already logged, or the seed itself
                try:
                    score = float(cand.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                structural = self._signal(cand, "structural")
                fp = frontier.get(aid)
                if fp is None:
                    fp = FrontierPaper(
                        arxiv_id=aid,
                        title=cand.get("title", "Untitled"),
                        abstract=cand.get("abstract", "")[:280],
                    )
                    frontier[aid] = fp
                fp.seed_hits += 1
                fp.from_seeds.append(seed.arxiv_id)
                fp.best_score = max(fp.best_score, score)
                fp.structural = max(fp.structural, structural)

        ranked = sorted(
            frontier.values(), key=lambda p: p.frontier_score, reverse=True
        )[:limit]

        return {
            "topic": topic,
            "mode": mode,
            "seeds": [{"arxiv_id": s.arxiv_id, "title": s.title} for s in seed_list],
            "corpus_size": len(known),
            "candidates_found": len(frontier),
            "frontier": [
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "url": p.url,
                    "seed_hits": p.seed_hits,
                    "score": round(p.best_score, 4),
                    "from_seeds": p.from_seeds,
                    "abstract": p.abstract,
                }
                for p in ranked
            ],
        }

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _cand_arxiv_id(cand: dict) -> Optional[str]:
        pid = cand.get("primaryId", "") or ""
        if pid.startswith("arxiv:"):
            return pid.split("arxiv:", 1)[1].split("v")[0]
        m = ARXIV_RE.search(str(cand.get("ids", "")))
        return m.group(1) if m else None

    @staticmethod
    def _signal(cand: dict, key: str) -> float:
        raw = cand.get("signals")
        if not raw:
            return 0.0
        m = re.search(rf"'{key}':\s*([\d.]+)", str(raw))
        return float(m.group(1)) if m else 0.0


def _print_report(out: dict) -> None:
    if not out.get("frontier"):
        print(f"No new frontier papers found. {out.get('note', '')}")
        if out.get("seeds"):
            print("Seeds used:", ", ".join(s["arxiv_id"] for s in out["seeds"]))
        return
    topic = out.get("topic") or "your corpus"
    print(f"\n=== FRONTIER SCOUT — new in '{topic}' ({out['mode']} mode) ===")
    print(
        f"Seeded from {len(out['seeds'])} of your papers · "
        f"{out['candidates_found']} new candidates vs {out['corpus_size']} "
        f"already logged\n"
    )
    for i, p in enumerate(out["frontier"], 1):
        density = f"{p['seed_hits']} seeds" if p["seed_hits"] > 1 else "1 seed"
        print(f"{i}. {p['title']}")
        print(f"   {p['url']}  · {density} · score {p['score']}")
        if p["abstract"]:
            print(f"   {p['abstract'][:140].strip()}...")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Frontier Scout for ResearchGravity")
    ap.add_argument("--topic", help="scope seeds + intent to a topic")
    ap.add_argument("--project", help="scope seeds to a lineage project")
    ap.add_argument("--seeds", type=int, default=5, help="seed papers to expand")
    ap.add_argument(
        "--mode",
        default="citers",
        choices=["citers", "similar", "references"],
        help="citers=frontier watch, similar=lateral, references=foundations",
    )
    ap.add_argument("--limit", type=int, default=10, help="frontier papers to surface")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    scout = FrontierScout()
    out = asyncio.run(
        scout.scout(
            topic=args.topic,
            project=args.project,
            seeds=args.seeds,
            mode=args.mode,
            limit=args.limit,
        )
    )
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        _print_report(out)


if __name__ == "__main__":
    main()
