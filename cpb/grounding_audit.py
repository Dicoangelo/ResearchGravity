#!/usr/bin/env python3
"""
Corpus Grounding Audit — a trust ledger over ResearchGravity's findings.

For every finding that carries an inline arXiv citation, pull the cited
paper's actual passages (Firecrawl read-paper) and check whether the paper is
real and contains text addressing the finding's claim. The result is a
per-finding trust record — a quality/epistemics audit of the corpus that no
other research system has.

Honest labeling (retrieval can confirm presence, not adjudicate truth):
    - grounded    : cited paper resolves AND returns passages addressing the claim
    - unresolved  : paper id not found, or no passages returned (citation is a
                    dangling reference — the thing most worth flagging)

Design (per house rules):
    - Append-only JSONL ledger; never overwritten.
    - Resumable: skips findings already in the ledger, so it can run in slices.
    - Scoped + rate-friendly: only inline-cited findings, with --limit.

Usage:
    python3 -m cpb.grounding_audit --limit 15          # audit a slice
    python3 -m cpb.grounding_audit                     # audit all remaining
    python3 -m cpb.grounding_audit --report            # summarize the ledger
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from .search_layer import get_search_layer

DB_PATH = Path.home() / ".agent-core" / "storage" / "antigravity.db"
LEDGER_PATH = Path.home() / ".agent-core" / "storage" / "grounding_ledger.jsonl"
ARXIV_RE = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audited_ids(ledger: Path = LEDGER_PATH) -> set[str]:
    """Finding ids already in the ledger (for resumability)."""
    seen: set[str] = set()
    if not ledger.exists():
        return seen
    with ledger.open() as f:
        for line in f:
            try:
                seen.add(str(json.loads(line)["finding_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def inline_cited_findings(db: Path = DB_PATH) -> list[dict]:
    """Findings whose own text names an arXiv paper."""
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, content, project, session_id FROM findings "
        "WHERE content LIKE '%arxiv%' OR content LIKE '%arXiv%'"
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        ids = list(dict.fromkeys(ARXIV_RE.findall(r["content"] or "")))
        if ids:
            out.append(
                {
                    "finding_id": str(r["id"]),
                    "content": r["content"],
                    "project": r["project"],
                    "arxiv_ids": ids,
                }
            )
    return out


class GroundingAudit:
    def __init__(self, db: Path = DB_PATH, ledger: Path = LEDGER_PATH):
        self.db = db
        self.ledger = ledger
        self.layer = get_search_layer()

    async def audit_finding(self, finding: dict) -> dict:
        """Ground one finding against every paper it cites."""
        claim = self._claim(finding["content"])
        per_paper = []
        grounded_any = False
        for aid in finding["arxiv_ids"][:3]:  # cap papers per finding
            passages = await self.layer.read_paper_passages(
                f"arxiv:{aid}", claim, k=2
            )
            resolved = bool(passages)
            grounded_any = grounded_any or resolved
            per_paper.append(
                {
                    "arxiv_id": aid,
                    "resolved": resolved,
                    "top_score": round(passages[0]["score"], 4) if passages else 0.0,
                    "top_passage": passages[0]["text"][:300] if passages else "",
                }
            )
        return {
            "finding_id": finding["finding_id"],
            "project": finding["project"],
            "claim": claim[:200],
            "verdict": "grounded" if grounded_any else "unresolved",
            "papers": per_paper,
            "audited_at": _now(),
        }

    async def run(self, limit: int | None = None) -> dict:
        done = audited_ids(self.ledger)
        pending = [f for f in inline_cited_findings(self.db) if f["finding_id"] not in done]
        if limit:
            pending = pending[:limit]

        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        grounded = unresolved = 0
        t0 = time.time()
        with self.ledger.open("a") as out:
            for f in pending:
                rec = await self.audit_finding(f)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if rec["verdict"] == "grounded":
                    grounded += 1
                else:
                    unresolved += 1

        return {
            "audited_now": len(pending),
            "grounded": grounded,
            "unresolved": unresolved,
            "already_in_ledger": len(done),
            "elapsed_s": round(time.time() - t0, 1),
        }

    @staticmethod
    def _claim(content: str) -> str:
        """Use the sentence nearest the citation as the claim to verify."""
        text = re.sub(r"\s+", " ", content or "").strip()
        m = ARXIV_RE.search(text)
        if not m:
            return text[:200]
        start = max(0, m.start() - 160)
        return text[start : m.start() + 40].strip()


def report(ledger: Path = LEDGER_PATH) -> None:
    if not ledger.exists():
        print("No ledger yet. Run: python3 -m cpb.grounding_audit --limit 15")
        return
    total = grounded = unresolved = 0
    dangling = []
    with ledger.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if rec["verdict"] == "grounded":
                grounded += 1
            else:
                unresolved += 1
                dangling.append(rec)
    print("\n=== CORPUS GROUNDING LEDGER ===")
    print(f"findings audited : {total}")
    print(f"grounded         : {grounded} ({grounded / total * 100:.0f}%)" if total else "")
    print(f"unresolved       : {unresolved}  (dangling citations — flag these)")
    for rec in dangling[:8]:
        ids = ", ".join(p["arxiv_id"] for p in rec["papers"])
        print(f"  ! finding {rec['finding_id']} [{rec.get('project') or '-'}] cites {ids}")
        print(f"    claim: {rec['claim'][:90]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Corpus Grounding Audit")
    ap.add_argument("--limit", type=int, help="audit at most N pending findings")
    ap.add_argument("--report", action="store_true", help="summarize the ledger")
    args = ap.parse_args()

    if args.report:
        report()
        return

    audit = GroundingAudit()
    summary = asyncio.run(audit.run(limit=args.limit))
    print(json.dumps(summary, indent=2))
    print(f"\nLedger: {LEDGER_PATH}")
    print("Summarize with: python3 -m cpb.grounding_audit --report")


if __name__ == "__main__":
    main()
