#!/usr/bin/env python3
"""
Contradiction Miner — surface disagreement the knowledge graph never recorded.

Motivation (Atlas deep dig, 2026-07-17): the cognitive knowledge graph holds
11,186 edges and only 2 `contradicts`. A corpus that never disagrees with
itself is an echo chamber. This pass mines the findings table for
self-declared contradiction markers and pairs each marker finding with the
prior findings it most plausibly contradicts (shared arXiv id, then shared
session), writing candidates to an append-only ledger for HUMAN review.

Candidates are never auto-promoted to KG edges: disagreement is a judgment
call, and auto-writing it would recreate the agreeableness problem in
reverse. Review the ledger, then promote accepted pairs.

Usage:
    python3 scripts/coherence/contradiction_miner.py            # mine, append new
    python3 scripts/coherence/contradiction_miner.py --report   # summarize ledger
    python3 scripts/coherence/contradiction_miner.py --limit 50
"""

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".agent-core" / "storage" / "antigravity.db"
LEDGER = Path.home() / ".agent-core" / "storage" / "contradiction_candidates.jsonl"

# Markers of self-declared disagreement, strongest first. Word-boundary
# regexes; case-insensitive. Kept conservative: generic negativity ("wrong",
# "bad") is deliberately excluded to keep precision high.
MARKERS = [
    ("falsified", r"\bfalsif\w+"),
    ("contradicts", r"\bcontradict\w+"),
    ("refutes", r"\brefut\w+"),
    ("debunks", r"\bdebunk\w+"),
    ("disproves", r"\bdisprov\w+"),
    ("overstated", r"\boverstat\w+"),
    ("myth", r"\bmyth\b"),
    ("scope-correction", r"\bscope correction\b"),
    ("do-not-repeat", r"\bdo not repeat the\b"),
    ("actually-wrong", r"\bturned? out (?:to be )?(?:wrong|false)\b"),
]

ARXIV_RE = re.compile(r"arxiv[:/\s]*(\d{4}\.\d{4,5})", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mined_ids(ledger: Path = LEDGER) -> set:
    if not ledger.exists():
        return set()
    ids = set()
    for line in ledger.read_text().splitlines():
        try:
            ids.add(json.loads(line)["marker_finding_id"])
        except Exception:
            continue
    return ids


CODE_SNIPPET_RE = re.compile(
    r"class \w+\(Enum\)|def \w+\(|=\s*\"\w+\"\\n|self\.\w+\s*=", re.IGNORECASE
)


def _marker_hits(content: str):
    # Source code that merely mentions a marker word (enum values, feature
    # specs like "Contradiction Detection") is not a contradiction. First
    # triage pass: 3 of 7 candidates were exactly this.
    if CODE_SNIPPET_RE.search(content):
        return []
    hits = []
    low = content.lower()
    for name, pattern in MARKERS:
        if re.search(pattern, low):
            hits.append(name)
    return hits


def mine(limit=None, include_imports=False):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    # By default exclude the chatgpt-import backfill: personal-conversation
    # markers ("myth", "contradicts" in social contexts) are noise here. The
    # research signal lives in engine-era findings.
    import_filter = (
        ""
        if include_imports
        else "AND (s.project IS NULL OR s.project != 'chatgpt-import') "
    )
    rows = con.execute(
        "SELECT f.id, f.session_id, f.content, f.type, f.created_at "
        "FROM findings f LEFT JOIN sessions s ON s.id = f.session_id "
        f"WHERE f.content IS NOT NULL {import_filter}"
        "ORDER BY f.created_at DESC"
    ).fetchall()

    done = _mined_ids()
    candidates = []
    for r in rows:
        fid = r["id"] or f"rowid-{r['session_id']}-{hash(r['content']) & 0xffffff:x}"
        if fid in done:
            continue
        hits = _marker_hits(r["content"])
        if not hits:
            continue

        # Pair with what it contradicts: same arXiv id first, same session second.
        arxiv_ids = list(dict.fromkeys(ARXIV_RE.findall(r["content"])))
        paired = []
        if arxiv_ids:
            q = " OR ".join(["content LIKE ?"] * len(arxiv_ids))
            params = [f"%{a}%" for a in arxiv_ids]
            for p in con.execute(
                f"SELECT id, substr(content,1,200) c, created_at FROM findings "
                f"WHERE ({q}) AND created_at < ? ORDER BY created_at DESC LIMIT 3",
                (*params, r["created_at"]),
            ):
                paired.append(
                    {"finding_id": p["id"], "via": "shared-arxiv", "excerpt": p["c"]}
                )
        if not paired and r["session_id"]:
            for p in con.execute(
                "SELECT id, substr(content,1,200) c FROM findings "
                "WHERE session_id = ? AND id IS NOT ? AND created_at < ? "
                "ORDER BY created_at DESC LIMIT 2",
                (r["session_id"], r["id"], r["created_at"]),
            ):
                paired.append(
                    {"finding_id": p["id"], "via": "same-session", "excerpt": p["c"]}
                )

        candidates.append(
            {
                "mined_at": _now(),
                "marker_finding_id": fid,
                "markers": hits,
                "type": r["type"],
                "created_at": r["created_at"],
                "excerpt": re.sub(r"\s+", " ", r["content"])[:280],
                "contradicts_candidates": paired,
                "review": "pending",
            }
        )
        if limit and len(candidates) >= limit:
            break
    con.close()

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    return candidates


def report():
    if not LEDGER.exists():
        print("No ledger yet. Run the miner first.")
        return
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    recs = [r for r in rows if "markers" in r]
    reviews = {r["review_of"]: r for r in rows if "review_of" in r}
    by_marker = {}
    paired = 0
    for r in recs:
        for m in r["markers"]:
            by_marker[m] = by_marker.get(m, 0) + 1
        if r["contradicts_candidates"]:
            paired += 1
    verdicts = {}
    for v in reviews.values():
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    unreviewed = sum(1 for r in recs if r["marker_finding_id"] not in reviews)
    print(f"=== Contradiction Ledger — {len(recs)} candidates, {paired} paired ===")
    for m, n in sorted(by_marker.items(), key=lambda x: -x[1]):
        print(f"  {m:<18} {n}")
    print(f"\nReviews: {verdicts or 'none'} | unreviewed: {unreviewed}")
    print("\nMost recent candidates:")
    for r in recs[-5:]:
        v = reviews.get(r["marker_finding_id"], {}).get("verdict", "pending")
        print(f"  [{','.join(r['markers'])}] ({v}) {r['excerpt'][:100]}")


def main():
    ap = argparse.ArgumentParser(description="Mine contradiction candidates")
    ap.add_argument("--limit", type=int, help="max new candidates this run")
    ap.add_argument("--report", action="store_true", help="summarize ledger")
    ap.add_argument(
        "--all",
        action="store_true",
        help="include chatgpt-import sessions (noisy; excluded by default)",
    )
    args = ap.parse_args()

    if args.report:
        report()
        return
    out = mine(limit=args.limit, include_imports=args.all)
    print(f"Mined {len(out)} new contradiction candidates -> {LEDGER}")
    for c in out[:8]:
        pair = c["contradicts_candidates"]
        via = pair[0]["via"] if pair else "unpaired"
        print(f"  [{','.join(c['markers'])}] ({via}) {c['excerpt'][:100]}")


if __name__ == "__main__":
    main()
