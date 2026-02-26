#!/usr/bin/env python3
"""HuggingFace Daily Papers Importer — ResearchGravity Integration

Fetches papers from the HuggingFace Daily Papers API and stores them in
antigravity.db (papers table) with full metadata, relevance scoring, and
optional vector indexing.

Designed to run daily via LaunchAgent or manually.

Usage:
  python3 huggingface_daily_papers.py                     # Fetch today's papers
  python3 huggingface_daily_papers.py --date 2026-02-24   # Specific date
  python3 huggingface_daily_papers.py --days 7            # Last 7 days
  python3 huggingface_daily_papers.py --status             # Show ingestion stats
  python3 huggingface_daily_papers.py --dry-run            # Preview without storing

API Docs:
  https://huggingface.co/api/daily_papers          — Today's papers
  https://huggingface.co/api/daily_papers?date=YYYY-MM-DD  — Specific date
"""

import argparse
import json
import sqlite3
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DB_PATH = Path.home() / ".agent-core" / "storage" / "antigravity.db"
HF_API_BASE = "https://huggingface.co/api/daily_papers"
USER_AGENT = "ResearchGravity/1.0 (sovereign-research-agent)"

# Topics that boost relevance for this ecosystem
RELEVANCE_KEYWORDS = {
    5: [
        "multi-agent", "agent orchestration", "cognitive", "sovereign",
        "self-improving", "routing", "model selection", "agentic",
    ],
    4: [
        "reinforcement learning", "tool use", "code generation",
        "rag", "retrieval", "embeddings", "fine-tuning", "reasoning",
        "planning", "knowledge graph", "memory",
    ],
    3: [
        "language model", "llm", "transformer", "attention",
        "benchmark", "evaluation", "alignment", "safety",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# API CLIENT
# ═══════════════════════════════════════════════════════════════════════════

def fetch_daily_papers(date: Optional[str] = None) -> list[dict]:
    """Fetch papers from HuggingFace Daily Papers API.

    Args:
        date: YYYY-MM-DD string, or None for today's papers.

    Returns:
        List of paper dicts from the API.
    """
    url = HF_API_BASE
    if date:
        url += f"?date={date}"

    headers = {"User-Agent": USER_AGENT}
    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data if isinstance(data, list) else []
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  No papers found for date: {date or 'today'}", file=sys.stderr)
            return []
        print(f"  HTTP error {e.code}: {e.reason}", file=sys.stderr)
        return []
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  Network error: {e}", file=sys.stderr)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_relevance(paper: dict) -> int:
    """Score paper relevance (1-5) based on keywords and community signal.

    Combines keyword matching with HF community upvotes.
    """
    inner = paper.get("paper", {})
    title = (inner.get("title") or "").lower()
    summary = (inner.get("summary") or "").lower()
    ai_keywords = [k.lower() for k in (inner.get("ai_keywords") or [])]
    text = f"{title} {summary} {' '.join(ai_keywords)}"

    # Keyword-based score
    keyword_score = 2  # default baseline
    for score, keywords in sorted(RELEVANCE_KEYWORDS.items(), reverse=True):
        if any(kw in text for kw in keywords):
            keyword_score = score
            break

    # Community signal boost: high upvotes = community validation
    upvotes = inner.get("upvotes", 0)
    if upvotes >= 50:
        keyword_score = min(5, keyword_score + 1)
    elif upvotes >= 20:
        keyword_score = min(5, keyword_score + 1)

    return keyword_score


def normalize_paper(raw: dict) -> dict:
    """Transform HF API response into our papers table schema."""
    inner = raw.get("paper", {})
    paper_id = inner.get("id", "")

    # Build author list
    authors = []
    for a in inner.get("authors", []):
        name = a.get("name", "")
        if name:
            authors.append(name)

    # Build metadata
    metadata = {
        "source": "huggingface_daily_papers",
        "upvotes": inner.get("upvotes", 0),
        "num_comments": raw.get("numComments", 0),
        "ai_summary": inner.get("ai_summary"),
        "ai_keywords": inner.get("ai_keywords", []),
        "github_repo": inner.get("githubRepo"),
        "github_stars": inner.get("githubStars", 0),
        "project_page": inner.get("projectPage"),
        "published_at": inner.get("publishedAt"),
        "submitted_daily_at": inner.get("submittedOnDailyAt"),
        "organization": inner.get("organization", {}).get("fullname"),
        "media_urls": inner.get("mediaUrls", []),
    }

    # Clean None values
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return {
        "id": paper_id,
        "title": inner.get("title"),
        "authors": json.dumps(authors),
        "abstract": inner.get("summary"),
        "url": f"https://huggingface.co/papers/{paper_id}" if paper_id else None,
        "relevance": score_relevance(raw),
        "applied": 0,
        "session_ids": json.dumps([]),
        "metadata": json.dumps(metadata),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════

def ensure_db() -> sqlite3.Connection:
    """Open DB connection, ensure papers table exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            url TEXT,
            relevance INTEGER,
            applied INTEGER DEFAULT 0,
            session_ids TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


def store_papers(papers: list[dict], dry_run: bool = False) -> tuple[int, int]:
    """Store papers in antigravity.db.

    Returns (inserted, updated) counts.
    """
    if dry_run:
        return len(papers), 0

    conn = ensure_db()
    inserted = 0
    updated = 0

    try:
        for p in papers:
            cursor = conn.execute("SELECT id, metadata FROM papers WHERE id = ?", (p["id"],))
            existing = cursor.fetchone()

            if existing:
                # Merge metadata — preserve session_ids and applied status
                old_meta = json.loads(existing[1] or "{}")
                new_meta = json.loads(p["metadata"])
                # Keep old source if not HF, add HF data
                old_meta.update(new_meta)
                conn.execute("""
                    UPDATE papers SET
                        title = COALESCE(?, title),
                        authors = COALESCE(?, authors),
                        abstract = COALESCE(?, abstract),
                        url = COALESCE(?, url),
                        relevance = MAX(relevance, ?),
                        metadata = ?
                    WHERE id = ?
                """, (
                    p["title"], p["authors"], p["abstract"], p["url"],
                    p["relevance"], json.dumps(old_meta), p["id"]
                ))
                updated += 1
            else:
                conn.execute("""
                    INSERT INTO papers (id, title, authors, abstract, url, relevance, applied, session_ids, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p["id"], p["title"], p["authors"], p["abstract"], p["url"],
                    p["relevance"], p["applied"], p["session_ids"], p["metadata"],
                    datetime.now(tz=None).isoformat()
                ))
                inserted += 1

        conn.commit()
    finally:
        conn.close()

    return inserted, updated


def get_stats() -> dict:
    """Get ingestion statistics."""
    if not DB_PATH.exists():
        return {"total": 0, "hf_papers": 0, "by_relevance": {}}

    conn = sqlite3.connect(str(DB_PATH))
    try:
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

        hf_count = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE metadata LIKE '%huggingface_daily_papers%'"
        ).fetchone()[0]

        # By relevance
        rows = conn.execute(
            "SELECT relevance, COUNT(*) FROM papers WHERE metadata LIKE '%huggingface_daily_papers%' GROUP BY relevance ORDER BY relevance DESC"
        ).fetchall()
        by_relevance = {str(r[0]): r[1] for r in rows}

        # Last ingestion
        last = conn.execute(
            "SELECT created_at FROM papers WHERE metadata LIKE '%huggingface_daily_papers%' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        # Recent dates
        recent = conn.execute(
            "SELECT DISTINCT substr(json_extract(metadata, '$.submitted_daily_at'), 1, 10) as d FROM papers WHERE metadata LIKE '%huggingface_daily_papers%' ORDER BY d DESC LIMIT 7"
        ).fetchall()

        return {
            "total_papers": total,
            "hf_papers": hf_count,
            "by_relevance": by_relevance,
            "last_ingested": last[0] if last else None,
            "recent_dates": [r[0] for r in recent if r[0]],
        }
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Daily Papers → ResearchGravity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # Fetch today's papers
  %(prog)s --date 2026-02-24       # Specific date
  %(prog)s --days 7                # Last 7 days backfill
  %(prog)s --status                # Show ingestion stats
  %(prog)s --min-relevance 4       # Only store high-relevance papers
        """
    )

    parser.add_argument("--date", type=str, help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Fetch last N days")
    parser.add_argument("--min-relevance", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Minimum relevance score to store (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without storing")
    parser.add_argument("--status", action="store_true", help="Show ingestion statistics")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Status mode
    if args.status:
        stats = get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"HuggingFace Daily Papers — Ingestion Stats")
            print(f"{'=' * 45}")
            print(f"  Total papers in DB:  {stats['total_papers']}")
            print(f"  From HF Daily:       {stats['hf_papers']}")
            print(f"  Last ingested:       {stats.get('last_ingested', 'never')}")
            print(f"  By relevance:")
            for rel, count in sorted(stats.get("by_relevance", {}).items(), reverse=True):
                print(f"    {'*' * int(rel)} ({rel}): {count}")
            if stats.get("recent_dates"):
                print(f"  Recent dates:        {', '.join(stats['recent_dates'][:5])}")
        return

    # Build date list
    dates = []
    if args.days:
        for i in range(args.days):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(d)
    elif args.date:
        dates.append(args.date)
    else:
        dates.append(None)  # None = today

    # Fetch and store
    total_inserted = 0
    total_updated = 0
    total_skipped = 0
    total_fetched = 0

    for date in dates:
        label = date or "today"
        print(f"Fetching HF Daily Papers for {label}...")

        raw_papers = fetch_daily_papers(date)
        if not raw_papers:
            print(f"  No papers returned for {label}")
            continue

        total_fetched += len(raw_papers)

        # Normalize and score
        papers = []
        for raw in raw_papers:
            normalized = normalize_paper(raw)
            if normalized["relevance"] >= args.min_relevance:
                papers.append(normalized)
            else:
                total_skipped += 1

        if not papers:
            print(f"  {len(raw_papers)} papers fetched, none above min relevance {args.min_relevance}")
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would store {len(papers)} papers:")
            for p in sorted(papers, key=lambda x: -x["relevance"])[:10]:
                print(f"    [rel={p['relevance']}] {p['id']}: {p['title'][:70]}")
            total_inserted += len(papers)
        else:
            inserted, updated = store_papers(papers)
            total_inserted += inserted
            total_updated += updated
            print(f"  {len(raw_papers)} fetched → {len(papers)} qualify → {inserted} new, {updated} updated")

    # Summary
    print(f"\nDone: {total_fetched} fetched, {total_inserted} new, {total_updated} updated, {total_skipped} below relevance threshold")

    if args.json:
        print(json.dumps({
            "fetched": total_fetched,
            "inserted": total_inserted,
            "updated": total_updated,
            "skipped": total_skipped,
        }))


if __name__ == "__main__":
    main()
