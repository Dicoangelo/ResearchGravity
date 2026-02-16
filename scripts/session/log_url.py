#!/usr/bin/env python3
"""
Log a URL to the current research session with Metaventions-grade metadata.

Usage:
  python3 log_url.py URL --tier 1 --category research --relevance 5 --used
  python3 log_url.py URL --skipped --notes "reason"
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def get_local_agent_dir() -> Path:
    return Path.cwd() / ".agent"


def get_current_session() -> Optional[dict]:
    local_dir = get_local_agent_dir() / "research"
    session_file = local_dir / "session.json"
    if session_file.exists():
        return json.loads(session_file.read_text())
    return None


def detect_source_and_category(url: str) -> Tuple[str, int, str]:
    """
    Detect the source name, tier, and category from URL.
    Returns (source_name, tier, category)
    """
    url_lower = url.lower()

    # Tier 1: Research
    if "arxiv.org" in url_lower:
        return ("arXiv", 1, "research")
    if "huggingface.co/papers" in url_lower:
        return ("HuggingFace", 1, "research")
    if "openreview.net" in url_lower:
        return ("OpenReview", 1, "research")

    # Tier 1: Labs
    if "openai.com" in url_lower:
        return ("OpenAI", 1, "labs")
    if "anthropic.com" in url_lower:
        return ("Anthropic", 1, "labs")
    if "blog.google" in url_lower or "deepmind.google" in url_lower:
        return ("Google AI", 1, "labs")
    if "ai.meta.com" in url_lower:
        return ("Meta AI", 1, "labs")

    # Tier 1: Industry
    if "techcrunch.com" in url_lower:
        return ("TechCrunch", 1, "industry")
    if "theverge.com" in url_lower:
        return ("The Verge", 1, "industry")
    if "arstechnica.com" in url_lower:
        return ("Ars Technica", 1, "industry")
    if "wired.com" in url_lower:
        return ("Wired", 1, "industry")

    # Tier 2: GitHub
    if "github.com" in url_lower:
        return ("GitHub", 2, "github")

    # Tier 2: Benchmarks
    if "metr.org" in url_lower:
        return ("METR", 2, "benchmarks")
    if "arcprize.org" in url_lower:
        return ("ARC Prize", 2, "benchmarks")
    if "paperswithcode.com" in url_lower:
        return ("PapersWithCode", 2, "benchmarks")
    if "lmarena.ai" in url_lower or "lmsys.org" in url_lower:
        return ("LMSYS Arena", 2, "benchmarks")

    # Tier 2: Social
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return ("X/Twitter", 2, "social")
    if "news.ycombinator.com" in url_lower:
        return ("Hacker News", 2, "social")
    if "reddit.com" in url_lower:
        return ("Reddit", 2, "social")

    # Tier 3: Newsletters
    if "substack.com" in url_lower:
        return ("Substack", 3, "newsletters")
    if "deeplearning.ai" in url_lower:
        return ("The Batch", 3, "newsletters")

    # Tier 3: Forums
    if "lesswrong.com" in url_lower:
        return ("LessWrong", 3, "forums")
    if "alignmentforum.org" in url_lower:
        return ("Alignment Forum", 3, "forums")
    if "eaforum.org" in url_lower:
        return ("EA Forum", 3, "forums")

    # Default
    return ("Web", 2, "other")


def log_url(
    url: str,
    used: bool = False,
    skipped: bool = False,
    tier: Optional[int] = None,
    category: Optional[str] = None,
    relevance: int = 0,
    signal: str = "",
    notes: str = ""
):
    """Log a URL to the session with full metadata."""
    session = get_current_session()
    if not session:
        print("No active session found. Run init_session.py first.")
        return False

    local_dir = get_local_agent_dir() / "research"
    now = datetime.now()
    time_str = now.strftime("%H:%M")

    # Auto-detect source, tier, category if not provided
    detected_source, detected_tier, detected_category = detect_source_and_category(url)
    tier = tier or detected_tier
    category = category or detected_category

    # Determine status
    if used:
        status = "used"
        status_mark = "Yes"
    elif skipped:
        status = "skipped"
        status_mark = "No"
    else:
        status = "visited"
        status_mark = "-"

    # Log to sources.csv (new schema)
    sources_file = local_dir / "sources.csv"
    file_exists = sources_file.exists() and sources_file.stat().st_size > 0

    with open(sources_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "name", "url", "tier", "category", "signal",
                "relevance", "used", "notes", "timestamp"
            ])
        writer.writerow([
            detected_source,
            url,
            tier,
            category,
            signal,
            relevance if relevance else "",
            status,
            notes,
            now.isoformat()
        ])

    # Log to session_log.md
    session_log = local_dir / "session_log.md"

    if session_log.exists():
        content = session_log.read_text()

        # Find the URLs Visited table and append
        if "## URLs Visited" in content:
            # Append to the table
            relevance_str = str(relevance) if relevance else "-"
            signal_str = signal if signal else "-"
            row = f"| {time_str} | {tier} | {category} | [{detected_source}]({url}) | {signal_str} | {relevance_str} | {status_mark} | {notes} |\n"

            # Find where to insert (after the table header)
            lines = content.split('\n')
            insert_idx = None
            for i, line in enumerate(lines):
                if line.startswith("|---") and "Tier" not in lines[i-1] if i > 0 else False:
                    # Skip the first table header we find that's for URLs
                    if i > 0 and "Category" in lines[i-1]:
                        insert_idx = i + 1
                        break

            if insert_idx:
                lines.insert(insert_idx, row.strip())
                session_log.write_text('\n'.join(lines))
            else:
                # Fallback: append to file
                with open(session_log, "a") as f:
                    f.write(row)
        else:
            # Create table if missing
            with open(session_log, "a") as f:
                f.write("\n\n## URLs Visited\n\n")
                f.write("| Time | Tier | Category | URL | Signal | Relevance | Used | Notes |\n")
                f.write("|------|------|----------|-----|--------|-----------|------|-------|\n")
                relevance_str = str(relevance) if relevance else "-"
                signal_str = signal if signal else "-"
                f.write(f"| {time_str} | {tier} | {category} | [{detected_source}]({url}) | {signal_str} | {relevance_str} | {status_mark} | {notes} |\n")

    # Update scratchpad
    scratchpad_file = local_dir / "scratchpad.json"
    if scratchpad_file.exists():
        scratchpad = json.loads(scratchpad_file.read_text())

        # Add to urls_visited
        url_entry = {
            "url": url,
            "source": detected_source,
            "tier": tier,
            "category": category,
            "signal": signal,
            "relevance": relevance,
            "status": status,
            "notes": notes,
            "timestamp": now.isoformat()
        }
        scratchpad["urls_visited"].append(url_entry)
        scratchpad["last_updated"] = now.isoformat()

        scratchpad_file.write_text(json.dumps(scratchpad, indent=2))

    # Print confirmation
    tier_emoji = {1: "1", 2: "2", 3: "3"}.get(tier, "?")
    print(f"Logged: {url}")
    print(f"   Source: {detected_source}")
    print(f"   Tier: {tier_emoji} | Category: {category}")
    print(f"   Status: {status}")
    if relevance:
        print(f"   Relevance: {relevance}/5")
    if signal:
        print(f"   Signal: {signal}")
    if notes:
        print(f"   Notes: {notes}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Log URL to research session (Metaventions-grade)"
    )
    parser.add_argument("url", help="The URL to log")
    parser.add_argument("--used", action="store_true", help="Mark as used in output")
    parser.add_argument("--skipped", action="store_true", help="Mark as skipped")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3],
                        help="Source tier (1=primary, 2=amplifier, 3=context)")
    parser.add_argument("--category",
                        choices=[
                            "research", "labs", "industry",  # Tier 1
                            "github", "benchmarks", "social",  # Tier 2
                            "newsletters", "forums",  # Tier 3
                            "frontier", "other"
                        ],
                        help="Source category")
    parser.add_argument("--relevance", type=int, choices=[1, 2, 3, 4, 5],
                        help="Relevance score (1-5)")
    parser.add_argument("--signal", default="",
                        help="Quantitative signal (e.g., '177k stars', 'cited 50x')")
    parser.add_argument("--notes", default="", help="Additional notes")

    args = parser.parse_args()

    if args.used and args.skipped:
        print("Error: Cannot be both --used and --skipped")
        return

    log_url(
        url=args.url,
        used=args.used,
        skipped=args.skipped,
        tier=args.tier,
        category=args.category,
        relevance=args.relevance or 0,
        signal=args.signal,
        notes=args.notes
    )


if __name__ == "__main__":
    main()
