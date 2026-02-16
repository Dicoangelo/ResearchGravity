#!/usr/bin/env python3
"""
Grok Export Importer → Unified Cognitive Database
===================================================

Imports scored Grok (xAI) conversations into the PostgreSQL cognitive database
(ucw_cognitive), enriching each event with UCW semantic layers.

Source: prod-grok-backend.json (Grok data export)
Target: postgresql://localhost:5432/ucw_cognitive
Tables: cognitive_sessions, cognitive_events

Requires:
    - grok_quality_scores.json in export_dir (from grok_quality_scorer.py)
    - prod-grok-backend.json in export_dir (Grok export)

Usage:
    python3 grok_importer.py <export_dir> [--tier all] [--dry-run] [--verbose]

Flags:
    --tier <mode>  Filter by cognitive mode: all, deep_work, exploration, casual
    --dry-run      Show what would be imported without writing
    --verbose      Show per-conversation details
    --limit N      Only process first N conversations
"""

import asyncio
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from mcp_raw.ucw_bridge import coherence_signature

# ─── Configuration ──────────────────────────────────────────────────────────

TARGET_DSN = "postgresql://localhost:5432/ucw_cognitive"
PLATFORM = "grok"
PROTOCOL = "import"
BATCH_SIZE = 500

# Intent signals for light layer
_INTENT_SIGNALS = {
    "search":   ["search", "find", "look", "where", "grep"],
    "create":   ["create", "build", "write", "make", "generate", "implement"],
    "analyze":  ["analyze", "review", "check", "explain", "why", "debug"],
    "retrieve": ["get", "read", "list", "show", "fetch"],
    "explore":  ["what", "how", "tell me", "explore", "discuss"],
    "strategize": ["strategy", "market", "invest", "growth", "forecast", "trend"],
}

# Domain keywords for topic classification
_DOMAIN_KEYWORDS = {
    "strategy":     ["strategy", "market", "trend", "geopolitics", "economics", "growth", "investment"],
    "ai_agents":    ["agent", "multi-agent", "orchestrat", "coordinat", "ai", "llm", "model"],
    "coding":       ["function", "class", "import", "variable", "refactor", "debug", "code"],
    "ucw":          ["ucw", "cognitive wallet", "coherence", "sovereignty"],
    "research":     ["research", "paper", "arxiv", "finding", "hypothesis"],
    "database":     ["database", "sql", "schema", "query", "postgres"],
    "career":       ["career", "job", "resume", "interview", "skills", "role"],
    "business":     ["business", "startup", "product", "customer", "revenue"],
    "philosophy":   ["philosophy", "consciousness", "meaning", "existence", "think"],
}

# Concepts to detect for instinct layer
_CONCEPT_TARGETS = [
    "mcp", "ucw", "database", "schema", "coherence", "protocol",
    "cognitive", "semantic", "embedding", "sovereign", "platform",
    "research", "session", "capture", "agent", "orchestrat",
    "strategy", "market", "innovation", "disruption", "growth",
    "career", "architecture", "emergence",
]


# ─── Timestamp Parsing ──────────────────────────────────────────────────────

def parse_grok_timestamp(create_time) -> int:
    """Parse Grok MongoDB-style timestamp to nanoseconds.

    Grok format: {"$date": {"$numberLong": "1770445520268"}} (epoch ms)
    """
    ts_ms = 0
    if isinstance(create_time, dict):
        date_obj = create_time.get('$date', {})
        if isinstance(date_obj, dict):
            ts_ms = int(date_obj.get('$numberLong', '0'))
        elif isinstance(date_obj, (int, float)):
            ts_ms = int(date_obj)
    elif isinstance(create_time, (int, float)):
        ts_ms = int(create_time)
    elif isinstance(create_time, str):
        # ISO format fallback
        try:
            dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
            ts_ms = int(dt.timestamp() * 1000)
        except (ValueError, OverflowError):
            pass

    return ts_ms * 1_000_000  # ms → ns


def parse_iso_timestamp(ts_str: str) -> int:
    """Parse ISO timestamp string to nanoseconds."""
    if not ts_str:
        return 0
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000_000)
    except (ValueError, OverflowError):
        return 0


# ─── Layer Extraction ───────────────────────────────────────────────────────

def _classify(text: str, mapping: Dict[str, List[str]], *, default: str) -> str:
    best, best_score = default, 0
    for label, keywords in mapping.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best, best_score = label, score
    return best


def _extract_concepts(text: str) -> List[str]:
    return [t for t in _CONCEPT_TARGETS if t in text]


def build_light_layer(text: str) -> Dict:
    cl = text.lower()[:4000]
    return {
        "intent": _classify(cl, _INTENT_SIGNALS, default="explore"),
        "topic": _classify(cl, _DOMAIN_KEYWORDS, default="general"),
        "concepts": _extract_concepts(cl),
        "summary": text[:200],
    }


def build_instinct_layer(light: Dict) -> Dict:
    concepts = light.get("concepts", [])
    topic = light.get("topic", "general")

    cp = 0.0
    if topic != "general":
        cp += 0.35
    if light.get("intent") in ("create", "analyze", "search", "strategize"):
        cp += 0.25
    cp += min(len(concepts) * 0.1, 0.4)
    cp = min(cp, 1.0)

    indicators = []
    if cp > 0.7:
        indicators.append("high_coherence_potential")
    if len(concepts) >= 3:
        indicators.append("concept_cluster")
    meta_terms = {"coherence", "cognitive", "emergence", "sovereign"}
    if meta_terms & set(concepts):
        indicators.append("meta_cognitive")
    strategic_terms = {"strategy", "market", "innovation", "disruption"}
    if strategic_terms & set(concepts):
        indicators.append("strategic_insight")

    return {
        "coherence_potential": round(cp, 3),
        "emergence_indicators": indicators,
        "gut_signal": (
            "breakthrough_potential" if len(indicators) >= 2
            else "interesting" if indicators
            else "routine"
        ),
    }


# ─── Conversation Processing ───────────────────────────────────────────────

def process_conversation(
    conv_entry: Dict,
    scores: Dict,
) -> Tuple[Optional[Dict], List[Dict]]:
    """Process a Grok conversation into a session + events."""
    conv = conv_entry.get("conversation", {})
    responses = conv_entry.get("responses", [])
    conv_id = conv.get("id", "")
    title = conv.get("title", "Untitled")
    metrics = scores.get("metrics", {})

    if not responses:
        return None, []

    # Session timestamps
    started_ns = parse_iso_timestamp(conv.get("create_time", ""))
    ended_ns = parse_iso_timestamp(conv.get("modify_time", ""))
    if not started_ns:
        started_ns = int(time.time() * 1_000_000_000)

    # Count user+assistant turns
    turn_count = sum(
        1 for r in responses
        if r.get("response", {}).get("sender") in ("human", "assistant")
    )

    cognitive_mode = metrics.get("cognitive_mode", "casual")
    quality_score = metrics.get("quality_score", 0)
    purpose = metrics.get("purpose", "random")

    full_session_id = f"grok-{conv_id}"

    # Build session record
    session_dict = {
        "session_id": full_session_id,
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "platform": PLATFORM,
        "status": "archived",
        "event_count": 0,
        "turn_count": turn_count,
        "topics": json.dumps({"summary": title, "purpose": purpose}),
        "summary": title,
        "cognitive_mode": cognitive_mode,
        "quality_score": quality_score,
        "metadata": json.dumps({
            "source": "grok_export",
            "conversation_id": conv_id,
            "model": metrics.get("model", ""),
            "depth": metrics.get("depth", 0),
            "focus": metrics.get("focus", 0),
            "signal": metrics.get("signal", 0),
            "signal_strength": metrics.get("signal_strength", 0),
            "message_count": metrics.get("message_count", turn_count),
            "total_chars": metrics.get("total_chars", 0),
            "starred": conv.get("starred", False),
            "system_prompt_name": conv.get("system_prompt_name", ""),
        }),
    }

    # Process responses into events
    events = []
    turn_counter = 0

    for resp_wrapper in responses:
        resp = resp_wrapper.get("response", {})
        sender = resp.get("sender", "")
        text = resp.get("message", "")
        model = resp.get("model", "")
        resp_id = resp.get("_id", "")

        if not text or not text.strip():
            continue

        text = text.strip()

        # Parse timestamp
        ts_ns = parse_grok_timestamp(resp.get("create_time", {}))
        if not ts_ns:
            ts_ns = started_ns

        # Direction
        direction = "in" if sender == "human" else "out"
        method = "user" if sender == "human" else "assistant"

        if sender == "human":
            turn_counter += 1

        # Build UCW layers
        light_layer = build_light_layer(text)
        instinct_layer = build_instinct_layer(light_layer)

        data_layer = {
            "method": method,
            "model": model,
            "content": text[:2000],
            "tokens_est": max(1, len(text) // 4),
        }

        # Coherence signature
        sig = coherence_signature(
            light_layer.get("intent", "explore"),
            light_layer.get("topic", "general"),
            ts_ns,
            text[:1024],
        )

        # Event ID: deterministic from session + response ID
        if resp_id:
            event_id = f"grok-{conv_id[:8]}-{resp_id[:8]}"
        else:
            content_hash = hashlib.sha256(
                f"{full_session_id}:{ts_ns}:{text[:512]}".encode()
            ).hexdigest()[:12]
            event_id = f"grok-{content_hash}"

        event = {
            "event_id": event_id,
            "session_id": full_session_id,
            "timestamp_ns": ts_ns,
            "direction": direction,
            "stage": "executed",
            "method": method,
            "request_id": resp_id,
            "parent_event_id": resp.get("parent_response_id"),
            "turn": turn_counter,
            "raw_bytes": text.encode("utf-8")[:10000],
            "parsed_json": json.dumps({
                "type": method,
                "role": method,
                "model": model,
                "content_preview": text[:500],
            }),
            "content_length": len(text),
            "error": None,
            "data_layer": json.dumps(data_layer),
            "light_layer": json.dumps(light_layer),
            "instinct_layer": json.dumps(instinct_layer),
            "coherence_sig": sig,
            "platform": PLATFORM,
            "protocol": PROTOCOL,
            "quality_score": quality_score,
            "cognitive_mode": cognitive_mode,
        }
        events.append(event)

    session_dict["event_count"] = len(events)
    return session_dict, events


# ─── Database Operations ────────────────────────────────────────────────────

async def insert_sessions(pool, sessions: List[Dict]) -> int:
    inserted = 0
    for i in range(0, len(sessions), BATCH_SIZE):
        batch = sessions[i:i + BATCH_SIZE]
        async with pool.acquire() as conn:
            for cs in batch:
                try:
                    result = await conn.execute(
                        """INSERT INTO cognitive_sessions
                           (session_id, started_ns, ended_ns, platform, status,
                            event_count, turn_count, topics, summary,
                            cognitive_mode, quality_score, metadata)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                           ON CONFLICT (session_id) DO NOTHING""",
                        cs["session_id"], cs["started_ns"], cs["ended_ns"],
                        cs["platform"], cs["status"], cs["event_count"],
                        cs["turn_count"], cs["topics"], cs["summary"],
                        cs["cognitive_mode"], cs["quality_score"], cs["metadata"],
                    )
                    if "INSERT 0 1" in result:
                        inserted += 1
                except Exception as e:
                    print(f"  Session error ({cs['session_id'][:30]}): {e}")
    return inserted


async def insert_events(pool, events: List[Dict]) -> int:
    inserted = 0
    for i in range(0, len(events), BATCH_SIZE):
        batch = events[i:i + BATCH_SIZE]
        async with pool.acquire() as conn:
            for ev in batch:
                try:
                    result = await conn.execute(
                        """INSERT INTO cognitive_events
                           (event_id, session_id, timestamp_ns, direction, stage,
                            method, request_id, parent_event_id, turn,
                            raw_bytes, parsed_json, content_length, error,
                            data_layer, light_layer, instinct_layer,
                            coherence_sig, platform, protocol,
                            quality_score, cognitive_mode)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                                   $10, $11, $12, $13, $14, $15, $16,
                                   $17, $18, $19, $20, $21)
                           ON CONFLICT (event_id) DO NOTHING""",
                        ev["event_id"], ev["session_id"], ev["timestamp_ns"],
                        ev["direction"], ev["stage"], ev["method"],
                        ev["request_id"], ev["parent_event_id"], ev["turn"],
                        ev["raw_bytes"], ev["parsed_json"], ev["content_length"],
                        ev["error"], ev["data_layer"], ev["light_layer"],
                        ev["instinct_layer"], ev["coherence_sig"],
                        ev["platform"], ev["protocol"],
                        ev["quality_score"], ev["cognitive_mode"],
                    )
                    if "INSERT 0 1" in result:
                        inserted += 1
                except Exception as e:
                    if "duplicate" not in str(e).lower():
                        print(f"  Event error ({ev['event_id'][:30]}): {e}")
    return inserted


# ─── Main ───────────────────────────────────────────────────────────────────

async def run_import(
    export_path: Path,
    tier: str = "all",
    dry_run: bool = False,
    verbose: bool = False,
    limit: int = None,
):
    import asyncpg

    print("=" * 70)
    print("Grok Export Importer -> Unified Cognitive Database")
    print("=" * 70)
    print(f"  Source:   {export_path}")
    print(f"  Target:   {TARGET_DSN}")
    print(f"  Tier:     {tier}")
    print(f"  Dry run:  {dry_run}")
    print(f"  Limit:    {limit or 'all'}")
    print()

    # Load scores
    scores_file = export_path / "grok_quality_scores.json"
    if not scores_file.exists():
        print(f"grok_quality_scores.json not found in {export_path}")
        print("Run grok_quality_scorer.py first.")
        sys.exit(1)

    print("Loading quality scores...")
    scores_data = json.loads(scores_file.read_text())
    scores_index = {item["conversation_id"]: item for item in scores_data}
    print(f"  Loaded {len(scores_index)} scores")

    # Load conversations
    grok_file = export_path / "prod-grok-backend.json"
    if not grok_file.exists():
        print(f"prod-grok-backend.json not found in {export_path}")
        sys.exit(1)

    print("Loading Grok conversations...")
    with open(grok_file, "r") as f:
        data = json.load(f)

    conversations = data.get("conversations", [])
    print(f"  Loaded {len(conversations)} conversations\n")

    # Filter by tier
    if tier != "all":
        target_ids = {
            cid for cid, s in scores_index.items()
            if s.get("metrics", {}).get("cognitive_mode") == tier
        }
    else:
        # Import all non-garbage (quality >= 0.40)
        target_ids = {
            cid for cid, s in scores_index.items()
            if s.get("metrics", {}).get("import_recommended", False)
        }

    print(f"Target conversations: {len(target_ids)}\n")

    if not target_ids:
        print("Nothing to import.")
        return

    # Build lookup
    conv_lookup = {}
    for conv_entry in conversations:
        cid = conv_entry.get("conversation", {}).get("id", "")
        if cid in target_ids:
            conv_lookup[cid] = conv_entry

    if limit:
        conv_lookup = dict(list(conv_lookup.items())[:limit])
        print(f"Limited to {limit} conversations\n")

    # Process all conversations
    print("Processing conversations...")
    all_sessions = []
    all_events = []
    mode_counts = defaultdict(int)
    purpose_counts = defaultdict(int)
    errors = []

    for i, (cid, conv_entry) in enumerate(conv_lookup.items()):
        scores = scores_index.get(cid, {})
        title = conv_entry.get("conversation", {}).get("title", "Untitled")

        try:
            session, events = process_conversation(conv_entry, scores)

            if session and events:
                all_sessions.append(session)
                all_events.extend(events)
                mode_counts[session["cognitive_mode"]] += 1
                purpose_counts[scores.get("metrics", {}).get("purpose", "random")] += 1

                if verbose:
                    print(f"  [{i+1:5d}] {title[:55]:55s} -> {len(events):4d} events ({session['cognitive_mode']})")

        except Exception as e:
            errors.append({"conversation_id": cid, "title": title, "error": str(e)})
            if len(errors) <= 10:
                print(f"  ERROR on '{title[:40]}': {e}")

        if (i + 1) % 500 == 0 and not verbose:
            print(f"  Processed {i+1}/{len(conv_lookup)}... ({len(all_events)} events)")

    print(f"\n  Sessions:  {len(all_sessions)}")
    print(f"  Events:    {len(all_events)}")
    print(f"  Errors:    {len(errors)}")
    print()

    print("Cognitive mode breakdown:")
    for mode in ["deep_work", "exploration", "casual"]:
        count = mode_counts.get(mode, 0)
        print(f"  {mode:15s}: {count:6d}")
    print()

    print("Purpose breakdown:")
    for purpose, count in sorted(purpose_counts.items(), key=lambda x: -x[1]):
        print(f"  {purpose:15s}: {count:6d}")
    print()

    # Direction breakdown
    dir_counts = defaultdict(int)
    for ev in all_events:
        dir_counts[ev["direction"]] += 1
    print("Direction breakdown:")
    for d in ["in", "out"]:
        print(f"  {d:5s}: {dir_counts.get(d, 0):8d}")
    print()

    if dry_run:
        print("=" * 70)
        print("DRY RUN — no data written to database")
        print("=" * 70)

        if all_events:
            print("\nSample events (first 5):")
            for ev in all_events[:5]:
                pj = json.loads(ev["parsed_json"])
                print(f"  [{ev['direction']:3s}] {ev['method']:10s} | "
                      f"{pj.get('content_preview', '')[:80]}...")
        return

    # Connect to PostgreSQL
    print("Connecting to PostgreSQL...")
    pool = await asyncpg.create_pool(TARGET_DSN, min_size=2, max_size=10)

    # Check existing data
    async with pool.acquire() as conn:
        existing_sessions = await conn.fetchval("SELECT COUNT(*) FROM cognitive_sessions")
        existing_events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        existing_grok = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE platform = 'grok'"
        )
    print(f"  Existing: {existing_sessions} sessions, {existing_events} events ({existing_grok} grok)")
    print()

    # Insert sessions
    print(f"Inserting {len(all_sessions)} sessions...")
    inserted_sessions = await insert_sessions(pool, all_sessions)
    print(f"  Inserted: {inserted_sessions} new sessions (skipped {len(all_sessions) - inserted_sessions} duplicates)")

    # Insert events
    print(f"\nInserting {len(all_events)} events...")
    t0 = time.time()
    inserted_events = await insert_events(pool, all_events)
    elapsed = time.time() - t0
    print(f"  Inserted: {inserted_events} new events (skipped {len(all_events) - inserted_events} duplicates)")
    print(f"  Time: {elapsed:.1f}s ({len(all_events) / max(elapsed, 0.1):.0f} events/sec)")

    # Verification
    print("\nVerification:")
    async with pool.acquire() as conn:
        total_sessions = await conn.fetchval("SELECT COUNT(*) FROM cognitive_sessions")
        total_events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        grok_events = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE platform = 'grok'"
        )
        grok_sessions = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_sessions WHERE platform = 'grok'"
        )
        platform_dist = await conn.fetch(
            "SELECT platform, COUNT(*) as cnt FROM cognitive_events GROUP BY platform ORDER BY cnt DESC"
        )
        grok_modes = await conn.fetch(
            "SELECT cognitive_mode, COUNT(*) as cnt FROM cognitive_events WHERE platform = 'grok' GROUP BY cognitive_mode ORDER BY cnt DESC"
        )

    print(f"  Total sessions:       {total_sessions}")
    print(f"  Total events:         {total_events}")
    print(f"  Grok sessions:        {grok_sessions}")
    print(f"  Grok events:          {grok_events}")
    print()
    print("  Platform distribution:")
    for row in platform_dist:
        print(f"    {row['platform']:20s}: {row['cnt']}")
    print()
    print("  Grok cognitive mode distribution:")
    for row in grok_modes:
        print(f"    {row['cognitive_mode'] or 'null':15s}: {row['cnt']}")

    await pool.close()

    # Save import log
    log_path = export_path / "grok_import_log.json"
    log_data = {
        "imported_at": datetime.now().isoformat(),
        "tier": tier,
        "sessions_imported": inserted_sessions,
        "events_imported": inserted_events,
        "mode_breakdown": dict(mode_counts),
        "purpose_breakdown": dict(purpose_counts),
        "errors": errors[:50],
    }
    log_path.write_text(json.dumps(log_data, indent=2))
    print(f"\nImport log: {log_path}")

    print("\n" + "=" * 70)
    print("IMPORT COMPLETE")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 grok_importer.py <export_dir> [--tier all] [--dry-run] [--verbose] [--limit N]")
        sys.exit(1)

    export_path = Path(sys.argv[1]).expanduser()
    if not export_path.exists():
        print(f"Export path not found: {export_path}")
        sys.exit(1)

    tier = "all"
    dry_run = "--dry-run" in sys.argv
    verbose = "--verbose" in sys.argv
    limit = None

    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--tier" and i + 1 < len(args):
            tier = args[i + 1]
        elif arg == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])

    asyncio.run(run_import(export_path, tier=tier, dry_run=dry_run, verbose=verbose, limit=limit))


if __name__ == "__main__":
    main()
