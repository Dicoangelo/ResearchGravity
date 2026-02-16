#!/usr/bin/env python3
"""
Claude CLI Session Importer → Unified Cognitive Database
=========================================================

Imports Claude Code CLI session transcripts (JSONL) into the PostgreSQL
cognitive database (ucw_cognitive), enriching each event with UCW layers.

Source: ~/.claude/projects/*/*.jsonl (+ subagents/**/*.jsonl)
Target: postgresql://localhost:5432/ucw_cognitive
Tables: cognitive_sessions, cognitive_events

Usage:
    python3 import_cli_sessions.py [--dry-run] [--limit N] [--verbose]

Flags:
    --dry-run   Show what would be imported without writing to the database
    --limit N   Only process the first N JSONL files
    --verbose   Print per-file progress details

Records imported: user, assistant, system, summary
Records skipped:  progress, file-history-snapshot, queue-operation
"""

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add researchgravity to path for UCW bridge imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mcp_raw.ucw_bridge import coherence_signature

# ─── Configuration ───────────────────────────────────────────────────────────

PROJECTS_BASE = Path.home() / ".claude" / "projects"
TARGET_DSN = "postgresql://localhost:5432/ucw_cognitive"
PLATFORM = "claude-cli"
PROTOCOL = "import"
BATCH_SIZE = 500

# Record types to import (skip progress, file-history-snapshot, queue-operation)
IMPORT_TYPES = {"user", "assistant", "system", "summary"}

# Direction mapping
DIRECTION_MAP = {
    "user": "in",
    "assistant": "out",
    "system": "in",
    "summary": "out",
}

# Intent signals for light layer classification
_INTENT_SIGNALS = {
    "search":   ["search", "find", "look", "where", "grep", "glob"],
    "create":   ["create", "build", "write", "make", "generate", "implement"],
    "analyze":  ["analyze", "review", "check", "explain", "why", "debug"],
    "retrieve": ["get", "read", "list", "show", "fetch", "cat"],
    "execute":  ["call", "run", "execute", "invoke", "bash", "test"],
}

# Domain keywords for topic classification
_DOMAIN_KEYWORDS = {
    "mcp_protocol": ["mcp", "protocol", "stdio", "json-rpc", "transport"],
    "database":     ["database", "sql", "schema", "query", "postgres", "sqlite"],
    "ucw":          ["ucw", "cognitive wallet", "coherence", "sovereignty"],
    "ai_agents":    ["agent", "multi-agent", "orchestrat", "coordinat"],
    "research":     ["research", "paper", "arxiv", "finding", "hypothesis"],
    "coding":       ["function", "class", "import", "variable", "refactor", "debug"],
    "devops":       ["git", "deploy", "docker", "ci", "cd", "build", "npm"],
    "frontend":     ["react", "component", "css", "html", "vite", "next"],
}

# Concepts to detect for instinct layer
_CONCEPT_TARGETS = [
    "mcp", "ucw", "database", "schema", "coherence", "protocol",
    "cognitive", "semantic", "embedding", "sovereign", "platform",
    "research", "session", "capture", "agent", "orchestrat",
    "kernel", "routing", "memory", "prefetch",
]


# ─── Text Extraction ────────────────────────────────────────────────────────

def extract_text_content(message: Any) -> str:
    """Extract text content from a JSONL message field.

    Handles:
      - String content (user messages)
      - List of content blocks (assistant messages with text/thinking/tool_use)
      - Nested message objects with role/content structure
    """
    if not message:
        return ""

    # Direct string content field
    content = message.get("content", "") if isinstance(message, dict) else ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                texts.append(block.get("text", ""))
            elif block_type == "thinking":
                # Include thinking as it represents cognitive work
                thinking = block.get("thinking", "")
                if thinking:
                    texts.append(f"[thinking] {thinking}")
            # Skip tool_use and tool_result blocks (they're operational, not cognitive)
        return "\n".join(t for t in texts if t).strip()

    return ""


def extract_tools_used(records: List[Dict]) -> List[str]:
    """Extract unique tool names used in assistant messages."""
    tools = set()
    for rec in records:
        if rec.get("type") != "assistant":
            continue
        msg = rec.get("message", {})
        content = msg.get("content", []) if isinstance(msg, dict) else []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    if name:
                        tools.add(name)
    return sorted(tools)


def extract_files_touched(records: List[Dict]) -> List[str]:
    """Extract file paths from tool_use blocks (Read, Write, Edit, Glob)."""
    files = set()
    for rec in records:
        if rec.get("type") != "assistant":
            continue
        msg = rec.get("message", {})
        content = msg.get("content", []) if isinstance(msg, dict) else []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                inp = block.get("input", {})
                if not isinstance(inp, dict):
                    continue
                if name in ("Read", "Write", "Edit"):
                    fp = inp.get("file_path", "")
                    if fp:
                        files.add(fp)
                elif name == "Glob":
                    pattern = inp.get("pattern", "")
                    if pattern:
                        files.add(f"glob:{pattern}")
                elif name == "Bash":
                    cmd = inp.get("command", "")
                    if cmd and len(cmd) < 200:
                        files.add(f"bash:{cmd[:80]}")
    return sorted(files)[:50]  # Cap at 50 to avoid bloat


# ─── Layer Extraction (adapted from ucw_bridge for CLI context) ──────────────

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
    """Build light layer from text content."""
    cl = text.lower()[:4000]
    intent = _classify(cl, _INTENT_SIGNALS, default="explore")
    topic = _classify(cl, _DOMAIN_KEYWORDS, default="general")
    concepts = _extract_concepts(cl)
    return {
        "intent": intent,
        "topic": topic,
        "concepts": concepts,
        "summary": text[:200],
    }


def build_instinct_layer(light: Dict) -> Dict:
    """Build instinct layer from light layer."""
    concepts = light.get("concepts", [])
    topic = light.get("topic", "general")

    cp = 0.0
    if topic != "general":
        cp += 0.35
    if light.get("intent") in ("create", "analyze", "search"):
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

    return {
        "coherence_potential": round(cp, 3),
        "emergence_indicators": indicators,
        "gut_signal": (
            "breakthrough_potential" if len(indicators) >= 2
            else "interesting" if indicators
            else "routine"
        ),
    }


# ─── Session Processing ─────────────────────────────────────────────────────

def classify_cognitive_mode(message_count: int) -> str:
    """Classify cognitive mode based on session message count."""
    if message_count > 50:
        return "deep_work"
    elif message_count >= 10:
        return "exploration"
    else:
        return "casual"


def parse_jsonl_file(filepath: Path) -> List[Dict]:
    """Parse a JSONL file, returning all valid records."""
    records = []
    with open(filepath, "r", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                # Skip malformed lines silently
                pass
    return records


def parse_timestamp(ts_str: str) -> Optional[int]:
    """Parse ISO timestamp string to nanoseconds."""
    if not ts_str:
        return None
    try:
        # Handle both Z and +00:00 formats
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000_000)
    except (ValueError, OverflowError):
        return None


def process_session_file(filepath: Path) -> Tuple[Optional[Dict], List[Dict]]:
    """Process a single JSONL file into a session + events.

    Returns (session_dict, list_of_event_dicts) or (None, []) on failure.
    """
    records = parse_jsonl_file(filepath)
    if not records:
        return None, []

    # Collect session metadata from first record with sessionId
    session_id = None
    first_timestamp_ns = None
    last_timestamp_ns = None
    cwd = None
    git_branch = None
    summary_text = None

    # First pass: gather session-level info
    importable_records = []
    all_message_records = []  # for counting user+assistant messages

    for rec in records:
        rec_type = rec.get("type", "")

        # Extract session info from any record that has it
        if not session_id and rec.get("sessionId"):
            session_id = rec["sessionId"]
        if not cwd and rec.get("cwd"):
            cwd = rec["cwd"]
        if not git_branch and rec.get("gitBranch"):
            git_branch = rec["gitBranch"]

        # Track timestamps
        ts_str = rec.get("timestamp", "")
        ts_ns = parse_timestamp(ts_str)
        if ts_ns:
            if first_timestamp_ns is None or ts_ns < first_timestamp_ns:
                first_timestamp_ns = ts_ns
            if last_timestamp_ns is None or ts_ns > last_timestamp_ns:
                last_timestamp_ns = ts_ns

        # Collect summary
        if rec_type == "summary":
            summary_text = rec.get("summary", "")

        # Track message records for counting
        if rec_type in ("user", "assistant"):
            msg = rec.get("message", {})
            if isinstance(msg, dict):
                role = msg.get("role", "")
                if role in ("user", "assistant"):
                    all_message_records.append(rec)

        # Collect importable records
        if rec_type in IMPORT_TYPES:
            importable_records.append(rec)

    if not session_id:
        # Try to extract from filename (UUID pattern)
        stem = filepath.stem
        # Files like "2026-01-09_#0002_replicate-readme_id-d3f00861" have UUID after _id-
        if "_id-" in stem:
            session_id = stem.split("_id-")[-1]
        elif len(stem) == 36 and stem.count("-") == 4:
            session_id = stem
        elif "agent-" in stem:
            session_id = stem  # subagent files

    if not session_id:
        return None, []

    # Prefix session_id with cli- for namespace separation
    full_session_id = f"cli-{session_id}"

    if not first_timestamp_ns:
        first_timestamp_ns = int(time.time() * 1_000_000_000)

    # Count user+assistant messages for cognitive mode classification
    user_assistant_count = len(all_message_records)
    cognitive_mode = classify_cognitive_mode(user_assistant_count)

    # Extract tools and files for data layer
    tools_used = extract_tools_used(records)
    files_touched = extract_files_touched(records)

    # Build session record
    session_dict = {
        "session_id": full_session_id,
        "started_ns": first_timestamp_ns,
        "ended_ns": last_timestamp_ns,
        "platform": PLATFORM,
        "status": "archived",
        "event_count": 0,  # Updated after event processing
        "turn_count": user_assistant_count,
        "topics": json.dumps({"summary": summary_text or filepath.stem}),
        "summary": summary_text or filepath.stem,
        "cognitive_mode": cognitive_mode,
        "quality_score": None,  # No scoring yet
        "metadata": json.dumps({
            "source_file": str(filepath),
            "cwd": cwd,
            "git_branch": git_branch,
            "tools_used": tools_used,
            "files_touched": files_touched[:10],  # Top 10 for session metadata
            "total_records": len(records),
            "imported_records": len(importable_records),
        }),
    }

    # Second pass: convert importable records to events
    events = []
    turn_counter = 0

    for rec in importable_records:
        rec_type = rec.get("type", "")
        ts_str = rec.get("timestamp", "")
        ts_ns = parse_timestamp(ts_str) or first_timestamp_ns
        rec_uuid = rec.get("uuid", "")

        # Determine direction
        direction = DIRECTION_MAP.get(rec_type, "in")

        # Extract text content
        msg = rec.get("message", {}) if isinstance(rec.get("message"), dict) else {}

        if rec_type == "summary":
            # Summary records have different structure
            text = rec.get("summary", "")
            method = "summary"
        else:
            text = extract_text_content(msg)
            method = rec_type  # user, assistant, system

        # Skip records with no text content
        if not text:
            continue

        # Build event_id: deterministic from session + uuid
        if rec_uuid:
            event_id = f"cli-{session_id}-{rec_uuid[:8]}"
        else:
            content_hash = hashlib.sha256(
                f"{full_session_id}:{ts_ns}:{text[:512]}".encode()
            ).hexdigest()[:12]
            event_id = f"cli-{content_hash}"

        # Track turns for user messages
        if rec_type == "user":
            # Only count real user messages, not tool_results
            content = msg.get("content", "")
            if isinstance(content, str):
                turn_counter += 1

        # Build UCW layers
        light_layer = build_light_layer(text)
        instinct_layer = build_instinct_layer(light_layer)

        data_layer = {
            "method": method,
            "tools_used": tools_used if rec_type == "assistant" else [],
            "files_touched": [],
            "git_branch": git_branch or "",
            "cwd": cwd or "",
            "content": text[:2000],
            "tokens_est": max(1, len(text) // 4),
        }

        # Extract per-message file touches for assistant tool_use
        if rec_type == "assistant" and isinstance(msg.get("content"), list):
            per_msg_files = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        fp = inp.get("file_path", "")
                        if fp:
                            per_msg_files.append(fp)
            data_layer["files_touched"] = per_msg_files[:20]

        # Generate coherence signature
        sig = coherence_signature(
            light_layer.get("intent", "explore"),
            light_layer.get("topic", "general"),
            ts_ns,
            text[:1024],
        )

        event = {
            "event_id": event_id,
            "session_id": full_session_id,
            "timestamp_ns": ts_ns,
            "direction": direction,
            "stage": "executed",
            "method": method,
            "request_id": rec.get("requestId"),
            "parent_event_id": rec.get("parentUuid"),
            "turn": turn_counter,
            "raw_bytes": text.encode("utf-8")[:10000],
            "parsed_json": json.dumps({
                "type": rec_type,
                "role": msg.get("role", rec_type),
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
            "quality_score": None,
            "cognitive_mode": cognitive_mode,
        }
        events.append(event)

    session_dict["event_count"] = len(events)
    return session_dict, events


# ─── File Discovery ──────────────────────────────────────────────────────────

def discover_jsonl_files() -> List[Path]:
    """Find all JSONL files across all project folders.

    Includes main session files and subagent files.
    """
    all_files = []

    if not PROJECTS_BASE.exists():
        print(f"Projects base not found: {PROJECTS_BASE}")
        return []

    for folder in sorted(PROJECTS_BASE.iterdir()):
        if not folder.is_dir():
            continue
        for root, dirs, files in os.walk(folder):
            for fname in files:
                if fname.endswith(".jsonl"):
                    all_files.append(Path(root) / fname)

    return sorted(all_files, key=lambda p: p.stat().st_mtime)


# ─── Database Operations ─────────────────────────────────────────────────────

async def insert_sessions(pool, sessions: List[Dict]) -> int:
    """Batch insert sessions with ON CONFLICT DO NOTHING."""
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
    """Batch insert events with ON CONFLICT DO NOTHING."""
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


# ─── Main ────────────────────────────────────────────────────────────────────

async def run_import(dry_run: bool = False, limit: int = None, verbose: bool = False):
    """Main import pipeline."""
    import asyncpg

    print("=" * 70)
    print("Claude CLI Session Importer -> Unified Cognitive Database")
    print("=" * 70)
    print(f"  Source:   {PROJECTS_BASE}")
    print(f"  Target:   {TARGET_DSN}")
    print(f"  Dry run:  {dry_run}")
    print(f"  Limit:    {limit or 'all'}")
    print()

    # Discover files
    print("Discovering JSONL files...")
    all_files = discover_jsonl_files()
    print(f"  Found {len(all_files)} JSONL files")

    if limit:
        all_files = all_files[:limit]
        print(f"  Limited to {limit} files")
    print()

    # Process files
    print("Processing session files...")
    all_sessions = []
    all_events = []
    skipped_files = 0
    errors = []
    mode_counts = defaultdict(int)
    folder_counts = defaultdict(int)

    for i, filepath in enumerate(all_files):
        try:
            session, events = process_session_file(filepath)

            if session and events:
                all_sessions.append(session)
                all_events.extend(events)
                mode_counts[session["cognitive_mode"]] += 1

                # Track which project folder
                rel = filepath.relative_to(PROJECTS_BASE)
                folder = str(rel).split("/")[0]
                folder_counts[folder] += 1

                if verbose:
                    print(f"  [{i+1:5d}] {filepath.name[:60]:60s} -> {len(events):4d} events ({session['cognitive_mode']})")
            else:
                skipped_files += 1
                if verbose:
                    print(f"  [{i+1:5d}] {filepath.name[:60]:60s} -> SKIPPED (no importable content)")

        except Exception as e:
            errors.append({"file": str(filepath), "error": str(e)})
            if len(errors) <= 10:
                print(f"  ERROR on {filepath.name[:50]}: {e}")

        if (i + 1) % 200 == 0 and not verbose:
            print(f"  Processed {i+1}/{len(all_files)} files... ({len(all_events)} events)")

    print(f"\n  Processed: {len(all_files)} files")
    print(f"  Sessions:  {len(all_sessions)}")
    print(f"  Events:    {len(all_events)}")
    print(f"  Skipped:   {skipped_files} files (no content)")
    print(f"  Errors:    {len(errors)}")
    print()

    # Cognitive mode breakdown
    print("Cognitive mode breakdown:")
    for mode in ["deep_work", "exploration", "casual"]:
        count = mode_counts.get(mode, 0)
        print(f"  {mode:15s}: {count:6d} sessions")
    print()

    # Folder breakdown
    print("Project folder breakdown:")
    for folder, count in sorted(folder_counts.items(), key=lambda x: -x[1]):
        print(f"  {folder:50s}: {count:5d} sessions")
    print()

    # Direction breakdown
    dir_counts = defaultdict(int)
    method_counts = defaultdict(int)
    for ev in all_events:
        dir_counts[ev["direction"]] += 1
        method_counts[ev["method"]] += 1

    print("Event breakdown:")
    print(f"  Direction:")
    for d in ["in", "out"]:
        print(f"    {d:5s}: {dir_counts.get(d, 0):8d}")
    print(f"  Method:")
    for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"    {m:15s}: {c:8d}")
    print()

    if dry_run:
        print("=" * 70)
        print("DRY RUN -- no data written to database")
        print("=" * 70)

        # Show sample events
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
        existing_cli = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE platform = 'claude-cli'"
        )
    print(f"  Existing: {existing_sessions} sessions, {existing_events} events ({existing_cli} claude-cli)")
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
        cli_events = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE platform = 'claude-cli'"
        )
        chatgpt_events = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE platform = 'chatgpt'"
        )
        cli_sessions = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_sessions WHERE platform = 'claude-cli'"
        )
        coherence_sigs = await conn.fetchval(
            "SELECT COUNT(DISTINCT coherence_sig) FROM cognitive_events WHERE coherence_sig IS NOT NULL"
        )
        mode_dist = await conn.fetch(
            "SELECT cognitive_mode, COUNT(*) as cnt FROM cognitive_events WHERE platform = 'claude-cli' GROUP BY cognitive_mode ORDER BY cnt DESC"
        )

    print(f"  Total sessions:       {total_sessions}")
    print(f"  Total events:         {total_events}")
    print(f"  Claude CLI sessions:  {cli_sessions}")
    print(f"  Claude CLI events:    {cli_events}")
    print(f"  ChatGPT events:       {chatgpt_events}")
    print(f"  Unique coherence sigs:{coherence_sigs}")
    print(f"  CLI mode distribution:")
    for row in mode_dist:
        print(f"    {row['cognitive_mode'] or 'null':15s}: {row['cnt']}")

    await pool.close()

    # Save import log
    log_path = Path(__file__).parent / "cli_import_log.json"
    log_data = {
        "imported_at": datetime.now().isoformat(),
        "files_processed": len(all_files),
        "sessions_imported": inserted_sessions,
        "events_imported": inserted_events,
        "sessions_skipped": len(all_files) - len(all_sessions),
        "mode_breakdown": dict(mode_counts),
        "folder_breakdown": dict(folder_counts),
        "errors": errors[:50],  # Cap error log
    }
    log_path.write_text(json.dumps(log_data, indent=2))
    print(f"\nImport log: {log_path}")

    print("\n" + "=" * 70)
    print("IMPORT COMPLETE")
    print("=" * 70)


def main():
    dry_run = "--dry-run" in sys.argv
    verbose = "--verbose" in sys.argv
    limit = None
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    asyncio.run(run_import(dry_run=dry_run, limit=limit, verbose=verbose))


if __name__ == "__main__":
    main()
