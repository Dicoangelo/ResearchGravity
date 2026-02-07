#!/usr/bin/env python3
"""
Migrate existing ResearchGravity data → Unified Cognitive Database (PostgreSQL)

Migrates:
  - 8,156 sessions → cognitive_sessions
  - 33,232 findings → cognitive_events (with UCW layer enrichment)

Source: ~/.agent-core/storage/antigravity.db (SQLite)
Target: postgresql://localhost:5432/ucw_cognitive

Run: python3 migrate_to_cognitive_db.py [--dry-run] [--limit N]
"""

import asyncio
import hashlib
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

# Add researchgravity to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_raw.ucw_bridge import extract_layers, coherence_signature

SOURCE_DB = Path.home() / ".agent-core" / "storage" / "antigravity.db"
TARGET_DSN = "postgresql://localhost:5432/ucw_cognitive"


def load_source():
    """Load sessions and findings from source SQLite."""
    conn = sqlite3.connect(str(SOURCE_DB))
    conn.row_factory = sqlite3.Row

    sessions = conn.execute("SELECT * FROM sessions ORDER BY created_at").fetchall()
    findings = conn.execute("SELECT * FROM findings ORDER BY created_at").fetchall()

    conn.close()
    return sessions, findings


def session_to_cognitive(session) -> dict:
    """Convert a ResearchGravity session → cognitive_sessions row."""
    meta_raw = session["metadata"] or "{}"
    try:
        meta = json.loads(meta_raw)
    except json.JSONDecodeError:
        meta = {}

    # Determine platform from session ID prefix or metadata
    sid = session["id"] or ""
    if sid.startswith("chatgpt-"):
        platform = "chatgpt"
    elif sid.startswith("claude-") or sid.startswith("backfill-"):
        platform = "claude-desktop"
    else:
        platform = "claude-desktop"

    # Parse started_at to nanoseconds
    started_at = session["started_at"] or ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        started_ns = int(dt.timestamp() * 1_000_000_000)
    except Exception:
        started_ns = int(time.time() * 1_000_000_000)

    ended_ns = None
    if session["archived_at"]:
        try:
            dt = datetime.fromisoformat(session["archived_at"].replace("Z", "+00:00"))
            ended_ns = int(dt.timestamp() * 1_000_000_000)
        except Exception:
            pass

    return {
        "session_id": sid,
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "platform": platform,
        "status": session["status"] or "archived",
        "event_count": session["finding_count"] or 0,
        "turn_count": 0,
        "topics": json.dumps({"primary": session["topic"]}),
        "summary": session["topic"],
        "cognitive_mode": meta.get("cognitive_mode", "exploration"),
        "quality_score": meta.get("quality_score"),
        "metadata": meta_raw,
    }


def finding_to_event(finding, session_platform: str, session_started_ns: int) -> dict:
    """Convert a ResearchGravity finding → cognitive_events row."""
    content = finding["content"] or ""
    finding_type = finding["type"] or "general"

    # Build a pseudo-MCP message for UCW bridge enrichment
    pseudo_msg = {
        "method": "tools/call",
        "params": {"name": "log_finding", "arguments": {"finding": content[:500]}},
        "result": {"content": [{"type": "text", "text": content}]},
    }

    # Extract UCW layers
    data_layer, light_layer, instinct_layer = extract_layers(pseudo_msg, "out")

    # Override with better data from the finding itself
    light_layer["summary"] = content[:200]
    data_layer["content"] = content[:2000]
    data_layer["tokens_est"] = max(1, len(content) // 4)

    # Map finding type to intent
    type_to_intent = {
        "technical": "create",
        "research": "analyze",
        "insight": "analyze",
        "learning": "retrieve",
        "observation": "explore",
        "general": "explore",
    }
    light_layer["intent"] = type_to_intent.get(finding_type, "explore")

    # Generate coherence signature
    sig = coherence_signature(
        light_layer.get("intent", "explore"),
        light_layer.get("topic", "general"),
        session_started_ns,
        content[:1024],
    )

    # Parse confidence
    evidence_raw = finding["evidence"] or "{}"
    try:
        evidence = json.loads(evidence_raw)
    except json.JSONDecodeError:
        evidence = {}

    quality = evidence.get("confidence", finding["confidence"] or 0.5)

    # Determine cognitive mode from quality
    if quality >= 0.75:
        mode = "deep_work"
    elif quality >= 0.5:
        mode = "exploration"
    elif quality >= 0.3:
        mode = "casual"
    else:
        mode = "garbage"

    # Parse created_at to nanoseconds
    created_at = finding["created_at"] or ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        timestamp_ns = int(dt.timestamp() * 1_000_000_000)
    except Exception:
        timestamp_ns = session_started_ns

    return {
        "event_id": finding["id"] or uuid.uuid4().hex[:16],
        "session_id": finding["session_id"],
        "timestamp_ns": timestamp_ns,
        "direction": "out",
        "stage": "executed",
        "method": "finding",
        "request_id": None,
        "parent_event_id": None,
        "turn": 0,
        "raw_bytes": content.encode("utf-8")[:10000],
        "parsed_json": json.dumps({"type": finding_type, "content": content[:2000]}),
        "content_length": len(content),
        "error": None,
        "data_layer": json.dumps(data_layer),
        "light_layer": json.dumps(light_layer),
        "instinct_layer": json.dumps(instinct_layer),
        "coherence_sig": sig,
        "platform": session_platform,
        "protocol": "import",
        "quality_score": quality,
        "cognitive_mode": mode,
    }


async def migrate(dry_run=False, limit=None):
    import asyncpg

    print(f"Source: {SOURCE_DB}")
    print(f"Target: {TARGET_DSN}")
    print(f"Dry run: {dry_run}")
    print()

    # Load source data
    print("Loading source data...")
    sessions, findings = load_source()
    print(f"  Sessions: {len(sessions)}")
    print(f"  Findings: {len(findings)}")
    print()

    if limit:
        sessions = sessions[:limit]
        print(f"  Limited to {limit} sessions")

    # Build session lookup for platform info
    session_map = {}
    cognitive_sessions = []
    for s in sessions:
        cs = session_to_cognitive(s)
        cognitive_sessions.append(cs)
        session_map[cs["session_id"]] = cs

    print(f"Converted {len(cognitive_sessions)} sessions")

    # Convert findings
    print("Converting findings (with UCW enrichment)...")
    cognitive_events = []
    skipped = 0
    for i, f in enumerate(findings):
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(findings)}...")

        sid = f["session_id"]
        session_info = session_map.get(sid)
        if not session_info:
            skipped += 1
            continue

        if limit and sid not in {cs["session_id"] for cs in cognitive_sessions}:
            continue

        event = finding_to_event(f, session_info["platform"], session_info["started_ns"])
        cognitive_events.append(event)

    print(f"Converted {len(cognitive_events)} events (skipped {skipped} orphans)")
    print()

    # Platform breakdown
    platforms = {}
    for e in cognitive_events:
        p = e["platform"]
        platforms[p] = platforms.get(p, 0) + 1
    print("Platform breakdown:")
    for p, c in sorted(platforms.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c} events")
    print()

    if dry_run:
        print("DRY RUN — no data written")
        return

    # Connect to PostgreSQL
    print("Connecting to PostgreSQL...")
    pool = await asyncpg.create_pool(TARGET_DSN, min_size=2, max_size=10)

    # Check for existing data
    async with pool.acquire() as conn:
        existing_sessions = await conn.fetchval("SELECT COUNT(*) FROM cognitive_sessions")
        existing_events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        print(f"Existing data: {existing_sessions} sessions, {existing_events} events")

    # Insert sessions in batches
    print(f"\nInserting {len(cognitive_sessions)} sessions...")
    batch_size = 500
    inserted_sessions = 0
    for i in range(0, len(cognitive_sessions), batch_size):
        batch = cognitive_sessions[i:i + batch_size]
        async with pool.acquire() as conn:
            for cs in batch:
                try:
                    await conn.execute(
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
                    inserted_sessions += 1
                except Exception as e:
                    print(f"  Session error: {e}")
        if (i + batch_size) % 2000 == 0:
            print(f"  Inserted {min(i + batch_size, len(cognitive_sessions))}/{len(cognitive_sessions)} sessions")

    print(f"  Done: {inserted_sessions} sessions inserted")

    # Insert events in batches
    print(f"\nInserting {len(cognitive_events)} events...")
    inserted_events = 0
    for i in range(0, len(cognitive_events), batch_size):
        batch = cognitive_events[i:i + batch_size]
        async with pool.acquire() as conn:
            for ev in batch:
                try:
                    await conn.execute(
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
                    inserted_events += 1
                except Exception as e:
                    if "duplicate" not in str(e).lower():
                        print(f"  Event error: {e}")
        if (i + batch_size) % 5000 == 0:
            print(f"  Inserted {min(i + batch_size, len(cognitive_events))}/{len(cognitive_events)} events")

    print(f"  Done: {inserted_events} events inserted")

    # Verify
    print("\nVerification:")
    async with pool.acquire() as conn:
        total_sessions = await conn.fetchval("SELECT COUNT(*) FROM cognitive_sessions")
        total_events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        chatgpt = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE platform = 'chatgpt'"
        )
        claude = await conn.fetchval(
            "SELECT COUNT(*) FROM cognitive_events WHERE platform = 'claude-desktop'"
        )
        coherence_sigs = await conn.fetchval(
            "SELECT COUNT(DISTINCT coherence_sig) FROM cognitive_events WHERE coherence_sig IS NOT NULL"
        )

    print(f"  Total sessions: {total_sessions}")
    print(f"  Total events: {total_events}")
    print(f"  ChatGPT events: {chatgpt}")
    print(f"  Claude events: {claude}")
    print(f"  Unique coherence signatures: {coherence_sigs}")

    await pool.close()
    print("\nMigration complete!")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    limit = None
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    asyncio.run(migrate(dry_run=dry_run, limit=limit))
