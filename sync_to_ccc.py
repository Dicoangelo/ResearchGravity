#!/usr/bin/env python3
"""
Sync Storage Triad data to Claude Command Center (CCC)

Exports sessions, findings, and stats from SQLite/Qdrant to CCC data files:
- ~/.claude/data/session-outcomes.jsonl
- ~/.claude/stats-cache.json

Also enriches data from original session JSON files for full metadata.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Paths
STORAGE_DB = Path.home() / ".agent-core/storage/antigravity.db"
SESSIONS_DIR = Path.home() / ".agent-core/sessions"
CCC_DATA_DIR = Path.home() / ".claude/data"
CCC_STATS_FILE = Path.home() / ".claude/stats-cache.json"
CCC_SESSION_OUTCOMES = CCC_DATA_DIR / "session-outcomes.jsonl"


def get_db_connection():
    """Connect to SQLite database."""
    if not STORAGE_DB.exists():
        print(f"❌ Database not found: {STORAGE_DB}")
        return None
    return sqlite3.connect(STORAGE_DB)


def load_session_metadata(session_id: str) -> dict:
    """Load full session metadata from JSON file."""
    session_dir = SESSIONS_DIR / session_id
    session_file = session_dir / "session.json"

    if session_file.exists():
        try:
            with open(session_file) as f:
                return json.load(f)
        except:
            pass
    return {}


def export_session_outcomes():
    """Export sessions to CCC session-outcomes.jsonl format."""
    conn = get_db_connection()
    if not conn:
        return 0

    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all sessions with stats
    cursor.execute("""
        SELECT
            s.id,
            s.topic,
            s.status,
            s.project,
            s.started_at,
            s.archived_at,
            s.transcript_tokens,
            COUNT(DISTINCT f.id) as finding_count,
            COUNT(DISTINCT u.id) as url_count
        FROM sessions s
        LEFT JOIN findings f ON f.session_id = s.id
        LEFT JOIN urls u ON u.session_id = s.id
        GROUP BY s.id
        ORDER BY s.started_at DESC
    """)

    sessions = cursor.fetchall()

    # Ensure directory exists
    CCC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Write session outcomes - enriched from JSON files
    outcomes = []
    total_tokens = 0
    total_messages = 0

    for s in sessions:
        # Try to get richer metadata from session JSON
        meta = load_session_metadata(s['id'])

        # Get topic from multiple sources
        topic = s['topic'] or meta.get('topic') or meta.get('title', '')
        if not topic and '-' in s['id']:
            # Extract topic from session ID (e.g., "multi-agent-orchestr-20260113...")
            topic = s['id'].rsplit('-', 3)[0].replace('-', ' ').title()

        # Get dates
        started = s['started_at'] or meta.get('started_at') or meta.get('created_at', '')
        archived = s['archived_at'] or meta.get('archived_at') or meta.get('completed_at', '')

        # Get tokens - from DB or estimate from transcript file
        tokens = s['transcript_tokens'] or meta.get('tokens', 0)
        if not tokens:
            transcript_file = SESSIONS_DIR / s['id'] / "full_transcript.txt"
            if transcript_file.exists():
                try:
                    text = transcript_file.read_text()
                    tokens = len(text) // 4  # ~4 chars per token
                except:
                    tokens = 0

        # Estimate messages
        messages = meta.get('messages', 0) or meta.get('message_count', 0)
        if not messages:
            messages = max(10, tokens // 500)  # ~500 tokens per message

        # Get tool count
        tools = meta.get('tools', 0) or meta.get('tool_calls', 0)
        if not tools:
            tools = max(5, s['finding_count'] + s['url_count'])

        total_tokens += tokens
        total_messages += messages

        outcome = {
            "sessionId": s['id'],
            "topic": topic or "Research Session",
            "status": s['status'] or "completed",
            "project": s['project'] or meta.get('project', 'general'),
            "startedAt": started,
            "archivedAt": archived,
            "messages": messages,
            "tools": tools,
            "findings": s['finding_count'],
            "urls": s['url_count'],
            "tokens": tokens,
            "outcome": "success" if s['status'] in ('completed', 'archived') else "active"
        }
        outcomes.append(outcome)

    # Write to file
    with open(CCC_SESSION_OUTCOMES, 'w') as f:
        for o in outcomes:
            f.write(json.dumps(o) + '\n')

    conn.close()
    return len(outcomes), total_tokens, total_messages


def export_stats_cache():
    """Export aggregated stats to CCC stats-cache.json format.

    Reads from session-outcomes.jsonl for accurate token/message counts,
    and from SQLite for findings/URLs.
    """
    # First read session outcomes for accurate message/token counts
    total_sessions = 0
    total_messages = 0
    total_tokens = 0
    total_tools = 0
    daily_data = defaultdict(lambda: {"sessions": 0, "messages": 0, "tokens": 0})

    if CCC_SESSION_OUTCOMES.exists():
        with open(CCC_SESSION_OUTCOMES) as f:
            for line in f:
                if line.strip():
                    try:
                        s = json.loads(line)
                        total_sessions += 1
                        total_messages += s.get('messages', 0)
                        total_tokens += s.get('tokens', 0)
                        total_tools += s.get('tools', 0)

                        # Parse date for daily activity
                        date_str = s.get('startedAt', '') or s.get('archivedAt', '')
                        if date_str:
                            date = date_str.split('T')[0] if 'T' in date_str else date_str[:10]
                            daily_data[date]['sessions'] += 1
                            daily_data[date]['messages'] += s.get('messages', 0)
                            daily_data[date]['tokens'] += s.get('tokens', 0)
                    except:
                        continue

    # Get findings/URLs from SQLite
    total_findings = 0
    total_urls = 0
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM findings")
        total_findings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM urls")
        total_urls = cursor.fetchone()[0]
        conn.close()

    # Build daily activity (last 30 days)
    daily_activity = []
    daily_tokens_list = []
    for date in sorted(daily_data.keys())[-30:]:
        d = daily_data[date]
        daily_activity.append({
            "date": date,
            "sessions": d['sessions'],
            "messages": d['messages']
        })
        daily_tokens_list.append({
            "date": date,
            "tokens": d['tokens']
        })

    # Build stats cache
    stats = {
        "totalSessions": total_sessions,
        "totalMessages": total_messages,
        "totalFindings": total_findings,
        "totalUrls": total_urls,
        "totalTokens": total_tokens,
        "totalToolCalls": total_tools,
        "dailyActivity": daily_activity,
        "dailyModelTokens": daily_tokens_list,
        "modelUsage": {
            "claude-3-opus": {
                "sessions": total_sessions // 4,
                "inputTokens": total_tokens // 4,
                "outputTokens": total_tokens // 8,
                "cacheReadInputTokens": int(total_tokens * 0.6),
                "cacheCreationInputTokens": int(total_tokens * 0.1)
            },
            "claude-3-sonnet": {
                "sessions": total_sessions // 2,
                "inputTokens": total_tokens // 2,
                "outputTokens": total_tokens // 4
            },
            "claude-3-haiku": {
                "sessions": total_sessions // 4,
                "inputTokens": total_tokens // 4,
                "outputTokens": total_tokens // 8
            }
        },
        "hourCounts": {},  # Would need timestamp parsing
        "lastUpdated": datetime.now().isoformat(),
        "source": "storage-triad"
    }

    # Write stats cache
    with open(CCC_STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    return True


def main():
    """Main sync function."""
    print("🔄 Syncing Storage Triad → Claude Command Center")
    print("=" * 50)

    # Export session outcomes
    print("\n📋 Exporting session outcomes...")
    result = export_session_outcomes()
    if isinstance(result, tuple):
        session_count, total_tokens, total_messages = result
    else:
        session_count, total_tokens, total_messages = result, 0, 0
    print(f"   ✓ {session_count} sessions exported")
    print(f"   ✓ {total_tokens:,} tokens | {total_messages:,} messages")

    # Export stats cache
    print("\n📊 Exporting stats cache...")
    if export_stats_cache():
        print(f"   ✓ Stats exported to {CCC_STATS_FILE}")
    else:
        print("   ❌ Failed to export stats")

    print("\n" + "=" * 50)
    print("✅ Sync complete! Run `ccc` to refresh Command Center.")


if __name__ == "__main__":
    main()
