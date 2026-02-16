#!/usr/bin/env python3
"""
Sync session state between CLI and Antigravity environments.

Usage:
  python3 sync_environments.py push    # Push local → global
  python3 sync_environments.py pull    # Pull global → local
  python3 sync_environments.py status  # Show sync status
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_agent_core_dir() -> Path:
    return Path.home() / ".agent-core"


def get_local_agent_dir() -> Path:
    return Path.cwd() / ".agent"


def get_current_session() -> Optional[dict]:
    """Get the current active session metadata."""
    local_dir = get_local_agent_dir() / "research"
    session_file = local_dir / "session.json"

    if session_file.exists():
        return json.loads(session_file.read_text())
    return None


def push_session():
    """Push local session state to global storage."""
    session = get_current_session()
    if not session:
        print("❌ No active session found in .agent/research/")
        return False

    local_dir = get_local_agent_dir() / "research"
    global_dir = get_agent_core_dir() / "sessions" / session["session_id"]

    global_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from local to global
    files_synced = []
    for file in local_dir.iterdir():
        if file.is_file():
            dest = global_dir / file.name
            shutil.copy2(file, dest)
            files_synced.append(file.name)
        elif file.is_dir() and file.name == "snippets":
            # Copy snippets directory
            dest_dir = global_dir / "snippets"
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(file, dest_dir)
            files_synced.append("snippets/")

    # Update sync timestamp
    session["stats"]["last_sync"] = datetime.now().isoformat()
    session["stats"]["sync_direction"] = "push"

    session_file = local_dir / "session.json"
    session_file.write_text(json.dumps(session, indent=2))

    global_session_file = global_dir / "session.json"
    global_session_file.write_text(json.dumps(session, indent=2))

    print(f"✅ Pushed session: {session['session_id']}")
    print(f"   Files synced: {', '.join(files_synced)}")
    print(f"   Destination: {global_dir}")
    return True


def pull_session(session_id: str = None):
    """Pull session state from global storage to local."""
    global_base = get_agent_core_dir() / "sessions"

    if session_id:
        global_dir = global_base / session_id
    else:
        # Find most recent session
        sessions = sorted(global_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            print("❌ No sessions found in global storage")
            return False
        global_dir = sessions[0]
        session_id = global_dir.name

    if not global_dir.exists():
        print(f"❌ Session not found: {session_id}")
        return False

    local_dir = get_local_agent_dir() / "research"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from global to local
    files_synced = []
    for file in global_dir.iterdir():
        if file.is_file():
            dest = local_dir / file.name
            shutil.copy2(file, dest)
            files_synced.append(file.name)
        elif file.is_dir() and file.name == "snippets":
            dest_dir = local_dir / "snippets"
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(file, dest_dir)
            files_synced.append("snippets/")

    # Update sync timestamp
    session_file = local_dir / "session.json"
    if session_file.exists():
        session = json.loads(session_file.read_text())
        session["stats"]["last_sync"] = datetime.now().isoformat()
        session["stats"]["sync_direction"] = "pull"
        session_file.write_text(json.dumps(session, indent=2))

    print(f"✅ Pulled session: {session_id}")
    print(f"   Files synced: {', '.join(files_synced)}")
    print(f"   Destination: {local_dir}")
    return True


def sync_status():
    """Show current sync status."""
    local_session = get_current_session()
    global_base = get_agent_core_dir() / "sessions"

    print("📊 Sync Status")
    print("=" * 50)

    # Local status
    print("\n📁 Local (.agent/research/):")
    if local_session:
        print(f"   Session: {local_session['session_id']}")
        print(f"   Topic: {local_session['topic']}")
        print(f"   Status: {local_session['status']}")
        last_sync = local_session["stats"].get("last_sync", "Never")
        print(f"   Last sync: {last_sync}")
    else:
        print("   No active session")

    # Global status
    print(f"\n🌐 Global ({global_base}):")
    if global_base.exists():
        sessions = list(global_base.iterdir())
        print(f"   Total sessions: {len(sessions)}")

        if sessions:
            recent = sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)[:3]
            print("   Recent sessions:")
            for s in recent:
                session_file = s / "session.json"
                if session_file.exists():
                    meta = json.loads(session_file.read_text())
                    print(f"     - {meta['session_id']}: {meta['topic']}")
    else:
        print("   Global storage not initialized")
        print(f"   Run: mkdir -p {global_base}")

    # Memory status
    memory_dir = get_agent_core_dir() / "memory"
    print(f"\n💾 Memory ({memory_dir}):")
    if memory_dir.exists():
        for mem_file in memory_dir.glob("*.md"):
            lines = len(mem_file.read_text().splitlines())
            print(f"   {mem_file.name}: {lines} lines")
    else:
        print("   Memory not initialized")


def main():
    parser = argparse.ArgumentParser(description="Sync agent sessions between environments")
    parser.add_argument("command", choices=["push", "pull", "status"],
                        help="Sync command")
    parser.add_argument("--session", help="Specific session ID (for pull)")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite without confirmation")

    args = parser.parse_args()

    if args.command == "push":
        push_session()
    elif args.command == "pull":
        pull_session(args.session)
    elif args.command == "status":
        sync_status()


if __name__ == "__main__":
    main()
