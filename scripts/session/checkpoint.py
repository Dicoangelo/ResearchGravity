#!/usr/bin/env python3
"""
Checkpoint System - Create restore points during sessions.

Creates periodic checkpoints for:
- Context preservation
- Progress tracking
- Failure recovery
- Task state management

Usage:
    python3 checkpoint.py create "description"    # Create checkpoint
    python3 checkpoint.py list                    # List checkpoints
    python3 checkpoint.py restore <checkpoint-id> # Restore to checkpoint
    python3 checkpoint.py auto                    # Auto-checkpoint current session
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
CHECKPOINTS_DIR = AGENT_CORE_DIR / "checkpoints"
TRACKER_FILE = AGENT_CORE_DIR / "session_tracker.json"


def get_active_session() -> Optional[str]:
    """Get currently active session ID."""
    if not TRACKER_FILE.exists():
        return None

    try:
        tracker = json.loads(TRACKER_FILE.read_text())
        return tracker.get("active_session_id")
    except:
        return None


def create_checkpoint(
    description: str,
    session_id: Optional[str] = None,
    tasks: list = None,
    context: dict = None
) -> dict:
    """
    Create a checkpoint for the current session.

    Args:
        description: Human-readable description of checkpoint
        session_id: Session to checkpoint (defaults to active)
        tasks: List of tasks and their states
        context: Additional context to preserve

    Returns:
        Checkpoint data dict
    """
    session_id = session_id or get_active_session()

    if not session_id:
        return {"error": "No active session"}

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create checkpoint
    now = datetime.now()
    checkpoint_id = f"cp-{now.strftime('%Y%m%d-%H%M%S')}-{session_id[:8]}"

    checkpoint = {
        "id": checkpoint_id,
        "session_id": session_id,
        "description": description,
        "created_at": now.isoformat() + "Z",
        "tasks": tasks or [],
        "context": context or {},
        "metadata": {
            "auto": False
        }
    }

    # Load session state if available
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists():
        # Capture current findings count
        findings_file = session_dir / "findings_captured.json"
        if findings_file.exists():
            try:
                findings = json.loads(findings_file.read_text())
                checkpoint["state"] = {
                    "findings_count": len(findings) if isinstance(findings, list) else 0
                }
            except:
                pass

        # Capture URL count
        urls_file = session_dir / "urls_captured.json"
        if urls_file.exists():
            try:
                urls = json.loads(urls_file.read_text())
                if "state" not in checkpoint:
                    checkpoint["state"] = {}
                checkpoint["state"]["urls_count"] = len(urls) if isinstance(urls, list) else 0
            except:
                pass

    # Save checkpoint
    checkpoint_file = CHECKPOINTS_DIR / f"{checkpoint_id}.json"
    checkpoint_file.write_text(json.dumps(checkpoint, indent=2))

    # Update tracker with last checkpoint
    if TRACKER_FILE.exists():
        try:
            tracker = json.loads(TRACKER_FILE.read_text())
            tracker["last_checkpoint"] = checkpoint_id
            tracker["last_checkpoint_at"] = checkpoint["created_at"]
            TRACKER_FILE.write_text(json.dumps(tracker, indent=2))
        except:
            pass

    return checkpoint


def list_checkpoints(session_id: Optional[str] = None, limit: int = 20) -> list:
    """List checkpoints, optionally filtered by session."""
    if not CHECKPOINTS_DIR.exists():
        return []

    checkpoints = []

    for cp_file in sorted(CHECKPOINTS_DIR.glob("cp-*.json"), reverse=True):
        try:
            cp = json.loads(cp_file.read_text())

            # Filter by session if specified
            if session_id and cp.get("session_id") != session_id:
                continue

            checkpoints.append({
                "id": cp.get("id"),
                "session_id": cp.get("session_id"),
                "description": cp.get("description"),
                "created_at": cp.get("created_at"),
                "tasks_count": len(cp.get("tasks", []))
            })

            if len(checkpoints) >= limit:
                break

        except:
            continue

    return checkpoints


def get_checkpoint(checkpoint_id: str) -> Optional[dict]:
    """Get a specific checkpoint by ID."""
    checkpoint_file = CHECKPOINTS_DIR / f"{checkpoint_id}.json"

    if not checkpoint_file.exists():
        return None

    try:
        return json.loads(checkpoint_file.read_text())
    except:
        return None


def restore_checkpoint(checkpoint_id: str) -> dict:
    """
    Get restoration context from checkpoint.

    Returns the checkpoint data formatted for reinvigoration.
    """
    checkpoint = get_checkpoint(checkpoint_id)

    if not checkpoint:
        return {"error": f"Checkpoint not found: {checkpoint_id}"}

    # Build restoration context
    context = f"""## CHECKPOINT RESTORATION: {checkpoint_id}

### Description
{checkpoint.get('description', 'No description')}

### Session
- ID: {checkpoint.get('session_id', 'Unknown')}
- Created: {checkpoint.get('created_at', 'Unknown')}
"""

    # Add tasks
    tasks = checkpoint.get("tasks", [])
    if tasks:
        context += "\n### Tasks at Checkpoint\n"
        for task in tasks:
            status = task.get("status", "unknown")
            icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}.get(status, "•")
            context += f"{icon} {task.get('content', task.get('description', 'Unknown task'))}\n"

    # Add state
    state = checkpoint.get("state", {})
    if state:
        context += "\n### State at Checkpoint\n"
        for key, value in state.items():
            context += f"- {key}: {value}\n"

    # Add custom context
    custom = checkpoint.get("context", {})
    if custom:
        context += "\n### Additional Context\n"
        for key, value in custom.items():
            context += f"- {key}: {value}\n"

    context += """
---
**Restore to this checkpoint and continue from here.**
"""

    return {
        "checkpoint": checkpoint,
        "restoration_context": context
    }


def auto_checkpoint(session_id: Optional[str] = None) -> dict:
    """Create automatic checkpoint for current state."""
    session_id = session_id or get_active_session()

    if not session_id:
        return {"error": "No active session for auto-checkpoint"}

    checkpoint = create_checkpoint(
        description=f"Auto-checkpoint at {datetime.now().strftime('%H:%M')}",
        session_id=session_id
    )

    if "error" not in checkpoint:
        checkpoint["metadata"]["auto"] = True

        # Update the saved checkpoint
        checkpoint_file = CHECKPOINTS_DIR / f"{checkpoint['id']}.json"
        checkpoint_file.write_text(json.dumps(checkpoint, indent=2))

    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint System - Create and restore session checkpoints"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create checkpoint")
    create_parser.add_argument("description", help="Checkpoint description")
    create_parser.add_argument("--session", "-s", help="Session ID (defaults to active)")

    # List command
    list_parser = subparsers.add_parser("list", help="List checkpoints")
    list_parser.add_argument("--session", "-s", help="Filter by session ID")
    list_parser.add_argument("--limit", "-l", type=int, default=20)

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore checkpoint")
    restore_parser.add_argument("checkpoint_id", help="Checkpoint ID to restore")

    # Auto command
    auto_parser = subparsers.add_parser("auto", help="Auto-checkpoint current session")
    auto_parser.add_argument("--session", "-s", help="Session ID (defaults to active)")

    args = parser.parse_args()

    if args.command == "create":
        result = create_checkpoint(
            description=args.description,
            session_id=args.session
        )

        if "error" in result:
            print(f"❌ {result['error']}")
            return 1

        print(f"✅ Checkpoint created: {result['id']}")
        print(f"   Session: {result['session_id'][:40]}")
        print(f"   Description: {result['description']}")

    elif args.command == "list":
        checkpoints = list_checkpoints(
            session_id=args.session,
            limit=args.limit
        )

        if not checkpoints:
            print("No checkpoints found")
            return 0

        print("Checkpoints:")
        print("=" * 60)

        for cp in checkpoints:
            print(f"  {cp['id']}")
            print(f"    Session: {cp['session_id'][:30]}...")
            print(f"    Description: {cp['description'][:40]}")
            print(f"    Created: {cp['created_at']}")
            print()

    elif args.command == "restore":
        result = restore_checkpoint(args.checkpoint_id)

        if "error" in result:
            print(f"❌ {result['error']}")
            return 1

        print(result["restoration_context"])

    elif args.command == "auto":
        result = auto_checkpoint(session_id=args.session)

        if "error" in result:
            print(f"❌ {result['error']}")
            return 1

        print(f"✅ Auto-checkpoint created: {result['id']}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
