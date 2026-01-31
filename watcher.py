#!/usr/bin/env python3
"""
ResearchGravity File Watcher
============================

Watches for new Claude Code sessions and automatically:
1. Detects new session files
2. Auto-creates linked ResearchGravity sessions
3. Infers topic from initial messages
4. Runs as a background daemon

Usage:
  python3 watcher.py start           # Start watching in foreground
  python3 watcher.py daemon          # Start as background daemon
  python3 watcher.py stop            # Stop daemon
  python3 watcher.py status          # Show daemon status
"""

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass, field


# Try to import watchdog for file system events
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not installed. Run: pip install watchdog")


# Paths
CLAUDE_DIR = Path.home() / ".claude"
CLAUDE_PROJECTS_DIR = CLAUDE_DIR / "projects"
AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
WATCHER_STATE_FILE = AGENT_CORE_DIR / "watcher_state.json"
WATCHER_PID_FILE = AGENT_CORE_DIR / "watcher.pid"
WATCHER_LOG_FILE = AGENT_CORE_DIR / "watcher.log"


@dataclass
class WatcherState:
    """Persistent watcher state."""
    started_at: Optional[str] = None
    last_activity: Optional[str] = None
    sessions_created: int = 0
    files_watched: Set[str] = field(default_factory=set)
    linked_sessions: Dict[str, str] = field(default_factory=dict)  # claude_file -> rg_session

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "last_activity": self.last_activity,
            "sessions_created": self.sessions_created,
            "files_watched": list(self.files_watched),
            "linked_sessions": self.linked_sessions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatcherState":
        return cls(
            started_at=data.get("started_at"),
            last_activity=data.get("last_activity"),
            sessions_created=data.get("sessions_created", 0),
            files_watched=set(data.get("files_watched", [])),
            linked_sessions=data.get("linked_sessions", {}),
        )


def load_state() -> WatcherState:
    """Load watcher state from disk."""
    if WATCHER_STATE_FILE.exists():
        try:
            data = json.loads(WATCHER_STATE_FILE.read_text())
            return WatcherState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            pass
    return WatcherState()


def save_state(state: WatcherState):
    """Save watcher state to disk."""
    WATCHER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHER_STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))


def log(message: str):
    """Log a message to the log file and stdout."""
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] {message}"
    print(log_line)

    try:
        with open(WATCHER_LOG_FILE, 'a') as f:
            f.write(log_line + "\n")
    except Exception:
        pass


def extract_text_from_jsonl(file_path: Path, max_lines: int = 50) -> str:
    """Extract text from first N lines of a JSONL file."""
    text_parts = []

    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    entry = json.loads(line.strip())
                    texts = extract_text_from_entry(entry)
                    text_parts.extend(texts)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return '\n'.join(text_parts)


def extract_text_from_entry(entry: Dict) -> list[str]:
    """Extract text from a JSONL entry."""
    texts = []

    if "content" in entry:
        content = entry["content"]
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])

    if "message" in entry:
        texts.extend(extract_text_from_entry(entry["message"]))

    return texts


def infer_topic(text: str) -> Optional[str]:
    """Infer research topic from initial text."""
    # Look for explicit patterns
    patterns = [
        r"research(?:ing)?\s+(?:on\s+)?['\"]?([^'\".\n]{10,60})['\"]?",
        r"investigating\s+([^.\n]{10,60})",
        r"exploring\s+([^.\n]{10,60})",
        r"looking\s+(?:into|at)\s+([^.\n]{10,60})",
        r"help\s+(?:me\s+)?(?:with|understand)\s+([^.\n]{10,60})",
        r"implement(?:ing)?\s+([^.\n]{10,60})",
        r"build(?:ing)?\s+([^.\n]{10,60})",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text[:3000], re.IGNORECASE)
        if matches:
            topic = matches[0].strip()
            if len(topic) >= 10:
                return topic[:60]

    return None


def generate_session_id(claude_file: Path, topic: str) -> str:
    """Generate unique session ID linked to Claude file."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_hash = hashlib.md5(str(claude_file).encode()).hexdigest()[:6]
    safe_topic = re.sub(r'[^a-z0-9]+', '-', topic.lower())[:25]
    return f"watch-{safe_topic}-{timestamp}-{file_hash}"


def create_linked_session(claude_file: Path, topic: str, state: WatcherState) -> str:
    """Create a new ResearchGravity session linked to Claude file."""
    session_id = generate_session_id(claude_file, topic)

    # Create session directory
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create session data
    session_data = {
        "session_id": session_id,
        "topic": topic,
        "started_at": datetime.now().isoformat(),
        "status": "active",
        "source": "watcher",
        "claude_file": str(claude_file),
        "auto_linked": True,
        "urls": [],
        "findings": [],
    }

    # Save session
    (session_dir / "session.json").write_text(json.dumps(session_data, indent=2))
    (session_dir / "urls_captured.json").write_text("[]")
    (session_dir / "findings_captured.json").write_text("[]")

    # Update state
    state.linked_sessions[str(claude_file)] = session_id
    state.sessions_created += 1
    state.last_activity = datetime.now().isoformat()
    save_state(state)

    log(f"Created linked session: {session_id}")
    log(f"  Topic: {topic}")
    log(f"  Claude file: {claude_file.name}")

    return session_id


class ClaudeSessionHandler(FileSystemEventHandler):
    """Handle Claude session file events."""

    def __init__(self, state: WatcherState):
        self.state = state
        self._processing: Set[str] = set()
        self._last_processed: Dict[str, float] = {}
        self._debounce_seconds = 5.0

    def on_created(self, event):
        """Handle new file creation."""
        if not event.is_directory and event.src_path.endswith('.jsonl'):
            self._handle_file(Path(event.src_path))

    def on_modified(self, event):
        """Handle file modification."""
        if not event.is_directory and event.src_path.endswith('.jsonl'):
            self._handle_file(Path(event.src_path))

    def _handle_file(self, file_path: Path):
        """Handle a new or modified session file."""
        file_key = str(file_path)

        # Debounce rapid events
        now = datetime.now().timestamp()
        if file_key in self._last_processed:
            if now - self._last_processed[file_key] < self._debounce_seconds:
                return
        self._last_processed[file_key] = now

        # Skip if already linked
        if file_key in self.state.linked_sessions:
            return

        # Skip if currently processing
        if file_key in self._processing:
            return
        self._processing.add(file_key)

        try:
            # Wait a moment for file to be written
            import time
            time.sleep(1)

            # Check file size (skip tiny files)
            if file_path.stat().st_size < 1000:
                return

            # Extract text and infer topic
            text = extract_text_from_jsonl(file_path)
            if not text or len(text) < 100:
                return

            topic = infer_topic(text)
            if not topic:
                topic = f"Session {file_path.stem[:20]}"

            # Create linked session
            create_linked_session(file_path, topic, self.state)

        except Exception as e:
            log(f"Error handling file {file_path}: {e}")
        finally:
            self._processing.discard(file_key)


def start_watcher(foreground: bool = True):
    """Start the file watcher."""
    if not WATCHDOG_AVAILABLE:
        print("Error: watchdog not installed. Run: pip install watchdog")
        sys.exit(1)

    if not CLAUDE_PROJECTS_DIR.exists():
        print(f"Warning: Claude projects directory not found: {CLAUDE_PROJECTS_DIR}")

    state = load_state()
    state.started_at = datetime.now().isoformat()
    save_state(state)

    # Save PID
    WATCHER_PID_FILE.write_text(str(os.getpid()))

    log(f"Starting watcher on {CLAUDE_PROJECTS_DIR}")

    handler = ClaudeSessionHandler(state)
    observer = Observer()
    observer.schedule(handler, str(CLAUDE_PROJECTS_DIR), recursive=True)

    def shutdown(signum, frame):
        log("Shutting down watcher...")
        observer.stop()
        if WATCHER_PID_FILE.exists():
            WATCHER_PID_FILE.unlink()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    observer.start()

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
    log("Watcher stopped")


def start_daemon():
    """Start watcher as a background daemon."""
    if not WATCHDOG_AVAILABLE:
        print("Error: watchdog not installed. Run: pip install watchdog")
        sys.exit(1)

    # Check if already running
    if WATCHER_PID_FILE.exists():
        try:
            pid = int(WATCHER_PID_FILE.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            print(f"Watcher already running (PID: {pid})")
            return
        except (OSError, ValueError):
            # Process not running, remove stale PID file
            WATCHER_PID_FILE.unlink()

    # Fork to background
    pid = os.fork()
    if pid > 0:
        print(f"Watcher daemon started (PID: {pid})")
        return

    # Child process
    os.setsid()
    os.chdir('/')

    # Redirect stdout/stderr to log file
    sys.stdout = open(WATCHER_LOG_FILE, 'a')
    sys.stderr = sys.stdout

    start_watcher(foreground=False)


def stop_daemon():
    """Stop the watcher daemon."""
    if not WATCHER_PID_FILE.exists():
        print("Watcher not running")
        return

    try:
        pid = int(WATCHER_PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped watcher (PID: {pid})")
        WATCHER_PID_FILE.unlink()
    except (OSError, ValueError) as e:
        print(f"Error stopping watcher: {e}")
        if WATCHER_PID_FILE.exists():
            WATCHER_PID_FILE.unlink()


def show_status():
    """Show watcher status."""
    state = load_state()

    print("=" * 50)
    print("  ResearchGravity Watcher Status")
    print("=" * 50)

    # Check if running
    running = False
    pid = None
    if WATCHER_PID_FILE.exists():
        try:
            pid = int(WATCHER_PID_FILE.read_text().strip())
            os.kill(pid, 0)
            running = True
        except (OSError, ValueError):
            pass

    if running:
        print(f"Status: RUNNING (PID: {pid})")
    else:
        print("Status: STOPPED")

    print(f"Started: {state.started_at or 'Never'}")
    print(f"Last activity: {state.last_activity or 'None'}")
    print(f"Sessions created: {state.sessions_created}")
    print(f"Linked sessions: {len(state.linked_sessions)}")

    if state.linked_sessions:
        print("\nRecent links:")
        for cf, rgs in list(state.linked_sessions.items())[-5:]:
            print(f"  {rgs[:40]}")
            print(f"    -> {Path(cf).name}")

    print()
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="ResearchGravity File Watcher"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("start", help="Start watching in foreground")
    subparsers.add_parser("daemon", help="Start as background daemon")
    subparsers.add_parser("stop", help="Stop daemon")
    subparsers.add_parser("status", help="Show daemon status")

    args = parser.parse_args()

    if args.command == "start":
        start_watcher()
    elif args.command == "daemon":
        start_daemon()
    elif args.command == "stop":
        stop_daemon()
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
