#!/usr/bin/env python3
"""
ResearchGravity Interactive REPL
================================

Real-time interactive research session management.
Replaces the 4-step manual workflow with a unified CLI.

Usage:
  python3 repl.py                    # Start interactive REPL
  python3 repl.py --resume SESSION   # Resume existing session
  python3 repl.py --status           # Show status only

Commands available in REPL:
  start <topic>     - Initialize session
  url <url>         - Log URL (auto-classify)
  finding <text>    - Capture insight
  thesis/gap/direction - Set synthesis
  status            - Show progress
  search <query>    - Semantic search
  predict           - Show predictions
  checkpoint        - Save state
  archive           - Finalize
  quit              - Exit
"""

import argparse
import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli.commands import CommandHandler
from cli.ui import SessionUI, create_ui, SessionStats


class ResearchREPL:
    """Interactive REPL for research sessions."""

    def __init__(self):
        self.handler = CommandHandler()
        self.ui = create_ui()
        self.running = False
        self._auto_capture_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the REPL."""
        await self.handler.initialize()

    async def close(self):
        """Clean up resources."""
        if self._auto_capture_task:
            self._auto_capture_task.cancel()
            try:
                await self._auto_capture_task
            except asyncio.CancelledError:
                pass

        await self.handler.close()

    async def run(self, resume_session: Optional[str] = None):
        """Run the interactive REPL."""
        self.running = True

        # Print banner
        self.ui.print_banner()

        # Resume session if specified
        if resume_session:
            await self._resume_session(resume_session)

        # Start auto-capture background task
        self._start_auto_capture()

        # Main REPL loop
        while self.running:
            try:
                # Get session ID for prompt
                session_id = None
                if self.handler.session:
                    session_id = self.handler.session.session_id

                # Get input
                line = self.ui.print_prompt(session_id)

                if not line.strip():
                    continue

                # Execute command
                result = await self.handler.execute(line)

                # Print result
                if result.message:
                    if result.success:
                        self.ui.print(result.message)
                    else:
                        self.ui.print_error(result.message)

                # Check if we should exit
                if not result.continue_repl:
                    self.running = False

            except KeyboardInterrupt:
                self.ui.print("\nUse 'quit' to exit.")
            except EOFError:
                self.running = False

        self.ui.print("Goodbye!")

    async def _resume_session(self, session_id: str):
        """Resume an existing session."""
        import json

        sessions_dir = Path.home() / ".agent-core" / "sessions"
        session_dir = sessions_dir / session_id

        if not session_dir.exists():
            self.ui.print_error(f"Session not found: {session_id}")
            return

        session_file = session_dir / "session.json"
        if not session_file.exists():
            self.ui.print_error(f"Session file not found: {session_file}")
            return

        try:
            session_data = json.loads(session_file.read_text())

            # Recreate session
            from cli.commands import ActiveSession
            self.handler.session = ActiveSession(
                session_id=session_data["session_id"],
                topic=session_data["topic"],
                started_at=session_data["started_at"],
                status=session_data.get("status", "active"),
                urls=session_data.get("urls", []),
                findings=session_data.get("findings", []),
                thesis=session_data.get("thesis"),
                gap=session_data.get("gap"),
                innovation_direction=session_data.get("innovation_direction"),
                checkpoints=session_data.get("checkpoints", []),
                project=session_data.get("project"),
                metadata=session_data.get("metadata", {}),
            )

            self.ui.print_success(f"Resumed session: {session_id}")
            self.ui.print(f"Topic: {session_data['topic']}")

        except Exception as e:
            self.ui.print_error(f"Failed to resume session: {e}")

    def _start_auto_capture(self):
        """Start background auto-capture task."""
        async def auto_capture_loop():
            """Periodically check for new URLs from Claude sessions."""
            while self.running:
                try:
                    await asyncio.sleep(60)  # Check every minute

                    if not self.handler.session:
                        continue

                    # Import auto-capture
                    try:
                        from auto_capture_v2 import find_claude_sessions, extract_urls, extract_text_from_jsonl, load_state
                    except ImportError:
                        continue

                    # Find recent sessions
                    state = load_state()
                    sessions = find_claude_sessions(hours=1)

                    for session_file in sessions[:3]:  # Check latest 3
                        text, _ = extract_text_from_jsonl(session_file)
                        if text:
                            urls = extract_urls(text, session_file, state)
                            if urls:
                                self.ui.print_auto_capture_notification(
                                    len(urls), session_file.name
                                )

                except asyncio.CancelledError:
                    break
                except Exception:
                    pass  # Silently continue

        self._auto_capture_task = asyncio.create_task(auto_capture_loop())


def show_status():
    """Show current status without entering REPL."""
    import json

    sessions_dir = Path.home() / ".agent-core" / "sessions"
    ui = create_ui()

    ui.print("=" * 60)
    ui.print("  ResearchGravity Status")
    ui.print("=" * 60)

    # Count sessions
    session_count = 0
    recent_sessions = []

    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    session_count += 1
                    try:
                        data = json.loads(session_file.read_text())
                        recent_sessions.append({
                            "id": data.get("session_id", session_dir.name),
                            "topic": data.get("topic", "Unknown"),
                            "status": data.get("status", "unknown"),
                            "started_at": data.get("started_at", ""),
                        })
                    except Exception:
                        pass

    ui.print(f"\nTotal sessions: {session_count}")

    # Show recent sessions
    recent = sorted(recent_sessions, key=lambda x: x["started_at"], reverse=True)[:5]

    if recent:
        ui.print("\nRecent sessions:")
        for s in recent:
            status_icon = "[green]active[/green]" if s["status"] == "active" else "[dim]archived[/dim]"
            ui.print(f"  {s['id'][:40]}")
            ui.print(f"    Topic: {s['topic'][:50]}")

    ui.print()
    ui.print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="ResearchGravity Interactive REPL"
    )
    parser.add_argument("--resume", help="Resume existing session by ID")
    parser.add_argument("--status", action="store_true", help="Show status only")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    repl = ResearchREPL()

    # Handle signals
    def signal_handler(sig, frame):
        repl.running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await repl.initialize()
        await repl.run(resume_session=args.resume)
    finally:
        await repl.close()


if __name__ == "__main__":
    asyncio.run(main())
