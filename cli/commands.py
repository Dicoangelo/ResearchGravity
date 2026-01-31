"""
REPL Command Handlers
Handle all interactive commands for the ResearchGravity REPL.
"""

import json
import re
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    continue_repl: bool = True


@dataclass
class ActiveSession:
    """Current active research session state."""
    session_id: str
    topic: str
    started_at: str
    status: str = "active"
    urls: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    thesis: Optional[str] = None
    gap: Optional[str] = None
    innovation_direction: Optional[str] = None
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    project: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "started_at": self.started_at,
            "status": self.status,
            "urls": self.urls,
            "findings": self.findings,
            "thesis": self.thesis,
            "gap": self.gap,
            "innovation_direction": self.innovation_direction,
            "checkpoints": self.checkpoints,
            "project": self.project,
            "metadata": self.metadata,
        }


class CommandHandler:
    """Handle REPL commands for research sessions."""

    def __init__(self):
        self.session: Optional[ActiveSession] = None
        self.storage_engine = None
        self._commands: Dict[str, Callable] = {
            "start": self.cmd_start,
            "url": self.cmd_url,
            "finding": self.cmd_finding,
            "thesis": self.cmd_thesis,
            "gap": self.cmd_gap,
            "direction": self.cmd_direction,
            "status": self.cmd_status,
            "checkpoint": self.cmd_checkpoint,
            "archive": self.cmd_archive,
            "search": self.cmd_search,
            "predict": self.cmd_predict,
            "errors": self.cmd_errors,
            "research": self.cmd_research,
            "help": self.cmd_help,
            "quit": self.cmd_quit,
            "exit": self.cmd_quit,
        }
        self.sessions_dir = Path.home() / ".agent-core" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize storage engine."""
        try:
            from storage.engine import get_engine
            self.storage_engine = await get_engine()
        except ImportError:
            print("Warning: Storage engine not available. Using file-based storage.")

    async def close(self):
        """Close storage engine."""
        if self.storage_engine:
            await self.storage_engine.close()

    def parse_command(self, line: str) -> tuple[str, str]:
        """Parse a command line into command and arguments."""
        line = line.strip()
        if not line:
            return "", ""

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        return cmd, args

    async def execute(self, line: str) -> CommandResult:
        """Execute a command line."""
        cmd, args = self.parse_command(line)

        if not cmd:
            return CommandResult(True, "")

        if cmd not in self._commands:
            return CommandResult(
                False,
                f"Unknown command: {cmd}. Type 'help' for available commands."
            )

        handler = self._commands[cmd]
        return await handler(args)

    # --- Session Commands ---

    async def cmd_start(self, args: str) -> CommandResult:
        """Start a new research session."""
        if not args:
            return CommandResult(False, "Usage: start <topic> [--project PROJECT]")

        # Parse project flag
        project = None
        topic = args
        if "--project" in args:
            match = re.search(r'--project\s+(\S+)', args)
            if match:
                project = match.group(1)
                topic = re.sub(r'--project\s+\S+', '', args).strip()

        if not topic:
            return CommandResult(False, "Please provide a topic for the session.")

        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_topic = re.sub(r'[^a-z0-9]+', '-', topic.lower())[:25]
        session_id = f"repl-{safe_topic}-{timestamp}"

        # Create session
        self.session = ActiveSession(
            session_id=session_id,
            topic=topic,
            started_at=datetime.now().isoformat(),
            project=project,
        )

        # Create session directory
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save initial state
        self._save_session()

        msg = f"Session started: {session_id}\n  Topic: {topic}"
        if project:
            msg += f"\n  Project: {project}"

        return CommandResult(True, msg, {"session_id": session_id})

    async def cmd_url(self, args: str) -> CommandResult:
        """Log a URL to the current session."""
        if not self.session:
            return CommandResult(False, "No active session. Use 'start <topic>' first.")

        if not args:
            return CommandResult(False, "Usage: url <URL> [--tier N] [--notes TEXT]")

        # Parse URL and options
        parts = args.split()
        url = parts[0]

        # Parse tier
        tier = None
        if "--tier" in args:
            match = re.search(r'--tier\s+(\d)', args)
            if match:
                tier = int(match.group(1))

        # Parse notes
        notes = ""
        if "--notes" in args:
            match = re.search(r'--notes\s+"([^"]+)"', args)
            if not match:
                match = re.search(r'--notes\s+(\S+)', args)
            if match:
                notes = match.group(1)

        # Auto-classify URL
        classification = self._classify_url(url)
        if tier:
            classification["tier"] = tier

        url_entry = {
            "id": str(uuid.uuid4())[:8],
            "url": url,
            "tier": classification["tier"],
            "category": classification["category"],
            "source": classification["source"],
            "notes": notes,
            "captured_at": datetime.now().isoformat(),
        }

        self.session.urls.append(url_entry)
        self._save_session()

        return CommandResult(
            True,
            f"Logged: {classification['source']} (Tier {classification['tier']})\n  {url}",
            url_entry
        )

    async def cmd_finding(self, args: str) -> CommandResult:
        """Capture a finding/insight."""
        if not self.session:
            return CommandResult(False, "No active session. Use 'start <topic>' first.")

        if not args:
            return CommandResult(False, "Usage: finding <text>")

        finding = {
            "id": f"finding-{uuid.uuid4().hex[:8]}",
            "content": args,
            "type": "finding",
            "confidence": 0.8,
            "captured_at": datetime.now().isoformat(),
        }

        self.session.findings.append(finding)
        self._save_session()

        return CommandResult(
            True,
            f"Finding captured (#{len(self.session.findings)})",
            finding
        )

    async def cmd_thesis(self, args: str) -> CommandResult:
        """Set the session thesis."""
        if not self.session:
            return CommandResult(False, "No active session. Use 'start <topic>' first.")

        if not args:
            if self.session.thesis:
                return CommandResult(True, f"Current thesis: {self.session.thesis}")
            return CommandResult(False, "Usage: thesis <text>")

        self.session.thesis = args
        self._save_session()

        # Also add as a finding
        finding = {
            "id": f"thesis-{uuid.uuid4().hex[:8]}",
            "content": args,
            "type": "thesis",
            "confidence": 0.9,
            "captured_at": datetime.now().isoformat(),
        }
        self.session.findings.append(finding)

        return CommandResult(True, "Thesis set", {"thesis": args})

    async def cmd_gap(self, args: str) -> CommandResult:
        """Set the identified gap."""
        if not self.session:
            return CommandResult(False, "No active session. Use 'start <topic>' first.")

        if not args:
            if self.session.gap:
                return CommandResult(True, f"Current gap: {self.session.gap}")
            return CommandResult(False, "Usage: gap <text>")

        self.session.gap = args
        self._save_session()

        finding = {
            "id": f"gap-{uuid.uuid4().hex[:8]}",
            "content": args,
            "type": "gap",
            "confidence": 0.85,
            "captured_at": datetime.now().isoformat(),
        }
        self.session.findings.append(finding)

        return CommandResult(True, "Gap set", {"gap": args})

    async def cmd_direction(self, args: str) -> CommandResult:
        """Set the innovation direction."""
        if not self.session:
            return CommandResult(False, "No active session. Use 'start <topic>' first.")

        if not args:
            if self.session.innovation_direction:
                return CommandResult(True, f"Current direction: {self.session.innovation_direction}")
            return CommandResult(False, "Usage: direction <text>")

        self.session.innovation_direction = args
        self._save_session()

        finding = {
            "id": f"direction-{uuid.uuid4().hex[:8]}",
            "content": args,
            "type": "innovation",
            "confidence": 0.85,
            "captured_at": datetime.now().isoformat(),
        }
        self.session.findings.append(finding)

        return CommandResult(True, "Innovation direction set", {"direction": args})

    async def cmd_status(self, args: str) -> CommandResult:
        """Show current session status."""
        if not self.session:
            return CommandResult(True, "No active session. Use 'start <topic>' to begin.")

        # Count URLs by tier
        tier_counts = {1: 0, 2: 0, 3: 0}
        for url in self.session.urls:
            tier = url.get("tier", 3)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Count findings by type
        finding_types = {}
        for f in self.session.findings:
            t = f.get("type", "finding")
            finding_types[t] = finding_types.get(t, 0) + 1

        status_lines = [
            f"Session: {self.session.session_id}",
            f"Topic: {self.session.topic}",
            f"Started: {self.session.started_at}",
            "",
            f"URLs: {len(self.session.urls)} total",
            f"  Tier 1: {tier_counts[1]} | Tier 2: {tier_counts[2]} | Tier 3: {tier_counts[3]}",
            "",
            f"Findings: {len(self.session.findings)} total",
        ]

        if finding_types:
            types_str = " | ".join([f"{k}: {v}" for k, v in finding_types.items()])
            status_lines.append(f"  {types_str}")

        status_lines.append("")

        # Synthesis status
        synthesis_status = []
        if self.session.thesis:
            synthesis_status.append("Thesis")
        if self.session.gap:
            synthesis_status.append("Gap")
        if self.session.innovation_direction:
            synthesis_status.append("Direction")

        if synthesis_status:
            status_lines.append(f"Synthesis: {', '.join(synthesis_status)}")
        else:
            status_lines.append("Synthesis: Not started (use thesis/gap/direction)")

        status_lines.append(f"Checkpoints: {len(self.session.checkpoints)}")

        return CommandResult(True, "\n".join(status_lines))

    async def cmd_checkpoint(self, args: str) -> CommandResult:
        """Save a checkpoint."""
        if not self.session:
            return CommandResult(False, "No active session.")

        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "urls_count": len(self.session.urls),
            "findings_count": len(self.session.findings),
            "has_thesis": bool(self.session.thesis),
            "has_gap": bool(self.session.gap),
            "has_direction": bool(self.session.innovation_direction),
            "notes": args if args else None,
        }

        self.session.checkpoints.append(checkpoint)
        self._save_session()

        return CommandResult(
            True,
            f"Checkpoint #{len(self.session.checkpoints)} saved",
            checkpoint
        )

    async def cmd_archive(self, args: str) -> CommandResult:
        """Archive the current session."""
        if not self.session:
            return CommandResult(False, "No active session to archive.")

        # Validate session
        warnings = []
        if not self.session.urls:
            warnings.append("No URLs captured")
        if not self.session.findings:
            warnings.append("No findings captured")
        if not self.session.thesis:
            warnings.append("No thesis set")

        if warnings and "--force" not in args:
            warning_str = "\n  - ".join(warnings)
            return CommandResult(
                False,
                f"Session incomplete:\n  - {warning_str}\n\nUse 'archive --force' to archive anyway."
            )

        # Update status
        self.session.status = "archived"
        self.session.metadata["archived_at"] = datetime.now().isoformat()

        # Save final state
        self._save_session()

        # Store to storage engine if available
        if self.storage_engine:
            try:
                session_data = self.session.to_dict()
                session_data["id"] = self.session.session_id
                await self.storage_engine.store_session(session_data, source="repl")

                for url in self.session.urls:
                    url["session_id"] = self.session.session_id
                await self.storage_engine.store_urls_batch(self.session.urls)

                for f in self.session.findings:
                    f["session_id"] = self.session.session_id
                await self.storage_engine.store_findings_batch(self.session.findings, source="repl")

            except Exception as e:
                return CommandResult(
                    True,
                    f"Session archived (storage sync failed: {e})\n  {self.session.session_id}"
                )

        session_id = self.session.session_id
        self.session = None

        return CommandResult(
            True,
            f"Session archived successfully\n  {session_id}",
            {"session_id": session_id}
        )

    # --- Search and Intelligence Commands ---

    async def cmd_search(self, args: str) -> CommandResult:
        """Semantic search past sessions."""
        if not args:
            return CommandResult(False, "Usage: search <query>")

        if not self.storage_engine:
            return CommandResult(False, "Storage engine not available for search.")

        try:
            results = await self.storage_engine.semantic_search(args, limit=5)

            if not any(results.values()):
                return CommandResult(True, "No results found.")

            lines = ["Search results:"]

            if results.get("findings"):
                lines.append("\nFindings:")
                for f in results["findings"][:3]:
                    content = f.get("content", "")[:100]
                    score = f.get("score", 0)
                    lines.append(f"  [{score:.2f}] {content}...")

            if results.get("sessions"):
                lines.append("\nSessions:")
                for s in results["sessions"][:3]:
                    topic = s.get("topic", "Unknown")
                    score = s.get("score", 0)
                    lines.append(f"  [{score:.2f}] {topic}")

            return CommandResult(True, "\n".join(lines))

        except Exception as e:
            return CommandResult(False, f"Search failed: {e}")

    async def cmd_predict(self, args: str) -> CommandResult:
        """Show session quality predictions."""
        try:
            from intelligence import predict_session_quality
            result = await predict_session_quality(args or "current task")
            return CommandResult(True, f"Prediction:\n{json.dumps(result, indent=2)}")
        except ImportError:
            return CommandResult(False, "Intelligence module not available.")
        except Exception as e:
            return CommandResult(False, f"Prediction failed: {e}")

    async def cmd_errors(self, args: str) -> CommandResult:
        """Show likely errors for context."""
        if not self.storage_engine:
            return CommandResult(False, "Storage engine not available.")

        query = args or (self.session.topic if self.session else "general")

        try:
            errors = await self.storage_engine.search_error_patterns(query, limit=5)

            if not errors:
                return CommandResult(True, "No error patterns found.")

            lines = ["Likely errors:"]
            for e in errors:
                error_type = e.get("error_type", "Unknown")
                solution = e.get("solution", "No solution")[:80]
                success = e.get("success_rate", 0) * 100
                lines.append(f"  [{success:.0f}%] {error_type}")
                lines.append(f"       {solution}...")

            return CommandResult(True, "\n".join(lines))

        except Exception as e:
            return CommandResult(False, f"Error lookup failed: {e}")

    async def cmd_research(self, args: str) -> CommandResult:
        """Suggest relevant research papers."""
        if not self.storage_engine:
            return CommandResult(False, "Storage engine not available.")

        query = args or (self.session.topic if self.session else "")

        if not query:
            return CommandResult(False, "Usage: research <query>")

        try:
            findings = await self.storage_engine.search_findings(
                query, limit=5, filter_type="research"
            )

            if not findings:
                findings = await self.storage_engine.search_findings(query, limit=5)

            if not findings:
                return CommandResult(True, "No relevant research found.")

            lines = ["Related research:"]
            for f in findings:
                content = f.get("content", "")[:100]
                score = f.get("score", 0)
                lines.append(f"  [{score:.2f}] {content}...")

            return CommandResult(True, "\n".join(lines))

        except Exception as e:
            return CommandResult(False, f"Research lookup failed: {e}")

    # --- Utility Commands ---

    async def cmd_help(self, args: str) -> CommandResult:
        """Show help information."""
        help_text = """
ResearchGravity REPL Commands:

SESSION MANAGEMENT
  start <topic> [--project P]  Start new research session
  status                       Show session status
  checkpoint [notes]           Save checkpoint
  archive [--force]            Archive and close session

URL LOGGING
  url <URL> [--tier N] [--notes TEXT]  Log URL (auto-classifies)

FINDINGS & SYNTHESIS
  finding <text>    Capture a finding/insight
  thesis <text>     Set session thesis
  gap <text>        Set identified gap
  direction <text>  Set innovation direction

INTELLIGENCE
  search <query>    Semantic search past sessions
  predict [task]    Session quality prediction
  errors [context]  Show likely errors
  research <query>  Find related research

OTHER
  help              Show this help
  quit/exit         Exit REPL (prompts to archive)
"""
        return CommandResult(True, help_text)

    async def cmd_quit(self, args: str) -> CommandResult:
        """Quit the REPL."""
        if self.session and self.session.status == "active":
            # Check if there's content
            if self.session.urls or self.session.findings:
                return CommandResult(
                    True,
                    "Active session with content. Use 'archive' to save or 'quit --force' to discard.",
                    continue_repl=True
                )

        return CommandResult(True, "Goodbye!", continue_repl=False)

    # --- Helper Methods ---

    def _classify_url(self, url: str) -> Dict[str, Any]:
        """Classify a URL by tier and category."""
        url_lower = url.lower()

        classifications = [
            # Tier 1: Research
            (["arxiv.org"], {"tier": 1, "category": "research", "source": "arXiv"}),
            (["huggingface.co/papers"], {"tier": 1, "category": "research", "source": "HuggingFace"}),
            (["openreview.net"], {"tier": 1, "category": "research", "source": "OpenReview"}),

            # Tier 1: Labs
            (["openai.com"], {"tier": 1, "category": "labs", "source": "OpenAI"}),
            (["anthropic.com"], {"tier": 1, "category": "labs", "source": "Anthropic"}),
            (["deepmind.google", "blog.google/technology/ai"], {"tier": 1, "category": "labs", "source": "Google AI"}),

            # Tier 1: Industry
            (["techcrunch.com"], {"tier": 1, "category": "industry", "source": "TechCrunch"}),
            (["theverge.com"], {"tier": 1, "category": "industry", "source": "The Verge"}),

            # Tier 2: GitHub
            (["github.com"], {"tier": 2, "category": "github", "source": "GitHub"}),

            # Tier 2: Benchmarks
            (["paperswithcode.com"], {"tier": 2, "category": "benchmarks", "source": "Papers With Code"}),
            (["lmarena.ai", "lmsys.org"], {"tier": 2, "category": "benchmarks", "source": "LMSYS"}),

            # Tier 2: Social
            (["twitter.com", "x.com"], {"tier": 2, "category": "social", "source": "X/Twitter"}),
            (["news.ycombinator.com"], {"tier": 2, "category": "social", "source": "Hacker News"}),

            # Tier 3: Forums
            (["lesswrong.com"], {"tier": 3, "category": "forums", "source": "LessWrong"}),
            (["substack.com"], {"tier": 3, "category": "newsletters", "source": "Substack"}),
        ]

        for patterns, classification in classifications:
            if any(p in url_lower for p in patterns):
                return classification.copy()

        return {"tier": 3, "category": "other", "source": "Web"}

    def _save_session(self):
        """Save current session to disk."""
        if not self.session:
            return

        session_dir = self.sessions_dir / self.session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save main session file
        session_file = session_dir / "session.json"
        session_file.write_text(json.dumps(self.session.to_dict(), indent=2))

        # Save URLs
        urls_file = session_dir / "urls_captured.json"
        urls_file.write_text(json.dumps(self.session.urls, indent=2))

        # Save findings
        findings_file = session_dir / "findings_captured.json"
        findings_file.write_text(json.dumps(self.session.findings, indent=2))

        # Save scratchpad (for compatibility)
        scratchpad = {
            "session_id": self.session.session_id,
            "topic": self.session.topic,
            "thesis": self.session.thesis,
            "gap": self.session.gap,
            "innovation_direction": self.session.innovation_direction,
            "urls_visited": self.session.urls,
            "findings": self.session.findings,
            "last_updated": datetime.now().isoformat(),
        }
        scratchpad_file = session_dir / "scratchpad.json"
        scratchpad_file.write_text(json.dumps(scratchpad, indent=2))
