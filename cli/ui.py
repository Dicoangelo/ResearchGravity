"""
Rich Terminal UI Components
Provides visual components for the ResearchGravity REPL.
"""

import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Try to import rich for enhanced terminal UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.live import Live
    from rich.text import Text
    from rich.style import Style
    from rich.box import ROUNDED, SIMPLE
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class SessionStats:
    """Session statistics for display."""
    session_id: str
    topic: str
    urls_count: int
    tier1_count: int
    tier2_count: int
    tier3_count: int
    findings_count: int
    has_thesis: bool
    has_gap: bool
    has_direction: bool
    checkpoints: int
    confidence: float = 0.0


class SessionUI:
    """Rich terminal UI for research sessions."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self._live: Optional[Live] = None

    def print(self, message: str, style: str = ""):
        """Print a message."""
        if self.console and style:
            self.console.print(message, style=style)
        else:
            print(message)

    def print_success(self, message: str):
        """Print a success message."""
        if self.console:
            self.console.print(f"[green]{message}[/green]")
        else:
            print(f"[OK] {message}")

    def print_error(self, message: str):
        """Print an error message."""
        if self.console:
            self.console.print(f"[red]{message}[/red]")
        else:
            print(f"[ERROR] {message}")

    def print_warning(self, message: str):
        """Print a warning message."""
        if self.console:
            self.console.print(f"[yellow]{message}[/yellow]")
        else:
            print(f"[WARN] {message}")

    def print_info(self, message: str):
        """Print an info message."""
        if self.console:
            self.console.print(f"[blue]{message}[/blue]")
        else:
            print(f"[INFO] {message}")

    def print_banner(self):
        """Print the REPL banner."""
        banner = r"""
 ____                              _      ____                 _ _
|  _ \ ___  ___  ___  __ _ _ __ ___| |__  / ___|_ __ __ ___   _(_) |_ _   _
| |_) / _ \/ __|/ _ \/ _` | '__/ __| '_ \| |  _| '__/ _` \ \ / / | __| | | |
|  _ <  __/\__ \  __/ (_| | | | (__| | | | |_| | | | (_| |\ V /| | |_| |_| |
|_| \_\___||___/\___|\__,_|_|  \___|_| |_|\____|_|  \__,_| \_/ |_|\__|\__, |
                                                                      |___/
        Interactive Research Session Manager v2.0
        Type 'help' for commands, 'quit' to exit
"""
        if self.console:
            self.console.print(Panel(
                banner,
                title="ResearchGravity REPL",
                border_style="blue",
                box=ROUNDED
            ))
        else:
            print(banner)
            print("-" * 60)

    def print_prompt(self, session_id: Optional[str] = None) -> str:
        """Print the command prompt and get input."""
        if session_id:
            prompt = f"rg:{session_id[:20]}> "
        else:
            prompt = "rg> "

        if self.console:
            try:
                return self.console.input(f"[bold cyan]{prompt}[/bold cyan]")
            except EOFError:
                return "quit"
        else:
            try:
                return input(prompt)
            except EOFError:
                return "quit"

    def print_session_status(self, stats: SessionStats):
        """Print a session status dashboard."""
        if not self.console:
            self._print_session_status_plain(stats)
            return

        # Create main table
        table = Table(show_header=False, box=SIMPLE, padding=(0, 1))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Session", stats.session_id)
        table.add_row("Topic", stats.topic)
        table.add_row("", "")

        # URLs breakdown
        urls_text = Text()
        urls_text.append(f"{stats.urls_count} total  ")
        urls_text.append(f"T1:{stats.tier1_count} ", style="green bold")
        urls_text.append(f"T2:{stats.tier2_count} ", style="yellow")
        urls_text.append(f"T3:{stats.tier3_count}", style="dim")
        table.add_row("URLs", urls_text)

        table.add_row("Findings", str(stats.findings_count))

        # Synthesis status
        synthesis = []
        if stats.has_thesis:
            synthesis.append("[green]Thesis[/green]")
        else:
            synthesis.append("[dim]Thesis[/dim]")
        if stats.has_gap:
            synthesis.append("[green]Gap[/green]")
        else:
            synthesis.append("[dim]Gap[/dim]")
        if stats.has_direction:
            synthesis.append("[green]Direction[/green]")
        else:
            synthesis.append("[dim]Direction[/dim]")

        table.add_row("Synthesis", " | ".join(synthesis))
        table.add_row("Checkpoints", str(stats.checkpoints))

        # Confidence bar
        confidence_pct = int(stats.confidence * 100)
        bar_filled = int(stats.confidence * 20)
        bar_empty = 20 - bar_filled
        bar = f"[green]{'█' * bar_filled}[/green][dim]{'░' * bar_empty}[/dim] {confidence_pct}%"
        table.add_row("Confidence", bar)

        self.console.print(Panel(
            table,
            title="[bold]Session Status[/bold]",
            border_style="cyan",
            box=ROUNDED
        ))

    def _print_session_status_plain(self, stats: SessionStats):
        """Print session status without rich."""
        print("=" * 50)
        print(f"Session: {stats.session_id}")
        print(f"Topic: {stats.topic}")
        print()
        print(f"URLs: {stats.urls_count} (T1:{stats.tier1_count} T2:{stats.tier2_count} T3:{stats.tier3_count})")
        print(f"Findings: {stats.findings_count}")
        print()
        synthesis = []
        if stats.has_thesis:
            synthesis.append("Thesis")
        if stats.has_gap:
            synthesis.append("Gap")
        if stats.has_direction:
            synthesis.append("Direction")
        print(f"Synthesis: {', '.join(synthesis) or 'Not started'}")
        print(f"Checkpoints: {stats.checkpoints}")
        print(f"Confidence: {int(stats.confidence * 100)}%")
        print("=" * 50)

    def print_url_logged(self, source: str, tier: int, url: str):
        """Print confirmation of URL logging."""
        tier_colors = {1: "green", 2: "yellow", 3: "dim"}
        color = tier_colors.get(tier, "white")

        if self.console:
            self.console.print(f"[{color}][T{tier}][/{color}] {source}: {url[:60]}...")
        else:
            print(f"[T{tier}] {source}: {url[:60]}...")

    def print_finding_captured(self, finding_num: int, text: str):
        """Print confirmation of finding capture."""
        preview = text[:80] + "..." if len(text) > 80 else text

        if self.console:
            self.console.print(f"[green]Finding #{finding_num}:[/green] {preview}")
        else:
            print(f"Finding #{finding_num}: {preview}")

    def print_search_results(self, results: Dict[str, List[Dict[str, Any]]]):
        """Print search results."""
        if not self.console:
            self._print_search_results_plain(results)
            return

        if not any(results.values()):
            self.console.print("[dim]No results found.[/dim]")
            return

        for category, items in results.items():
            if not items:
                continue

            table = Table(title=category.title(), box=SIMPLE)
            table.add_column("Score", style="cyan", width=6)
            table.add_column("Content")

            for item in items[:5]:
                score = item.get("score", item.get("relevance_score", 0))
                content = item.get("content", item.get("topic", ""))[:80]
                table.add_row(f"{score:.2f}", content + "...")

            self.console.print(table)

    def _print_search_results_plain(self, results: Dict[str, List[Dict[str, Any]]]):
        """Print search results without rich."""
        for category, items in results.items():
            if not items:
                continue
            print(f"\n{category.title()}:")
            for item in items[:5]:
                score = item.get("score", 0)
                content = item.get("content", "")[:80]
                print(f"  [{score:.2f}] {content}...")

    def print_predictions(self, predictions: Dict[str, Any]):
        """Print session predictions."""
        if not self.console:
            for key, value in predictions.items():
                print(f"  {key}: {value}")
            return

        table = Table(show_header=False, box=SIMPLE)
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        for key, value in predictions.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.2f}")
            else:
                table.add_row(key, str(value))

        self.console.print(Panel(
            table,
            title="[bold]Predictions[/bold]",
            border_style="magenta",
            box=ROUNDED
        ))

    def print_auto_capture_notification(self, urls_found: int, session_file: str):
        """Print notification about auto-captured URLs."""
        if self.console:
            self.console.print(
                f"[dim][Auto-capture][/dim] Found {urls_found} URLs from {session_file}",
                style="dim"
            )
        else:
            print(f"[Auto-capture] Found {urls_found} URLs from {session_file}")

    def start_spinner(self, message: str):
        """Start a progress spinner."""
        if not self.console:
            print(f"{message}...")
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=self.console
        )
        self._progress.start()
        self._task = self._progress.add_task(message)

    def stop_spinner(self):
        """Stop the progress spinner."""
        if hasattr(self, '_progress'):
            self._progress.stop()

    def print_help(self):
        """Print help in a nice format."""
        if not self.console:
            print(self._get_help_text())
            return

        self.console.print(Panel(
            self._get_help_text(),
            title="[bold]ResearchGravity REPL Help[/bold]",
            border_style="blue",
            box=ROUNDED
        ))

    def _get_help_text(self) -> str:
        return """
[bold]SESSION MANAGEMENT[/bold]
  start <topic> [--project P]  Start new research session
  status                       Show session status
  checkpoint [notes]           Save checkpoint
  archive [--force]            Archive and close session

[bold]URL LOGGING[/bold]
  url <URL> [--tier N] [--notes TEXT]  Log URL (auto-classifies)

[bold]FINDINGS & SYNTHESIS[/bold]
  finding <text>    Capture a finding/insight
  thesis <text>     Set session thesis
  gap <text>        Set identified gap
  direction <text>  Set innovation direction

[bold]INTELLIGENCE[/bold]
  search <query>    Semantic search past sessions
  predict [task]    Session quality prediction
  errors [context]  Show likely errors
  research <query>  Find related research

[bold]OTHER[/bold]
  help              Show this help
  quit/exit         Exit REPL
"""


def create_ui() -> SessionUI:
    """Create a SessionUI instance."""
    return SessionUI()
