"""
Claude Adapter — Integration with Claude Code / Anthropic platforms.

This adapter handles:
- Exporting sessions from Claude Code .jsonl format
- Generating CLAUDE.md-compatible context injection
- Detecting Claude Code environment
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .base import PlatformAdapter
from ..schema import Session, CognitiveWallet, URL


CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


class ClaudeAdapter(PlatformAdapter):
    """Adapter for Claude Code / Anthropic platforms."""

    @property
    def platform_name(self) -> str:
        return "Claude"

    def detect_platform(self) -> bool:
        """Check if we're in a Claude Code environment."""
        # Check for .claude directory
        if CLAUDE_PROJECTS_DIR.exists():
            return True

        # Check for Claude Code specific env vars
        import os
        if os.environ.get("CLAUDE_CODE_VERSION"):
            return True

        return False

    def export_sessions(self) -> List[Session]:
        """Extract sessions from Claude Code .jsonl files."""
        sessions = []

        if not CLAUDE_PROJECTS_DIR.exists():
            return sessions

        # Find all .jsonl session files
        for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue

            for jsonl_file in project_dir.glob("*.jsonl"):
                session = self._parse_jsonl_session(jsonl_file)
                if session:
                    sessions.append(session)

        return sessions

    def _parse_jsonl_session(self, jsonl_path: Path) -> Optional[Session]:
        """Parse a Claude Code .jsonl file into a Session."""
        try:
            messages = []
            urls = []
            arxiv_pattern = re.compile(r'(\d{4}\.\d{4,5})')

            with open(jsonl_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "message":
                            content = entry.get("message", {}).get("content", "")
                            if isinstance(content, list):
                                content = " ".join(
                                    c.get("text", "") for c in content
                                    if isinstance(c, dict)
                                )
                            messages.append(content)

                            # Extract URLs
                            url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
                            found_urls = url_pattern.findall(content)
                            for url in found_urls:
                                tier = 1 if "arxiv.org" in url else (2 if "github.com" in url else 3)
                                urls.append(URL(
                                    url=url,
                                    tier=tier,
                                    category="research" if tier == 1 else "other",
                                    source="arXiv" if "arxiv" in url else "Web",
                                    context="",
                                    captured_at=datetime.now(),
                                ))
                    except json.JSONDecodeError:
                        continue

            if not messages:
                return None

            # Extract topic from first meaningful message
            topic = "Claude Session"
            for msg in messages[:5]:
                if len(msg) > 20:
                    topic = msg[:50].strip()
                    break

            # Extract papers
            papers = []
            full_text = " ".join(messages)
            papers = list(set(arxiv_pattern.findall(full_text)))

            return Session(
                id=jsonl_path.stem,
                topic=topic,
                date=datetime.fromtimestamp(jsonl_path.stat().st_mtime),
                findings=[],
                papers=papers,
                urls=urls,
                project=jsonl_path.parent.name,
                status="captured",
                metadata={
                    "source_file": str(jsonl_path),
                    "message_count": len(messages),
                },
            )

        except Exception as e:
            print(f"Warning: Could not parse {jsonl_path}: {e}")
            return None

    def generate_context(self, wallet: CognitiveWallet, max_tokens: int = 4000) -> str:
        """
        Generate Claude-compatible context for injection.

        This creates markdown that can be injected into CLAUDE.md or
        used as session context.
        """
        lines = [
            "## Cognitive Wallet Context",
            "",
            f"*Wallet Value: ${wallet.value_metrics.total_value:,.2f}*",
            f"*Sessions: {len(wallet.sessions)} | Concepts: {len(wallet.concepts)} | Papers: {len(wallet.papers)}*",
            "",
        ]

        # Recent sessions (most recent first)
        recent_sessions = sorted(
            wallet.sessions.values(),
            key=lambda s: s.date,
            reverse=True
        )[:5]

        if recent_sessions:
            lines.append("### Recent Research Sessions")
            lines.append("")
            for session in recent_sessions:
                date_str = session.date.strftime("%Y-%m-%d")
                lines.append(f"- **{date_str}**: {session.topic[:60]}")
                if session.papers:
                    papers_str = ", ".join(session.papers[:3])
                    lines.append(f"  - Papers: {papers_str}")
            lines.append("")

        # Key concepts
        concepts = list(wallet.concepts.values())[:10]
        if concepts:
            lines.append("### Key Concepts")
            lines.append("")
            for concept in concepts:
                type_emoji = {
                    "thesis": "📌",
                    "gap": "🔍",
                    "innovation": "💡",
                    "finding": "📝",
                }.get(concept.concept_type.value, "•")
                lines.append(f"- {type_emoji} {concept.content[:100]}")
            lines.append("")

        # Papers
        if wallet.papers:
            lines.append("### Referenced Papers")
            lines.append("")
            for arxiv_id in list(wallet.papers.keys())[:10]:
                lines.append(f"- [arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
            lines.append("")

        # Domain breakdown
        if wallet.value_metrics.domains:
            lines.append("### Domain Focus")
            lines.append("")
            for domain, weight in sorted(
                wallet.value_metrics.domains.items(),
                key=lambda x: -x[1]
            )[:3]:
                pct = weight * 100
                lines.append(f"- {domain}: {pct:.0f}%")
            lines.append("")

        return "\n".join(lines)

    def inject_into_claude_md(
        self,
        wallet: CognitiveWallet,
        claude_md_path: Optional[Path] = None,
    ) -> bool:
        """
        Inject wallet context into CLAUDE.md file.

        Uses markers to replace existing prefetched content.
        """
        if claude_md_path is None:
            claude_md_path = Path.home() / "CLAUDE.md"

        if not claude_md_path.exists():
            print(f"CLAUDE.md not found at {claude_md_path}")
            return False

        content = claude_md_path.read_text()
        context = self.generate_context(wallet)

        # Look for markers
        start_marker = "<!-- PREFETCHED CONTEXT START -->"
        end_marker = "<!-- PREFETCHED CONTEXT END -->"

        if start_marker in content and end_marker in content:
            # Replace between markers
            before = content.split(start_marker)[0]
            after = content.split(end_marker)[1]
            new_content = f"{before}{start_marker}\n{context}\n{end_marker}{after}"
        else:
            # Append at end
            new_content = f"{content}\n\n{start_marker}\n{context}\n{end_marker}\n"

        claude_md_path.write_text(new_content)
        print(f"Injected wallet context into {claude_md_path}")
        return True
