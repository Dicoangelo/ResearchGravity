#!/usr/bin/env python3
"""
Context Prefetcher - Memory injection for Claude sessions.

Loads relevant learnings, project memory, and research papers based on:
- Current project (auto-detected from working directory)
- Topic filter
- Recency filter (days)

Usage:
  python3 prefetch.py                          # Auto-detect project
  python3 prefetch.py --project os-app         # Specific project
  python3 prefetch.py --topic multi-agent      # Filter by topic
  python3 prefetch.py --days 7                 # Last 7 days only
  python3 prefetch.py --papers                 # Include arXiv papers
  python3 prefetch.py --clipboard              # Copy to clipboard
  python3 prefetch.py --inject                 # Inject into CLAUDE.md
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any


AGENT_CORE_DIR = Path.home() / ".agent-core"
PROJECTS_FILE = AGENT_CORE_DIR / "projects.json"
LEARNINGS_FILE = AGENT_CORE_DIR / "memory" / "learnings.md"
MEMORY_DIR = AGENT_CORE_DIR / "memory" / "projects"
HOME_CLAUDE_MD = Path.home() / "CLAUDE.md"

# Markers for injection
CONTEXT_START = "<!-- PREFETCHED CONTEXT START -->"
CONTEXT_END = "<!-- PREFETCHED CONTEXT END -->"


class ContextPrefetcher:
    def __init__(self):
        self.projects_data = self._load_projects()

    def _load_projects(self) -> Dict[str, Any]:
        """Load projects.json registry."""
        if PROJECTS_FILE.exists():
            try:
                return json.loads(PROJECTS_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {"projects": {}, "paper_index": {}, "topic_index": {}}

    def detect_project(self) -> Optional[str]:
        """
        Detect project from current working directory.
        Uses projects.json path matching and name inference.
        """
        cwd = str(Path.cwd()).lower()

        # Check against registered project paths
        for project_id, project in self.projects_data.get("projects", {}).items():
            project_path = project.get("path", "")
            if project_path:
                expanded = str(Path(project_path).expanduser()).lower()
                if expanded in cwd or cwd in expanded:
                    return project_id

        # Check by project name in path
        for project_id, project in self.projects_data.get("projects", {}).items():
            name = project.get("name", "").lower().replace(" ", "").replace("-", "")
            if name:
                cwd_clean = cwd.replace("/", "").replace("-", "").replace("_", "")
                if name in cwd_clean:
                    return project_id

        # Fallback: check common patterns
        if "os-app" in cwd or "osapp" in cwd:
            return "os-app"
        elif "careercoach" in cwd:
            return "careercoach"
        elif "researchgravity" in cwd:
            return "researchgravity"
        elif "metaventions" in cwd:
            return "metaventions"

        return None

    def load_learnings(
        self,
        project: Optional[str] = None,
        topic: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Load and filter learnings from learnings.md.

        Filtering logic:
        1. If project specified: filter by project mention
        2. If topic specified: filter by topic keywords
        3. If days specified: filter by date
        4. Always sort by recency
        """
        if not LEARNINGS_FILE.exists():
            return []

        content = LEARNINGS_FILE.read_text()
        learnings = []

        # Parse learnings by section (## headers)
        sections = re.split(r'\n## ', content)

        for section in sections[1:]:  # Skip header
            if not section.strip():
                continue

            # Extract date from first line
            first_line = section.split('\n')[0]
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', first_line)
            if not date_match:
                continue

            date_str = date_match.group(1)

            # Apply date filter
            if days:
                try:
                    section_date = datetime.strptime(date_str, "%Y-%m-%d")
                    cutoff = datetime.now() - timedelta(days=days)
                    if section_date < cutoff:
                        continue
                except ValueError:
                    pass

            # Apply project filter
            if project:
                project_lower = project.lower()
                if f"**project:** {project_lower}" not in section.lower() and project_lower not in section.lower():
                    # Check if project is linked in lineage
                    project_data = self.projects_data.get("projects", {}).get(project, {})
                    sessions = project_data.get("sessions", [])
                    session_in_section = any(s[:30] in section for s in sessions)
                    if not session_in_section:
                        continue

            # Apply topic filter
            if topic:
                topic_lower = topic.lower()
                if topic_lower not in section.lower():
                    continue

            learnings.append({
                "date": date_str,
                "content": "## " + section.strip(),
                "raw": section
            })

        # Sort by date (most recent first) and limit
        learnings.sort(key=lambda x: x["date"], reverse=True)
        return learnings[:limit]

    def load_project_memory(self, project_id: str) -> Optional[str]:
        """Load project-specific memory from memory/projects/[project].md"""
        memory_path = MEMORY_DIR / f"{project_id}.md"
        if memory_path.exists():
            return memory_path.read_text()

        # Check if projects.json has a memory path
        project = self.projects_data.get("projects", {}).get(project_id, {})
        mem_path = project.get("memory")
        if mem_path:
            expanded = Path(mem_path).expanduser()
            if expanded.exists():
                return expanded.read_text()

        return None

    def load_relevant_papers(
        self,
        project: Optional[str] = None,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load relevant arXiv papers from paper_index."""
        papers = []
        paper_index = self.projects_data.get("paper_index", {})

        for arxiv_id, info in paper_index.items():
            include = False

            # Include if matches project
            if project and project in info.get("projects", []):
                include = True

            # Include if matches topic
            if topic:
                topic_lower = topic.lower()
                if topic_lower in str(info).lower():
                    include = True

            # If no filters, include all
            if not project and not topic:
                include = True

            if include:
                papers.append({
                    "id": arxiv_id,
                    "projects": info.get("projects", []),
                    "sessions": info.get("sessions", []),
                    "url": f"https://arxiv.org/abs/{arxiv_id}"
                })

        # Also get papers from project's key_papers
        if project:
            project_data = self.projects_data.get("projects", {}).get(project, {})
            for paper in project_data.get("key_papers", []):
                if paper.get("id") not in [p["id"] for p in papers]:
                    papers.append({
                        "id": paper.get("id", ""),
                        "title": paper.get("title", ""),
                        "topic": paper.get("topic", ""),
                        "url": f"https://arxiv.org/abs/{paper.get('id', '')}"
                    })

        return papers

    def get_project_lineage(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get lineage data for a project."""
        project = self.projects_data.get("projects", {}).get(project_id, {})
        return project.get("lineage")

    def get_project_info(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get basic project info."""
        return self.projects_data.get("projects", {}).get(project_id)

    def format_context_block(
        self,
        project: Optional[str],
        learnings: List[Dict[str, Any]],
        project_memory: Optional[str],
        papers: List[Dict[str, Any]],
        include_papers: bool = True
    ) -> str:
        """Format all context as Claude-ready markdown block."""
        now = datetime.now().strftime("%Y-%m-%dT%H:%M")
        lines = []

        lines.append(CONTEXT_START)
        lines.append(f"<!-- Generated: {now} -->")
        lines.append("")

        # Project header
        if project:
            project_info = self.get_project_info(project)
            if project_info:
                lines.append(f"## Active Project: {project_info.get('name', project)}")
                lines.append("")

                if project_info.get("focus"):
                    lines.append(f"**Focus:** {', '.join(project_info['focus'])}")
                if project_info.get("tech_stack"):
                    lines.append(f"**Tech Stack:** {', '.join(project_info['tech_stack'])}")
                if project_info.get("status"):
                    lines.append(f"**Status:** {project_info['status']}")
                lines.append("")

        # Project memory/identity
        if project_memory:
            lines.append("### Project Identity")
            lines.append("")
            # Truncate if too long
            if len(project_memory) > 1500:
                lines.append(project_memory[:1500] + "\n\n*[truncated]*")
            else:
                lines.append(project_memory)
            lines.append("")

        # Recent learnings
        if learnings:
            days_str = f" (Last {len(learnings)} entries)"
            lines.append(f"### Recent Learnings{days_str}")
            lines.append("")

            for learning in learnings[:5]:
                # Extract just the key parts
                content = learning["content"]
                # Get first 500 chars of each section
                lines.append(content[:800] if len(content) <= 800 else content[:800] + "\n\n*[truncated]*")
                lines.append("")

        # Research papers
        if include_papers and papers:
            lines.append("### Relevant Research Papers")
            lines.append("")
            lines.append("| arXiv ID | Title/Topic | Projects |")
            lines.append("|----------|-------------|----------|")

            for paper in papers[:10]:
                arxiv_id = paper.get("id", "")
                title = paper.get("title") or paper.get("topic") or "—"
                projects = ", ".join(paper.get("projects", [])) or "—"
                url = paper.get("url", f"https://arxiv.org/abs/{arxiv_id}")
                lines.append(f"| [{arxiv_id}]({url}) | {title} | {projects} |")

            lines.append("")

        # Lineage
        if project:
            lineage = self.get_project_lineage(project)
            if lineage:
                lines.append("### Research Lineage")
                lines.append("")

                if lineage.get("research_sessions"):
                    sessions = lineage["research_sessions"][:3]
                    lines.append(f"**Research Sessions:** {', '.join(s[:40] for s in sessions)}")

                if lineage.get("features_implemented"):
                    features = lineage["features_implemented"]
                    lines.append(f"**Features Implemented:** {', '.join(features)}")

                lines.append("")

        lines.append(CONTEXT_END)

        return "\n".join(lines)

    def prefetch(
        self,
        project: Optional[str] = None,
        topic: Optional[str] = None,
        days: Optional[int] = 7,
        limit: int = 10,
        include_papers: bool = True,
        output_mode: str = "stdout"
    ) -> str:
        """Main prefetch orchestration."""

        # Auto-detect project if not specified
        if not project:
            project = self.detect_project()

        # Load components
        learnings = self.load_learnings(
            project=project,
            topic=topic,
            days=days,
            limit=limit
        )

        project_memory = None
        if project:
            project_memory = self.load_project_memory(project)

        papers = []
        if include_papers:
            papers = self.load_relevant_papers(project=project, topic=topic)

        # Format context
        context = self.format_context_block(
            project=project,
            learnings=learnings,
            project_memory=project_memory,
            papers=papers,
            include_papers=include_papers
        )

        # Output handling
        if output_mode == "clipboard":
            self._copy_to_clipboard(context)
        elif output_mode == "inject":
            self._inject_into_claude_md(context)

        return context

    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard (macOS)."""
        try:
            process = subprocess.Popen(
                ['pbcopy'],
                stdin=subprocess.PIPE,
                env={**os.environ, 'LANG': 'en_US.UTF-8'}
            )
            process.communicate(text.encode('utf-8'))
        except Exception as e:
            print(f"Warning: Could not copy to clipboard: {e}")

    def _inject_into_claude_md(self, context: str):
        """Inject context into ~/CLAUDE.md between markers."""
        if not HOME_CLAUDE_MD.exists():
            print(f"Warning: {HOME_CLAUDE_MD} not found")
            return

        content = HOME_CLAUDE_MD.read_text()

        # Check if markers exist
        if CONTEXT_START in content and CONTEXT_END in content:
            # Replace between markers
            pattern = re.escape(CONTEXT_START) + r'.*?' + re.escape(CONTEXT_END)
            new_content = re.sub(pattern, context, content, flags=re.DOTALL)
        else:
            # Add at the end of the file
            new_content = content.rstrip() + "\n\n## Dynamic Context\n\n" + context + "\n"

        HOME_CLAUDE_MD.write_text(new_content)
        print(f"✓ Context injected into {HOME_CLAUDE_MD}")


def main():
    parser = argparse.ArgumentParser(
        description="Context Prefetcher - Memory injection for Claude sessions"
    )
    parser.add_argument("--project", "-p",
                        help="Project ID to load context for")
    parser.add_argument("--topic", "-t",
                        help="Filter by topic")
    parser.add_argument("--days", "-d", type=int, default=14,
                        help="Limit to last N days (default: 14)")
    parser.add_argument("--limit", "-l", type=int, default=5,
                        help="Max learning entries to include (default: 5)")
    parser.add_argument("--papers", action="store_true",
                        help="Include relevant arXiv papers")
    parser.add_argument("--clipboard", "-c", action="store_true",
                        help="Copy to clipboard (macOS)")
    parser.add_argument("--inject", "-i", action="store_true",
                        help="Inject into ~/CLAUDE.md")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of markdown")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress informational output")

    args = parser.parse_args()

    prefetcher = ContextPrefetcher()

    # Determine output mode
    output_mode = "stdout"
    if args.clipboard:
        output_mode = "clipboard"
    elif args.inject:
        output_mode = "inject"

    result = prefetcher.prefetch(
        project=args.project,
        topic=args.topic,
        days=args.days,
        limit=args.limit,
        include_papers=args.papers,
        output_mode=output_mode
    )

    if args.json:
        # JSON output for programmatic use
        output = {
            "project": args.project or prefetcher.detect_project(),
            "context": result,
            "generated_at": datetime.now().isoformat()
        }
        print(json.dumps(output, indent=2))
    else:
        if not args.quiet:
            print(result)

        if args.clipboard:
            print("\n✓ Context copied to clipboard")


if __name__ == "__main__":
    main()
