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

# Co-evolution integration
CLAUDE_DIR = Path.home() / ".claude"
STATS_CACHE = CLAUDE_DIR / "stats-cache.json"
DETECTED_PATTERNS = CLAUDE_DIR / "kernel" / "detected-patterns.json"
COEVO_CONFIG = CLAUDE_DIR / "kernel" / "coevo-config.json"

# Markers for injection
CONTEXT_START = "<!-- PREFETCHED CONTEXT START -->"
CONTEXT_END = "<!-- PREFETCHED CONTEXT END -->"

# Pattern-specific research foundations for recursive novelty
PATTERN_RESEARCH_PAPERS = {
    "debugging": {
        "papers": ["2512.20845", "2506.08410"],  # MAR, multi-agent reflexion
        "focus": ["error patterns", "root cause analysis", "fix verification"],
        "suggested_tools": ["/debug", "git diff"]
    },
    "research": {
        "papers": ["2511.16931", "2512.12686", "2512.12818"],  # OmniScientist, Memoria, Hindsight
        "focus": ["papers", "learnings", "thesis gaps", "synthesis"],
        "suggested_tools": ["log_url.py", "archive_session.py"]
    },
    "refactoring": {
        "papers": ["2505.02888", "2503.00735"],  # LADDER - recursive refinement
        "focus": ["code patterns", "test coverage", "before/after"],
        "suggested_tools": ["/refactor", "npm test"]
    },
    "testing": {
        "papers": ["2510.24797", "2601.03511"],  # IntroLM - self-evaluation
        "focus": ["coverage", "edge cases", "test patterns"],
        "suggested_tools": ["/test", "npm run test:coverage"]
    },
    "architecture": {
        "papers": ["2507.14241", "2501.12689"],  # Promptomatix, IC-Cache
        "focus": ["system design", "component boundaries", "trade-offs"],
        "suggested_tools": ["/arch", "prefetch --papers"]
    },
    "performance": {
        "papers": ["2501.12689", "2502.00299"],  # IC-Cache, ChunkKV
        "focus": ["profiling", "bottlenecks", "optimization"],
        "suggested_tools": ["npm run build", "lighthouse"]
    },
    "deployment": {
        "papers": [],
        "focus": ["CI/CD", "production checks", "rollback"],
        "suggested_tools": ["/pr", "git status"]
    },
    "learning": {
        "papers": ["2512.12686", "2512.12818"],  # Memoria, Hindsight
        "focus": ["concepts", "examples", "documentation"],
        "suggested_tools": ["prefetch --topic"]
    }
}


class ContextPrefetcher:
    def __init__(self):
        self.projects_data = self._load_projects()
        self.stats_cache = self._load_stats_cache()
        self.detected_patterns = self._load_detected_patterns()
        self.coevo_config = self._load_coevo_config()

    def _load_stats_cache(self) -> Dict[str, Any]:
        """Load stats-cache.json for temporal analysis."""
        if STATS_CACHE.exists():
            try:
                return json.loads(STATS_CACHE.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _load_detected_patterns(self) -> Dict[str, Any]:
        """Load detected-patterns.json."""
        if DETECTED_PATTERNS.exists():
            try:
                return json.loads(DETECTED_PATTERNS.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _load_coevo_config(self) -> Dict[str, Any]:
        """Load co-evolution config."""
        if COEVO_CONFIG.exists():
            try:
                return json.loads(COEVO_CONFIG.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {"proactive": {"predictPatterns": True}}

    def predict_pattern(self) -> Optional[str]:
        """
        Predict likely session pattern using:
        1. Current time (peak hours from stats)
        2. Recently detected patterns
        3. Day of week patterns (if available)

        Returns predicted pattern or None if uncertain.
        """
        # Check if prediction is enabled
        if not self.coevo_config.get("proactive", {}).get("predictPatterns", True):
            return None

        predictions = {}

        # 1. Time-based prediction
        current_hour = datetime.now().hour
        hour_counts = self.stats_cache.get("hourCounts", {})

        # Map hours to likely patterns based on typical workflow
        hour_pattern_weights = {
            range(6, 10): {"research": 0.4, "learning": 0.3, "debugging": 0.3},
            range(10, 12): {"architecture": 0.4, "debugging": 0.3, "refactoring": 0.3},
            range(12, 14): {"learning": 0.4, "research": 0.3, "testing": 0.3},
            range(14, 17): {"architecture": 0.5, "refactoring": 0.3, "debugging": 0.2},
            range(17, 20): {"debugging": 0.4, "testing": 0.3, "deployment": 0.3},
            range(20, 24): {"research": 0.4, "learning": 0.4, "refactoring": 0.2},
            range(0, 6): {"research": 0.5, "learning": 0.5}
        }

        for hour_range, weights in hour_pattern_weights.items():
            if current_hour in hour_range:
                for pattern, weight in weights.items():
                    predictions[pattern] = predictions.get(pattern, 0) + weight * 0.3
                break

        # 2. Recently detected pattern (strongest signal)
        detected = self.detected_patterns.get("patterns", [])
        if detected:
            top_pattern = detected[0].get("id")
            if top_pattern:
                predictions[top_pattern] = predictions.get(top_pattern, 0) + 0.5

        # 3. Historical pattern distribution
        # This would come from activity-events analysis but we approximate

        if not predictions:
            return None

        # Return highest confidence pattern
        best_pattern = max(predictions, key=predictions.get)
        if predictions[best_pattern] >= 0.3:  # Confidence threshold
            return best_pattern

        return None

    def load_pattern_based_context(self, pattern: str) -> str:
        """
        Load context optimized for detected/specified session type.

        Patterns: debugging, research, refactoring, testing,
                  architecture, performance, deployment, learning

        Returns markdown context block tailored to the pattern.
        """
        pattern_data = PATTERN_RESEARCH_PAPERS.get(pattern, {})
        if not pattern_data:
            return ""

        lines = []
        lines.append(f"### Pattern-Aware Context: {pattern.title()}")
        lines.append("")

        # Focus areas
        focus = pattern_data.get("focus", [])
        if focus:
            lines.append(f"**Focus Areas:** {', '.join(focus)}")
            lines.append("")

        # Suggested tools
        tools = pattern_data.get("suggested_tools", [])
        if tools:
            lines.append(f"**Suggested Tools:** `{', '.join(tools)}`")
            lines.append("")

        # Research papers (for recursive novelty)
        papers = pattern_data.get("papers", [])
        if papers:
            lines.append("**Research Foundations:**")
            for paper_id in papers:
                lines.append(f"- [arXiv:{paper_id}](https://arxiv.org/abs/{paper_id})")
            lines.append("")

        # Pattern-specific memories
        memory_context = self._load_pattern_memories(pattern)
        if memory_context:
            lines.append("**Relevant Memories:**")
            lines.append(memory_context)
            lines.append("")

        return '\n'.join(lines)

    def _load_pattern_memories(self, pattern: str) -> str:
        """Load pattern-specific memories from learnings."""
        if not LEARNINGS_FILE.exists():
            return ""

        content = LEARNINGS_FILE.read_text()

        # Keywords associated with each pattern
        pattern_keywords = {
            "debugging": ["error", "fix", "bug", "debug", "issue"],
            "research": ["paper", "arxiv", "study", "finding", "synthesis"],
            "refactoring": ["refactor", "clean", "extract", "rename"],
            "testing": ["test", "coverage", "spec", "assert"],
            "architecture": ["design", "system", "component", "architecture"],
            "performance": ["performance", "optimize", "slow", "memory"],
            "deployment": ["deploy", "release", "production", "CI"],
            "learning": ["learn", "understand", "concept", "tutorial"]
        }

        keywords = pattern_keywords.get(pattern, [])
        if not keywords:
            return ""

        # Find relevant sections
        relevant = []
        sections = re.split(r'\n## ', content)

        for section in sections[:20]:  # Limit scan
            section_lower = section.lower()
            if any(kw in section_lower for kw in keywords):
                # Extract first meaningful line
                first_lines = section.split('\n')[:3]
                summary = ' '.join(first_lines)[:150]
                if len(summary) > 50:
                    relevant.append(f"- {summary}...")
                if len(relevant) >= 3:
                    break

        return '\n'.join(relevant)

    def get_proactive_suggestions(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        Get proactive suggestions based on predicted or detected pattern.

        Returns dict with:
        - predicted_pattern: the pattern we think is coming
        - suggestions: list of proactive actions
        - research_papers: relevant papers for the pattern
        """
        if pattern is None:
            pattern = self.predict_pattern()

        if not pattern:
            return {
                "predicted_pattern": None,
                "suggestions": [],
                "research_papers": []
            }

        pattern_data = PATTERN_RESEARCH_PAPERS.get(pattern, {})

        return {
            "predicted_pattern": pattern,
            "confidence": 0.7,  # Could be calculated more precisely
            "suggestions": pattern_data.get("suggested_tools", []),
            "focus_areas": pattern_data.get("focus", []),
            "research_papers": [
                {"id": p, "url": f"https://arxiv.org/abs/{p}"}
                for p in pattern_data.get("papers", [])
            ]
        }

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
        output_mode: str = "stdout",
        pattern: Optional[str] = None,
        proactive: bool = False
    ) -> str:
        """Main prefetch orchestration."""

        # Auto-detect project if not specified
        if not project:
            project = self.detect_project()

        # Proactive mode: predict pattern if not specified
        if proactive and not pattern:
            pattern = self.predict_pattern()

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

        # Add pattern-based context if pattern specified or predicted
        if pattern:
            pattern_context = self.load_pattern_based_context(pattern)
            if pattern_context:
                # Insert pattern context before the closing marker
                context = context.replace(
                    CONTEXT_END,
                    f"\n{pattern_context}\n{CONTEXT_END}"
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
    parser.add_argument("--pattern", "--pat",
                        choices=["debugging", "research", "refactoring", "testing",
                                "architecture", "performance", "deployment", "learning"],
                        help="Load pattern-specific context")
    parser.add_argument("--proactive", action="store_true",
                        help="Auto-predict pattern from time/history")
    parser.add_argument("--suggest", action="store_true",
                        help="Show proactive suggestions for current pattern")

    args = parser.parse_args()

    prefetcher = ContextPrefetcher()

    # Handle proactive suggestions mode
    if args.suggest:
        suggestions = prefetcher.get_proactive_suggestions(args.pattern)
        if args.json:
            print(json.dumps(suggestions, indent=2))
        else:
            pattern = suggestions.get("predicted_pattern", "unknown")
            print(f"\nProactive Suggestions for: {pattern}")
            print("=" * 40)
            if suggestions.get("focus_areas"):
                print(f"Focus: {', '.join(suggestions['focus_areas'])}")
            if suggestions.get("suggestions"):
                print(f"Tools: {', '.join(suggestions['suggestions'])}")
            if suggestions.get("research_papers"):
                print("Research Papers:")
                for p in suggestions["research_papers"]:
                    print(f"  - {p['url']}")
        return

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
        output_mode=output_mode,
        pattern=args.pattern,
        proactive=args.proactive
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
