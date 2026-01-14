#!/usr/bin/env python3
"""
ResearchGravity Project Context Loader v1.0
Automatically loads project context, research, and memory for any registered project.

Features:
1. Detects current project from working directory
2. Loads project-specific research files
3. Loads project memory/identity
4. Shows related sessions and papers
5. Provides quick context for Claude sessions

Usage:
  python3 project_context.py              # Auto-detect project
  python3 project_context.py --project os-app
  python3 project_context.py --list       # List all projects
  python3 project_context.py --index      # Show research index
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List


AGENT_CORE_DIR = Path.home() / ".agent-core"
PROJECTS_FILE = AGENT_CORE_DIR / "projects.json"
RESEARCH_INDEX = AGENT_CORE_DIR / "research" / "INDEX.md"


def load_projects() -> Dict[str, Any]:
    """Load projects registry."""
    if PROJECTS_FILE.exists():
        return json.loads(PROJECTS_FILE.read_text())
    return {"projects": {}, "paper_index": {}, "topic_index": {}}


def detect_project() -> Optional[str]:
    """Detect project from current working directory."""
    cwd = str(Path.cwd()).lower()
    data = load_projects()

    for project_id, project in data.get("projects", {}).items():
        project_path = project.get("path", "")
        if project_path:
            # Expand ~ and compare
            expanded = str(Path(project_path).expanduser()).lower()
            if expanded in cwd or cwd in expanded:
                return project_id

    # Check by name in path
    for project_id, project in data.get("projects", {}).items():
        name = project.get("name", "").lower().replace(" ", "")
        if name and name in cwd.replace("/", "").replace("-", "").replace("_", ""):
            return project_id

    return None


def get_project_context(project_id: str) -> Dict[str, Any]:
    """Get full context for a project."""
    data = load_projects()
    project = data.get("projects", {}).get(project_id)

    if not project:
        return {"error": f"Project not found: {project_id}"}

    context = {
        "project_id": project_id,
        "name": project.get("name"),
        "description": project.get("description"),
        "status": project.get("status"),
        "focus": project.get("focus", []),
        "tech_stack": project.get("tech_stack", []),
        "research_files": [],
        "memory": None,
        "sessions": [],
        "key_papers": project.get("key_papers", []),
        "lineage": project.get("lineage", {})
    }

    # Load research files content
    research = project.get("research", {})
    research_folder = research.get("folder")
    if research_folder:
        folder_path = Path(research_folder).expanduser()
        for filename in research.get("files", []):
            file_path = folder_path / filename
            if file_path.exists():
                content = file_path.read_text()
                context["research_files"].append({
                    "name": filename,
                    "path": str(file_path),
                    "content": content[:5000] + "..." if len(content) > 5000 else content
                })

    # Load memory
    memory_path = project.get("memory")
    if memory_path:
        memory_file = Path(memory_path).expanduser()
        if memory_file.exists():
            context["memory"] = {
                "path": str(memory_file),
                "content": memory_file.read_text()
            }

    # Load session summaries
    for session_id in project.get("sessions", []):
        session_dir = AGENT_CORE_DIR / "sessions" / session_id
        session_file = session_dir / "session.json"
        if session_file.exists():
            session_data = json.loads(session_file.read_text())
            context["sessions"].append({
                "id": session_id,
                "topic": session_data.get("topic"),
                "started": session_data.get("started"),
                "urls_count": len(session_data.get("urls_captured", [])),
                "findings_count": len(session_data.get("findings_captured", []))
            })

    return context


def format_context_for_display(context: Dict[str, Any]) -> str:
    """Format project context for terminal display."""
    if "error" in context:
        return f"Error: {context['error']}"

    lines = []
    lines.append("=" * 70)
    lines.append(f"  PROJECT CONTEXT: {context['name']}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"ID:          {context['project_id']}")
    lines.append(f"Status:      {context['status']}")
    lines.append(f"Description: {context['description']}")
    lines.append(f"Focus:       {', '.join(context['focus'])}")
    lines.append(f"Tech Stack:  {', '.join(context['tech_stack'])}")
    lines.append("")

    # Research files
    lines.append("-" * 70)
    lines.append("RESEARCH FILES")
    lines.append("-" * 70)
    for rf in context["research_files"]:
        lines.append(f"\n{rf['name']}:")
        lines.append(f"  Path: {rf['path']}")
        # Show first 500 chars
        preview = rf['content'][:500].replace('\n', '\n  ')
        lines.append(f"  Preview:\n  {preview}...")

    # Memory
    if context["memory"]:
        lines.append("")
        lines.append("-" * 70)
        lines.append("PROJECT MEMORY")
        lines.append("-" * 70)
        lines.append(f"Path: {context['memory']['path']}")
        lines.append(context['memory']['content'][:1000])

    # Sessions
    if context["sessions"]:
        lines.append("")
        lines.append("-" * 70)
        lines.append("RELATED SESSIONS")
        lines.append("-" * 70)
        for sess in context["sessions"]:
            lines.append(f"  {sess['id'][:40]}")
            lines.append(f"    Topic: {sess['topic']}")
            lines.append(f"    URLs: {sess['urls_count']} | Findings: {sess['findings_count']}")

    # Key papers
    if context["key_papers"]:
        lines.append("")
        lines.append("-" * 70)
        lines.append("KEY PAPERS")
        lines.append("-" * 70)
        for paper in context["key_papers"]:
            lines.append(f"  [{paper['id']}] {paper['title']} — {paper['topic']}")

    # Lineage
    if context["lineage"]:
        lines.append("")
        lines.append("-" * 70)
        lines.append("LINEAGE")
        lines.append("-" * 70)
        if context["lineage"].get("research_sessions"):
            lines.append(f"  Research Sessions: {', '.join(context['lineage']['research_sessions'])}")
        if context["lineage"].get("features_implemented"):
            lines.append(f"  Features Implemented: {', '.join(context['lineage']['features_implemented'])}")

    lines.append("")
    lines.append("=" * 70)

    return '\n'.join(lines)


def list_projects() -> str:
    """List all registered projects."""
    data = load_projects()

    lines = []
    lines.append("=" * 60)
    lines.append("  REGISTERED PROJECTS")
    lines.append("=" * 60)
    lines.append("")

    for project_id, project in data.get("projects", {}).items():
        status_icon = "" if project.get("status") == "active" else ""
        lines.append(f"{status_icon} {project_id}")
        lines.append(f"   Name: {project.get('name')}")
        lines.append(f"   Description: {project.get('description', 'N/A')[:50]}")
        lines.append(f"   Focus: {', '.join(project.get('focus', []))}")
        lines.append("")

    lines.append("-" * 60)
    lines.append(f"Total projects: {len(data.get('projects', {}))}")

    return '\n'.join(lines)


def show_index() -> str:
    """Show research index."""
    if RESEARCH_INDEX.exists():
        return RESEARCH_INDEX.read_text()
    return "Research index not found. Run migration first."


def generate_claude_context(project_id: str) -> str:
    """Generate context block for Claude session."""
    context = get_project_context(project_id)

    if "error" in context:
        return f"<!-- Error: {context['error']} -->"

    lines = []
    lines.append(f"<!-- PROJECT CONTEXT: {context['name']} -->")
    lines.append("")
    lines.append(f"## Project: {context['name']}")
    lines.append(f"**Focus:** {', '.join(context['focus'])}")
    lines.append(f"**Tech:** {', '.join(context['tech_stack'])}")
    lines.append("")

    if context["memory"]:
        lines.append("### Project Identity")
        lines.append(context["memory"]["content"])
        lines.append("")

    if context["key_papers"]:
        lines.append("### Key Research Papers")
        for paper in context["key_papers"]:
            lines.append(f"- [{paper['id']}] {paper['title']} — {paper['topic']}")
        lines.append("")

    if context["lineage"]:
        lines.append("### Research Lineage")
        if context["lineage"].get("features_implemented"):
            lines.append(f"Features implemented from research: {', '.join(context['lineage']['features_implemented'])}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="ResearchGravity Project Context Loader"
    )
    parser.add_argument("--project", "-p", help="Project ID to load")
    parser.add_argument("--list", "-l", action="store_true", help="List all projects")
    parser.add_argument("--index", "-i", action="store_true", help="Show research index")
    parser.add_argument("--claude", "-c", action="store_true", help="Output Claude-ready context")

    args = parser.parse_args()

    if args.list:
        print(list_projects())
    elif args.index:
        print(show_index())
    else:
        # Detect or use specified project
        project_id = args.project or detect_project()

        if not project_id:
            print("Could not detect project from current directory.")
            print("Use --project <id> or --list to see available projects.")
            return

        if args.claude:
            print(generate_claude_context(project_id))
        else:
            context = get_project_context(project_id)
            print(format_context_for_display(context))


if __name__ == "__main__":
    main()
