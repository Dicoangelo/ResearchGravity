#!/usr/bin/env python3
"""
Principle Injector for Antigravity Chief of Staff.

Injects context-appropriate principles into agent prompts based on:
- Current layer (capture, sorting, intelligence, storage, retrieval)
- Task type (build, research, archive, query)
- Specific categories (reliability, architecture, quality, maintenance)

Usage:
    python3 principle_injector.py                     # All principles
    python3 principle_injector.py --layer capture    # Capture layer principles
    python3 principle_injector.py --task synthesis   # Synthesis task principles
    python3 principle_injector.py --categories reliability,quality
    python3 principle_injector.py --inject           # Inject into CLAUDE.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def get_principles_dir() -> Path:
    return Path.home() / ".agent-core" / "principles"


def load_yaml_or_json(path: Path) -> dict:
    """Load YAML file, falling back to JSON parsing if PyYAML unavailable."""
    content = path.read_text()

    if YAML_AVAILABLE:
        return yaml.safe_load(content)

    # Fallback: Basic YAML parsing for our simple format
    # This handles the specific YAML structure we use
    import re

    result = {"principles": {}}
    current_principle = None
    current_key = None
    current_indent = 0
    buffer = []

    for line in content.split('\n'):
        # Skip comments and empty lines
        if line.strip().startswith('#') or not line.strip():
            continue

        # Detect principle name (e.g., "  dont_swallow_errors:")
        if re.match(r'^  [a-z_]+:$', line):
            if current_principle and buffer:
                # Save previous principle
                pass
            current_principle = line.strip().rstrip(':')
            result["principles"][current_principle] = {}
            continue

        # Detect key-value pairs
        if current_principle:
            match = re.match(r'^    ([a-z_]+):\s*(.*)$', line)
            if match:
                key, value = match.groups()
                if value.strip():
                    # Inline value
                    value = value.strip().strip('"\'')
                    result["principles"][current_principle][key] = value
                else:
                    # Multi-line value starts
                    current_key = key
                    buffer = []
            elif current_key and line.startswith('      '):
                # Continuation of multi-line
                buffer.append(line.strip())

    return result


def load_definitions() -> dict:
    """Load principle definitions from YAML."""
    definitions_path = get_principles_dir() / "definitions.yaml"
    if not definitions_path.exists():
        raise FileNotFoundError(f"Principles not found at {definitions_path}")
    return load_yaml_or_json(definitions_path)


def load_manifest() -> dict:
    """Load principle manifest from YAML."""
    manifest_path = get_principles_dir() / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    return load_yaml_or_json(manifest_path)


def get_principles_for_context(
    layer: Optional[str] = None,
    task_type: Optional[str] = None,
    categories: Optional[list[str]] = None,
    include_examples: bool = False
) -> str:
    """
    Generate principle guidance block for agent context.

    Args:
        layer: Current layer (capture, sorting, intelligence, storage, retrieval)
        task_type: Type of task (build, research, archive, synthesis, query)
        categories: List of categories (reliability, architecture, quality, maintenance)
        include_examples: Whether to include code examples

    Returns:
        Formatted markdown string for agent injection
    """
    definitions = load_definitions()
    manifest = load_manifest()

    selected = []

    for name, principle in definitions.get("principles", {}).items():
        include = False

        # Filter by layer
        if layer:
            applications = principle.get("applications", {})
            layer_key = f"{layer}_layer"
            if layer_key in applications or layer in applications:
                include = True

        # Filter by task type
        if task_type:
            triggers = principle.get("triggers", [])
            if isinstance(triggers, list):
                trigger_text = " ".join(triggers).lower()
                if task_type.lower() in trigger_text:
                    include = True
            # Also check applications
            applications = principle.get("applications", {})
            if task_type in str(applications).lower():
                include = True

        # Filter by category
        if categories:
            if principle.get("category") in categories:
                include = True

        # Include all if no filters specified
        if not layer and not task_type and not categories:
            include = True

        if include:
            selected.append((name, principle))

    # Format output
    output = "## ACTIVE PRINCIPLES\n\n"
    output += f"_Context: layer={layer or 'all'}, task={task_type or 'all'}, "
    output += f"categories={categories or 'all'}_\n\n"

    for name, p in selected:
        output += f"### {p.get('name', name)}\n"
        output += f"**Category:** {p.get('category', 'general')}\n\n"

        # Description
        desc = p.get('description', '')
        if desc:
            output += f"{desc.strip()}\n\n"

        # Agent instruction (most important)
        instruction = p.get('agent_instruction', '')
        if instruction:
            output += f"**Agent Instruction:**\n{instruction.strip()}\n\n"

        # Layer-specific application
        if layer:
            applications = p.get('applications', {})
            layer_app = applications.get(f"{layer}_layer") or applications.get(layer)
            if layer_app:
                output += f"**{layer.title()} Layer Application:** {layer_app}\n\n"

        # Examples (optional)
        if include_examples:
            examples = p.get('examples', {})
            if examples.get('good'):
                output += f"**Good Example:**\n```python\n{examples['good'].strip()}\n```\n\n"
            if examples.get('bad'):
                output += f"**Bad Example:**\n```python\n{examples['bad'].strip()}\n```\n\n"

        output += "---\n\n"

    return output


def get_oracle_integration_guidance() -> str:
    """
    Get guidance for Oracle (multi-stream consensus) integration.
    Used for Writer-Critic validation with existing Oracle infrastructure.
    """
    definitions = load_definitions()

    # Find writer_critic_validation principle
    writer_critic = definitions.get("principles", {}).get("writer_critic_validation", {})
    oracle_config = writer_critic.get("oracle_integration", {})

    output = "## ORACLE INTEGRATION FOR VALIDATION\n\n"
    output += "Use existing Oracle multi-stream consensus for Writer-Critic pattern:\n\n"
    output += f"- **Streams:** {oracle_config.get('streams', 3)} concurrent perspectives\n"
    output += f"- **Method:** {oracle_config.get('method', 'intersection → synthesis')}\n"
    output += f"- **Confidence Threshold:** {oracle_config.get('confidence_threshold', 0.7)}\n\n"
    output += "**Protocol:**\n"
    output += "1. Writer generates initial output\n"
    output += "2. Fork to 3 Oracle streams with different perspectives\n"
    output += "3. Compute intersection of validated claims\n"
    output += "4. Synthesize with confidence scores\n"
    output += "5. Only publish if aggregate confidence > 0.7\n"

    return output


def inject_into_claude_md(principles_content: str, target: str = "global"):
    """
    Inject principles into CLAUDE.md file.

    Args:
        principles_content: Formatted principles markdown
        target: 'global' for ~/CLAUDE.md, 'project' for ./CLAUDE.md
    """
    if target == "global":
        claude_md = Path.home() / "CLAUDE.md"
    else:
        claude_md = Path.cwd() / "CLAUDE.md"

    if not claude_md.exists():
        print(f"Warning: {claude_md} does not exist, creating new file")
        claude_md.write_text("")

    content = claude_md.read_text()

    # Define markers
    start_marker = "<!-- PRINCIPLES CONTEXT START -->"
    end_marker = "<!-- PRINCIPLES CONTEXT END -->"

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
    injection = f"{start_marker}\n<!-- Generated: {timestamp} -->\n\n"
    injection += principles_content
    injection += f"\n{end_marker}"

    if start_marker in content:
        # Replace existing
        import re
        pattern = f"{re.escape(start_marker)}.*?{re.escape(end_marker)}"
        content = re.sub(pattern, injection, content, flags=re.DOTALL)
    else:
        # Append new
        content += f"\n\n{injection}"

    claude_md.write_text(content)
    print(f"Injected principles into {claude_md}")


def get_layer_summary() -> str:
    """Get a summary of which principles apply to each layer."""
    manifest = load_manifest()

    output = "## PRINCIPLE-LAYER MAPPING\n\n"
    output += "| Layer | Active Principles |\n"
    output += "|-------|------------------|\n"

    for layer in manifest.get("layers", []):
        name = layer.get("name", "unknown")
        principles = layer.get("active_principles", [])
        output += f"| {name.title()} | {', '.join(principles)} |\n"

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Inject principles into agent context"
    )
    parser.add_argument(
        "--layer",
        choices=["capture", "sorting", "intelligence", "storage", "retrieval"],
        help="Filter by layer"
    )
    parser.add_argument(
        "--task",
        help="Filter by task type (build, research, archive, synthesis, query)"
    )
    parser.add_argument(
        "--categories",
        help="Comma-separated categories (reliability, architecture, quality, maintenance)"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Include code examples"
    )
    parser.add_argument(
        "--inject",
        action="store_true",
        help="Inject into ~/CLAUDE.md"
    )
    parser.add_argument(
        "--inject-project",
        action="store_true",
        help="Inject into ./CLAUDE.md"
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Show Oracle integration guidance"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show layer-principle mapping summary"
    )

    args = parser.parse_args()

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # Generate content
    if args.oracle:
        content = get_oracle_integration_guidance()
    elif args.summary:
        content = get_layer_summary()
    else:
        content = get_principles_for_context(
            layer=args.layer,
            task_type=args.task,
            categories=categories,
            include_examples=args.examples
        )

    # Output
    if args.inject:
        inject_into_claude_md(content, target="global")
    elif args.inject_project:
        inject_into_claude_md(content, target="project")
    else:
        print(content)


if __name__ == "__main__":
    main()
