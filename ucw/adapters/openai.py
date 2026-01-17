"""
OpenAI/GPT Adapter — Integration with ChatGPT and OpenAI platforms.

This adapter handles:
- Importing sessions from ChatGPT data export (JSON format)
- Generating context for GPT-4/GPT-4o sessions
- Formatting for OpenAI system prompts

STATUS: STUB — Full implementation planned for Phase 3
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import PlatformAdapter
from ..schema import Session, CognitiveWallet, URL


class OpenAIAdapter(PlatformAdapter):
    """Adapter for OpenAI/ChatGPT platforms."""

    @property
    def platform_name(self) -> str:
        return "OpenAI"

    def detect_platform(self) -> bool:
        """
        Check if we're in an OpenAI environment.

        Note: Unlike Claude Code, ChatGPT doesn't have a local CLI.
        This checks for exported data or environment variables.
        """
        # Check for ChatGPT export directory
        chatgpt_export = Path.home() / "Downloads" / "chatgpt_export"
        if chatgpt_export.exists():
            return True

        # Check for OpenAI API key (suggests OpenAI usage)
        import os
        if os.environ.get("OPENAI_API_KEY"):
            return True

        return False

    def export_sessions(self) -> List[Session]:
        """
        Extract sessions from ChatGPT data export.

        ChatGPT exports conversations in a JSON format with structure:
        {
            "title": "Conversation Title",
            "create_time": timestamp,
            "mapping": {node_id: {message: {...}}}
        }

        STATUS: STUB — Returns empty list until full implementation
        """
        sessions = []

        # Look for ChatGPT export files
        export_paths = [
            Path.home() / "Downloads" / "chatgpt_export",
            Path.home() / "Downloads",
        ]

        for export_dir in export_paths:
            if not export_dir.exists():
                continue

            # Find conversations.json files
            for json_file in export_dir.glob("**/conversations.json"):
                try:
                    conversations = json.loads(json_file.read_text())
                    for conv in conversations:
                        session = self._parse_conversation(conv)
                        if session:
                            sessions.append(session)
                except Exception as e:
                    print(f"Warning: Could not parse {json_file}: {e}")

        return sessions

    def _parse_conversation(self, conv: Dict[str, Any]) -> Optional[Session]:
        """
        Parse a ChatGPT conversation into a UCW Session.

        STATUS: STUB — Basic structure only
        """
        try:
            conv_id = conv.get("id", "")
            title = conv.get("title", "Untitled")

            # Parse timestamp
            create_time = conv.get("create_time", 0)
            if isinstance(create_time, (int, float)):
                date = datetime.fromtimestamp(create_time)
            else:
                date = datetime.now()

            # Extract messages from mapping
            messages = []
            urls = []
            mapping = conv.get("mapping", {})

            for node_id, node in mapping.items():
                message = node.get("message")
                if not message:
                    continue

                content = message.get("content", {})
                parts = content.get("parts", [])

                for part in parts:
                    if isinstance(part, str):
                        messages.append(part)

                        # Extract URLs
                        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
                        found_urls = url_pattern.findall(part)
                        for url in found_urls:
                            urls.append(URL(
                                url=url,
                                tier=2,
                                category="research" if "arxiv" in url else "other",
                                source="ChatGPT",
                                context="",
                                captured_at=date,
                            ))

            # Extract arXiv paper IDs
            arxiv_pattern = re.compile(r'(\d{4}\.\d{4,5})')
            full_text = " ".join(messages)
            papers = list(set(arxiv_pattern.findall(full_text)))

            return Session(
                id=f"gpt_{conv_id[:8]}" if conv_id else f"gpt_{hash(title) % 10000:04d}",
                topic=title,
                date=date,
                findings=[],
                papers=papers,
                urls=urls,
                project=None,
                status="imported",
                metadata={
                    "source": "chatgpt_export",
                    "message_count": len(messages),
                },
            )

        except Exception as e:
            print(f"Warning: Could not parse conversation: {e}")
            return None

    def generate_context(self, wallet: CognitiveWallet, max_tokens: int = 4000) -> str:
        """
        Generate OpenAI-compatible context for injection.

        This creates a system prompt or context that can be used in:
        - Custom GPT instructions
        - OpenAI API system messages
        - ChatGPT session preamble

        The format is optimized for GPT-4's context window and style.
        """
        lines = [
            "## Cognitive Context from Universal Cognitive Wallet",
            "",
            f"Wallet Value: ${wallet.value_metrics.total_value:,.2f}",
            f"Knowledge Base: {len(wallet.sessions)} sessions, {len(wallet.concepts)} concepts, {len(wallet.papers)} papers",
            "",
        ]

        # Recent sessions
        recent_sessions = sorted(
            wallet.sessions.values(),
            key=lambda s: s.date,
            reverse=True
        )[:5]

        if recent_sessions:
            lines.append("### Recent Research")
            lines.append("")
            for session in recent_sessions:
                date_str = session.date.strftime("%Y-%m-%d")
                lines.append(f"- [{date_str}] {session.topic[:60]}")
                if session.papers:
                    lines.append(f"  Papers: {', '.join(session.papers[:3])}")
            lines.append("")

        # Key concepts (prioritize high-confidence)
        concepts = sorted(
            wallet.concepts.values(),
            key=lambda c: c.confidence,
            reverse=True
        )[:10]

        if concepts:
            lines.append("### Key Insights")
            lines.append("")
            for concept in concepts:
                type_marker = concept.concept_type.value.upper()
                lines.append(f"- [{type_marker}] {concept.content[:100]}")
            lines.append("")

        # Domain focus
        if wallet.value_metrics.domains:
            lines.append("### Expertise Domains")
            lines.append("")
            for domain, weight in sorted(
                wallet.value_metrics.domains.items(),
                key=lambda x: -x[1]
            )[:3]:
                pct = weight * 100
                lines.append(f"- {domain}: {pct:.0f}%")
            lines.append("")

        # Add instruction for AI
        lines.extend([
            "---",
            "Use this cognitive context to inform responses. Reference relevant",
            "concepts and papers when applicable. Build on established insights.",
        ])

        return "\n".join(lines)

    def export_for_custom_gpt(self, wallet: CognitiveWallet) -> str:
        """
        Generate instructions for a Custom GPT configuration.

        This creates a complete instruction set that can be pasted into
        the "Instructions" field of a Custom GPT.
        """
        context = self.generate_context(wallet)

        instructions = f"""You are an AI assistant with access to the user's cognitive context.

{context}

## Behavior Guidelines

1. When the user asks about topics in the Knowledge Base, reference relevant sessions and papers
2. Build on established concepts and insights rather than starting from scratch
3. Connect new information to the existing knowledge graph
4. Maintain continuity with the user's research trajectory
5. Acknowledge the accumulated expertise in the user's domains

## Response Style

- Reference specific papers by arXiv ID when relevant
- Connect responses to established concepts
- Suggest connections between new information and existing knowledge
- Maintain the user's preferred terminology and frameworks
"""

        return instructions


def import_chatgpt_export(export_path: str) -> List[Session]:
    """
    Convenience function to import ChatGPT export.

    Args:
        export_path: Path to conversations.json or export directory

    Returns:
        List of imported sessions
    """
    adapter = OpenAIAdapter()

    path = Path(export_path)
    if path.is_file():
        try:
            conversations = json.loads(path.read_text())
            return [
                adapter._parse_conversation(conv)
                for conv in conversations
                if adapter._parse_conversation(conv) is not None
            ]
        except Exception as e:
            print(f"Error importing: {e}")
            return []

    return adapter.export_sessions()


if __name__ == "__main__":
    # Test adapter
    adapter = OpenAIAdapter()
    print(f"Platform: {adapter.platform_name}")
    print(f"Detected: {adapter.detect_platform()}")

    # Test with mock wallet
    from ..export import build_wallet_from_agent_core
    try:
        wallet = build_wallet_from_agent_core()
        print("\nGenerated context:")
        print(adapter.generate_context(wallet))
    except Exception as e:
        print(f"Could not load wallet: {e}")
