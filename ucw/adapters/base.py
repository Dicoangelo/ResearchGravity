"""
Base Platform Adapter — Abstract interface for AI platform integrations.

Each platform adapter must implement:
- export_sessions(): Extract sessions from platform format
- generate_context(): Create platform-specific context injection
- detect_platform(): Check if we're in this platform's environment
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..schema import Session, CognitiveWallet


class PlatformAdapter(ABC):
    """Abstract base class for AI platform adapters."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the platform (e.g., 'Claude', 'GPT')."""
        pass

    @abstractmethod
    def export_sessions(self) -> List[Session]:
        """Extract sessions from this platform's format."""
        pass

    @abstractmethod
    def generate_context(self, wallet: CognitiveWallet, max_tokens: int = 4000) -> str:
        """
        Generate platform-specific context for injection.

        Args:
            wallet: The cognitive wallet to generate context from.
            max_tokens: Approximate max tokens for context.

        Returns:
            Formatted context string for the platform.
        """
        pass

    @abstractmethod
    def detect_platform(self) -> bool:
        """Check if we're currently in this platform's environment."""
        pass

    def get_relevant_context(
        self,
        wallet: CognitiveWallet,
        topic: Optional[str] = None,
        project: Optional[str] = None,
        max_concepts: int = 10,
        max_papers: int = 5,
    ) -> str:
        """
        Get relevant context filtered by topic/project.

        This is a convenience method that filters wallet contents
        before generating context.
        """
        # Filter concepts by topic if provided
        relevant_concepts = list(wallet.concepts.values())
        if topic:
            topic_lower = topic.lower()
            relevant_concepts = [
                c for c in relevant_concepts
                if topic_lower in c.content.lower()
            ]

        # Filter sessions by project if provided
        relevant_sessions = list(wallet.sessions.values())
        if project:
            relevant_sessions = [
                s for s in relevant_sessions
                if s.project and project.lower() in s.project.lower()
            ]

        # Limit to most recent/relevant
        relevant_concepts = relevant_concepts[:max_concepts]
        relevant_sessions = sorted(
            relevant_sessions,
            key=lambda s: s.date,
            reverse=True
        )[:10]

        # Build filtered wallet view
        from ..schema import CognitiveWallet
        filtered = CognitiveWallet(
            version=wallet.version,
            concepts={c.id: c for c in relevant_concepts},
            sessions={s.id: s for s in relevant_sessions},
            papers={
                k: v for k, v in wallet.papers.items()
                if any(k in s.papers for s in relevant_sessions)
            }[:max_papers],
            value_metrics=wallet.value_metrics,
        )

        return self.generate_context(filtered)
