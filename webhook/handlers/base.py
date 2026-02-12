"""
WebhookHandler ABC — Template for provider-specific handlers.

Each handler must:
1. Parse the raw webhook body into structured events
2. Determine the event type (push, issue, message, etc.)
3. Return a list of WebhookEvents for normalization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class WebhookEvent:
    """Parsed webhook event before normalization to CapturedEvent."""

    event_type: str             # e.g., "push", "pull_request", "message"
    content: str                # Human-readable content string
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0     # Unix timestamp
    session_id: str = ""       # Provider-specific grouping key
    role: str = "system"       # "user", "assistant", "system"


class WebhookHandler(ABC):
    """Abstract base for provider-specific webhook handlers."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name (e.g., 'github', 'slack')."""
        ...

    @property
    @abstractmethod
    def platform(self) -> str:
        """Platform identifier for cognitive_events.platform field."""
        ...

    @abstractmethod
    async def handle(
        self, headers: Mapping[str, str], body: bytes
    ) -> List[WebhookEvent]:
        """Parse raw webhook into structured events."""
        ...

    def supported_events(self) -> List[str]:
        """List event types this handler can process."""
        return ["*"]
