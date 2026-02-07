"""
Base classes for cross-platform live capture.

CapturedEvent: Raw event from any external platform.
PlatformAdapter: Abstract base for platform-specific polling.
AdapterStatus: Health check result.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CapturedEvent:
    """A single captured event from an external platform."""

    platform: str                # 'chatgpt', 'cursor', 'grok'
    session_id: str              # Platform-specific conversation/session ID
    content: str                 # Message text
    role: str                    # 'user' or 'assistant'
    timestamp: float             # Unix timestamp (seconds)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    cognitive_mode: str = "exploration"

    # Generated fields
    event_id: str = ""
    timestamp_ns: int = 0
    content_hash: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.platform}-{uuid.uuid4().hex[:12]}"
        if not self.timestamp_ns:
            self.timestamp_ns = int(self.timestamp * 1_000_000_000)


@dataclass
class AdapterStatus:
    """Health check result for a platform adapter."""

    healthy: bool
    last_poll: float            # Unix timestamp of last successful poll
    events_captured: int        # Total events captured this session
    error: Optional[str] = None


class PlatformAdapter(ABC):
    """Abstract base class for platform-specific capture adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""
        ...

    @property
    @abstractmethod
    def platform(self) -> str:
        """Platform identifier for cognitive_events.platform field."""
        ...

    @abstractmethod
    async def initialize(self, pool) -> bool:
        """
        Initialize the adapter with a shared asyncpg pool.
        Returns True if ready to poll.
        """
        ...

    @abstractmethod
    async def poll(self) -> List[CapturedEvent]:
        """
        Fetch new events since last poll.
        Returns list of CapturedEvent (empty if nothing new).
        """
        ...

    @abstractmethod
    async def normalize(self, events: List[CapturedEvent]) -> List[dict]:
        """
        Convert CapturedEvents to cognitive_events rows (dicts).
        Each dict matches the cognitive_events schema.
        """
        ...

    @abstractmethod
    async def health_check(self) -> AdapterStatus:
        """Check adapter health and return status."""
        ...
