"""
Cross-Platform Live Capture — Phase 6 of UCW

Captures cognitive events from ChatGPT, Cursor, and Grok/X
into the unified cognitive database for real-time coherence detection.
"""

from .base import CapturedEvent, PlatformAdapter, AdapterStatus
from .manager import CaptureManager

__all__ = ["CapturedEvent", "PlatformAdapter", "AdapterStatus", "CaptureManager"]
