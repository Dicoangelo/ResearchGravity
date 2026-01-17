"""
UCW Platform Adapters
Export/import between AI platforms (Claude, GPT, etc.)
"""

from .base import PlatformAdapter
from .claude import ClaudeAdapter
from .openai import OpenAIAdapter

__all__ = ["PlatformAdapter", "ClaudeAdapter", "OpenAIAdapter"]
