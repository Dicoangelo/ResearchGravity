"""
UCW Raw MCP Server — Sovereign Cognitive Capture Infrastructure

No SDK. Full protocol control. Every byte captured.
"""

__version__ = "1.0.0"

from .server import RawMCPServer
from .capture import CaptureEngine, CaptureEvent
from .db import CaptureDB
from .database import CognitiveDatabase
from .router import Router
from .config import Config
