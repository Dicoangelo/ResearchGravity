"""Configuration for raw MCP server"""

import os
from pathlib import Path


class Config:
    # Server identity
    SERVER_NAME = "researchgravity-ucw"
    SERVER_VERSION = "1.0.0"
    PROTOCOL_VERSION = "2024-11-05"

    # Platform identification
    PLATFORM = "claude-desktop"
    PROTOCOL = "mcp"

    # Paths
    AGENT_CORE = Path.home() / ".agent-core"
    UCW_DIR = Path.home() / ".ucw"
    LOG_DIR = UCW_DIR / "logs"
    DB_PATH = UCW_DIR / "cognitive_capture.db"

    # ResearchGravity paths (for tool implementations)
    SESSION_TRACKER = AGENT_CORE / "session_tracker.json"
    PROJECTS_FILE = AGENT_CORE / "projects.json"
    LEARNINGS_FILE = AGENT_CORE / "memory" / "learnings.md"
    RESEARCH_DIR = AGENT_CORE / "research"

    # Logging (NEVER to stdout)
    LOG_FILE = LOG_DIR / "mcp-raw.log"
    ERROR_LOG = LOG_DIR / "mcp-errors.log"
    CAPTURE_LOG = LOG_DIR / "capture.log"

    # Capture settings
    CAPTURE_RAW_BYTES = True
    ENABLE_UCW_LAYERS = True

    @classmethod
    def ensure_dirs(cls):
        """Create required directories"""
        cls.UCW_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
