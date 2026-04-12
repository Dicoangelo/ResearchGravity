"""Configuration for NotebookLM MCP Server"""

import os
from pathlib import Path
from urllib.parse import urlparse


class NotebookLMConfig:
    """NotebookLM-specific configuration extending base UCW config"""

    # Server identity
    SERVER_NAME = "notebooklm-ucw"
    SERVER_VERSION = "2.0.0"  # v2: HTTP/RPC + Cognitive Intelligence
    PROTOCOL_VERSION = "2024-11-05"

    # Platform identification
    PLATFORM = "notebooklm"
    PROTOCOL = "mcp"

    # Paths
    AGENT_CORE = Path.home() / ".agent-core"
    UCW_DIR = Path.home() / ".ucw"
    LOG_DIR = UCW_DIR / "logs"
    DB_PATH = UCW_DIR / "cognitive_capture.db"

    # NotebookLM-specific paths
    NOTEBOOKLM_DIR = UCW_DIR / "notebooklm"
    AUTH_STATE_DIR = NOTEBOOKLM_DIR / "auth_state"
    SESSIONS_DIR = NOTEBOOKLM_DIR / "sessions"

    # Auth profile paths (compatible with jacob-bd notebooklm-mcp-cli)
    NOTEBOOKLM_CLI_DIR = Path.home() / ".notebooklm-mcp-cli"
    AUTH_PROFILES_DIR = NOTEBOOKLM_CLI_DIR / "profiles"

    # Logging (NEVER to stdout — MCP protocol requirement)
    LOG_FILE = LOG_DIR / "notebooklm-mcp.log"
    ERROR_LOG = LOG_DIR / "notebooklm-errors.log"
    CAPTURE_LOG = LOG_DIR / "notebooklm-capture.log"

    # NotebookLM URLs
    NOTEBOOKLM_BASE_URL = "https://notebooklm.google.com"

    # Base URL allowlist — prevents requests to non-Google hosts
    # (upstream v0.5.17 security hardening)
    BASE_URL_ALLOWLIST = frozenset({
        "notebooklm.google.com",
        "notebooklm-pa.clients6.google.com",
        "notebooklm-pa.googleapis.com",
        "accounts.google.com",
        "myaccount.google.com",
    })

    @classmethod
    def validate_base_url(cls, url: str) -> str:
        """Validate a URL against the allowlist. Raises ValueError if not allowed.

        Only HTTPS URLs targeting *.google.com hosts in BASE_URL_ALLOWLIST pass.
        """
        if not url:
            raise ValueError("empty URL")
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"non-https scheme rejected: {parsed.scheme!r}")
        host = (parsed.hostname or "").lower()
        if host not in cls.BASE_URL_ALLOWLIST:
            raise ValueError(f"host not in allowlist: {host!r}")
        return url

    # Capture settings (inherit from UCW)
    CAPTURE_RAW_BYTES = True
    ENABLE_UCW_LAYERS = True

    # Database connection (PostgreSQL for UCW)
    DATABASE_URL = os.getenv(
        "UCW_DATABASE_URL", "postgresql://localhost:5432/ucw_cognitive"
    )

    @classmethod
    def ensure_dirs(cls):
        """Create required directories. Auth state dirs are chmod 0o700."""
        cls.UCW_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.NOTEBOOKLM_DIR.mkdir(parents=True, exist_ok=True)
        cls.AUTH_STATE_DIR.mkdir(parents=True, exist_ok=True)
        cls.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        # Auth state is sensitive — tighten perms (upstream v0.5.17)
        try:
            cls.AUTH_STATE_DIR.chmod(0o700)
            cls.NOTEBOOKLM_DIR.chmod(0o700)
        except OSError:
            pass
