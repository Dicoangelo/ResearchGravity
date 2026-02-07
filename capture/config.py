"""
Capture Configuration — Environment-based settings for all adapters.

Reuses PG_DSN from coherence_engine.config for shared database access.
"""

import os
import sys
from pathlib import Path

# ── Database (shared with coherence engine) ──────────────────
PG_DSN = os.environ.get(
    "UCW_DATABASE_URL",
    "postgresql://localhost:5432/ucw_cognitive",
)
PG_MIN_POOL = int(os.environ.get("UCW_PG_MIN_POOL", "2"))
PG_MAX_POOL = int(os.environ.get("UCW_PG_MAX_POOL", "10"))

# ── ChatGPT adapter ─────────────────────────────────────────
CHATGPT_EXPORT_PATH = os.environ.get(
    "UCW_CHATGPT_EXPORT",
    str(Path.home() / "Downloads" / "chatgpt-export"),
)
CHATGPT_POLL_INTERVAL_S = int(os.environ.get("UCW_CHATGPT_POLL_INTERVAL", "300"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Cursor adapter ──────────────────────────────────────────
if sys.platform == "darwin":
    _cursor_default = str(
        Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage"
    )
else:
    _cursor_default = str(Path.home() / ".config" / "Cursor" / "User" / "workspaceStorage")

CURSOR_DATA_DIR = os.environ.get("UCW_CURSOR_DATA_DIR", _cursor_default)
CURSOR_POLL_INTERVAL_S = int(os.environ.get("UCW_CURSOR_POLL_INTERVAL", "60"))

# ── Grok / X adapter ────────────────────────────────────────
GROK_API_KEY = os.environ.get("UCW_GROK_API_KEY", "")
GROK_POLL_INTERVAL_S = int(os.environ.get("UCW_GROK_POLL_INTERVAL", "600"))

# ── Deduplication ────────────────────────────────────────────
DEDUP_WINDOW_HOURS = int(os.environ.get("UCW_DEDUP_WINDOW_HOURS", "72"))

# ── Capture state directory ─────────────────────────────────
STATE_DIR = Path.home() / ".ucw" / "capture_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
