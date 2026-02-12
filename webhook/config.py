"""
Webhook Configuration — Environment-based settings.

Reuses PG_DSN from capture.config for shared database access.
"""

import os
from pathlib import Path

# ── Server ────────────────────────────────────────────────
WEBHOOK_HOST = os.environ.get("UCW_WEBHOOK_HOST", "127.0.0.1")
WEBHOOK_PORT = int(os.environ.get("UCW_WEBHOOK_PORT", "3848"))

# ── Database (shared with capture + coherence) ────────────
PG_DSN = os.environ.get(
    "UCW_DATABASE_URL",
    "postgresql://localhost:5432/ucw_cognitive",
)
PG_MIN_POOL = int(os.environ.get("UCW_PG_MIN_POOL", "2"))
PG_MAX_POOL = int(os.environ.get("UCW_PG_MAX_POOL", "5"))

# ── Provider Secrets ──────────────────────────────────────
GITHUB_WEBHOOK_SECRET = os.environ.get("UCW_GITHUB_WEBHOOK_SECRET", "")
SLACK_SIGNING_SECRET = os.environ.get("UCW_SLACK_SIGNING_SECRET", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("UCW_STRIPE_WEBHOOK_SECRET", "")
GENERIC_WEBHOOK_SECRET = os.environ.get("UCW_GENERIC_WEBHOOK_SECRET", "")

# ── Relay ─────────────────────────────────────────────────
RELAY_SHARED_SECRET = os.environ.get("UCW_RELAY_SECRET", "")

# ── Enabled Providers ─────────────────────────────────────
ENABLED_PROVIDERS = [
    p.strip()
    for p in os.environ.get("UCW_WEBHOOK_PROVIDERS", "github,slack,generic").split(",")
    if p.strip()
]

# ── State / Logs ──────────────────────────────────────────
STATE_DIR = Path.home() / ".ucw" / "webhook_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path.home() / ".ucw" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
