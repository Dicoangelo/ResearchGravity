"""
Coherence Engine — Configuration

Central config for all engine components.
Reads from environment with sensible defaults.
"""

import os
from pathlib import Path


# ── Database ──────────────────────────────────────────────
PG_DSN = os.environ.get(
    "UCW_DATABASE_URL",
    "postgresql://localhost:5432/ucw_cognitive",
)
PG_MIN_POOL = int(os.environ.get("UCW_PG_MIN_POOL", "2"))
PG_MAX_POOL = int(os.environ.get("UCW_PG_MAX_POOL", "10"))

# ── Embedding ─────────────────────────────────────────────
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_DIMENSIONS = 384
COHERE_MODEL = "embed-v4.0"
COHERE_DIMENSIONS = 1024
EMBED_BATCH_SIZE = 256

# ── Detection thresholds ──────────────────────────────────
# NOTE: Cross-platform similarity is lower (~0.55-0.74) than same-platform
# because MCP captures and imported findings have different text formats.
# Thresholds calibrated to actual data distribution (2026-02-07).
SIGNATURE_CONFIDENCE = 0.95
SEMANTIC_THRESHOLD = 0.65           # Cross-platform high (PRD: 0.85)
SEMANTIC_MEDIUM_THRESHOLD = 0.55    # Cross-platform medium (PRD: 0.75)
SEMANTIC_CONFIDENCE_FACTOR = 0.9
SYNCHRONICITY_THRESHOLD = 0.60      # Multi-signal (PRD: 0.70)
TIME_WINDOW_MINUTES = 30
TIME_WINDOW_NS = TIME_WINDOW_MINUTES * 60 * 1_000_000_000

# ── Synchronicity signal weights ──────────────────────────
SYNC_WEIGHTS = {
    "temporal": 0.15,
    "semantic": 0.30,
    "meta_cognitive": 0.25,
    "instinct_alignment": 0.15,
    "concept_overlap": 0.15,
}

# ── Meta-cognitive keywords ───────────────────────────────
META_COGNITIVE_TERMS = {
    "coherence", "cognitive", "emergence", "unify", "sovereign",
    "ucw", "wallet", "consciousness", "synchronicity", "alignment",
    "meta", "breakthrough", "convergence",
}

# ── Daemon ────────────────────────────────────────────────
POLL_INTERVAL_S = int(os.environ.get("UCW_POLL_INTERVAL", "10"))
RECENT_WINDOW_HOURS = 24
MAX_CANDIDATES_PER_EVENT = 50

# ── Alerts ────────────────────────────────────────────────
DESKTOP_NOTIFICATIONS = True
MIN_ALERT_CONFIDENCE = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.80
WEBHOOK_URL = os.environ.get("UCW_WEBHOOK_URL")
LOG_DIR = Path.home() / ".ucw" / "logs"
LOG_FILE = LOG_DIR / "coherence.log"

# ── Dashboard ─────────────────────────────────────────────
DASHBOARD_REFRESH_S = 5
DASHBOARD_HISTORY_HOURS = 24
