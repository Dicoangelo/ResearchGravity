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
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIMENSIONS = 768
EMBED_COLUMN = "embedding_768"
LEGACY_MODEL = "all-MiniLM-L6-v2"
LEGACY_DIMENSIONS = 384
COHERE_MODEL = "embed-v4.0"
COHERE_DIMENSIONS = 1024
EMBED_BATCH_SIZE = 256

# ── Detection thresholds ──────────────────────────────────
# Calibrated 2026-02-08 after moment flood analysis (20K noise moments).
# Previous thresholds were too loose: 0.70 MIN_ALERT let ~90% of matches through.
# New thresholds target <100 genuine moments/day instead of 12,000+ noise.
SIGNATURE_CONFIDENCE = 0.95
SEMANTIC_THRESHOLD = 0.80           # Cross-platform high (was 0.65)
SEMANTIC_MEDIUM_THRESHOLD = 0.72    # Cross-platform medium (was 0.55)
SEMANTIC_CONFIDENCE_FACTOR = 0.9
SYNCHRONICITY_THRESHOLD = 0.78      # Multi-signal (was 0.60)
TIME_WINDOW_MINUTES = 30
TIME_WINDOW_NS = TIME_WINDOW_MINUTES * 60 * 1_000_000_000

# ── Multi-scale temporal windows (Task #10) ──────────────
# Enable/disable multi-scale detection (graceful fallback to single-window)
MULTI_SCALE_ENABLED = True

# Canonical window definitions used by temporal.MultiScaleDetector.
# Each entry: name, duration in seconds, minimum confidence threshold.
# Recalibrated 2026-02-08: much tighter thresholds to prevent noise.
# Only micro/short/session are active — wider windows produce mostly noise.
TIME_WINDOWS = [
    {"name": "micro",   "seconds": 120,    "min_confidence": 0.85},
    {"name": "short",   "seconds": 600,    "min_confidence": 0.83},
    {"name": "session", "seconds": 3600,   "min_confidence": 0.82},
]

# ── Synchronicity signal weights ──────────────────────────
SYNC_WEIGHTS = {
    "temporal": 0.15,
    "semantic": 0.30,
    "meta_cognitive": 0.25,
    "instinct_alignment": 0.15,
    "concept_overlap": 0.15,
}

# ── Meta-cognitive keywords ───────────────────────────────
# NOTE: Terms that appear in nearly ALL UCW events (ucw, cognitive, meta,
# sovereign) were removed — they inflated every score. Only keep terms
# that genuinely signal emergence/breakthrough.
META_COGNITIVE_TERMS = {
    "emergence", "consciousness", "synchronicity",
    "breakthrough", "convergence", "epiphany",
    "insight", "revelation",
}

# ── Content noise filters ───────────────────────────────
# Skip events whose content matches these patterns (case-insensitive).
# These are structural/boilerplate, not meaningful cognitive content.
NOISE_PREFIXES = [
    "<task-notification>",
    "<task-id>",
    "git commit:",
    "continue where we left off",
    "continue here",
    "picking up where",
    "this session is being continued",
    "let me pick up",
    "cost-aware mode active",
    "session window",
]
MIN_CONTENT_LENGTH = 40  # Skip very short content

# ── Platform families ────────────────────────────────────
# Platforms in the same family are NOT cross-platform coherence.
# claude-code, claude-cli, claude-desktop, ccc are all "Claude".
PLATFORM_FAMILIES = {
    "claude-code": "claude",
    "claude-cli": "claude",
    "claude-desktop": "claude",
    "ccc": "claude",
    "chatgpt": "openai",
    "grok": "xai",
}

# ── Daemon ────────────────────────────────────────────────
POLL_INTERVAL_S = int(os.environ.get("UCW_POLL_INTERVAL", "10"))
RECENT_WINDOW_HOURS = 24
MAX_CANDIDATES_PER_EVENT = 50

# ── Alerts ────────────────────────────────────────────────
DESKTOP_NOTIFICATIONS = True
MIN_ALERT_CONFIDENCE = 0.82
HIGH_CONFIDENCE_THRESHOLD = 0.90
WEBHOOK_URL = os.environ.get("UCW_WEBHOOK_URL")
LOG_DIR = Path.home() / ".ucw" / "logs"
LOG_FILE = LOG_DIR / "coherence.log"

# ── Dashboard ─────────────────────────────────────────────
DASHBOARD_REFRESH_S = 5
DASHBOARD_HISTORY_HOURS = 24
