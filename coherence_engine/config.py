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

# Per-connection init: set pgvector HNSW search depth for better recall
PG_INIT_SQL = "SET hnsw.ef_search = 100;"

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
SEMANTIC_THRESHOLD = 0.80  # Cross-platform high (was 0.65)
SEMANTIC_MEDIUM_THRESHOLD = 0.72  # Cross-platform medium (was 0.55)
SEMANTIC_CONFIDENCE_FACTOR = 0.9
SYNCHRONICITY_THRESHOLD = 0.78  # Multi-signal (was 0.60)
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
    {"name": "micro", "seconds": 120, "min_confidence": 0.85},
    {"name": "short", "seconds": 600, "min_confidence": 0.83},
    {"name": "session", "seconds": 3600, "min_confidence": 0.82},
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
# Vocabulary for the meta_cognitive signal (weight 0.25 in SYNC_WEIGHTS).
#
# The domain terms below were dropped in d9aa778, a commit whose stated purpose
# was the realtime/embedding upgrade. That silently zeroed the signal for this
# corpus: it is written in exactly this vocabulary ("the universal cognitive
# wallet", "the meta-engine", "coherence"), and none of those words survived
# the edit. Every sampled synchronicity pair scores meta_cognitive = 0.0 today
# against a stored value of 1.0 in February.
#
# Restoring them recovers 0.3-0.7 of the signal — not the full 1.0, because the
# embedding space and preview text also changed in the same period. Partial
# recovery of a signal that is currently a hard zero is still the right move.
#
# If you narrow this set again, re-measure synchronicity detection first.
META_COGNITIVE_TERMS = {
    # Generic meta-cognitive vocabulary
    "emergence",
    "consciousness",
    "synchronicity",
    "breakthrough",
    "convergence",
    "epiphany",
    "insight",
    "revelation",
    # Domain vocabulary this corpus actually uses (restored from c977997)
    "coherence",
    "cognitive",
    "unify",
    "sovereign",
    "ucw",
    "wallet",
    "alignment",
    "meta",
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
