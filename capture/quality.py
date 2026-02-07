"""
Platform-Aware Quality Scoring — Score individual messages/events.

Reuses scoring logic from chatgpt_quality_scorer.py adapted for
per-message scoring (vs per-conversation) used in live capture.
"""

from typing import Tuple


# Depth keywords (from chatgpt_quality_scorer.py)
_DEPTH_KEYWORDS = {
    "code", "function", "algorithm", "database", "schema", "implementation",
    "architecture", "design", "analysis", "research", "theory", "model",
    "system", "protocol", "framework", "optimization", "performance",
    "explain", "understand", "concept", "principle", "mechanism",
    "cognitive", "semantic", "temporal", "coherence", "emergence",
    "infrastructure", "sovereign", "autonomous", "intelligence",
}

# Garbage signals
_GARBAGE_SIGNALS = {
    "hi", "hello", "thanks", "thank you", "ok", "okay", "yes", "no",
    "lol", "haha", "cool", "nice", "great", "awesome", "test", "testing",
}

# Quality thresholds
DEEP_WORK_THRESHOLD = 0.75
EXPLORATION_THRESHOLD = 0.50
CASUAL_THRESHOLD = 0.30

# Platform adjustments (curated platforms get bonus)
_PLATFORM_BONUS = {
    "claude-desktop": 0.10,   # MCP capture = high-quality by design
    "chatgpt": 0.0,           # Neutral
    "cursor": 0.05,           # Coding context = above average
    "grok": 0.05,             # Strategic/world-level thinking
}


def score_event(content: str, role: str, platform: str) -> Tuple[float, str]:
    """
    Score a single message for quality.

    Returns:
        (quality_score, cognitive_mode)
    """
    if not content or len(content.strip()) < 5:
        return 0.0, "garbage"

    cl = content.lower()

    # Depth (0.4 weight): message length + technical keywords
    length_score = min(len(content) / 500, 1.0)
    keyword_count = sum(1 for kw in _DEPTH_KEYWORDS if kw in cl)
    keyword_score = min(keyword_count / 8, 1.0)
    depth = (length_score * 0.5) + (keyword_score * 0.5)

    # Signal (0.3 weight): substance vs noise
    stripped = cl.strip()
    if stripped in _GARBAGE_SIGNALS or len(stripped) < 10:
        signal = 0.1
    elif keyword_count > 0:
        signal = min(0.5 + (keyword_count * 0.1), 1.0)
    elif len(content) > 100:
        signal = 0.5
    else:
        signal = 0.3

    # Focus (0.3 weight): for single messages, approximate via keyword density
    if len(content) > 50 and keyword_count >= 2:
        focus = min(keyword_count / 5, 1.0)
    elif len(content) > 200:
        focus = 0.5
    else:
        focus = 0.3

    # Weighted average
    raw_score = (depth * 0.4) + (focus * 0.3) + (signal * 0.3)

    # Platform bonus
    bonus = _PLATFORM_BONUS.get(platform, 0.0)
    quality_score = min(raw_score + bonus, 1.0)

    # Classify
    if quality_score >= DEEP_WORK_THRESHOLD:
        cognitive_mode = "deep_work"
    elif quality_score >= EXPLORATION_THRESHOLD:
        cognitive_mode = "exploration"
    elif quality_score >= CASUAL_THRESHOLD:
        cognitive_mode = "casual"
    else:
        cognitive_mode = "garbage"

    return round(quality_score, 3), cognitive_mode
