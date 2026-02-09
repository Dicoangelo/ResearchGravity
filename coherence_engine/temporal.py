"""
Coherence Engine -- Multi-Scale Temporal Detection

Replaces the fixed 5-minute time window with multi-scale temporal
windows for coherence detection. Different coherence patterns
emerge at different scales:

  - micro  (2 min):  Real-time synchronicity (platform switching)
  - short  (10 min): Active parallel work
  - session (1 hr):  Working-session themes
  - block  (4 hr):   Daily work-block patterns
  - daily  (24 hr):  Daily cognitive arcs
  - weekly (7 d):    Weekly / recurring themes

Each scale has its own minimum confidence threshold -- shorter
windows require higher confidence because the temporal proximity
already carries strong signal, while longer windows accept lower
confidence since temporal evidence is weaker.

Deduplication: when the same event pair is detected at multiple
scales, only the tightest (shortest) window is kept.
"""

import json as _json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .similarity import SimilarityIndex, SimilarityResult
from .scorer import CoherenceMoment, CoherenceScorer
from . import config as cfg

log = logging.getLogger("coherence.temporal")


# -- Temporal Window Definition ------------------------------------

@dataclass(frozen=True)
class TemporalWindow:
    """A single time-scale window for coherence detection."""
    name: str
    seconds: int
    min_confidence: float


# Ordered from tightest to widest -- dedup keeps first hit
WINDOWS: List[TemporalWindow] = [
    TemporalWindow(name="micro",   seconds=120,    min_confidence=0.80),
    TemporalWindow(name="short",   seconds=600,    min_confidence=0.72),
    TemporalWindow(name="session", seconds=3600,   min_confidence=0.68),
    TemporalWindow(name="block",   seconds=14400,  min_confidence=0.65),
    TemporalWindow(name="daily",   seconds=86400,  min_confidence=0.60),
    TemporalWindow(name="weekly",  seconds=604800, min_confidence=0.55),
]


# -- Multi-Scale Detector ------------------------------------------

class MultiScaleDetector:
    """
    Run coherence detection at multiple temporal scales for a
    single event, deduplicating across scales so that each
    event pair is only reported at the tightest window where
    it meets the confidence threshold.
    """

    def __init__(self, pool, similarity: SimilarityIndex, scorer: CoherenceScorer):
        self._pool = pool
        self._similarity = similarity
        self._scorer = scorer

    async def detect_multi_scale(
        self,
        event: Dict[str, Any],
        embedding: List[float],
    ) -> List[Tuple[CoherenceMoment, str]]:
        """
        Detect coherence moments at every temporal scale.

        Returns a list of (CoherenceMoment, window_scale_name) tuples,
        deduplicated so that each event pair only appears once (at the
        tightest scale where it was detected).
        """
        seen_pairs: Dict[frozenset, str] = {}
        results: List[Tuple[CoherenceMoment, str]] = []

        event_ts_ns = event.get("timestamp_ns", 0)
        platform = event.get("platform", "")

        for window in WINDOWS:
            similar = await self._find_similar_in_window(
                event, embedding, window, event_ts_ns, platform,
            )

            if not similar:
                continue

            moments = await self._scorer.score(
                event, embedding, similar, window_scale=window.name,
            )

            for moment in moments:
                if moment.confidence < window.min_confidence:
                    continue

                pair_key = frozenset(moment.event_ids)
                if pair_key in seen_pairs:
                    continue

                seen_pairs[pair_key] = window.name
                results.append((moment, window.name))

            log.debug(
                "Window '%s' (%ds): %d candidates, %d new moments",
                window.name,
                window.seconds,
                len(similar),
                sum(1 for m, w in results if w == window.name),
            )

        return results

    async def _find_similar_in_window(
        self,
        event: Dict[str, Any],
        embedding: List[float],
        window: TemporalWindow,
        event_ts_ns: int,
        platform: str,
    ) -> List[SimilarityResult]:
        """
        Find cross-platform similar events constrained to a time window.

        Uses pgvector similarity search with an additional timestamp
        filter so that only events within [ts - window, ts + window]
        are considered.
        """
        window_ns = window.seconds * 1_000_000_000
        ts_lower = event_ts_ns - window_ns
        ts_upper = event_ts_ns + window_ns

        threshold = cfg.SEMANTIC_MEDIUM_THRESHOLD
        vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
        dist_threshold = 1.0 - threshold

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT ec.content_hash, ec.content_preview,
                              ec.source_event_id,
                              ce.platform, ce.session_id, ce.light_layer,
                              ce.instinct_layer, ce.coherence_sig,
                              ce.timestamp_ns,
                              (ec.embedding_768 <=> $1::vector) AS distance
                       FROM embedding_cache ec
                       JOIN cognitive_events ce
                            ON ec.source_event_id = ce.event_id
                       WHERE ce.platform != $2
                         AND ec.embedding_768 IS NOT NULL
                         AND ce.timestamp_ns >= $3
                         AND ce.timestamp_ns <= $4
                         AND (ec.embedding_768 <=> $1::vector) < $5
                       ORDER BY distance
                       LIMIT $6""",
                    vec_str,
                    platform,
                    ts_lower,
                    ts_upper,
                    dist_threshold,
                    cfg.MAX_CANDIDATES_PER_EVENT,
                )
        except Exception as e:
            log.warning(
                "Time-windowed similarity search failed for "
                "window '%s': %s",
                window.name,
                e,
            )
            return []

        results = []
        for row in rows:
            light = row["light_layer"]
            if isinstance(light, str):
                light = _json.loads(light)
            instinct = row.get("instinct_layer")
            if isinstance(instinct, str):
                instinct = _json.loads(instinct)

            results.append(SimilarityResult(
                event_id=row["source_event_id"],
                platform=row["platform"],
                session_id=row["session_id"],
                similarity=round(1.0 - row["distance"], 4),
                preview=row["content_preview"],
                coherence_sig=row["coherence_sig"],
                light_layer=light,
                instinct_layer=instinct,
                timestamp_ns=row["timestamp_ns"],
            ))

        return results
