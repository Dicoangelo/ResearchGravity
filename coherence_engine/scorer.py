"""
Coherence Engine — Coherence Scorer

Combines all detection layers into final coherence scores.
Generates CoherenceMoment records for storage and alerting.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config as cfg
from .detector import (
    DetectionResult,
    SignatureDetector,
    SemanticDetector,
    SynchronicityDetector,
)
from .similarity import SimilarityResult

import logging

log = logging.getLogger("coherence.scorer")


COHERENCE_TYPES = {
    "signature_match": "Exact topic+intent alignment across platforms",
    "semantic_echo": "Conceptual alignment (different words, same idea)",
    "synchronicity": "Multi-signal emergence pattern",
    "temporal_cluster": "Multiple events in tight time window",
}


@dataclass
class CoherenceMoment:
    """A detected moment of cross-platform coherence."""
    moment_id: str
    detected_ns: int
    event_ids: List[str]
    platforms: List[str]
    coherence_type: str
    confidence: float
    description: str
    time_window_s: float
    signals: Dict[str, float] = field(default_factory=dict)


class CoherenceScorer:
    """
    Multi-signal coherence scoring.

    Runs all three detection layers against candidates and
    produces CoherenceMoment records for significant coherence.
    """

    def __init__(self, pool=None):
        self._pool = pool
        self._signature_detector = SignatureDetector()
        self._semantic_detector = SemanticDetector()
        self._synchronicity_detector = SynchronicityDetector()

    async def score(
        self,
        event_row: Dict[str, Any],
        embedding: List[float],
        similar_results: List[SimilarityResult],
    ) -> List[CoherenceMoment]:
        """
        Score an event against all similar candidates.

        Returns list of CoherenceMoment for detections above threshold.
        """
        moments = []
        seen_pairs = set()

        # Layer 1: Signature matches (queries DB directly)
        sig_detections = await self._signature_detector.detect(event_row, self._pool)
        for det in sig_detections:
            pair = tuple(sorted([det.event_a_id, det.event_b_id]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                moments.append(self._to_moment(det))

        # Layer 2: Semantic echo
        sem_detections = self._semantic_detector.detect(event_row, similar_results)
        for det in sem_detections:
            pair = tuple(sorted([det.event_a_id, det.event_b_id]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                moments.append(self._to_moment(det))

        # Layer 3: Synchronicity (on high-similarity cross-platform matches)
        for sr in similar_results:
            if sr.platform == event_row.get("platform", ""):
                continue

            pair = tuple(sorted([event_row["event_id"], sr.event_id]))
            if pair in seen_pairs:
                continue

            sync = self._synchronicity_detector.detect(
                event_row, sr, sr.similarity
            )
            if sync.is_synchronicity:
                seen_pairs.add(pair)
                moments.append(CoherenceMoment(
                    moment_id=f"cm-{uuid.uuid4().hex[:12]}",
                    detected_ns=time.time_ns(),
                    event_ids=[event_row["event_id"], sr.event_id],
                    platforms=[event_row.get("platform", ""), sr.platform],
                    coherence_type="synchronicity",
                    confidence=sync.confidence,
                    description=(
                        f"Synchronicity: {event_row.get('platform', '')} <-> {sr.platform} "
                        f"| {sr.preview[:60]}"
                    ),
                    time_window_s=abs(
                        event_row.get("timestamp_ns", 0) - sr.timestamp_ns
                    ) / 1e9,
                    signals=sync.signals,
                ))

        # Sort by confidence descending
        moments.sort(key=lambda m: m.confidence, reverse=True)
        return moments

    def _to_moment(self, det: DetectionResult) -> CoherenceMoment:
        """Convert a DetectionResult to a CoherenceMoment."""
        return CoherenceMoment(
            moment_id=f"cm-{uuid.uuid4().hex[:12]}",
            detected_ns=time.time_ns(),
            event_ids=[det.event_a_id, det.event_b_id],
            platforms=[det.platform_a, det.platform_b],
            coherence_type=det.coherence_type,
            confidence=det.confidence,
            description=det.description,
            time_window_s=det.time_gap_s,
            signals=det.signals,
        )

    async def store_moment(self, moment: CoherenceMoment):
        """Persist a coherence moment to the database."""
        if not self._pool:
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO coherence_moments
                       (moment_id, detected_ns, event_ids, platforms,
                        coherence_type, confidence, description, time_window_s)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                       ON CONFLICT (moment_id) DO NOTHING""",
                    moment.moment_id,
                    moment.detected_ns,
                    moment.event_ids,
                    moment.platforms,
                    moment.coherence_type,
                    moment.confidence,
                    moment.description,
                    int(moment.time_window_s),
                )
        except Exception as e:
            log.error(f"Failed to store moment {moment.moment_id}: {e}")
