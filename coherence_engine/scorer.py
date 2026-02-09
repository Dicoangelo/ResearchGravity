"""
Coherence Engine — Coherence Scorer

Combines all detection layers into final coherence scores.
Generates CoherenceMoment records for storage and alerting.
"""

import hashlib
import json
import time
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
    window_scale: str = "short"


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
        window_scale: str = "short",
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
                moments.append(self._to_moment(det, window_scale=window_scale))

        # Layer 2: Semantic echo
        sem_detections = self._semantic_detector.detect(event_row, similar_results)
        for det in sem_detections:
            pair = tuple(sorted([det.event_a_id, det.event_b_id]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                moments.append(self._to_moment(det, window_scale=window_scale))

        # Layer 3: Synchronicity (on high-similarity cross-platform matches)
        families = cfg.PLATFORM_FAMILIES
        event_family = families.get(event_row.get("platform", ""), "")
        for sr in similar_results:
            if sr.platform == event_row.get("platform", ""):
                continue
            # Skip same-family platforms (claude-code ↔ claude-cli = same ecosystem)
            if families.get(sr.platform, sr.platform) == event_family:
                continue

            pair = tuple(sorted([event_row["event_id"], sr.event_id]))
            if pair in seen_pairs:
                continue

            sync = self._synchronicity_detector.detect(
                event_row, sr, sr.similarity
            )
            if sync.is_synchronicity:
                seen_pairs.add(pair)
                event_ids = [event_row["event_id"], sr.event_id]
                moments.append(CoherenceMoment(
                    moment_id=self._deterministic_moment_id(event_ids, "synchronicity"),
                    detected_ns=time.time_ns(),
                    event_ids=event_ids,
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
                    window_scale=window_scale,
                ))

        # Sort by confidence descending
        moments.sort(key=lambda m: m.confidence, reverse=True)
        return moments

    @staticmethod
    def _deterministic_moment_id(event_ids: List[str], coherence_type: str) -> str:
        """Generate a deterministic moment_id from the event pair + type."""
        pair_key = "|".join(sorted(event_ids)) + "|" + coherence_type
        return f"cm-{hashlib.sha256(pair_key.encode()).hexdigest()[:16]}"

    @staticmethod
    def _moment_signature(event_ids: List[str], coherence_type: str, confidence: float) -> str:
        """Generate a signature for deduplication."""
        pair_key = "|".join(sorted(event_ids)) + "|" + coherence_type
        return hashlib.sha256(f"{pair_key}|{confidence:.4f}".encode()).hexdigest()

    def _to_moment(self, det: DetectionResult, window_scale: str = "short") -> CoherenceMoment:
        """Convert a DetectionResult to a CoherenceMoment."""
        event_ids = [det.event_a_id, det.event_b_id]
        return CoherenceMoment(
            moment_id=self._deterministic_moment_id(event_ids, det.coherence_type),
            detected_ns=time.time_ns(),
            event_ids=event_ids,
            platforms=[det.platform_a, det.platform_b],
            coherence_type=det.coherence_type,
            confidence=det.confidence,
            description=det.description,
            time_window_s=det.time_gap_s,
            signals=det.signals,
            window_scale=window_scale,
        )

    async def store_moment(self, moment: CoherenceMoment):
        """Persist a coherence moment to the database."""
        if not self._pool:
            return

        signature = self._moment_signature(
            moment.event_ids, moment.coherence_type, moment.confidence
        )

        metadata = json.dumps({"signals": moment.signals}) if moment.signals else None

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO coherence_moments
                       (moment_id, detected_ns, event_ids, platforms,
                        coherence_type, confidence, description, time_window_s,
                        signature, metadata, window_scale)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                       ON CONFLICT (moment_id) DO UPDATE SET
                           confidence = GREATEST(coherence_moments.confidence, EXCLUDED.confidence),
                           detected_ns = EXCLUDED.detected_ns,
                           metadata = COALESCE(EXCLUDED.metadata, coherence_moments.metadata),
                           window_scale = EXCLUDED.window_scale""",
                    moment.moment_id,
                    moment.detected_ns,
                    moment.event_ids,
                    moment.platforms,
                    moment.coherence_type,
                    moment.confidence,
                    moment.description,
                    int(moment.time_window_s),
                    signature,
                    metadata,
                    moment.window_scale,
                )
        except Exception as e:
            log.error(f"Failed to store moment {moment.moment_id}: {e}")
