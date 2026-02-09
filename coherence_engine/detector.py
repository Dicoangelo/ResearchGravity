"""
Coherence Engine — Multi-Layer Detector

Three detection layers:
  1. Signature Match   — SHA-256 coherence signatures (5-min buckets)
  2. Semantic Similarity — Cosine sim > threshold (cross-platform)
  3. Synchronicity      — Multi-signal emergence pattern

Each layer produces candidates scored by the CoherenceScorer.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from mcp_raw.embeddings import cosine_similarity
from . import config as cfg
from .similarity import SimilarityResult

import logging

log = logging.getLogger("coherence.detector")


@dataclass
class SynchronicityScore:
    """Result of synchronicity detection between two events."""
    confidence: float
    signals: Dict[str, float]
    is_synchronicity: bool


@dataclass
class DetectionResult:
    """A coherence detection from any layer."""
    coherence_type: str  # signature_match, semantic_echo, synchronicity
    event_a_id: str
    event_b_id: str
    platform_a: str
    platform_b: str
    confidence: float
    description: str
    time_gap_s: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)


class SignatureDetector:
    """
    Layer 1: Exact coherence signature matching.

    SHA-256 signatures from UCW bridge use 5-minute time buckets.
    Matching signatures across platforms = near-certain alignment.
    """

    async def detect(
        self,
        event_row: Dict[str, Any],
        pool,
    ) -> List[DetectionResult]:
        """Find events with matching coherence signature on other platforms."""
        sig = event_row.get("coherence_sig")
        platform = event_row.get("platform", "")
        if not sig or not pool:
            return []

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT event_id, platform, session_id, timestamp_ns,
                          light_layer, data_layer
                   FROM cognitive_events
                   WHERE coherence_sig = $1 AND platform != $2
                   LIMIT 20""",
                sig, platform,
            )

        # Filter out same-family platforms
        families = cfg.PLATFORM_FAMILIES
        event_family = families.get(platform, platform)

        results = []
        event_ts = event_row.get("timestamp_ns", 0)
        for row in rows:
            match_family = families.get(row["platform"], row["platform"])
            if match_family == event_family:
                continue
            gap = abs(event_ts - row["timestamp_ns"]) / 1e9
            light = row["light_layer"]
            if isinstance(light, str):
                light = json.loads(light)
            topic = light.get("topic", "unknown") if light else "unknown"

            results.append(DetectionResult(
                coherence_type="signature_match",
                event_a_id=event_row["event_id"],
                event_b_id=row["event_id"],
                platform_a=platform,
                platform_b=row["platform"],
                confidence=cfg.SIGNATURE_CONFIDENCE,
                description=f"Exact signature match on topic '{topic}'",
                time_gap_s=gap,
            ))

        return results


class SemanticDetector:
    """
    Layer 2: Semantic similarity detection.

    Finds events with high cosine similarity across platforms.
    Uses pre-computed results from SimilarityIndex.
    """

    def detect(
        self,
        event_row: Dict[str, Any],
        similar_results: List[SimilarityResult],
    ) -> List[DetectionResult]:
        """Score semantic similarity results as coherence detections."""
        results = []
        platform = event_row.get("platform", "")
        event_ts = event_row.get("timestamp_ns", 0)
        families = cfg.PLATFORM_FAMILIES
        event_family = families.get(platform, platform)

        for sr in similar_results:
            if sr.platform == platform:
                continue
            # Skip same-family platforms
            if families.get(sr.platform, sr.platform) == event_family:
                continue

            # Check time window
            gap = abs(event_ts - sr.timestamp_ns) / 1e9
            if gap > cfg.TIME_WINDOW_MINUTES * 60:
                continue

            confidence = sr.similarity * cfg.SEMANTIC_CONFIDENCE_FACTOR

            light = sr.light_layer or {}
            topic = light.get("topic", "unknown")

            results.append(DetectionResult(
                coherence_type="semantic_echo",
                event_a_id=event_row["event_id"],
                event_b_id=sr.event_id,
                platform_a=platform,
                platform_b=sr.platform,
                confidence=round(confidence, 4),
                description=(
                    f"Semantic echo: '{sr.preview[:80]}' "
                    f"(sim={sr.similarity:.3f}, topic='{topic}')"
                ),
                time_gap_s=gap,
                signals={"similarity": sr.similarity},
            ))

        return results


class SynchronicityDetector:
    """
    Layer 3: Multi-signal synchronicity detection.

    Detects meaningful coincidence across platforms by combining:
    1. Temporal proximity (within time window)
    2. Semantic similarity (above threshold)
    3. Meta-cognitive content (UCW, coherence, emergence, etc.)
    4. Instinct layer alignment (both events have high coherence_potential)
    5. Concept cluster overlap (shared concepts)
    """

    def detect(
        self,
        event_row: Dict[str, Any],
        candidate: SimilarityResult,
        similarity: float,
    ) -> SynchronicityScore:
        """Compute synchronicity score between two events."""
        signals = {
            "temporal": self._temporal_score(event_row, candidate),
            "semantic": self._semantic_score(similarity),
            "meta_cognitive": self._meta_cognitive_score(event_row, candidate),
            "instinct_alignment": self._instinct_score(event_row, candidate),
            "concept_overlap": self._concept_score(event_row, candidate),
        }

        confidence = sum(
            signals[k] * cfg.SYNC_WEIGHTS[k] for k in cfg.SYNC_WEIGHTS
        )

        return SynchronicityScore(
            confidence=round(confidence, 4),
            signals=signals,
            is_synchronicity=confidence > cfg.SYNCHRONICITY_THRESHOLD,
        )

    def _temporal_score(self, event_row: Dict, candidate: SimilarityResult) -> float:
        """Score based on time proximity. Closer = higher."""
        event_ts = event_row.get("timestamp_ns", 0)
        gap_s = abs(event_ts - candidate.timestamp_ns) / 1e9
        window_s = cfg.TIME_WINDOW_MINUTES * 60

        if gap_s <= 0:
            return 1.0
        if gap_s > window_s:
            return 0.0
        # Linear decay within window
        return 1.0 - (gap_s / window_s)

    def _semantic_score(self, similarity: float) -> float:
        """Normalize semantic similarity to 0-1 for scoring."""
        if similarity >= 0.95:
            return 1.0
        if similarity >= cfg.SEMANTIC_THRESHOLD:
            return 0.8 + (similarity - cfg.SEMANTIC_THRESHOLD) * 2
        if similarity >= cfg.SEMANTIC_MEDIUM_THRESHOLD:
            return 0.5 + (similarity - cfg.SEMANTIC_MEDIUM_THRESHOLD) * 3
        return max(0.0, similarity - 0.5) * 2

    def _meta_cognitive_score(self, event_row: Dict, candidate: SimilarityResult) -> float:
        """Score based on meta-cognitive keyword presence in both events."""
        event_light = event_row.get("light_layer", {})
        if isinstance(event_light, str):
            event_light = json.loads(event_light)

        cand_light = candidate.light_layer or {}

        # Collect concepts from both
        event_concepts = set(event_light.get("concepts", []))
        cand_concepts = set(cand_light.get("concepts", []))

        # Check content too
        event_summary = (event_light.get("summary", "") or "").lower()
        cand_preview = (candidate.preview or "").lower()

        meta_in_event = len(cfg.META_COGNITIVE_TERMS & event_concepts)
        meta_in_cand = len(cfg.META_COGNITIVE_TERMS & cand_concepts)
        meta_in_text = sum(
            1 for t in cfg.META_COGNITIVE_TERMS
            if t in event_summary or t in cand_preview
        )

        total = meta_in_event + meta_in_cand + meta_in_text
        if total >= 6:
            return 1.0
        if total >= 4:
            return 0.7
        if total >= 2:
            return 0.3
        return 0.0

    def _instinct_score(self, event_row: Dict, candidate: SimilarityResult) -> float:
        """Score based on instinct layer alignment."""
        event_instinct = event_row.get("instinct_layer", {})
        if isinstance(event_instinct, str):
            event_instinct = json.loads(event_instinct)

        cand_instinct = candidate.instinct_layer or {}

        event_cp = event_instinct.get("coherence_potential", 0)
        cand_cp = cand_instinct.get("coherence_potential", 0)

        # Both high coherence potential = strong alignment
        if event_cp > 0.7 and cand_cp > 0.7:
            return 1.0
        avg = (event_cp + cand_cp) / 2
        return min(avg * 1.2, 1.0)

    def _concept_score(self, event_row: Dict, candidate: SimilarityResult) -> float:
        """Score based on concept overlap between events."""
        event_light = event_row.get("light_layer", {})
        if isinstance(event_light, str):
            event_light = json.loads(event_light)

        cand_light = candidate.light_layer or {}

        event_concepts = set(event_light.get("concepts", []))
        cand_concepts = set(cand_light.get("concepts", []))

        if not event_concepts or not cand_concepts:
            return 0.0

        overlap = event_concepts & cand_concepts
        union = event_concepts | cand_concepts

        # Jaccard similarity
        if not union:
            return 0.0
        return len(overlap) / len(union)
