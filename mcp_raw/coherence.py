"""
Coherence Detection Engine — Cross-platform pattern recognition

Detects:
- Temporal alignment: events close in time across platforms
- Semantic similarity: similar content across contexts
- Synchronicity: the UCW "founding moment" patterns

Uses 5-minute time buckets and SHA-256 signatures from ucw_bridge.
"""

import hashlib
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .logger import get_logger
from .ucw_bridge import coherence_signature

log = get_logger("coherence")


class CoherenceEngine:
    """
    Detects coherence patterns across cognitive events.

    Coherence = meaningful alignment across platforms, time, or concepts.
    Three detection levels:
      1. Temporal — events close in time across platforms
      2. Semantic — similar content/concepts across contexts
      3. Synchronicity — temporal + semantic + meta-cognitive (highest order)
    """

    # 5-minute bucket for temporal alignment (nanoseconds)
    DEFAULT_WINDOW_NS = 5 * 60 * 1_000_000_000

    def __init__(self):
        self._moments: List[Dict] = []

    async def detect_temporal_alignment(
        self,
        events: List[Dict],
        window_ns: Optional[int] = None,
    ) -> List[Dict]:
        """
        Find events close in time across different platforms.
        Groups events into time buckets and finds cross-platform clusters.
        """
        window = window_ns or self.DEFAULT_WINDOW_NS

        # Group by time bucket
        buckets: Dict[int, List[Dict]] = defaultdict(list)
        for event in events:
            ts = event.get("timestamp_ns", 0)
            bucket_key = ts // window
            buckets[bucket_key].append(event)

        alignments = []
        for bucket_key, bucket_events in buckets.items():
            platforms = set(e.get("platform", "unknown") for e in bucket_events)
            if len(platforms) > 1:
                confidence = min(1.0, len(platforms) * 0.3 + len(bucket_events) * 0.05)
                alignments.append({
                    "type": "temporal",
                    "bucket_ns": bucket_key * window,
                    "platforms": sorted(platforms),
                    "event_count": len(bucket_events),
                    "confidence": round(confidence, 3),
                    "events": [e.get("event_id", "") for e in bucket_events],
                })

        alignments.sort(key=lambda a: a["confidence"], reverse=True)
        return alignments

    async def detect_semantic_similarity(
        self,
        event: Dict,
        candidates: List[Dict],
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Find events with similar content/concepts.
        Uses concept overlap + topic matching as lightweight similarity.
        """
        source_concepts = _parse_concepts(event)
        source_topic = event.get("light_topic", "")
        source_intent = event.get("light_intent", "")

        matches = []
        for candidate in candidates:
            if candidate.get("event_id") == event.get("event_id"):
                continue

            cand_concepts = _parse_concepts(candidate)
            cand_topic = candidate.get("light_topic", "")
            cand_intent = candidate.get("light_intent", "")

            score = 0.0

            # Concept overlap (Jaccard-like)
            if source_concepts and cand_concepts:
                overlap = source_concepts & cand_concepts
                union = source_concepts | cand_concepts
                score += 0.5 * (len(overlap) / len(union)) if union else 0

            # Topic match (non-general)
            if source_topic and source_topic == cand_topic and source_topic != "general":
                score += 0.3

            # Intent match
            if source_intent and source_intent == cand_intent:
                score += 0.2

            if score >= threshold:
                matches.append({
                    "event_id": candidate.get("event_id", ""),
                    "similarity": round(score, 3),
                    "shared_concepts": sorted(
                        source_concepts & cand_concepts
                    ) if source_concepts and cand_concepts else [],
                    "topic_match": source_topic == cand_topic,
                    "platform": candidate.get("platform", "unknown"),
                })

        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches

    async def detect_synchronicity(
        self,
        events: List[Dict],
    ) -> List[Dict]:
        """
        Find UCW "founding moment" patterns — synchronicity across platforms.

        Synchronicity = temporal alignment + semantic similarity + meta-cognitive signals.
        This is the highest-order coherence pattern.
        """
        signals = []

        # Step 1: Find temporal clusters (tight 2-minute window)
        tight_window = 2 * 60 * 1_000_000_000
        temporal = await self.detect_temporal_alignment(events, window_ns=tight_window)

        for cluster in temporal:
            if cluster["confidence"] < 0.3:
                continue

            # Step 2: Check for meta-cognitive and high-coherence signals
            cluster_event_ids = set(cluster["events"])
            cluster_events = [
                e for e in events
                if e.get("event_id", "") in cluster_event_ids
            ]

            has_meta = False
            has_high_coherence = False

            for e in cluster_events:
                indicators = _parse_json_list(e.get("instinct_indicators", []))

                if "meta_cognitive" in indicators:
                    has_meta = True

                coherence_potential = e.get("instinct_coherence", 0)
                if isinstance(coherence_potential, (int, float)) and coherence_potential > 0.7:
                    has_high_coherence = True

            if has_meta or has_high_coherence:
                confidence = cluster["confidence"]
                if has_meta:
                    confidence = min(1.0, confidence + 0.3)
                if has_high_coherence:
                    confidence = min(1.0, confidence + 0.2)

                signals.append({
                    "type": "synchronicity",
                    "platforms": cluster["platforms"],
                    "event_count": cluster["event_count"],
                    "confidence": round(confidence, 3),
                    "has_meta_cognitive": has_meta,
                    "has_high_coherence": has_high_coherence,
                    "events": cluster["events"],
                    "timestamp_ns": cluster["bucket_ns"],
                })

        signals.sort(key=lambda s: s["confidence"], reverse=True)
        return signals

    async def generate_moment(
        self,
        events: List[Dict],
        coherence_type: str,
        confidence: float,
    ) -> Dict:
        """Create a coherence_moment record from detected events."""
        event_ids = [e.get("event_id", "") for e in events]
        platforms = sorted(set(e.get("platform", "unknown") for e in events))
        timestamps = [e.get("timestamp_ns", 0) for e in events if e.get("timestamp_ns")]

        all_concepts: set = set()
        for e in events:
            all_concepts.update(_parse_concepts(e))

        moment = {
            "moment_id": hashlib.sha256(
                f"{coherence_type}::{','.join(sorted(event_ids))}".encode()
            ).hexdigest()[:16],
            "type": coherence_type,
            "confidence": round(confidence, 3),
            "platforms": platforms,
            "event_ids": event_ids,
            "event_count": len(events),
            "concepts": sorted(all_concepts),
            "timestamp_ns": min(timestamps) if timestamps else time.time_ns(),
            "span_ns": (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0,
            "created_ns": time.time_ns(),
        }

        self._moments.append(moment)
        log.info(
            f"Coherence moment: type={coherence_type} "
            f"confidence={confidence:.3f} events={len(events)}"
        )
        return moment

    @property
    def moments(self) -> List[Dict]:
        return list(self._moments)


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_concepts(event: Dict) -> set:
    """Parse concepts from an event (handles both list and JSON string)."""
    raw = event.get("light_concepts", [])
    if isinstance(raw, list):
        return set(raw)
    return set(_parse_json_list(raw))


def _parse_json_list(value) -> List[str]:
    """Safely parse a JSON array string into a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []
