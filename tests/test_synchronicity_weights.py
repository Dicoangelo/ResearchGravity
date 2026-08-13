"""
Guards for the synchronicity scoring weights.

SYNC_WEIGHTS was tuned against all-MiniLM-L6-v2. When embeddings migrated to
768d nomic-embed-text-v1.5 the weights were never re-benchmarked against
detection, and synchronicity went from 40% of all engine output to zero — all
29 historical moments dropped from an average confidence of 0.834 to 0.634,
and 0/29 cleared the threshold. Nothing failed and no test covered it.

Recalibrated 2026-08-13 by measuring the 29 known-good pairs against 3,111 real
candidate pairs. These tests pin the outcome so the same silent drift cannot
recur.
"""

import pytest

from coherence_engine import config as cfg
from coherence_engine.detector import SynchronicityDetector
from coherence_engine.similarity import SimilarityResult

BASE_NS = 1_785_000_000_000_000_000


def _pair(
    *,
    concepts_a,
    concepts_b,
    summary,
    preview,
    similarity,
    gap_s,
    potential=0.8,
):
    """Build a (event_row, candidate) pair for the detector."""
    event = {
        "event_id": "ev-a",
        "platform": "chatgpt",
        "timestamp_ns": BASE_NS,
        "light_layer": {"summary": summary, "concepts": concepts_a},
        "instinct_layer": {"coherence_potential": potential},
    }
    candidate = SimilarityResult(
        event_id="ev-b",
        platform="claude-code",
        session_id="s1",
        similarity=similarity,
        preview=preview,
        light_layer={"concepts": concepts_b},
        instinct_layer={"coherence_potential": potential},
        timestamp_ns=BASE_NS + int(gap_s * 1e9),
        coherence_sig=None,
    )
    return event, candidate


def test_weights_sum_to_one():
    total = sum(cfg.SYNC_WEIGHTS.values())
    assert total == pytest.approx(1.0), f"SYNC_WEIGHTS sum to {total}, not 1.0"


def test_discriminating_signals_carry_the_weight():
    """meta_cognitive and concept_overlap are the only signals that separate.

    Measured pos-vs-neg medians: concept_overlap +1.000, meta_cognitive +0.700,
    instinct +0.150, semantic -0.035, temporal +0.000.
    """
    assert cfg.SYNC_WEIGHTS["meta_cognitive"] >= 0.4
    assert cfg.SYNC_WEIGHTS["concept_overlap"] >= 0.4


def test_semantic_is_not_reweighted_into_the_composite():
    """Cosine is a precondition, not a discriminator.

    Candidates are already filtered at cosine >= SEMANTIC_MEDIUM_THRESHOLD in
    temporal._find_similar_in_window. Within that filtered pool the signal
    measured *anti*-correlated with ground truth (neg median 0.664 > pos median
    0.629), so weighting it again both double-counts and inverts.
    """
    assert cfg.SYNC_WEIGHTS["semantic"] == 0.0


def test_temporal_is_not_weighted():
    """_temporal_score decays over the fixed 30-minute TIME_WINDOW_MINUTES,
    but synchronicity only fires in the block/daily/weekly windows (4h-7d).
    Every real pair is past the decay floor: all 29 historical moments stored
    temporal = 0.000. Weighting it spends 15% of the budget on a constant.
    """
    assert cfg.SYNC_WEIGHTS["temporal"] == 0.0

    event, candidate = _pair(
        concepts_a=["ucw"],
        concepts_b=["ucw"],
        summary="coherence and the cognitive wallet",
        preview="the sovereign wallet, unify the meta layer",
        similarity=0.76,
        gap_s=4 * 3600,  # inside the 'block' window, far past 30 minutes
    )
    score = SynchronicityDetector().detect(event, candidate, 0.76)
    assert score.signals["temporal"] == 0.0


def test_known_good_shaped_pair_fires():
    """A pair shaped like the 29 historical moments must clear both gates.

    Shape taken from the reproduction run: strong concept overlap, domain
    vocabulary in both halves, cosine ~0.76, hours apart.
    """
    event, candidate = _pair(
        concepts_a=["coherence", "ucw", "wallet"],
        concepts_b=["coherence", "ucw", "wallet"],
        summary="the universal cognitive wallet and coherence across platforms",
        preview="sovereign ucw wallet — unify coherence, meta alignment",
        similarity=0.76,
        gap_s=3 * 3600,
    )
    score = SynchronicityDetector().detect(event, candidate, 0.76)

    assert score.is_synchronicity, (
        f"known-good-shaped pair scored {score.confidence}, below "
        f"SYNCHRONICITY_THRESHOLD {cfg.SYNCHRONICITY_THRESHOLD} — the "
        f"regression that killed 40% of engine output. signals={score.signals}"
    )
    assert score.confidence >= cfg.MIN_ALERT_CONFIDENCE, (
        "pair clears the detector but would be dropped by the daemon's "
        f"MIN_ALERT_CONFIDENCE gate ({cfg.MIN_ALERT_CONFIDENCE})"
    )


def test_unrelated_pair_does_not_fire():
    """High cosine alone must not be enough — that is the false-positive path."""
    event, candidate = _pair(
        concepts_a=["deployment"],
        concepts_b=["invoicing"],
        summary="restarting the staging container after the rollout",
        preview="quarterly invoice reconciliation spreadsheet",
        similarity=0.93,  # deliberately high
        gap_s=2 * 3600,
        potential=0.2,
    )
    score = SynchronicityDetector().detect(event, candidate, 0.93)
    assert not score.is_synchronicity, (
        f"unrelated pair fired on cosine alone (confidence={score.confidence}). "
        "Cosine is a precondition, not evidence of synchronicity."
    )


def test_semantic_score_never_exceeds_one():
    """The piecewise ramp is discontinuous at 0.95 and used to return 1.10.

    Stored signals from the MiniLM era show semantic = 1.097; that over-unity
    output is what carried synchronicity over its threshold. A function
    documented as returning 0-1 must return 0-1.
    """
    detector = SynchronicityDetector()
    for i in range(0, 101):
        sim = i / 100.0
        val = detector._semantic_score(sim)
        assert 0.0 <= val <= 1.0, f"_semantic_score({sim}) returned {val}"
