"""
Guard for the meta_cognitive vocabulary.

d9aa778 ("realtime mode, knowledge graph, embedding upgrade") narrowed
META_COGNITIVE_TERMS from 13 terms to 8, dropping every domain word this
corpus is actually written in. The meta_cognitive signal carries weight 0.25;
after the edit it scored a hard 0.0 on the very pairs that had stored a 1.0.

Nothing failed and no test covered it, so it went unnoticed from February to
July. These tests make the vocabulary an explicit contract.
"""

import pytest

from coherence_engine import config as cfg
from coherence_engine.detector import SynchronicityDetector
from coherence_engine.similarity import SimilarityResult


# The words this corpus uses to describe itself. Dropping these is what
# silently zeroed the signal.
DOMAIN_TERMS = {
    "coherence",
    "cognitive",
    "ucw",
    "wallet",
    "meta",
    "sovereign",
    "unify",
    "alignment",
}


def test_domain_vocabulary_is_present():
    missing = DOMAIN_TERMS - cfg.META_COGNITIVE_TERMS
    assert not missing, (
        f"domain terms dropped from META_COGNITIVE_TERMS: {sorted(missing)}. "
        "Re-measure synchronicity detection before narrowing this set."
    )


def test_generic_vocabulary_is_retained():
    generic = {"emergence", "consciousness", "synchronicity", "breakthrough"}
    assert generic <= cfg.META_COGNITIVE_TERMS


def test_meta_cognitive_scores_above_zero_on_domain_content():
    """A pair written in this corpus's vocabulary must not score a hard zero."""
    detector = SynchronicityDetector()

    event = {
        "event_id": "ev-a",
        "platform": "chatgpt",
        "timestamp_ns": 1_785_000_000_000_000_000,
        "light_layer": {
            "summary": "thinking about the universal cognitive wallet and "
                       "the meta-engine behind it",
            "concepts": ["cognitive", "ucw"],
        },
        "instinct_layer": {"coherence_potential": 0.8},
    }
    candidate = SimilarityResult(
        event_id="ev-b",
        platform="claude-code",
        session_id="s1",
        similarity=0.78,
        preview="building the coherence engine — sovereign data, unify the wallet",
        light_layer={"concepts": ["coherence", "wallet"]},
        instinct_layer={"coherence_potential": 0.8},
        timestamp_ns=1_785_000_000_000_000_000 + 9_000_000_000_000,
        coherence_sig=None,
    )

    score = detector.detect(event, candidate, 0.78)
    assert score.signals["meta_cognitive"] > 0.0, (
        "meta_cognitive scored 0 on content written in this corpus's own "
        "vocabulary — the exact regression d9aa778 introduced"
    )
