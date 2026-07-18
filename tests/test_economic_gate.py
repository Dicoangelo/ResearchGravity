"""Tests for the economic delegation gate (arXiv:2603.02961 adaptation)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from delegation.economic_gate import Regime, economic_delegation_gate
from delegation.models import TaskProfile


def _profile(**kw) -> TaskProfile:
    return TaskProfile(**kw)


def test_legacy_anchor_low_verifiability_low_trust_is_manual():
    """The legacy MIN_VERIFIABILITY=0.3 rule survives as the alpha=0.5 case."""
    p = _profile(verifiability=0.2, criticality=0.9, reversibility=0.2)
    d = economic_delegation_gate(p, trust_score=0.5)
    assert d.regime is Regime.MANUAL


def test_high_trust_earns_delegation_on_hard_to_verify_task():
    """Generalization over the constant: track record buys delegation."""
    p = _profile(verifiability=0.2, criticality=0.9, reversibility=0.2)
    d = economic_delegation_gate(p, trust_score=0.95)
    assert d.regime is Regime.VERIFIED_DELEGATION
    assert d.verification_intensity > 0


def test_high_trust_low_stakes_is_pure_delegation():
    p = _profile(verifiability=0.9, criticality=0.1, reversibility=0.9)
    d = economic_delegation_gate(p, trust_score=0.9)
    assert d.regime is Regime.PURE_DELEGATION
    assert d.verification_intensity == 0.0


def test_mid_case_is_verified_with_interior_intensity():
    p = _profile(verifiability=0.6, criticality=0.6, reversibility=0.5)
    d = economic_delegation_gate(p, trust_score=0.5)
    assert d.regime is Regime.VERIFIED_DELEGATION
    assert 0.0 < d.verification_intensity <= 1.0


def test_intensity_decreases_with_trust():
    p = _profile(verifiability=0.6, criticality=0.7, reversibility=0.4)
    low = economic_delegation_gate(p, trust_score=0.3).verification_intensity
    high = economic_delegation_gate(p, trust_score=0.8).verification_intensity
    assert low > high


def test_intensity_increases_with_stakes():
    lo = _profile(verifiability=0.6, criticality=0.3, reversibility=0.8)
    hi = _profile(verifiability=0.6, criticality=0.9, reversibility=0.1)
    a = economic_delegation_gate(lo, trust_score=0.5).verification_intensity
    b = economic_delegation_gate(hi, trust_score=0.5).verification_intensity
    assert b > a


def test_trust_clamped_and_rationale_present():
    p = _profile()
    d = economic_delegation_gate(p, trust_score=1.7)
    assert d.alpha == 1.0
    assert d.rationale
