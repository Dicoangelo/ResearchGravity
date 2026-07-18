"""
Economic Delegation Gate — three-regime policy from arXiv:2603.02961.

"Delegation and Verification Under AI" derives the worker's optimal action
(d*, s*) — delegate? verify with what intensity? — as three regimes over
(alpha, beta):

    Manual work         (0, 0)   when verification is costly AND the agent's
                                  advantage is too small to risk delegation
    Pure delegation     (1, 0)   when delegation beats manual work and the
                                  marginal benefit of verification at zero is
                                  non-positive (s-dagger = 0)
    Verified delegation (1, s*)  otherwise, with interior intensity s*

Core mechanism (paper section 9): "delegation is beneficial only when paired
with sufficiently reliable verification, while rational private delegation can
generate institutional quality loss when verification is weak."

This module is a STRUCTURAL ADAPTATION, not the paper's closed form: we do not
reproduce psi0/psi1 analytically (their shape depends on task-profile
parameters we don't observe). We preserve the regime geometry and the
marginal-benefit logic, mapped onto the engine's 11-dimensional TaskProfile
and the Bayesian trust ledger:

    beta  (verification cost)  = 1 - profile.verifiability
    alpha (agent advantage)    = trust score from the ledger (0.5 = no data)
    stakes (loss parameter)    = 0.6*criticality + 0.4*(1 - reversibility)

Calibration anchor: at uninformative trust (alpha = 0.5) the MANUAL boundary
sits at verifiability 0.3 for critical tasks — the legacy MIN_VERIFIABILITY
constant becomes the special case of this gate, not a separate rule. What the
economic gate adds over the constant:

    - a high-trust agent can EARN delegation on a hard-to-verify task
      (the constant refused regardless of track record);
    - verification intensity is an output, not a fixed post-hoc choice —
      spend verification where stakes are high and trust is thin.

Usage:
    from delegation.economic_gate import economic_delegation_gate

    decision = economic_delegation_gate(profile, trust_score=0.82)
    if decision.regime is Regime.MANUAL:
        ...  # do not delegate
    else:
        ...  # delegate; verify with decision.verification_intensity
"""

from dataclasses import dataclass
from enum import Enum

from .models import TaskProfile

# ── Calibration (documented, single place) ───────────────────────────────────

# Verification-cost threshold t: beta >= T_BETA means "verification expensive".
# T_BETA = 0.7 <=> verifiability <= 0.3 — anchors the legacy constant.
T_BETA = 0.7

# MANUAL boundary psi0: minimum agent advantage required to delegate when
# verification is expensive. Rises with both extra verification cost and
# stakes. At beta=1.0, full stakes, an agent needs alpha > 0.85.
PSI0_BASE = 0.60
PSI0_SLOPE = 0.50  # per unit of (beta - T_BETA), scaled by stakes

# PURE boundary: verification is not worthwhile (s-dagger = 0) when the
# avoidable expected loss stakes*(1 - alpha) drops below this AND stakes are
# low outright. High-stakes work never skips verification regardless of trust
# — the paper's core mechanism is that delegation without verification is
# where institutional quality loss comes from.
PURE_THRESHOLD = 0.08
PURE_MAX_STAKES = 0.5

# Interior intensity scaling: s* proportional to avoidable expected loss.
INTENSITY_GAIN = 2.0
INTENSITY_FLOOR = 0.10


class Regime(Enum):
    MANUAL = "manual"
    PURE_DELEGATION = "pure_delegation"
    VERIFIED_DELEGATION = "verified_delegation"


@dataclass
class GateDecision:
    regime: Regime
    verification_intensity: float  # 0.0 for manual/pure, (0,1] for verified
    alpha: float  # agent advantage used
    beta: float  # verification cost used
    stakes: float  # loss parameter used
    rationale: str


def economic_delegation_gate(
    profile: TaskProfile, trust_score: float = 0.5
) -> GateDecision:
    """Decide (delegate?, verification intensity) for a task and agent.

    trust_score: Bayesian trust from the ledger for the candidate agent on
    this task type; 0.5 when no history exists (Beta(1,1) mean).
    """
    alpha = max(0.0, min(1.0, trust_score))
    beta = 1.0 - profile.verifiability
    stakes = 0.6 * profile.criticality + 0.4 * (1.0 - profile.reversibility)

    # Regime 1 — MANUAL: verification expensive and agent advantage below the
    # rising boundary psi0(beta, stakes).
    if beta >= T_BETA:
        psi0 = PSI0_BASE + PSI0_SLOPE * (beta - T_BETA) * (0.5 + stakes)
        if alpha < psi0:
            return GateDecision(
                regime=Regime.MANUAL,
                verification_intensity=0.0,
                alpha=alpha,
                beta=beta,
                stakes=stakes,
                rationale=(
                    f"Manual: verification cost {beta:.2f} >= {T_BETA} and "
                    f"trust {alpha:.2f} < required {psi0:.2f} at stakes "
                    f"{stakes:.2f}. Delegation without reliable verification "
                    f"risks quality loss (arXiv:2603.02961 §9)."
                ),
            )

    # Marginal benefit of verification at s=0: avoidable expected loss.
    avoidable_loss = stakes * (1.0 - alpha)

    # Regime 2 — PURE: verification not worthwhile (corner s-dagger = 0),
    # and only ever on low-stakes work.
    if avoidable_loss < PURE_THRESHOLD and stakes < PURE_MAX_STAKES:
        return GateDecision(
            regime=Regime.PURE_DELEGATION,
            verification_intensity=0.0,
            alpha=alpha,
            beta=beta,
            stakes=stakes,
            rationale=(
                f"Pure delegation: avoidable loss {avoidable_loss:.3f} < "
                f"{PURE_THRESHOLD} (trust {alpha:.2f}, stakes {stakes:.2f}) — "
                f"marginal benefit of verification at zero is non-positive."
            ),
        )

    # Regime 3 — VERIFIED: interior verification intensity.
    intensity = max(INTENSITY_FLOOR, min(1.0, INTENSITY_GAIN * avoidable_loss))
    return GateDecision(
        regime=Regime.VERIFIED_DELEGATION,
        verification_intensity=intensity,
        alpha=alpha,
        beta=beta,
        stakes=stakes,
        rationale=(
            f"Verified delegation: intensity {intensity:.2f} from avoidable "
            f"loss {avoidable_loss:.3f} (trust {alpha:.2f}, stakes "
            f"{stakes:.2f}, verification cost {beta:.2f})."
        ),
    )
