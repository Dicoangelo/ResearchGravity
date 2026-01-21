#!/usr/bin/env python3
"""
CPB Precision Mode - Configuration

PRECISION tier: Maximum quality configuration for research-grounded,
evidence-verified answers.

Key Features:
- dqThreshold: 0.95 (highest quality bar)
- 7-agent ACE ensemble with specialized personas
- Critic validation with EvidenceCritic + OracleConsensus
- Retry loop until DQ >= 0.95 or max retries

Research Foundation:
- arXiv:2512.24601 (RLM) - Context externalization
- arXiv:2511.15755 (DQ) - Quality measurement
- arXiv:2508.17536 (Voting vs Debate) - Consensus strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import CPBConfig, CPBPath, RLMConfig, ACEConfig


# =============================================================================
# PRECISION ACE CONFIGURATION
# =============================================================================

@dataclass
class PrecisionACEConfig(ACEConfig):
    """ACE configuration for PRECISION mode."""
    max_rounds: int = 25                # Extended consensus rounds
    agent_count: int = 7                # 7-agent ensemble
    enable_auction: bool = True
    enable_hop_grouping: bool = True
    require_citations: bool = True      # All claims must cite sources


# =============================================================================
# PRECISION CONFIGURATION
# =============================================================================

@dataclass
class PrecisionConfig(CPBConfig):
    """
    PRECISION tier configuration.

    Maximum quality configuration for research-grounded, evidence-verified answers.
    Uses ResearchGravity knowledge base for context enrichment.
    """
    # Path selection - always cascade in precision mode
    auto_route: bool = False
    default_path: CPBPath = CPBPath.CASCADE

    # Quality thresholds - highest bar
    dq_threshold: float = 0.95          # Must hit 95% quality
    force_cascade: bool = True          # Always full pipeline
    critic_validation: bool = True      # Use RG critics
    max_retries: int = 5                # Retry until threshold or max

    # Context thresholds
    context_threshold: int = 200000     # Handle larger contexts (~50k tokens)
    complexity_threshold: float = 0.20  # Lower threshold - more orchestration

    # Time budgets (ms) - extended for quality
    fast_path_ms: int = 15000
    standard_path_ms: int = 90000
    hybrid_path_ms: int = 180000

    # Quality settings
    enable_verification: bool = True
    enable_learning: bool = True
    retry_on_low_dq: bool = True

    # Engine configs
    rlm_config: RLMConfig = field(default_factory=lambda: RLMConfig(
        max_iterations=30,              # Deeper decomposition
        root_model='deep',              # Opus for root synthesis
        sub_model='balanced'            # Sonnet for sub-tasks
    ))
    ace_config: PrecisionACEConfig = field(default_factory=PrecisionACEConfig)


# =============================================================================
# 7-AGENT PRECISION PERSONAS
# =============================================================================

PRECISION_AGENT_PERSONAS = [
    {
        'name': 'Analyst',
        'role': 'evidence',
        'prompt': '''You are a rigorous research analyst. Examine this objectively, focusing on:
- Empirical data and measurable outcomes
- Logical consistency and valid reasoning chains
- Distinguishing proven facts from assumptions
- Citing specific evidence for all claims

Your expertise: Data analysis, evidence evaluation, logical reasoning.
Every claim must have a source. No unsupported assertions.'''
    },
    {
        'name': 'Skeptic',
        'role': 'critique',
        'prompt': '''You are a critical skeptic and devil's advocate. Your role is to:
- Challenge every assumption and claim
- Identify potential failure modes and edge cases
- Question the methodology and data quality
- Find what's missing or overlooked

Your expertise: Risk assessment, critical analysis, adversarial thinking.
Ask "what could go wrong?" and "what are we missing?" at every step.'''
    },
    {
        'name': 'Synthesizer',
        'role': 'integration',
        'prompt': '''You are a systems thinker and pattern integrator. Your focus:
- Finding deep connections across domains
- Identifying emergent patterns and principles
- Integrating diverse perspectives into coherent frameworks
- Revealing underlying structures and relationships

Your expertise: Cross-domain synthesis, pattern recognition, framework building.
Seek the unifying principles that tie everything together.'''
    },
    {
        'name': 'Pragmatist',
        'role': 'implementation',
        'prompt': '''You are a practical implementer focused on actionability. Consider:
- Real-world feasibility and constraints
- Resource requirements and trade-offs
- Step-by-step implementation paths
- What can actually be built and deployed

Your expertise: Implementation planning, resource estimation, practical problem-solving.
Ask "how would this actually work?" for every proposal.'''
    },
    {
        'name': 'Visionary',
        'role': 'strategy',
        'prompt': '''You are a strategic visionary focused on long-term implications. Explore:
- Second and third-order effects
- Paradigm shifts and transformational potential
- Future scenarios and evolutionary paths
- Novel possibilities and opportunities

Your expertise: Strategic foresight, trend analysis, opportunity identification.
Think beyond the immediate to what this could become.'''
    },
    {
        'name': 'Historian',
        'role': 'context',
        'prompt': '''You are a research historian providing deep context. Your focus:
- Historical precedents and analogies
- Evolution of ideas and approaches
- What has been tried before and why it succeeded/failed
- Lineage of concepts and their development

Your expertise: Research history, prior art analysis, contextual understanding.
Ground the discussion in what we've learned from the past.'''
    },
    {
        'name': 'Innovator',
        'role': 'novelty',
        'prompt': '''You are a creative innovator seeking novel solutions. Explore:
- Unconventional approaches and combinations
- Gaps in current thinking that could be exploited
- Novel framings of the problem
- Breakthrough possibilities others might miss

Your expertise: Creative problem-solving, lateral thinking, innovation.
What new approaches could change everything?'''
    }
]


# =============================================================================
# DQ WEIGHTS FOR PRECISION MODE
# =============================================================================

PRECISION_DQ_WEIGHTS = {
    'validity': 0.30,       # Down from 40% - evidence matters more
    'specificity': 0.25,    # Down from 30%
    'correctness': 0.45,    # Up from 30% - evidence-backed correctness
}

# =============================================================================
# DQ WEIGHTS FOR PIONEER MODE (v2.4)
# =============================================================================
# For cutting-edge research queries without established external validation.
# Reduces validity strictness (exploratory reasoning OK), increases ground truth
# weight (trust claimed ground truth more when external sources don't exist yet).

PIONEER_DQ_WEIGHTS = {
    'validity': 0.25,       # Reduced - exploratory reasoning acceptable
    'specificity': 0.25,    # Same - still need concrete details
    'correctness': 0.30,    # Reduced - may lack external validation
    'ground_truth': 0.20,   # Increased - trust user-provided ground truth
}

# =============================================================================
# DQ WEIGHTS FOR TRUST CONTEXT MODE (v2.4)
# =============================================================================
# For queries with user-provided context marked as Tier 1 trusted.
# Increases correctness weight (user context assumed credible), reduces
# external validation requirement.

TRUST_CONTEXT_DQ_WEIGHTS = {
    'validity': 0.28,       # Slightly reduced
    'specificity': 0.20,    # Reduced - user context provides specificity
    'correctness': 0.40,    # Increased - user context assumed credible
    'ground_truth': 0.12,   # Reduced - less need for external validation
}


# =============================================================================
# CRITIC WEIGHTS FOR PRECISION MODE
# =============================================================================

PRECISION_CRITIC_WEIGHTS = {
    'evidence_critic': 0.40,    # Evidence validation most important
    'oracle_consensus': 0.35,   # Multi-stream verification
    'confidence_scorer': 0.25,  # Confidence assessment
}


# =============================================================================
# VERIFICATION THRESHOLDS
# =============================================================================

@dataclass
class VerificationThresholds:
    """Thresholds for precision verification."""
    evidence_min: float = 0.85          # EvidenceCritic minimum
    oracle_min: float = 0.85            # OracleConsensus minimum
    confidence_min: float = 0.80        # ConfidenceScorer minimum
    combined_min: float = 0.95          # Combined DQ minimum
    citation_coverage_min: float = 0.90 # 90% of claims must cite


PRECISION_VERIFICATION_THRESHOLDS = VerificationThresholds()


# =============================================================================
# DEFAULT PRECISION INSTANCE
# =============================================================================

PRECISION_CONFIG = PrecisionConfig()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_agent_by_role(role: str) -> Optional[Dict[str, str]]:
    """Get agent persona by role."""
    for agent in PRECISION_AGENT_PERSONAS:
        if agent['role'] == role:
            return agent
    return None


def get_precision_agent_prompts(query: str, context: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Build prompts for all 7 precision agents.

    Args:
        query: The query to analyze
        context: Optional context from ResearchGravity

    Returns:
        List of agent prompts ready for parallel execution
    """
    prompts = []

    for agent in PRECISION_AGENT_PERSONAS:
        user_prompt_parts = []

        if context:
            user_prompt_parts.append(f"## CONTEXT (from ResearchGravity)\n{context[:15000]}")

        user_prompt_parts.append(f"## QUERY\n{query}")
        user_prompt_parts.append("""
## INSTRUCTIONS
Analyze this query from your specialized perspective. Provide:

1. **Key Assessment** (2-3 critical points from your viewpoint)
2. **Evidence & Sources** (cite specific sources for claims)
3. **Confidence Level** (0-100% with justification)
4. **Concerns & Caveats** (what to watch out for)
5. **Recommendations** (actionable next steps)

CRITICAL: Every factual claim must cite a source. Use format [source_id] or (arXiv:XXXX.XXXXX).
Be thorough but focused on your specialized domain.""")

        user_prompt = "\n\n".join(user_prompt_parts)

        prompts.append({
            'agent': agent['name'],
            'role': agent['role'],
            'system_prompt': agent['prompt'],
            'user_prompt': user_prompt,
            'full_prompt': f"{agent['prompt']}\n\n---\n\n{user_prompt}"
        })

    return prompts


def calculate_precision_dq(
    validity: float,
    specificity: float,
    correctness: float
) -> float:
    """
    Calculate DQ score using Precision mode weights.

    Precision mode weights correctness higher (45%) because
    evidence-backed claims are critical.

    Args:
        validity: Validity score (0-1)
        specificity: Specificity score (0-1)
        correctness: Correctness score (0-1)

    Returns:
        Weighted DQ score (0-1)
    """
    return (
        validity * PRECISION_DQ_WEIGHTS['validity'] +
        specificity * PRECISION_DQ_WEIGHTS['specificity'] +
        correctness * PRECISION_DQ_WEIGHTS['correctness']
    )


def validate_precision_config(config: PrecisionConfig) -> List[str]:
    """
    Validate precision configuration and return warnings.

    Args:
        config: PrecisionConfig to validate

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    if config.dq_threshold < 0.90:
        warnings.append(f"DQ threshold ({config.dq_threshold}) below 0.90 - not truly 'precision' mode")

    if config.ace_config.agent_count < 5:
        warnings.append(f"Agent count ({config.ace_config.agent_count}) low for precision consensus")

    if config.max_retries < 3:
        warnings.append(f"Max retries ({config.max_retries}) may not be enough to hit 0.95 threshold")

    if not config.critic_validation:
        warnings.append("Critic validation disabled - precision mode requires critic verification")

    if not config.force_cascade:
        warnings.append("Force cascade disabled - precision mode should use full pipeline")

    return warnings
