#!/usr/bin/env python3
"""
Cognitive Precision Bridge (CPB) - Router

Complexity analysis and path selection for optimal CPB execution.

ELITE TIER: Lower thresholds for more consensus and deeper reasoning.
"""

import re
import hashlib
from typing import Dict, Any, Optional
from .types import (
    CPBPath, CPBConfig, PathSignals, RoutingDecision, PathAlternative,
    DEFAULT_CPB_CONFIG
)


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

CODE_PATTERNS = re.compile(
    r'\b(implement|refactor|debug|function|class|api|code|typescript|'
    r'javascript|python|rust|write a|create a function|fix the bug|'
    r'generate code)\b',
    re.IGNORECASE
)

REASONING_PATTERNS = re.compile(
    r'\b(why|analyze|compare|trade-?off|design|architect|explain|evaluate|'
    r'assess|consider|think about|what if|implications|consequences)\b',
    re.IGNORECASE
)

CREATIVE_PATTERNS = re.compile(
    r'\b(brainstorm|imagine|creative|novel|idea|invent|suggest|propose|'
    r'alternative|what could|dream up)\b',
    re.IGNORECASE
)

NAVIGATION_PATTERNS = re.compile(
    r'\b(go to|navigate|open|show me|take me|switch to|display|view|'
    r'see|look at|pull up)\b',
    re.IGNORECASE
)

QUESTION_PATTERNS = re.compile(
    r'\b(what is|who is|where is|when did|how do|can you|could you|'
    r'would you|is there|are there)\b',
    re.IGNORECASE
)

DEEP_DOMAIN_PATTERNS = re.compile(
    r'\b(architecture|system design|multi-agent|consensus|orchestration|'
    r'distributed|scalability|performance optimization|security audit|'
    r'research synthesis|state machine)\b',
    re.IGNORECASE
)

CONSENSUS_PATTERNS = re.compile(
    r'\b(best approach|trade-?offs|pros and cons|should we|recommend|'
    r'decision|choose|select|evaluate options|compare approaches|'
    r'critical|important)\b',
    re.IGNORECASE
)


# =============================================================================
# COMPLEXITY SIGNALS
# =============================================================================

def extract_complexity_signals(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Extract complexity signals from a query"""
    tokens = query.strip().split()
    full_text = f"{query} {context or ''}"

    return {
        'token_count': len(tokens),
        'context_length': len(context) if context else 0,
        'has_code_indicators': bool(CODE_PATTERNS.search(query)),
        'has_reasoning_indicators': bool(REASONING_PATTERNS.search(query)),
        'has_creative_indicators': bool(CREATIVE_PATTERNS.search(query)),
        'has_navigation_indicators': bool(NAVIGATION_PATTERNS.search(query)),
        'has_question_indicators': bool(QUESTION_PATTERNS.search(query)),
        'has_consensus_indicators': bool(CONSENSUS_PATTERNS.search(query)),
        'domain_complexity': 0.3 if DEEP_DOMAIN_PATTERNS.search(full_text) else 0,
    }


def calculate_complexity_score(signals: Dict[str, Any]) -> float:
    """
    Calculate complexity score from signals.
    Returns a value between 0 (simple) and 1 (complex).

    ELITE TIER: Calibrated for more aggressive routing to higher tiers.
    """
    score = 0.0

    # Token count factor (longer = more complex, up to 0.25)
    score += min(signals['token_count'] / 100, 0.25)

    # Context length factor (larger context = more complex)
    if signals['context_length'] > 10000:
        score += 0.15
    elif signals['context_length'] > 5000:
        score += 0.08

    # Code indicators (moderately complex)
    if signals['has_code_indicators']:
        score += 0.25

    # Reasoning indicators (complex)
    if signals['has_reasoning_indicators']:
        score += 0.2

    # Creative indicators (moderately complex)
    if signals['has_creative_indicators']:
        score += 0.15

    # Consensus indicators (requires multi-perspective)
    if signals['has_consensus_indicators']:
        score += 0.2

    # Navigation indicators (reduce complexity - should be fast)
    if signals['has_navigation_indicators']:
        score -= 0.3

    # Simple questions (reduce complexity slightly)
    if signals['has_question_indicators'] and not signals['has_reasoning_indicators']:
        score -= 0.1

    # Domain complexity boost
    score += signals['domain_complexity']

    # Clamp to [0, 1]
    return max(0.0, min(score, 1.0))


# =============================================================================
# PATH SELECTION
# =============================================================================

def select_path(
    query: str,
    context: Optional[str] = None,
    config: CPBConfig = DEFAULT_CPB_CONFIG
) -> RoutingDecision:
    """
    Select optimal CPB execution path based on query analysis.

    ELITE TIER:
    - Complexity < 0.2 → direct (only truly simple queries)
    - Complexity 0.2-0.5 → rlm (context compression)
    - Complexity 0.5-0.7 → ace (consensus)
    - Complexity > 0.7 → hybrid/cascade (full pipeline)
    """
    signals_dict = extract_complexity_signals(query, context)
    complexity = calculate_complexity_score(signals_dict)
    context_length = len(context) if context else 0

    # Build PathSignals
    signals = PathSignals(
        context_length=context_length,
        query_complexity=complexity,
        requires_consensus=signals_dict['has_consensus_indicators'],
        requires_reasoning=signals_dict['has_reasoning_indicators'],
        has_ground_truth=False,  # Could be enhanced with knowledge base check
        time_budget_ms=config.hybrid_path_ms,
        quality_target=config.dq_threshold
    )

    # ELITE TIER: Path selection with lower thresholds
    alternatives = []
    reasoning_parts = []

    # Direct path - only for truly simple queries
    if complexity < 0.2 and context_length < config.context_threshold // 4:
        selected_path = CPBPath.DIRECT
        reasoning_parts.append(f"Low complexity ({complexity:.2f}) with minimal context")
        alternatives.append(PathAlternative(
            path=CPBPath.RLM,
            score=complexity + 0.1,
            tradeoff="Would add ~2s latency for marginal quality gain"
        ))

    # RLM path - for context-heavy but straightforward queries
    elif context_length > config.context_threshold or (0.2 <= complexity < 0.5):
        if context_length > config.context_threshold:
            selected_path = CPBPath.RLM
            reasoning_parts.append(f"Large context ({context_length:,} chars) requires compression")
        else:
            selected_path = CPBPath.RLM
            reasoning_parts.append(f"Moderate complexity ({complexity:.2f}) benefits from RLM decomposition")

        alternatives.append(PathAlternative(
            path=CPBPath.ACE,
            score=complexity + 0.15,
            tradeoff="Would enable consensus but increase latency"
        ))

    # ACE path - for queries requiring consensus
    elif signals.requires_consensus or (0.5 <= complexity < 0.7):
        selected_path = CPBPath.ACE
        reasoning_parts.append(f"Consensus indicators or moderate-high complexity ({complexity:.2f})")
        alternatives.append(PathAlternative(
            path=CPBPath.HYBRID,
            score=complexity + 0.1,
            tradeoff="Would add RLM preprocessing for fuller analysis"
        ))

    # Hybrid path - for complex queries with context
    elif complexity >= 0.7 and context_length > 5000:
        selected_path = CPBPath.HYBRID
        reasoning_parts.append(f"High complexity ({complexity:.2f}) with substantial context")
        alternatives.append(PathAlternative(
            path=CPBPath.CASCADE,
            score=complexity + 0.05,
            tradeoff="Would add verification pass for maximum quality"
        ))

    # Cascade path - for expert-level queries
    elif complexity >= 0.7 or signals_dict['domain_complexity'] > 0:
        selected_path = CPBPath.CASCADE
        reasoning_parts.append(f"Expert-level complexity ({complexity:.2f}) or domain expertise required")
        alternatives.append(PathAlternative(
            path=CPBPath.HYBRID,
            score=complexity - 0.05,
            tradeoff="Would skip verification for faster response"
        ))

    # Default to config's default path
    else:
        selected_path = config.default_path
        reasoning_parts.append(f"Using default path for complexity {complexity:.2f}")

    reasoning = "; ".join(reasoning_parts)

    return RoutingDecision(
        selected_path=selected_path,
        signals=signals,
        reasoning=reasoning,
        confidence=min(100, int((1 - abs(complexity - 0.5)) * 100)),  # Higher confidence near extremes
        alternatives=alternatives
    )


def should_orchestrate(query: str, context: Optional[str] = None) -> bool:
    """Check if a query would benefit from CPB orchestration (vs direct call)"""
    signals = extract_complexity_signals(query, context)
    complexity = calculate_complexity_score(signals)

    # ELITE TIER: Lower threshold for orchestration
    return complexity >= 0.2 or (context and len(context) > 25000)


def analyze_query(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a query without executing - useful for debugging/logging"""
    signals = extract_complexity_signals(query, context)
    complexity = calculate_complexity_score(signals)
    decision = select_path(query, context)

    return {
        'query': query[:100] + '...' if len(query) > 100 else query,
        'context_length': len(context) if context else 0,
        'signals': signals,
        'complexity_score': complexity,
        'selected_path': decision.selected_path.value,
        'reasoning': decision.reasoning,
        'confidence': decision.confidence,
        'alternatives': [
            {'path': a.path.value, 'score': a.score, 'tradeoff': a.tradeoff}
            for a in decision.alternatives
        ]
    }


def hash_query(query: str, context: Optional[str] = None) -> str:
    """Generate consistent hash for query (for caching/learning)"""
    full_text = f"{query}|{context[:1000] if context else ''}"
    return hashlib.sha256(full_text.encode()).hexdigest()[:12]


# =============================================================================
# TIER CLASSIFICATION (Voice Nexus compatible)
# =============================================================================

def get_reasoning_tier(complexity: float) -> str:
    """
    Map complexity score to reasoning tier.
    Compatible with Voice Nexus tier system.

    ELITE TIER: Lower thresholds
    """
    if complexity < 0.2:
        return 'fast'
    if complexity < 0.5:
        return 'balanced'
    return 'deep'


def get_model_recommendation(complexity: float) -> str:
    """
    Get recommended model based on complexity.

    ELITE TIER: Opus-first for anything non-trivial
    """
    if complexity < 0.2:
        return 'claude-sonnet'  # ELITE: Sonnet even for fast tier
    if complexity < 0.5:
        return 'claude-opus'    # ELITE: Opus for balanced
    return 'claude-opus'        # ELITE: Opus for deep


# =============================================================================
# COMPLEXITY THRESHOLDS
# =============================================================================

STANDARD_THRESHOLDS = {
    'balanced': 0.4,
    'deep': 0.75
}

ELITE_THRESHOLDS = {
    'balanced': 0.2,
    'deep': 0.5
}
