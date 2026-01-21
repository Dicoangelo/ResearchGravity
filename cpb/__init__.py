#!/usr/bin/env python3
"""
Cognitive Precision Bridge (CPB) - Python Implementation

Unified precision-aware AI orchestration for ResearchGravity.

## Quick Start

```python
from cpb import cpb, analyze, score_response

# Analyze query complexity
result = analyze("Design a distributed cache system")
print(f"Complexity: {result['complexity_score']:.2f}")
print(f"Recommended path: {result['selected_path']}")

# Get routing decision
decision = cpb.route("Compare REST vs GraphQL for our API")
print(f"Path: {decision.selected_path}")
print(f"Reasoning: {decision.reasoning}")

# Build ACE consensus prompts
prompts = cpb.build_ace_prompts("What's the best auth strategy?")
for p in prompts:
    print(f"[{p['agent']}] {p['system_prompt'][:50]}...")

# Score a response
dq = score_response(
    query="Explain microservices",
    response="Microservices are an architectural style..."
)
print(f"DQ Score: {dq.overall:.2f}")
```

## Architecture

CPB provides:
- **Router**: Complexity analysis and path selection
- **Orchestrator**: ACE consensus building, quality frameworks
- **DQ Scorer**: Response quality measurement

## ELITE TIER

Default configuration uses Elite tier settings:
- Lower complexity thresholds (more orchestration)
- 5-agent ACE ensemble
- Higher DQ quality bar (0.75)
- Opus-first model recommendations
"""

# Types
from .types import (
    # Path types
    CPBPath,
    CPBPhase,
    ReasoningModel,

    # Configuration
    CPBConfig,
    RLMConfig,
    ACEConfig,
    DEFAULT_CPB_CONFIG,
    STANDARD_CPB_CONFIG,

    # Signals & Results
    PathSignals,
    CPBStatus,
    CPBResult,
    DQScore,
    RoutingDecision,
    PathAlternative,

    # Requests
    CPBRequest,

    # Learning
    CPBPattern,
    LearnedRouting,

    # Constants
    ACE_AGENT_PERSONAS,
)

# Router functions
from .router import (
    extract_complexity_signals,
    calculate_complexity_score,
    select_path,
    should_orchestrate,
    analyze_query,
    hash_query,
    get_reasoning_tier,
    get_model_recommendation,
    STANDARD_THRESHOLDS,
    ELITE_THRESHOLDS,
)

# Orchestrator
from .orchestrator import (
    CPBOrchestrator,
    cpb,  # Singleton instance
    analyze,
    route,
    build_ace_prompts,
    score_response,
)

# DQ Scorer
from .dq_scorer import (
    DQScorer,
    dq_scorer,  # Singleton instance
    score,
    log_score,
    get_stats,
    meets_threshold,
)

# Precision Mode (lazy imports to avoid circular deps)
def get_precision_config():
    """Get precision mode configuration."""
    from .precision_config import PRECISION_CONFIG
    return PRECISION_CONFIG

def get_precision_orchestrator():
    """Get precision orchestrator singleton."""
    from .precision_orchestrator import precision_orchestrator
    return precision_orchestrator

async def execute_precision(query: str, context=None, on_status=None):
    """Execute precision mode query (v2 with tiered search)."""
    from .precision_orchestrator import execute_precision as _execute
    return await _execute(query, context, on_status)

# Search Layer (v2)
def get_search_layer():
    """Get tiered search layer singleton."""
    from .search_layer import get_search_layer as _get
    return _get()

async def search_tiered(query: str, max_results: int = 30):
    """Execute tiered search using ResearchGravity methodology."""
    from .search_layer import search_tiered as _search
    return await _search(query, max_results)

# Hooks
from .orchestrator import cpb_hooks

# Ground Truth (v2) + Corpus (v2.2)
from .ground_truth import (
    GroundTruthValidator,
    GroundTruthClaim,
    ValidationResult as GroundTruthResult,
    ClaimExtractor,
    CrossSourceValidator,
    SelfConsistencyChecker,
    FeedbackCollector,
    TruthSource,
    get_validator as get_ground_truth_validator,
    validate_against_ground_truth,
    record_feedback,
    # v2.2: Ground Truth Corpus
    GroundTruthCorpus,
    get_corpus as get_ground_truth_corpus,
    store_verified_claims,
)

__version__ = '2.5.0'  # v2.5: Comprehensive hardening (caching, retry, fallback, cost tracking)


# =============================================================================
# DEPENDENCY CHECK (v2.5)
# =============================================================================

def check_dependencies() -> dict:
    """
    Check installed CPB dependencies and their versions.

    Returns:
        Dict with dependency status:
        {
            'aiohttp': {'installed': True, 'version': '3.9.0'},
            'google-genai': {'installed': False, 'version': None, 'message': 'Not installed'},
            ...
        }
    """
    import importlib.metadata

    dependencies = [
        'aiohttp',
        'google-genai',
        'anthropic',
        'pytest',
        'pytest-asyncio',
        'arxiv',
        'cohere',
    ]

    result = {}

    for dep in dependencies:
        try:
            version = importlib.metadata.version(dep)
            result[dep] = {
                'installed': True,
                'version': version,
            }
        except importlib.metadata.PackageNotFoundError:
            result[dep] = {
                'installed': False,
                'version': None,
                'message': 'Not installed',
            }

    return result


def get_deep_research_status() -> dict:
    """
    Get comprehensive deep research provider status.

    Returns:
        Dict with provider availability:
        {
            'gemini': {'available': True, 'message': 'Ready'},
            'perplexity': {'available': False, 'message': 'API key not found'},
            'best_provider': 'gemini',
            'cache_stats': {...},
        }
    """
    from .deep_research import (
        check_deep_research_available,
        get_best_available_provider,
        get_cache_stats,
    )

    gemini_available, gemini_msg = check_deep_research_available("gemini")
    perplexity_available, perplexity_msg = check_deep_research_available("perplexity")
    best_provider, _ = get_best_available_provider()

    return {
        'gemini': {
            'available': gemini_available,
            'message': gemini_msg,
        },
        'perplexity': {
            'available': perplexity_available,
            'message': perplexity_msg,
        },
        'best_provider': best_provider,
        'cache_stats': get_cache_stats(),
    }
__all__ = [
    # Types
    'CPBPath',
    'CPBPhase',
    'ReasoningModel',
    'CPBConfig',
    'RLMConfig',
    'ACEConfig',
    'DEFAULT_CPB_CONFIG',
    'STANDARD_CPB_CONFIG',
    'PathSignals',
    'CPBStatus',
    'CPBResult',
    'DQScore',
    'RoutingDecision',
    'PathAlternative',
    'CPBRequest',
    'CPBPattern',
    'LearnedRouting',
    'ACE_AGENT_PERSONAS',

    # Router
    'extract_complexity_signals',
    'calculate_complexity_score',
    'select_path',
    'should_orchestrate',
    'analyze_query',
    'hash_query',
    'get_reasoning_tier',
    'get_model_recommendation',
    'STANDARD_THRESHOLDS',
    'ELITE_THRESHOLDS',

    # Orchestrator
    'CPBOrchestrator',
    'cpb',
    'analyze',
    'route',
    'build_ace_prompts',
    'score_response',

    # DQ Scorer
    'DQScorer',
    'dq_scorer',
    'score',
    'log_score',
    'get_stats',
    'meets_threshold',

    # Precision Mode v2
    'get_precision_config',
    'get_precision_orchestrator',
    'execute_precision',
    'get_search_layer',
    'search_tiered',

    # Hooks
    'cpb_hooks',

    # Ground Truth (v2.1) + Corpus (v2.2)
    'GroundTruthValidator',
    'GroundTruthClaim',
    'GroundTruthResult',
    'ClaimExtractor',
    'CrossSourceValidator',
    'SelfConsistencyChecker',
    'FeedbackCollector',
    'TruthSource',
    'get_ground_truth_validator',
    'validate_against_ground_truth',
    'record_feedback',
    # v2.2
    'GroundTruthCorpus',
    'get_ground_truth_corpus',
    'store_verified_claims',

    # v2.5: Dependency check
    'check_dependencies',
    'get_deep_research_status',
]
