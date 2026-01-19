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

__version__ = '1.0.0'
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
]
