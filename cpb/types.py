#!/usr/bin/env python3
"""
Cognitive Precision Bridge (CPB) - Types & Configuration

Unifies RLM, ACE, and DQ scoring into a single precision-aware pipeline.

CPB Pattern: COMPRESS → PRE-COMPUTE → PARALLEL EXPLORE → ACCUMULATE → RECONSTRUCT → VERIFY

Research Foundation:
- arXiv:2512.24601 (RLM) - Context externalization
- arXiv:2511.15755 (DQ) - Quality measurement
- arXiv:2508.17536 (Voting vs Debate) - Consensus strategies

ELITE TIER: Maximum quality configuration with Opus-first routing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Literal
from enum import Enum


# =============================================================================
# EXECUTION PATH TYPES
# =============================================================================

class CPBPath(str, Enum):
    """Available execution paths through the CPB"""
    DIRECT = 'direct'       # Simple query, no CPB needed
    RLM = 'rlm'            # Long context → RLM handles compression
    ACE = 'ace'            # Multi-perspective → ACE consensus
    HYBRID = 'hybrid'      # Complex → RLM for context, ACE for decision
    CASCADE = 'cascade'    # Expert → Full pipeline with verification


class CPBPhase(str, Enum):
    """CPB execution phases"""
    IDLE = 'idle'
    ANALYZING = 'analyzing'          # Determining optimal path
    COMPRESSING = 'compressing'      # RLM context compression
    EXPLORING = 'exploring'          # Parallel exploration
    CONVERGING = 'converging'        # ACE consensus
    VERIFYING = 'verifying'          # DQ verification
    RECONSTRUCTING = 'reconstructing'  # Final synthesis
    COMPLETE = 'complete'
    ERROR = 'error'


class ReasoningModel(str, Enum):
    """Model selection preferences"""
    GEMINI_FLASH = 'gemini-flash'
    GEMINI_PRO = 'gemini-pro'
    CLAUDE_HAIKU = 'claude-haiku'
    CLAUDE_SONNET = 'claude-sonnet'
    CLAUDE_OPUS = 'claude-opus'
    AUTO = 'auto'


# =============================================================================
# PATH SIGNALS
# =============================================================================

@dataclass
class PathSignals:
    """Characteristics that determine optimal path"""
    context_length: int = 0
    query_complexity: float = 0.0
    requires_consensus: bool = False
    requires_reasoning: bool = False
    has_ground_truth: bool = False
    time_budget_ms: int = 30000
    quality_target: float = 0.75  # 0-1 DQ threshold


# =============================================================================
# CPB CONFIGURATION
# =============================================================================

@dataclass
class RLMConfig:
    """RLM (Recursive Language Model) configuration"""
    max_iterations: int = 25          # ELITE: Deeper decomposition
    root_model: str = 'deep'          # ELITE: Opus for root synthesis
    sub_model: str = 'balanced'       # ELITE: Sonnet for sub-tasks


@dataclass
class ACEConfig:
    """ACE (Adaptive Consensus Engine) configuration"""
    max_rounds: int = 18              # ELITE: More consensus rounds
    agent_count: int = 5              # ELITE: 5-agent ensemble
    enable_auction: bool = True
    enable_hop_grouping: bool = True


@dataclass
class CPBConfig:
    """Full CPB configuration"""
    # Path selection
    auto_route: bool = True
    default_path: CPBPath = CPBPath.CASCADE  # ELITE: Full pipeline by default

    # Thresholds
    context_threshold: int = 100000        # ELITE: Handle larger contexts (~25k tokens)
    complexity_threshold: float = 0.35     # ELITE: Lower threshold → more consensus
    dq_threshold: float = 0.75             # ELITE: Higher quality bar

    # Time budgets (ms)
    fast_path_ms: int = 8000               # ELITE: More time for quality
    standard_path_ms: int = 45000          # ELITE: Extended for deep reasoning
    hybrid_path_ms: int = 90000            # ELITE: Full pipeline allowance

    # Quality settings
    enable_verification: bool = True       # Run DQ verification pass
    enable_learning: bool = True           # Store patterns for learning
    retry_on_low_dq: bool = True           # Auto-retry if DQ below threshold

    # Engine configs
    rlm_config: RLMConfig = field(default_factory=RLMConfig)
    ace_config: ACEConfig = field(default_factory=ACEConfig)


# Default configurations
DEFAULT_CPB_CONFIG = CPBConfig()

# Standard tier (cost-conscious)
STANDARD_CPB_CONFIG = CPBConfig(
    default_path=CPBPath.DIRECT,
    context_threshold=50000,
    complexity_threshold=0.5,
    dq_threshold=0.6,
    fast_path_ms=5000,
    standard_path_ms=30000,
    hybrid_path_ms=60000,
    rlm_config=RLMConfig(max_iterations=10, root_model='balanced', sub_model='fast'),
    ace_config=ACEConfig(max_rounds=8, agent_count=3)
)


# =============================================================================
# CPB STATUS & RESULTS
# =============================================================================

@dataclass
class CPBStatus:
    """Real-time status updates"""
    phase: CPBPhase = CPBPhase.IDLE
    path: CPBPath = CPBPath.DIRECT
    progress: int = 0                      # 0-100%
    current_engine: Optional[str] = None   # 'rlm' | 'ace' | 'dq'
    engine_status: Optional[Dict[str, Any]] = None
    elapsed_ms: int = 0
    estimated_remaining_ms: int = 0
    message: Optional[str] = None


@dataclass
class DQScore:
    """DQ (Decisional Quality) score breakdown"""
    overall: float = 0.0
    validity: float = 0.0       # 40% weight - Is the reasoning valid?
    specificity: float = 0.0    # 30% weight - Is it specific enough?
    correctness: float = 0.0    # 30% weight - Is it factually correct?


@dataclass
class CPBResult:
    """Final CPB result"""
    # Output
    output: str = ''
    confidence: float = 0.0          # 0-100

    # Execution metadata
    path: CPBPath = CPBPath.DIRECT
    execution_time_ms: int = 0
    tokens_used: int = 0

    # Quality metrics
    dq_score: DQScore = field(default_factory=DQScore)
    verified: bool = False
    retry_count: int = 0

    # Engine results
    rlm_result: Optional[Dict[str, Any]] = None
    ace_result: Optional[Dict[str, Any]] = None

    # Path analysis
    path_signals: PathSignals = field(default_factory=PathSignals)
    path_reasoning: str = ''

    # Learning
    pattern_stored: bool = False


# =============================================================================
# ROUTING DECISION
# =============================================================================

@dataclass
class PathAlternative:
    """Alternative path option"""
    path: CPBPath
    score: float
    tradeoff: str


@dataclass
class RoutingDecision:
    """Path selection decision with reasoning"""
    selected_path: CPBPath
    signals: PathSignals
    reasoning: str
    confidence: float
    alternatives: List[PathAlternative] = field(default_factory=list)


# =============================================================================
# ACE AGENT PERSONAS (5-Agent Elite Ensemble)
# =============================================================================

ACE_AGENT_PERSONAS = [
    {
        'name': 'Analyst',
        'prompt': '''You are a rigorous analyst. Examine this objectively, focusing on data,
evidence, and logical consistency. Identify what can be proven vs. assumed.
Be precise and cite specific evidence for claims.'''
    },
    {
        'name': 'Skeptic',
        'prompt': '''You are a critical skeptic. Challenge every assumption, identify potential
failure modes, edge cases, and risks. Ask "what could go wrong?" and "what are we missing?"
Don't accept claims without strong justification.'''
    },
    {
        'name': 'Synthesizer',
        'prompt': '''You are a systems thinker. Find deep connections, identify emergent patterns,
and integrate diverse perspectives. Look for underlying principles that unify different viewpoints.
Seek coherent frameworks.'''
    },
    {
        'name': 'Pragmatist',
        'prompt': '''You are a practical implementer. Focus on actionability, resource constraints,
and real-world feasibility. Ask "how would this actually work?" and "what's the simplest
approach that achieves the goal?"'''
    },
    {
        'name': 'Visionary',
        'prompt': '''You are a strategic visionary. Think long-term, consider second-order effects,
and explore novel possibilities. Ask "what could this become?" and "what paradigm shifts
might be relevant?"'''
    }
]


# =============================================================================
# CPB REQUEST
# =============================================================================

@dataclass
class CPBRequest:
    """Input request to CPB"""
    # Core request
    query: str
    context: Optional[str] = None

    # Optional overrides
    force_path: Optional[CPBPath] = None
    force_model: Optional[ReasoningModel] = None
    time_budget_ms: Optional[int] = None
    quality_target: Optional[float] = None

    # Callbacks
    on_status: Optional[Callable[[CPBStatus], None]] = None


# =============================================================================
# CPB MEMORY & LEARNING
# =============================================================================

@dataclass
class CPBPattern:
    """Historical execution pattern"""
    id: str
    query_hash: str
    timestamp: float

    # Request characteristics
    context_length: int
    query_complexity: float

    # Execution details
    path: CPBPath
    execution_time_ms: int
    tokens_used: int

    # Quality
    dq_score: float
    verified: bool

    # Outcome
    success: bool
    retries: int


@dataclass
class LearnedRouting:
    """Learned routing preferences"""
    domain: str
    preferred_path: CPBPath
    avg_dq: float
    avg_time: float
    sample_count: int
    confidence: float
