#!/usr/bin/env python3
"""
Cognitive Precision Bridge (CPB) - Orchestrator

Main coordinator for CPB pipeline execution.
Manages path routing, engine coordination, and quality verification.

ELITE TIER: 5-agent consensus ensemble, Opus-first routing, higher quality bar.

This is a context-aware orchestrator - it doesn't make direct LLM calls but
provides structured routing decisions and quality frameworks that can be
used by any LLM integration.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from .types import (
    CPBPath, CPBConfig, DQScore, ACE_AGENT_PERSONAS,
    DEFAULT_CPB_CONFIG
)
from .router import select_path, analyze_query, hash_query


# =============================================================================
# STORAGE PATHS
# =============================================================================

HOME = Path.home()
CPB_PATTERNS_FILE = HOME / ".claude/data/cpb-patterns.jsonl"
CPB_LEARNING_FILE = HOME / ".claude/data/cpb-learning.json"


# =============================================================================
# CPB ORCHESTRATOR
# =============================================================================

class CPBOrchestrator:
    """
    Main CPB orchestrator class.

    Provides structured routing, quality frameworks, and learning capabilities
    for precision-aware AI query execution.

    Usage:
        ```python
        from cpb import cpb

        # Analyze query complexity
        analysis = cpb.analyze("Design a distributed cache system")
        print(f"Path: {analysis['selected_path']} ({analysis['complexity_score']:.2f})")

        # Get routing decision
        decision = cpb.route("Compare microservices vs monolith")
        print(f"Use {decision.selected_path} path")

        # Build ACE consensus prompt
        prompts = cpb.build_ace_prompts("What's the best approach?")
        for p in prompts:
            print(f"{p['agent']}: {p['prompt'][:50]}...")

        # Score a response
        dq = cpb.score_response(query, response, context)
        print(f"DQ Score: {dq.overall:.2f}")
        ```
    """

    def __init__(self, config: CPBConfig = DEFAULT_CPB_CONFIG):
        self.config = config
        self._ensure_data_dirs()

    def _ensure_data_dirs(self):
        """Ensure data directories exist"""
        CPB_PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # ROUTING & ANALYSIS
    # =========================================================================

    def analyze(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a query without executing.
        Returns detailed complexity breakdown and path recommendation.
        """
        return analyze_query(query, context)

    def route(self, query: str, context: Optional[str] = None):
        """
        Get routing decision for a query.
        Returns RoutingDecision with selected path and alternatives.
        """
        return select_path(query, context, self.config)

    def should_orchestrate(self, query: str, context: Optional[str] = None) -> bool:
        """Check if query benefits from CPB orchestration"""
        analysis = self.analyze(query, context)
        return analysis['complexity_score'] >= 0.2

    # =========================================================================
    # ACE CONSENSUS BUILDING
    # =========================================================================

    def build_ace_prompts(
        self,
        query: str,
        context: Optional[str] = None,
        agent_count: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Build prompts for ACE (Adaptive Consensus Engine) multi-agent evaluation.

        ELITE TIER: 5-agent ensemble with diverse cognitive profiles:
        - Analyst: Evidence and logic focus
        - Skeptic: Challenges and risks
        - Synthesizer: Patterns and integration
        - Pragmatist: Feasibility and implementation
        - Visionary: Long-term and strategic

        Returns list of agent prompts ready for parallel execution.
        """
        count = agent_count or self.config.ace_config.agent_count
        agents = ACE_AGENT_PERSONAS[:count]

        prompts = []
        for agent in agents:
            system_prompt = agent['prompt']
            user_prompt = self._build_ace_user_prompt(query, context)

            prompts.append({
                'agent': agent['name'],
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'full_prompt': f"{system_prompt}\n\n---\n\n{user_prompt}"
            })

        return prompts

    def _build_ace_user_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build the user prompt for ACE agents"""
        parts = []

        if context:
            parts.append(f"## CONTEXT\n{context[:10000]}")  # Limit context size

        parts.append(f"## QUERY\n{query}")
        parts.append("""
## INSTRUCTIONS
Analyze this query from your unique perspective. Provide:
1. Your assessment (2-3 key points)
2. Confidence level (0-100%)
3. Key evidence or reasoning
4. Concerns or caveats

Be concise but thorough. Focus on your specialized viewpoint.""")

        return "\n\n".join(parts)

    def synthesize_ace_responses(
        self,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize multiple ACE agent responses into consensus.

        Args:
            responses: List of {agent, response, confidence} dicts

        Returns:
            Synthesis with agreement score, merged insights, and conflicts
        """
        if not responses:
            return {'error': 'No responses to synthesize'}

        # Extract confidences
        confidences = [r.get('confidence', 50) for r in responses]
        avg_confidence = sum(confidences) / len(confidences)

        # Calculate agreement (using text similarity heuristic)
        agreement = self._calculate_agreement([r.get('response', '') for r in responses])

        # Categorize by agent type
        by_agent = {r.get('agent', 'unknown'): r for r in responses}

        # Identify conflicts (responses with very different conclusions)
        conflicts = []
        if agreement < 0.5:
            conflicts.append("Significant disagreement detected between agents")

        return {
            'agent_count': len(responses),
            'avg_confidence': avg_confidence,
            'agreement_score': agreement,
            'by_agent': by_agent,
            'conflicts': conflicts,
            'consensus_strength': 'strong' if agreement > 0.7 else 'moderate' if agreement > 0.4 else 'weak'
        }

    def _calculate_agreement(self, responses: List[str]) -> float:
        """Calculate agreement level between responses (0-1)"""
        if len(responses) < 2:
            return 1.0

        # Extract keywords from each response
        def extract_keywords(text: str) -> set:
            words = text.lower().split()
            # Filter to meaningful words (>4 chars, not common)
            stopwords = {'this', 'that', 'with', 'from', 'they', 'have', 'will', 'would', 'could', 'should'}
            return {w for w in words if len(w) > 4 and w not in stopwords}

        keyword_sets = [extract_keywords(r) for r in responses]

        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                a, b = keyword_sets[i], keyword_sets[j]
                if a or b:
                    jaccard = len(a & b) / len(a | b) if (a | b) else 0
                    similarities.append(jaccard)

        return sum(similarities) / len(similarities) if similarities else 0.5

    # =========================================================================
    # DQ SCORING
    # =========================================================================

    def score_response(
        self,
        query: str,
        response: str,
        context: Optional[str] = None
    ) -> DQScore:
        """
        Score a response using DQ (Decisional Quality) framework.

        DQ Score = Validity (40%) + Specificity (30%) + Correctness (30%)

        Note: This provides structural scoring. For semantic scoring,
        integrate with an LLM to evaluate the response quality.
        """
        # Validity: Does the response address the query?
        validity = self._score_validity(query, response)

        # Specificity: Is the response specific enough?
        specificity = self._score_specificity(response)

        # Correctness: Does it seem factually grounded?
        correctness = self._score_correctness(response)

        # Weighted overall
        overall = (validity * 0.4) + (specificity * 0.3) + (correctness * 0.3)

        return DQScore(
            overall=round(overall, 3),
            validity=round(validity, 3),
            specificity=round(specificity, 3),
            correctness=round(correctness, 3)
        )

    def _score_validity(self, query: str, response: str) -> float:
        """Score how well response addresses the query"""
        query_keywords = set(query.lower().split())
        response_lower = response.lower()

        # Check if key query terms appear in response
        matches = sum(1 for kw in query_keywords if kw in response_lower and len(kw) > 3)
        keyword_coverage = min(1.0, matches / max(1, len([kw for kw in query_keywords if len(kw) > 3])))

        # Check for direct address patterns
        has_direct_address = any(
            phrase in response_lower
            for phrase in ['to answer', 'regarding your', 'the answer', 'in response']
        )

        return (keyword_coverage * 0.7) + (0.3 if has_direct_address else 0.15)

    def _score_specificity(self, response: str) -> float:
        """Score how specific/detailed the response is"""
        words = response.split()
        word_count = len(words)

        # Length factor (longer = more specific, up to a point)
        length_score = min(1.0, word_count / 200)

        # Check for specific indicators
        has_numbers = bool(__import__('re').search(r'\d+', response))
        has_examples = any(
            phrase in response.lower()
            for phrase in ['for example', 'such as', 'e.g.', 'specifically', 'in particular']
        )
        has_lists = response.count('\n-') > 1 or response.count('\n*') > 1

        specificity_bonus = (
            (0.15 if has_numbers else 0) +
            (0.15 if has_examples else 0) +
            (0.1 if has_lists else 0)
        )

        return min(1.0, (length_score * 0.6) + specificity_bonus + 0.1)

    def _score_correctness(self, response: str) -> float:
        """Score apparent correctness (heuristic without verification)"""
        response_lower = response.lower()

        # Hedging (slight negative - might indicate uncertainty)
        hedging_phrases = ['might', 'maybe', 'possibly', 'i think', 'not sure', 'uncertain']
        hedging_count = sum(1 for phrase in hedging_phrases if phrase in response_lower)

        # Confidence indicators (positive)
        confidence_phrases = ['clearly', 'certainly', 'definitely', 'the fact is', 'evidence shows']
        confidence_count = sum(1 for phrase in confidence_phrases if phrase in response_lower)

        # Citation patterns (positive)
        has_citations = bool(__import__('re').search(r'arxiv|doi|http|source:|according to', response_lower))

        base_score = 0.6  # Default assumption
        hedging_penalty = min(0.2, hedging_count * 0.05)
        confidence_bonus = min(0.2, confidence_count * 0.1)
        citation_bonus = 0.15 if has_citations else 0

        return min(1.0, base_score - hedging_penalty + confidence_bonus + citation_bonus)

    # =========================================================================
    # PATTERN LEARNING
    # =========================================================================

    def store_pattern(
        self,
        query: str,
        path: CPBPath,
        execution_time_ms: int,
        dq_score: float,
        success: bool,
        context_length: int = 0
    ):
        """Store execution pattern for learning"""
        if not self.config.enable_learning:
            return

        pattern = {
            'ts': time.time(),
            'query_hash': hash_query(query),
            'path': path.value,
            'execution_time_ms': execution_time_ms,
            'dq_score': dq_score,
            'success': success,
            'context_length': context_length,
            'complexity': self.analyze(query).get('complexity_score', 0)
        }

        with open(CPB_PATTERNS_FILE, 'a') as f:
            f.write(json.dumps(pattern) + '\n')

    def get_learned_preferences(self) -> Dict[str, Any]:
        """Get learned routing preferences from historical patterns"""
        if not CPB_PATTERNS_FILE.exists():
            return {'message': 'No patterns recorded yet'}

        patterns = []
        with open(CPB_PATTERNS_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        patterns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not patterns:
            return {'message': 'No valid patterns found'}

        # Analyze by path
        by_path = {}
        for p in patterns:
            path = p.get('path', 'unknown')
            if path not in by_path:
                by_path[path] = {'count': 0, 'dq_scores': [], 'times': [], 'successes': 0}

            by_path[path]['count'] += 1
            by_path[path]['dq_scores'].append(p.get('dq_score', 0))
            by_path[path]['times'].append(p.get('execution_time_ms', 0))
            if p.get('success'):
                by_path[path]['successes'] += 1

        # Calculate averages
        summary = {}
        for path, data in by_path.items():
            summary[path] = {
                'count': data['count'],
                'avg_dq': sum(data['dq_scores']) / len(data['dq_scores']) if data['dq_scores'] else 0,
                'avg_time_ms': sum(data['times']) / len(data['times']) if data['times'] else 0,
                'success_rate': data['successes'] / data['count'] if data['count'] else 0
            }

        return {
            'total_patterns': len(patterns),
            'by_path': summary,
            'recommended_default': max(summary.keys(), key=lambda p: summary[p]['avg_dq']) if summary else 'cascade'
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current CPB status and configuration"""
        return {
            'config': {
                'auto_route': self.config.auto_route,
                'default_path': self.config.default_path.value,
                'context_threshold': self.config.context_threshold,
                'complexity_threshold': self.config.complexity_threshold,
                'dq_threshold': self.config.dq_threshold,
                'ace_agent_count': self.config.ace_config.agent_count,
                'rlm_max_iterations': self.config.rlm_config.max_iterations
            },
            'tier': 'elite',
            'learning_enabled': self.config.enable_learning,
            'verification_enabled': self.config.enable_verification
        }

    def validate_config(self) -> List[str]:
        """Validate current configuration and return any warnings"""
        warnings = []

        if self.config.dq_threshold > 0.9:
            warnings.append("DQ threshold very high (>0.9) - may cause frequent retries")

        if self.config.ace_config.agent_count < 3:
            warnings.append("ACE agent count low (<3) - consensus may be unreliable")

        if self.config.complexity_threshold < 0.1:
            warnings.append("Complexity threshold very low (<0.1) - most queries will use heavy paths")

        return warnings


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

cpb = CPBOrchestrator()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Analyze query complexity and get routing recommendation"""
    return cpb.analyze(query, context)


def route(query: str, context: Optional[str] = None):
    """Get routing decision for query"""
    return cpb.route(query, context)


def should_orchestrate(query: str, context: Optional[str] = None) -> bool:
    """Check if query benefits from CPB orchestration"""
    return cpb.should_orchestrate(query, context)


def build_ace_prompts(query: str, context: Optional[str] = None) -> List[Dict[str, str]]:
    """Build ACE multi-agent consensus prompts"""
    return cpb.build_ace_prompts(query, context)


def score_response(query: str, response: str, context: Optional[str] = None) -> DQScore:
    """Score response quality using DQ framework"""
    return cpb.score_response(query, response, context)


# =============================================================================
# PRE/POST EXECUTION HOOKS
# =============================================================================

class CPBHooks:
    """
    Hooks for extending CPB execution.

    Register callbacks to run before/after various stages.
    Enables integration with precision mode and external systems.

    Usage:
        from cpb.orchestrator import cpb_hooks

        @cpb_hooks.on_pre_analyze
        def my_pre_analyze(query, context):
            # Enrich context before analysis
            return query, context

        @cpb_hooks.on_post_score
        def my_post_score(result):
            # Log or modify score
            return result
    """

    def __init__(self):
        self._pre_analyze: List[Callable] = []
        self._post_analyze: List[Callable] = []
        self._pre_route: List[Callable] = []
        self._post_route: List[Callable] = []
        self._pre_ace: List[Callable] = []
        self._post_ace: List[Callable] = []
        self._pre_score: List[Callable] = []
        self._post_score: List[Callable] = []

    def on_pre_analyze(self, func: Callable) -> Callable:
        """Decorator to register pre-analyze hook."""
        self._pre_analyze.append(func)
        return func

    def on_post_analyze(self, func: Callable) -> Callable:
        """Decorator to register post-analyze hook."""
        self._post_analyze.append(func)
        return func

    def on_pre_route(self, func: Callable) -> Callable:
        """Decorator to register pre-route hook."""
        self._pre_route.append(func)
        return func

    def on_post_route(self, func: Callable) -> Callable:
        """Decorator to register post-route hook."""
        self._post_route.append(func)
        return func

    def on_pre_ace(self, func: Callable) -> Callable:
        """Decorator to register pre-ACE hook."""
        self._pre_ace.append(func)
        return func

    def on_post_ace(self, func: Callable) -> Callable:
        """Decorator to register post-ACE hook."""
        self._post_ace.append(func)
        return func

    def on_pre_score(self, func: Callable) -> Callable:
        """Decorator to register pre-score hook."""
        self._pre_score.append(func)
        return func

    def on_post_score(self, func: Callable) -> Callable:
        """Decorator to register post-score hook."""
        self._post_score.append(func)
        return func

    def run_pre_hooks(self, hook_type: str, *args) -> tuple:
        """Run all pre-hooks of a given type."""
        hooks = getattr(self, f'_pre_{hook_type}', [])
        result = args
        for hook in hooks:
            try:
                result = hook(*result)
                if not isinstance(result, tuple):
                    result = (result,)
            except Exception:
                # Log but don't break execution
                pass
        return result

    def run_post_hooks(self, hook_type: str, result: Any) -> Any:
        """Run all post-hooks of a given type."""
        hooks = getattr(self, f'_post_{hook_type}', [])
        for hook in hooks:
            try:
                result = hook(result)
            except Exception:
                # Log but don't break execution
                pass
        return result

    def clear_hooks(self, hook_type: Optional[str] = None):
        """Clear registered hooks."""
        if hook_type:
            setattr(self, f'_pre_{hook_type}', [])
            setattr(self, f'_post_{hook_type}', [])
        else:
            self._pre_analyze = []
            self._post_analyze = []
            self._pre_route = []
            self._post_route = []
            self._pre_ace = []
            self._post_ace = []
            self._pre_score = []
            self._post_score = []


# Global hooks instance
cpb_hooks = CPBHooks()
