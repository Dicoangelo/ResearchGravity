#!/usr/bin/env python3
"""
Cognitive Precision Bridge (CPB) - DQ Scorer

Integrates with ResearchGravity's existing routing-metrics and confidence_scorer
to provide unified quality measurement.

DQ Score = Validity (40%) + Specificity (30%) + Correctness (30%)

Based on arXiv:2511.15755 (DQ) - Quality measurement framework.

ELITE TIER: Higher quality thresholds, more rigorous validation.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .types import DQScore, CPBPath


# =============================================================================
# STORAGE
# =============================================================================

HOME = Path.home()
DQ_SCORES_FILE = HOME / ".claude/kernel/dq-scores.jsonl"
ROUTING_METRICS_FILE = HOME / ".claude/data/routing-metrics.jsonl"


# =============================================================================
# DQ SCORER
# =============================================================================

class DQScorer:
    """
    DQ (Decisional Quality) Scorer for CPB.

    Provides quality measurement for AI responses with integration into
    the ResearchGravity routing metrics infrastructure.

    Quality Targets (ELITE TIER):
    - Minimum acceptable: 0.60
    - Good: 0.75
    - Excellent: 0.85+

    Components:
    - Validity (40%): Does the response address the query?
    - Specificity (30%): Is it detailed and actionable?
    - Correctness (30%): Is it factually grounded?
    """

    # ELITE TIER thresholds
    THRESHOLD_MIN = 0.60
    THRESHOLD_GOOD = 0.75
    THRESHOLD_EXCELLENT = 0.85

    def __init__(self):
        self._ensure_storage()

    def _ensure_storage(self):
        """Ensure storage directories exist"""
        DQ_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
        ROUTING_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    def score(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> DQScore:
        """
        Calculate DQ score for a query-response pair.

        Args:
            query: The original query
            response: The AI response
            context: Optional context provided
            ground_truth: Optional ground truth for correctness validation

        Returns:
            DQScore with overall and component scores
        """
        validity = self._score_validity(query, response, context)
        specificity = self._score_specificity(response)
        correctness = self._score_correctness(response, ground_truth)

        overall = (validity * 0.4) + (specificity * 0.3) + (correctness * 0.3)

        return DQScore(
            overall=round(overall, 3),
            validity=round(validity, 3),
            specificity=round(specificity, 3),
            correctness=round(correctness, 3)
        )

    def _score_validity(
        self,
        query: str,
        response: str,
        context: Optional[str] = None
    ) -> float:
        """
        Score validity: Does the response address the query?

        Factors:
        - Query keyword coverage in response
        - Direct address patterns
        - Context relevance (if provided)
        """
        import re

        query_lower = query.lower()
        response_lower = response.lower()

        # Extract meaningful query words (>3 chars)
        query_words = set(
            w for w in re.findall(r'\b\w+\b', query_lower)
            if len(w) > 3
        )

        # Count matches in response
        if query_words:
            matches = sum(1 for w in query_words if w in response_lower)
            keyword_coverage = min(1.0, matches / len(query_words))
        else:
            keyword_coverage = 0.5

        # Check for direct address patterns
        direct_patterns = [
            r'to answer your',
            r'regarding your question',
            r'the answer is',
            r'in response to',
            r'you asked about',
            r'addressing your query'
        ]
        has_direct_address = any(
            re.search(p, response_lower) for p in direct_patterns
        )

        # Context relevance bonus
        context_bonus = 0
        if context:
            context_words = set(
                w for w in re.findall(r'\b\w+\b', context.lower())
                if len(w) > 4
            )[:50]  # Limit to top 50 words
            if context_words:
                context_matches = sum(1 for w in context_words if w in response_lower)
                context_bonus = min(0.1, context_matches / len(context_words) * 0.2)

        validity = (keyword_coverage * 0.65) + (0.25 if has_direct_address else 0.1) + context_bonus

        return min(1.0, validity)

    def _score_specificity(self, response: str) -> float:
        """
        Score specificity: Is the response detailed and actionable?

        Factors:
        - Response length (reasonable length preferred)
        - Presence of specific details (numbers, examples)
        - Structure (lists, sections)
        - Technical depth indicators
        """
        import re

        words = response.split()
        word_count = len(words)

        # Length factor (optimal around 100-500 words)
        if word_count < 20:
            length_score = 0.2
        elif word_count < 50:
            length_score = 0.4
        elif word_count < 100:
            length_score = 0.6
        elif word_count < 500:
            length_score = 0.8
        else:
            length_score = 0.7  # Very long might be verbose

        # Specific details
        has_numbers = bool(re.search(r'\b\d+\b', response))
        has_percentages = bool(re.search(r'\d+%', response))
        has_examples = bool(re.search(r'(for example|such as|e\.g\.|i\.e\.|specifically|in particular)', response.lower()))
        has_code = bool(re.search(r'```|`[^`]+`', response))

        detail_score = (
            (0.15 if has_numbers else 0) +
            (0.1 if has_percentages else 0) +
            (0.15 if has_examples else 0) +
            (0.1 if has_code else 0)
        )

        # Structure
        has_lists = bool(re.search(r'\n[-*•]\s', response))
        has_sections = bool(re.search(r'\n#{1,3}\s|\n\*\*[^*]+\*\*\n', response))
        has_numbered = bool(re.search(r'\n\d+\.\s', response))

        structure_score = (
            (0.1 if has_lists else 0) +
            (0.1 if has_sections else 0) +
            (0.05 if has_numbered else 0)
        )

        # Technical depth (presence of technical terms)
        technical_patterns = [
            r'algorithm', r'implementation', r'architecture',
            r'performance', r'optimization', r'latency',
            r'throughput', r'complexity', r'scalab'
        ]
        technical_count = sum(
            1 for p in technical_patterns
            if re.search(p, response.lower())
        )
        technical_score = min(0.15, technical_count * 0.03)

        specificity = (length_score * 0.4) + detail_score + structure_score + technical_score

        return min(1.0, specificity)

    def _score_correctness(
        self,
        response: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Score correctness: Is the response factually grounded?

        Factors:
        - Citation/source references
        - Confidence language
        - Hedging/uncertainty indicators
        - Ground truth match (if provided)
        """
        import re

        response_lower = response.lower()

        base_score = 0.55  # Default assumption

        # Citations and sources (strong positive)
        citation_patterns = [
            r'arxiv', r'doi:', r'https?://',
            r'according to', r'research shows',
            r'studies indicate', r'source:', r'reference:'
        ]
        citation_count = sum(
            1 for p in citation_patterns
            if re.search(p, response_lower)
        )
        citation_bonus = min(0.2, citation_count * 0.05)

        # Confidence indicators (moderate positive)
        confidence_patterns = [
            r'clearly', r'certainly', r'definitely',
            r'the fact is', r'evidence shows', r'proven'
        ]
        confidence_count = sum(
            1 for p in confidence_patterns
            if re.search(p, response_lower)
        )
        confidence_bonus = min(0.15, confidence_count * 0.05)

        # Hedging indicators (slight negative)
        hedging_patterns = [
            r'\bmight\b', r'\bmaybe\b', r'\bpossibly\b',
            r"i think", r"not sure", r"uncertain",
            r"i'm not certain", r"it could be"
        ]
        hedging_count = sum(
            1 for p in hedging_patterns
            if re.search(p, response_lower)
        )
        hedging_penalty = min(0.15, hedging_count * 0.03)

        # Ground truth comparison (if provided)
        ground_truth_bonus = 0
        if ground_truth:
            # Simple overlap check
            gt_words = set(ground_truth.lower().split())
            resp_words = set(response_lower.split())
            overlap = len(gt_words & resp_words) / max(1, len(gt_words))
            ground_truth_bonus = overlap * 0.2

        correctness = (
            base_score +
            citation_bonus +
            confidence_bonus -
            hedging_penalty +
            ground_truth_bonus
        )

        return max(0.0, min(1.0, correctness))

    # =========================================================================
    # PERSISTENCE & TRACKING
    # =========================================================================

    def log_score(
        self,
        query: str,
        response: str,
        dq_score: DQScore,
        model: str = 'unknown',
        path: Optional[CPBPath] = None,
        complexity: float = 0.0
    ):
        """Log DQ score to metrics file for tracking"""
        entry = {
            'ts': int(time.time() * 1000),
            'query': query[:200],  # Truncate for storage
            'dq': dq_score.overall,
            'dqScore': dq_score.overall,  # Legacy field name
            'validity': dq_score.validity,
            'specificity': dq_score.specificity,
            'correctness': dq_score.correctness,
            'model': model,
            'complexity': complexity,
            'path': path.value if path else None
        }

        with open(DQ_SCORES_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_recent_scores(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent DQ scores"""
        if not DQ_SCORES_FILE.exists():
            return []

        scores = []
        with open(DQ_SCORES_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        scores.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return scores[-limit:]

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get DQ score statistics"""
        scores = self.get_recent_scores(1000)

        if not scores:
            return {'message': 'No scores recorded'}

        # Filter by time
        cutoff = time.time() * 1000 - (days * 24 * 60 * 60 * 1000)
        recent = [s for s in scores if s.get('ts', 0) > cutoff]

        if not recent:
            return {'message': f'No scores in last {days} days'}

        dq_values = [s.get('dq', 0) for s in recent]

        return {
            'period_days': days,
            'total_scored': len(recent),
            'avg_dq': sum(dq_values) / len(dq_values),
            'min_dq': min(dq_values),
            'max_dq': max(dq_values),
            'above_threshold': sum(1 for d in dq_values if d >= self.THRESHOLD_GOOD),
            'below_min': sum(1 for d in dq_values if d < self.THRESHOLD_MIN),
            'by_model': self._group_by_field(recent, 'model'),
            'by_path': self._group_by_field(recent, 'path')
        }

    def _group_by_field(self, scores: List[Dict], field: str) -> Dict[str, Any]:
        """Group scores by a field and calculate stats"""
        groups = {}
        for s in scores:
            key = s.get(field, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(s.get('dq', 0))

        return {
            k: {
                'count': len(v),
                'avg_dq': sum(v) / len(v) if v else 0
            }
            for k, v in groups.items()
        }

    # =========================================================================
    # QUALITY VALIDATION
    # =========================================================================

    def meets_threshold(self, dq_score: DQScore, threshold: Optional[float] = None) -> bool:
        """Check if DQ score meets threshold"""
        target = threshold if threshold is not None else self.THRESHOLD_GOOD
        return dq_score.overall >= target

    def get_quality_tier(self, dq_score: DQScore) -> str:
        """Get quality tier from DQ score"""
        if dq_score.overall >= self.THRESHOLD_EXCELLENT:
            return 'excellent'
        if dq_score.overall >= self.THRESHOLD_GOOD:
            return 'good'
        if dq_score.overall >= self.THRESHOLD_MIN:
            return 'acceptable'
        return 'below_threshold'

    def suggest_improvements(self, dq_score: DQScore) -> List[str]:
        """Suggest improvements based on component scores"""
        suggestions = []

        if dq_score.validity < 0.7:
            suggestions.append("Improve validity: Ensure response directly addresses the query")

        if dq_score.specificity < 0.7:
            suggestions.append("Improve specificity: Add examples, numbers, or structured details")

        if dq_score.correctness < 0.7:
            suggestions.append("Improve correctness: Add citations or reduce hedging language")

        return suggestions


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

dq_scorer = DQScorer()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def score(query: str, response: str, context: Optional[str] = None) -> DQScore:
    """Score a query-response pair"""
    return dq_scorer.score(query, response, context)


def log_score(query: str, response: str, dq_score: DQScore, **kwargs):
    """Log a DQ score"""
    dq_scorer.log_score(query, response, dq_score, **kwargs)


def get_stats(days: int = 7) -> Dict[str, Any]:
    """Get DQ statistics"""
    return dq_scorer.get_stats(days)


def meets_threshold(dq_score: DQScore, threshold: Optional[float] = None) -> bool:
    """Check if score meets threshold"""
    return dq_scorer.meets_threshold(dq_score, threshold)
