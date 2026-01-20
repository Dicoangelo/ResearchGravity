#!/usr/bin/env python3
"""
CPB Precision Mode - Critic Verifier

Integration layer for EvidenceCritic + OracleConsensus verification.
Provides evidence-backed quality validation for precision mode responses.

Verification Pipeline:
1. EvidenceCritic - Validates citations and source quality
2. OracleConsensus - Multi-stream validation (3 perspectives)
3. ConfidenceScorer - Calculates calibrated confidence

DQ Weights (Precision Mode):
- Validity: 30% (down from 40%)
- Specificity: 25% (down from 30%)
- Correctness: 45% (up from 30% - evidence-backed)
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from critic.base import ValidationResult, Issue, Severity, OracleConsensus as OracleBase
from critic.evidence_critic import EvidenceCritic

from .precision_config import (
    PRECISION_DQ_WEIGHTS,
    PRECISION_CRITIC_WEIGHTS,
    PRECISION_VERIFICATION_THRESHOLDS,
    calculate_precision_dq
)


# =============================================================================
# VERIFICATION RESULT
# =============================================================================

@dataclass
class VerificationResult:
    """Result from critic verification pipeline."""
    dq_score: float
    passed: bool

    # Component scores
    evidence_score: float = 0.0
    oracle_score: float = 0.0
    confidence_score: float = 0.0

    # Detailed metrics
    validity: float = 0.0
    specificity: float = 0.0
    correctness: float = 0.0

    # Issues and citations
    issues: List[Dict[str, Any]] = field(default_factory=list)
    citations_found: int = 0
    citations_verified: int = 0

    # Metadata
    verification_method: str = "precision"
    retries_recommended: int = 0
    feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dq_score': self.dq_score,
            'passed': self.passed,
            'evidence_score': self.evidence_score,
            'oracle_score': self.oracle_score,
            'confidence_score': self.confidence_score,
            'validity': self.validity,
            'specificity': self.specificity,
            'correctness': self.correctness,
            'issues': self.issues,
            'citations_found': self.citations_found,
            'citations_verified': self.citations_verified,
            'verification_method': self.verification_method,
            'retries_recommended': self.retries_recommended,
            'feedback': self.feedback,
        }


# =============================================================================
# CONFIDENCE SCORER
# =============================================================================

class ConfidenceScorer:
    """
    Calculates calibrated confidence scores for responses.

    Factors:
    - Citation presence and quality
    - Hedging language detection
    - Assertion strength
    - Response structure
    """

    def calculate(
        self,
        response: str,
        citations_verified: int,
        citations_total: int,
        evidence_score: float
    ) -> float:
        """
        Calculate confidence score for a response.

        Args:
            response: The response text
            citations_verified: Number of verified citations
            citations_total: Total citations found
            evidence_score: Score from evidence critic

        Returns:
            Confidence score (0-1)
        """
        # Base from evidence score
        base = evidence_score * 0.4

        # Citation coverage
        if citations_total > 0:
            citation_coverage = citations_verified / citations_total
            base += citation_coverage * 0.25
        else:
            # Penalty for no citations in precision mode
            base -= 0.1

        # Hedging analysis
        hedging_penalty = self._analyze_hedging(response)
        base -= hedging_penalty

        # Assertion strength
        strength_bonus = self._analyze_assertion_strength(response)
        base += strength_bonus

        # Structure bonus
        structure_bonus = self._analyze_structure(response)
        base += structure_bonus

        return max(0.0, min(1.0, base))

    def _analyze_hedging(self, response: str) -> float:
        """Detect hedging language and calculate penalty."""
        response_lower = response.lower()

        hedging_patterns = [
            r'\bmight\b', r'\bmaybe\b', r'\bpossibly\b',
            r'\bperhaps\b', r'\bcould be\b', r'\bi think\b',
            r'\bnot sure\b', r'\buncertain\b', r'\bi believe\b',
            r'\bit seems\b', r'\bappears to\b', r'\bprobably\b'
        ]

        hedging_count = sum(
            len(re.findall(pattern, response_lower))
            for pattern in hedging_patterns
        )

        # Normalize by response length
        words = len(response.split())
        if words > 0:
            hedging_ratio = hedging_count / (words / 100)  # Per 100 words
        else:
            hedging_ratio = 0

        return min(0.15, hedging_ratio * 0.03)

    def _analyze_assertion_strength(self, response: str) -> float:
        """Detect strong assertion language."""
        response_lower = response.lower()

        strength_patterns = [
            r'\bclearly\b', r'\bcertainly\b', r'\bdefinitely\b',
            r'\bevidence shows\b', r'\bresearch demonstrates\b',
            r'\bdata indicates\b', r'\bthe fact is\b',
            r'\bproven\b', r'\bestablished\b', r'\bconfirmed\b'
        ]

        strength_count = sum(
            len(re.findall(pattern, response_lower))
            for pattern in strength_patterns
        )

        return min(0.1, strength_count * 0.02)

    def _analyze_structure(self, response: str) -> float:
        """Analyze response structure quality."""
        bonus = 0.0

        # Lists and structure
        if re.search(r'\n[-*•]\s', response):
            bonus += 0.03

        # Numbered points
        if re.search(r'\n\d+\.\s', response):
            bonus += 0.02

        # Headers/sections
        if re.search(r'\n#{1,3}\s|\*\*[^*]+\*\*', response):
            bonus += 0.02

        # Code blocks
        if '```' in response or '`' in response:
            bonus += 0.02

        return min(0.1, bonus)


# =============================================================================
# CITATION EXTRACTOR
# =============================================================================

class CitationExtractor:
    """Extracts and validates citations from responses."""

    # Patterns for different citation formats
    PATTERNS = {
        'arxiv': re.compile(r'(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?', re.IGNORECASE),
        'doi': re.compile(r'10\.\d{4,}/[^\s\]]+'),
        'url': re.compile(r'https?://[^\s\]]+'),
        'bracket_ref': re.compile(r'\[([^\]]+)\]'),
        'session_ref': re.compile(r'(?:session[:\s]+)?([a-z]+-[a-z]+-\d{8}-\d{6})', re.IGNORECASE),
    }

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all citations from text.

        Args:
            text: Text to extract citations from

        Returns:
            List of citation dictionaries
        """
        citations = []

        # arXiv citations
        for match in self.PATTERNS['arxiv'].finditer(text):
            citations.append({
                'type': 'arxiv',
                'id': match.group(1),
                'raw': match.group(0)
            })

        # DOI citations
        for match in self.PATTERNS['doi'].finditer(text):
            citations.append({
                'type': 'doi',
                'id': match.group(0),
                'raw': match.group(0)
            })

        # URL citations
        for match in self.PATTERNS['url'].finditer(text):
            url = match.group(0)
            # Avoid duplicating arxiv/doi URLs
            if 'arxiv.org' not in url and 'doi.org' not in url:
                citations.append({
                    'type': 'url',
                    'id': url,
                    'raw': url
                })

        # Session references
        for match in self.PATTERNS['session_ref'].finditer(text):
            citations.append({
                'type': 'session',
                'id': match.group(1),
                'raw': match.group(0)
            })

        # Deduplicate
        seen = set()
        unique = []
        for c in citations:
            key = f"{c['type']}:{c['id']}"
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def count_claims(self, text: str) -> int:
        """
        Count factual claims in text (for citation coverage analysis).

        Args:
            text: Text to analyze

        Returns:
            Estimated number of factual claims
        """
        # Simple heuristic: count sentences with factual indicators
        claim_patterns = [
            r'(?:is|are|was|were|has|have|had)\s+(?:a|an|the|one|two|three)',
            r'\d+%',
            r'(?:research|studies|data|evidence)\s+(?:shows?|indicates?|suggests?)',
            r'(?:according to|based on)',
            r'(?:in \d{4}|since \d{4})',
        ]

        claim_count = 0
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claim_count += 1
                    break

        return max(claim_count, 1)


# =============================================================================
# CRITIC VERIFIER
# =============================================================================

class CriticVerifier:
    """
    Main verification pipeline for precision mode.

    Integrates:
    - EvidenceCritic for citation/source validation
    - OracleConsensus for multi-stream verification
    - ConfidenceScorer for calibrated confidence
    """

    def __init__(self):
        self.evidence_critic = EvidenceCritic()
        self.confidence_scorer = ConfidenceScorer()
        self.citation_extractor = CitationExtractor()
        self.thresholds = PRECISION_VERIFICATION_THRESHOLDS

    async def verify(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        query: Optional[str] = None,
        context: Optional[str] = None
    ) -> VerificationResult:
        """
        Run full verification pipeline on a response.

        Args:
            response: The response to verify
            sources: Sources/citations to validate
            query: Original query (for relevance checking)
            context: Context used for response

        Returns:
            VerificationResult with combined scores
        """
        # Extract citations from response
        citations = self.citation_extractor.extract_citations(response)
        citations_found = len(citations)

        # Verify citations against sources
        citations_verified = self._verify_citations(citations, sources)

        # Calculate component scores
        evidence_score = await self._calculate_evidence_score(
            response, citations, sources
        )

        oracle_score = await self._calculate_oracle_score(
            response, query, context
        )

        confidence_score = self.confidence_scorer.calculate(
            response, citations_verified, citations_found, evidence_score
        )

        # Calculate DQ components
        validity = self._score_validity(response, query)
        specificity = self._score_specificity(response)
        correctness = self._score_correctness(
            evidence_score, oracle_score, citations_verified, citations_found
        )

        # Calculate weighted DQ score
        dq_score = calculate_precision_dq(validity, specificity, correctness)

        # Also factor in critic scores
        combined_score = (
            dq_score * 0.5 +
            evidence_score * PRECISION_CRITIC_WEIGHTS['evidence_critic'] +
            oracle_score * PRECISION_CRITIC_WEIGHTS['oracle_consensus'] +
            confidence_score * PRECISION_CRITIC_WEIGHTS['confidence_scorer']
        ) / 1.5  # Normalize

        # Collect issues
        issues = self._collect_issues(
            response, citations, citations_verified, citations_found,
            validity, specificity, correctness
        )

        # Determine if passed
        passed = (
            combined_score >= self.thresholds.combined_min and
            evidence_score >= self.thresholds.evidence_min * 0.9 and  # Allow some slack
            not any(i.get('severity') == 'critical' for i in issues)
        )

        # Generate feedback for retry
        feedback = self._generate_feedback(
            issues, validity, specificity, correctness, citations_verified, citations_found
        )

        # Calculate recommended retries
        retries = self._calculate_retries_needed(combined_score)

        return VerificationResult(
            dq_score=round(combined_score, 3),
            passed=passed,
            evidence_score=round(evidence_score, 3),
            oracle_score=round(oracle_score, 3),
            confidence_score=round(confidence_score, 3),
            validity=round(validity, 3),
            specificity=round(specificity, 3),
            correctness=round(correctness, 3),
            issues=issues,
            citations_found=citations_found,
            citations_verified=citations_verified,
            retries_recommended=retries,
            feedback=feedback
        )

    def _verify_citations(
        self,
        citations: List[Dict[str, Any]],
        sources: List[Dict[str, Any]]
    ) -> int:
        """Count how many citations can be verified against sources."""
        if not citations or not sources:
            return 0

        verified = 0
        source_ids = set()

        # Build set of source identifiers
        for source in sources:
            if 'arxiv_id' in source:
                source_ids.add(source['arxiv_id'])
            if 'url' in source:
                source_ids.add(source['url'])
            if 'id' in source:
                source_ids.add(source['id'])
            if 'session_id' in source:
                source_ids.add(source['session_id'])

        # Check citations against sources
        for citation in citations:
            cid = citation.get('id', '')
            if cid in source_ids or any(cid in s for s in source_ids):
                verified += 1

        return verified

    async def _calculate_evidence_score(
        self,
        response: str,
        citations: List[Dict[str, Any]],
        sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate evidence score based on citation quality."""
        if not citations:
            return 0.3  # Penalty for no citations

        # Calculate claim count
        claim_count = self.citation_extractor.count_claims(response)

        # Citation coverage
        coverage = min(1.0, len(citations) / max(claim_count * 0.5, 1))

        # Source quality (prefer arxiv, doi)
        high_quality = sum(
            1 for c in citations
            if c['type'] in ('arxiv', 'doi')
        )
        quality_ratio = high_quality / len(citations) if citations else 0

        # Verification rate against provided sources
        verified = self._verify_citations(citations, sources)
        verification_rate = verified / len(citations) if citations else 0

        # Weighted score
        score = (
            coverage * 0.35 +
            quality_ratio * 0.35 +
            verification_rate * 0.30
        )

        return min(1.0, max(0.0, score))

    async def _calculate_oracle_score(
        self,
        response: str,
        query: Optional[str],
        context: Optional[str]
    ) -> float:
        """
        Calculate oracle consensus score (simplified for precision mode).

        In full implementation, this would run 3 parallel validation streams.
        Here we use heuristic approximation.
        """
        score = 0.5  # Base

        # Accuracy indicators
        if re.search(r'arXiv|doi|http|source:|according to', response.lower()):
            score += 0.1

        # Completeness indicators
        word_count = len(response.split())
        if word_count > 100:
            score += 0.1
        if word_count > 300:
            score += 0.05

        # Has structure
        if re.search(r'\n[-*•]\s|\n\d+\.\s', response):
            score += 0.1

        # Relevance to query
        if query:
            query_words = [w.lower() for w in query.split() if len(w) > 3]
            response_lower = response.lower()
            matches = sum(1 for w in query_words if w in response_lower)
            if query_words:
                relevance = matches / len(query_words)
                score += relevance * 0.15

        return min(1.0, max(0.0, score))

    def _score_validity(self, response: str, query: Optional[str]) -> float:
        """Score how well response addresses the query."""
        if not query:
            return 0.7  # Default without query

        query_lower = query.lower()
        response_lower = response.lower()

        # Extract meaningful query words
        query_words = [w for w in query_lower.split() if len(w) > 3]
        if not query_words:
            return 0.7

        # Check keyword coverage
        matches = sum(1 for w in query_words if w in response_lower)
        coverage = matches / len(query_words)

        # Check for direct address
        direct_address = any(
            phrase in response_lower
            for phrase in ['to answer', 'regarding', 'the answer', 'in response']
        )

        return (coverage * 0.7) + (0.3 if direct_address else 0.15)

    def _score_specificity(self, response: str) -> float:
        """Score response specificity."""
        words = response.split()
        word_count = len(words)

        # Length factor
        if word_count < 50:
            length_score = 0.3
        elif word_count < 100:
            length_score = 0.5
        elif word_count < 300:
            length_score = 0.7
        elif word_count < 500:
            length_score = 0.85
        else:
            length_score = 0.8

        # Detail indicators
        has_numbers = bool(re.search(r'\d+', response))
        has_examples = bool(re.search(r'for example|such as|e\.g\.|specifically', response.lower()))
        has_structure = bool(re.search(r'\n[-*•]\s|\n\d+\.\s', response))

        detail_bonus = (
            (0.1 if has_numbers else 0) +
            (0.1 if has_examples else 0) +
            (0.1 if has_structure else 0)
        )

        return min(1.0, length_score * 0.7 + detail_bonus + 0.1)

    def _score_correctness(
        self,
        evidence_score: float,
        oracle_score: float,
        citations_verified: int,
        citations_found: int
    ) -> float:
        """
        Score correctness (evidence-backed in precision mode).

        This is weighted higher (45%) in precision mode because
        evidence backing is critical.
        """
        # Base from evidence
        base = evidence_score * 0.5

        # Citation verification bonus
        if citations_found > 0:
            verification_rate = citations_verified / citations_found
            base += verification_rate * 0.3
        else:
            base -= 0.1  # Penalty for no citations

        # Oracle validation
        base += oracle_score * 0.2

        return min(1.0, max(0.0, base))

    def _collect_issues(
        self,
        response: str,
        citations: List[Dict[str, Any]],
        citations_verified: int,
        citations_found: int,
        validity: float,
        specificity: float,
        correctness: float
    ) -> List[Dict[str, Any]]:
        """Collect verification issues."""
        issues = []

        # Citation issues
        if not citations:
            issues.append({
                'code': 'NO_CITATIONS',
                'message': 'Response contains no citations',
                'severity': 'warning',
                'suggestion': 'Add arXiv IDs, URLs, or session references'
            })
        elif citations_found > 0 and citations_verified == 0:
            issues.append({
                'code': 'UNVERIFIED_CITATIONS',
                'message': f'{citations_found} citations could not be verified',
                'severity': 'warning',
                'suggestion': 'Ensure citations match provided sources'
            })

        # Component score issues
        if validity < 0.7:
            issues.append({
                'code': 'LOW_VALIDITY',
                'message': f'Response validity ({validity:.2f}) below threshold',
                'severity': 'warning' if validity > 0.5 else 'error',
                'suggestion': 'Ensure response directly addresses the query'
            })

        if specificity < 0.7:
            issues.append({
                'code': 'LOW_SPECIFICITY',
                'message': f'Response specificity ({specificity:.2f}) below threshold',
                'severity': 'warning' if specificity > 0.5 else 'error',
                'suggestion': 'Add concrete examples, numbers, or structured details'
            })

        if correctness < 0.7:
            issues.append({
                'code': 'LOW_CORRECTNESS',
                'message': f'Response correctness ({correctness:.2f}) below threshold',
                'severity': 'warning' if correctness > 0.5 else 'error',
                'suggestion': 'Add citations and reduce hedging language'
            })

        return issues

    def _generate_feedback(
        self,
        issues: List[Dict[str, Any]],
        validity: float,
        specificity: float,
        correctness: float,
        citations_verified: int,
        citations_found: int
    ) -> str:
        """Generate feedback for retry attempts."""
        feedback_parts = []

        if validity < 0.8:
            feedback_parts.append("- Ensure response directly addresses the query")

        if specificity < 0.8:
            feedback_parts.append("- Add more concrete examples and structured details")

        if correctness < 0.8:
            feedback_parts.append("- Add citations (arXiv:XXXX.XXXXX format) for factual claims")

        if citations_found == 0:
            feedback_parts.append("- Include citations for all factual claims")
        elif citations_verified < citations_found * 0.5:
            feedback_parts.append("- Verify citations match actual sources")

        for issue in issues:
            if issue.get('severity') == 'error':
                suggestion = issue.get('suggestion', '')
                if suggestion and suggestion not in '\n'.join(feedback_parts):
                    feedback_parts.append(f"- {suggestion}")

        return '\n'.join(feedback_parts) if feedback_parts else ""

    def _calculate_retries_needed(self, current_score: float) -> int:
        """Estimate retries needed to reach threshold."""
        target = self.thresholds.combined_min
        gap = target - current_score

        if gap <= 0:
            return 0
        elif gap <= 0.05:
            return 1
        elif gap <= 0.15:
            return 2
        elif gap <= 0.25:
            return 3
        else:
            return 5  # Max


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton instance
critic_verifier = CriticVerifier()


async def verify(
    response: str,
    sources: List[Dict[str, Any]],
    query: Optional[str] = None,
    context: Optional[str] = None
) -> VerificationResult:
    """Verify a response using the precision verification pipeline."""
    return await critic_verifier.verify(response, sources, query, context)


def format_critic_feedback(issues: List[Dict[str, Any]]) -> str:
    """Format issues into feedback for retry."""
    suggestions = []
    for issue in issues:
        if issue.get('suggestion'):
            suggestions.append(f"- {issue['suggestion']}")
    return '\n'.join(suggestions) if suggestions else ""
