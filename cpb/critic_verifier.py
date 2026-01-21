#!/usr/bin/env python3
"""
CPB Precision Mode - Critic Verifier

Integration layer for EvidenceCritic + OracleConsensus + GroundTruth verification.
Provides evidence-backed quality validation for precision mode responses.

Verification Pipeline:
1. EvidenceCritic - Validates citations and source quality
2. OracleConsensus - Multi-stream validation (3 perspectives)
3. ConfidenceScorer - Calculates calibrated confidence
4. GroundTruthValidator - Validates against extracted claims, cross-source agreement, self-consistency

DQ Weights (Precision Mode v2):
- Validity: 25% (down from 30%)
- Specificity: 20% (down from 25%)
- Correctness: 30% (down from 45%)
- GroundTruth: 25% (NEW - factual accuracy + cross-source + self-consistency)

Research Foundation:
- arXiv:2512.00047 (Emergent Convergence) - Self-consistency in multi-agent
- arXiv:2508.17536 (Voting vs Debate) - Agreement as quality signal
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from critic.evidence_critic import EvidenceCritic

from .precision_config import (
    PRECISION_CRITIC_WEIGHTS,
    PRECISION_VERIFICATION_THRESHOLDS
)
from .ground_truth import (
    get_validator as get_gt_validator,
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

    # Ground truth scores (v2)
    ground_truth_score: float = 0.0
    factual_accuracy: float = 0.0
    cross_source_score: float = 0.0
    self_consistency: float = 0.0

    # Detailed metrics
    validity: float = 0.0
    specificity: float = 0.0
    correctness: float = 0.0

    # Claims analysis (v2)
    claims_checked: int = 0
    claims_verified: int = 0
    claims_contradicted: int = 0
    verified_claims: List[str] = field(default_factory=list)
    contradicted_claims: List[str] = field(default_factory=list)

    # Issues and citations
    issues: List[Dict[str, Any]] = field(default_factory=list)
    citations_found: int = 0
    citations_verified: int = 0

    # Metadata
    verification_method: str = "precision_v2"
    retries_recommended: int = 0
    feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dq_score': self.dq_score,
            'passed': self.passed,
            'evidence_score': self.evidence_score,
            'oracle_score': self.oracle_score,
            'confidence_score': self.confidence_score,
            'ground_truth_score': self.ground_truth_score,
            'factual_accuracy': self.factual_accuracy,
            'cross_source_score': self.cross_source_score,
            'self_consistency': self.self_consistency,
            'validity': self.validity,
            'specificity': self.specificity,
            'correctness': self.correctness,
            'claims_checked': self.claims_checked,
            'claims_verified': self.claims_verified,
            'claims_contradicted': self.claims_contradicted,
            'verified_claims': self.verified_claims,
            'contradicted_claims': self.contradicted_claims,
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
        # Rebalanced formula - max achievable ~0.95+ for high-quality responses

        # Evidence forms the foundation (0-0.45)
        evidence_contribution = evidence_score * 0.45

        # Citation coverage (0-0.25)
        if citations_total > 0:
            citation_coverage = citations_verified / citations_total
            citation_contribution = citation_coverage * 0.25
        else:
            # Small penalty for no citations, but don't tank the score
            citation_contribution = -0.05

        # Structure indicates well-organized response (0-0.15)
        structure_bonus = self._analyze_structure(response)

        # Assertion strength indicates confidence in claims (0-0.1)
        strength_bonus = self._analyze_assertion_strength(response)

        # Hedging penalty (0-0.1) - reduced impact
        hedging_penalty = self._analyze_hedging(response)

        # Combine components
        score = (
            evidence_contribution +     # 0-0.45
            citation_contribution +     # 0-0.25
            structure_bonus +           # 0-0.15
            strength_bonus -            # 0-0.1
            hedging_penalty * 0.5       # Reduced penalty (0-0.05)
        )

        return max(0.0, min(1.0, score))

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

        # Lists and structure (common in quality reports)
        if re.search(r'\n[-*•]\s', response):
            bonus += 0.04

        # Numbered points
        if re.search(r'\n\d+\.\s', response):
            bonus += 0.03

        # Headers/sections (markdown)
        if re.search(r'\n#{1,3}\s', response):
            bonus += 0.04

        # Bold text for emphasis
        if re.search(r'\*\*[^*]+\*\*', response):
            bonus += 0.02

        # Code blocks or inline code
        if '```' in response or '`' in response:
            bonus += 0.02

        return min(0.15, bonus)


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
        'bracket_num': re.compile(r'\[(\d+)\]'),  # [1], [2], etc.
        'bracket_range': re.compile(r'\[(\d+)[-–,](\d+)\]'),  # [1-3], [9,10]
        'session_ref': re.compile(r'(?:session[:\s]+)?([a-z]+-[a-z]+-\d{8}-\d{6})', re.IGNORECASE),
    }

    def extract_citations(self, text: str, sources: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract all citations from text.

        Args:
            text: Text to extract citations from
            sources: Optional list of sources to map bracket references [N] to

        Returns:
            List of citation dictionaries
        """
        citations = []

        # arXiv citations (e.g., arXiv:2412.05449)
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

        # =================================================================
        # BRACKET CITATIONS [1], [2], etc. - KEY FIX FOR DQ
        # =================================================================
        bracket_nums = set()

        # Single bracket numbers: [1], [2], [15]
        for match in self.PATTERNS['bracket_num'].finditer(text):
            num = match.group(1)
            if num.isdigit() and 1 <= int(num) <= 50:  # Reasonable citation range
                bracket_nums.add(int(num))

        # Ranges: [1-3], [9-12]
        for match in self.PATTERNS['bracket_range'].finditer(text):
            try:
                start, end = int(match.group(1)), int(match.group(2))
                if 1 <= start <= 50 and 1 <= end <= 50:
                    for n in range(start, min(end + 1, 51)):
                        bracket_nums.add(n)
            except ValueError:
                pass

        # Convert bracket numbers to citations with source resolution
        for num in sorted(bracket_nums):
            citation = {
                'type': 'bracket',
                'id': str(num),
                'raw': f'[{num}]'
            }
            # Resolve bracket to actual source if sources provided
            if sources and num <= len(sources):
                source = sources[num - 1]  # 1-indexed to 0-indexed
                citation['resolved_url'] = source.get('url', '')
                citation['resolved_title'] = source.get('title', '')
                citation['resolved_type'] = source.get('type', '')
                # Extract arXiv ID if URL contains it
                url = citation.get('resolved_url', '')
                if 'arxiv.org' in url:
                    arxiv_match = re.search(r'(\d{4}\.\d{4,5})', url)
                    if arxiv_match:
                        citation['resolved_arxiv'] = arxiv_match.group(1)
            citations.append(citation)

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
    Main verification pipeline for precision mode v2.

    Integrates:
    - EvidenceCritic for citation/source validation
    - OracleConsensus for multi-stream verification
    - ConfidenceScorer for calibrated confidence
    - GroundTruthValidator for factual accuracy and consistency (v2)

    Research Foundation:
    - arXiv:2512.00047 (Emergent Convergence) - Self-consistency
    - arXiv:2508.17536 (Voting vs Debate) - Agreement as quality signal
    """

    def __init__(self):
        self.evidence_critic = EvidenceCritic()
        self.confidence_scorer = ConfidenceScorer()
        self.citation_extractor = CitationExtractor()
        self.ground_truth_validator = get_gt_validator()
        self.thresholds = PRECISION_VERIFICATION_THRESHOLDS

    async def verify(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        query: Optional[str] = None,
        context: Optional[str] = None
    ) -> VerificationResult:
        """
        Run full verification pipeline on a response (v2 with ground truth).

        Pipeline:
        1. Citation extraction and verification
        2. Evidence scoring
        3. Oracle consensus scoring
        4. Confidence scoring
        5. Ground truth validation (v2) - factual accuracy, cross-source, self-consistency
        6. Combined DQ calculation with ground truth weight

        Args:
            response: The response to verify
            sources: Sources/citations to validate
            query: Original query (for relevance checking)
            context: Context used for response

        Returns:
            VerificationResult with combined scores including ground truth
        """
        # Extract citations from response (pass sources for bracket resolution)
        citations = self.citation_extractor.extract_citations(response, sources)
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

        # =================================================================
        # GROUND TRUTH VALIDATION (v2)
        # =================================================================
        # Validate against extracted claims, cross-source agreement, self-consistency
        gt_result = await self.ground_truth_validator.validate(
            query=query or "",
            output=response,
            sources=sources
        )

        # Extract ground truth scores
        factual_accuracy = gt_result.factual_accuracy
        cross_source_score = gt_result.cross_source_score
        self_consistency = gt_result.self_consistency
        ground_truth_score = gt_result.ground_truth_score

        # Calculate DQ components
        validity = self._score_validity(response, query)
        specificity = self._score_specificity(response)
        correctness = self._score_correctness(
            evidence_score, oracle_score, citations_verified, citations_found
        )

        # =================================================================
        # DQ CALCULATION v2.2 (optimized weights)
        # =================================================================
        # Rebalanced: favor reliable metrics (validity, correctness)
        # Ground truth reduced since self-consistency is inherently noisy
        dq_v2_weights = {
            'validity': 0.30,       # Increased - very reliable
            'specificity': 0.20,    # Same
            'correctness': 0.35,    # Increased - citation-based, reliable
            'ground_truth': 0.15,   # Reduced - self-consistency noise
        }

        dq_score = (
            validity * dq_v2_weights['validity'] +
            specificity * dq_v2_weights['specificity'] +
            correctness * dq_v2_weights['correctness'] +
            ground_truth_score * dq_v2_weights['ground_truth']
        )

        # Combine DQ with critic scores for final score
        # DQ (75%) + weighted average of critic scores (25%)
        critic_avg = (
            evidence_score * PRECISION_CRITIC_WEIGHTS['evidence_critic'] +
            oracle_score * PRECISION_CRITIC_WEIGHTS['oracle_consensus'] +
            confidence_score * PRECISION_CRITIC_WEIGHTS['confidence_scorer']
        )  # Already sums to 1.0 by design

        combined_score = dq_score * 0.75 + critic_avg * 0.25

        # Collect issues (including ground truth issues)
        issues = self._collect_issues(
            response, citations, citations_verified, citations_found,
            validity, specificity, correctness
        )

        # Add ground truth issues
        if gt_result.claims_contradicted > 0:
            issues.append({
                'code': 'CONTRADICTED_CLAIMS',
                'message': f'{gt_result.claims_contradicted} claims contradicted by ground truth',
                'severity': 'error',
                'suggestion': 'Review and correct contradicted claims'
            })

        if cross_source_score < 0.5:
            issues.append({
                'code': 'LOW_SOURCE_AGREEMENT',
                'message': f'Sources show low agreement ({cross_source_score:.2f})',
                'severity': 'warning',
                'suggestion': 'Seek additional sources or qualify conflicting information'
            })

        if self_consistency < 0.5:
            issues.append({
                'code': 'LOW_SELF_CONSISTENCY',
                'message': f'Response inconsistent with previous runs ({self_consistency:.2f})',
                'severity': 'warning',
                'suggestion': 'Review for factual stability'
            })

        # Determine if passed
        passed = (
            combined_score >= self.thresholds.combined_min and
            evidence_score >= self.thresholds.evidence_min * 0.9 and
            gt_result.claims_contradicted == 0 and  # No contradictions allowed
            not any(i.get('severity') == 'critical' for i in issues)
        )

        # Generate feedback for retry (including ground truth feedback)
        feedback = self._generate_feedback(
            issues, validity, specificity, correctness, citations_verified, citations_found
        )

        if gt_result.claims_contradicted > 0:
            feedback += f"\n- Remove or correct contradicted claims: {', '.join(gt_result.contradicted_claims[:3])}"

        if ground_truth_score < 0.6:
            feedback += "\n- Improve factual accuracy by verifying claims against sources"

        # Calculate recommended retries
        retries = self._calculate_retries_needed(combined_score)

        return VerificationResult(
            dq_score=round(combined_score, 3),
            passed=passed,
            evidence_score=round(evidence_score, 3),
            oracle_score=round(oracle_score, 3),
            confidence_score=round(confidence_score, 3),
            ground_truth_score=round(ground_truth_score, 3),
            factual_accuracy=round(factual_accuracy, 3),
            cross_source_score=round(cross_source_score, 3),
            self_consistency=round(self_consistency, 3),
            validity=round(validity, 3),
            specificity=round(specificity, 3),
            correctness=round(correctness, 3),
            claims_checked=gt_result.claims_checked,
            claims_verified=gt_result.claims_verified,
            claims_contradicted=gt_result.claims_contradicted,
            verified_claims=gt_result.verified_claims[:10],  # Limit for output
            contradicted_claims=gt_result.contradicted_claims,
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
            ctype = citation.get('type', '')
            cid = citation.get('id', '')

            # Handle bracket citations [1], [2], etc. - verify by source index
            if ctype == 'bracket':
                try:
                    idx = int(cid) - 1  # Convert [1] to index 0
                    if 0 <= idx < len(sources):
                        # Bracket citation references a valid source
                        verified += 1
                except (ValueError, TypeError):
                    pass
            # Handle traditional citations (arxiv, doi, url, session)
            elif cid in source_ids or any(cid in s for s in source_ids if isinstance(s, str)):
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

        # Source quality (prefer arxiv, doi, or bracket refs to quality sources)
        high_quality = 0
        for c in citations:
            if c['type'] in ('arxiv', 'doi'):
                high_quality += 1
            elif c['type'] == 'bracket':
                # Check if bracket citation resolves to a quality source
                resolved_url = c.get('resolved_url', '')
                if 'arxiv' in resolved_url.lower() or 'doi.org' in resolved_url.lower():
                    high_quality += 1
                elif c.get('resolved_title', ''):
                    # Has a title = references a real source
                    high_quality += 0.5
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

        # Extract meaningful query words (ignore stopwords)
        stopwords = {'what', 'are', 'the', 'best', 'for', 'in', 'how', 'to', 'is', 'a', 'an', 'and', 'or'}
        query_words = [w.strip('?.,!') for w in query_lower.split() if len(w) > 2 and w not in stopwords]
        if not query_words:
            return 0.7

        # Check keyword coverage (more flexible matching)
        matches = 0
        for word in query_words:
            # Check exact match or partial match (for compound terms like "multi-agent")
            if word in response_lower:
                matches += 1
            elif '-' in word:
                # Handle hyphenated terms: "multi-agent" matches "multi agent" or "multiagent"
                parts = word.split('-')
                if all(part in response_lower for part in parts):
                    matches += 1
            elif len(word) > 5:
                # Check for word stem matching (e.g., "practices" matches "practice")
                stem = word[:len(word)-2] if word.endswith(('es', 'ed', 'er', 'ly', 'ing')) else word[:len(word)-1]
                if len(stem) > 3 and stem in response_lower:
                    matches += 0.8

        coverage = min(1.0, matches / len(query_words)) if query_words else 0.5

        # Check for structured response indicators (common in quality reports)
        structure_indicators = [
            '## ',            # Markdown headers
            '### ',           # Subheaders
            '**',             # Bold text
            '- ',             # Bullet points
            '1. ',            # Numbered lists
            'summary',        # Section headers
            'findings',
            'recommendations',
            'analysis',
            'key ',           # Key findings/points
            'sources',        # Citations section
        ]
        structure_score = sum(1 for ind in structure_indicators if ind in response_lower)
        structure_bonus = min(0.2, structure_score * 0.02)

        # Check for domain-relevant content
        domain_terms = ['agent', 'multi', 'orchestrat', 'consensus', 'coordinat', 'collaborat', 'framework']
        domain_matches = sum(1 for term in domain_terms if term in response_lower)
        domain_bonus = min(0.15, domain_matches * 0.02)

        # Base score from coverage + bonuses
        score = (coverage * 0.65) + structure_bonus + domain_bonus

        return min(1.0, max(0.0, score))

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
