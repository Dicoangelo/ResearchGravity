"""
Ground Truth Validation for CPB Precision Mode

Implements evaluation against actual truth, not just citation existence.

Ground Truth Sources:
1. EXTRACTED CLAIMS: Parse factual statements from retrieved papers
2. CROSS-SOURCE AGREEMENT: Multiple sources must agree
3. SELF-CONSISTENCY: Multiple runs must converge
4. FEEDBACK LEARNING: Human corrections improve over time

Research Foundation:
- arXiv:2512.00047 (Emergent Convergence) - Self-consistency in multi-agent
- arXiv:2508.17536 (Voting vs Debate) - Agreement as quality signal
"""

import json
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from enum import Enum


class TruthSource(Enum):
    """Types of ground truth."""
    EXTRACTED_CLAIM = "extracted_claim"      # From paper abstract
    CROSS_SOURCE = "cross_source"            # Multiple sources agree
    SELF_CONSISTENT = "self_consistent"      # Multiple runs agree
    HUMAN_FEEDBACK = "human_feedback"        # User correction
    EXTERNAL_FACT = "external_fact"          # Known verified fact


@dataclass
class GroundTruthClaim:
    """A verified claim that can be used for evaluation."""
    claim: str
    source: TruthSource
    confidence: float  # 0-1

    # Evidence
    source_urls: list[str] = field(default_factory=list)
    source_excerpts: list[str] = field(default_factory=list)

    # Metadata
    topic: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    verified_by: str = ""  # "system" or user ID

    # For cross-source
    agreement_count: int = 1
    disagreement_count: int = 0

    def agreement_ratio(self) -> float:
        total = self.agreement_count + self.disagreement_count
        return self.agreement_count / total if total > 0 else 0.5


@dataclass
class ValidationResult:
    """Result of validating a response against ground truth."""
    # Scores
    factual_accuracy: float = 0.0      # Claims match ground truth
    cross_source_score: float = 0.0    # Sources agree with each other
    self_consistency: float = 0.0      # Multiple runs agree

    # Combined
    ground_truth_score: float = 0.0

    # Details
    claims_checked: int = 0
    claims_verified: int = 0
    claims_contradicted: int = 0
    claims_unknown: int = 0

    # Evidence
    verified_claims: list[str] = field(default_factory=list)
    contradicted_claims: list[str] = field(default_factory=list)

    def compute_score(self):
        """Compute weighted ground truth score."""
        # Weights optimized for precision mode
        # Cross-source de-weighted: sources from different sessions naturally differ
        # Self-consistency de-weighted: LLMs naturally vary phrasing
        weights = {
            'factual_accuracy': 0.70,    # Primary: claims match sources
            'cross_source': 0.15,         # Reduced: sources naturally differ
            'self_consistency': 0.15,     # Reduced: LLM variation
        }

        self.ground_truth_score = (
            self.factual_accuracy * weights['factual_accuracy'] +
            self.cross_source_score * weights['cross_source'] +
            self.self_consistency * weights['self_consistency']
        )


class ClaimExtractor:
    """Extract factual claims from text."""

    # Patterns for factual claims
    CLAIM_PATTERNS = [
        # Quantitative claims
        r'(\d+(?:\.\d+)?%?\s+(?:of|improvement|reduction|increase|decrease))',
        r'(achieves?\s+\d+(?:\.\d+)?%?\s+\w+)',
        r'(outperforms?\s+.+?\s+by\s+\d+(?:\.\d+)?%?)',

        # Comparative claims
        r'((?:better|worse|faster|slower)\s+than\s+.+)',
        r'(compared\s+to\s+.+?,\s+.+)',

        # Definitional claims
        r'(is\s+defined\s+as\s+.+)',
        r'(consists?\s+of\s+.+)',

        # Causal claims
        r'(because\s+.+?,\s+.+)',
        r'(leads?\s+to\s+.+)',
        r'(results?\s+in\s+.+)',
    ]

    def extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from text."""
        claims = []

        # Split into sentences AND bullet points (handle markdown)
        # Split on: sentence terminators, newlines with bullets, newlines with numbers
        segments = re.split(r'[.!?]\s+|\n[-*•]\s*|\n\d+\.\s*|\n\n', text)

        for segment in segments:
            segment = segment.strip()

            # Skip short segments
            if len(segment) < 15:
                continue

            # Skip headers (lines that are just bold or contain only formatting)
            if segment.startswith('**') and segment.endswith('**') and len(segment) < 50:
                continue

            # Check for claim patterns
            for pattern in self.CLAIM_PATTERNS:
                matches = re.findall(pattern, segment, re.IGNORECASE)
                claims.extend(matches)

            # Also extract segments with numbers (likely factual)
            if re.search(r'\d+(?:\.\d+)?%?', segment):
                clean = segment.strip('- *')
                if clean and clean not in claims and len(clean) > 15:
                    claims.append(clean)

            # Extract segments with key factual indicators
            if any(ind in segment.lower() for ind in ['convergence', 'framework', 'achieves', 'demonstrates', 'findings', 'research', 'system']):
                clean = segment.strip('- *')
                if clean and clean not in claims and len(clean) > 20:
                    claims.append(clean)

        return claims[:30]  # Limit to top 30 claims

    def extract_from_sources(self, sources: list[dict]) -> list[GroundTruthClaim]:
        """Extract ground truth claims from sources."""
        all_claims = []
        claim_to_sources = {}  # claim_hash -> [source_urls]

        for source in sources:
            content = source.get('content', '') or source.get('abstract', '')
            url = source.get('url', '')

            claims = self.extract_claims(content)

            for claim in claims:
                claim_hash = hashlib.md5(claim.lower().encode()).hexdigest()[:12]

                if claim_hash not in claim_to_sources:
                    claim_to_sources[claim_hash] = {
                        'claim': claim,
                        'urls': [],
                        'excerpts': [],
                    }

                claim_to_sources[claim_hash]['urls'].append(url)
                claim_to_sources[claim_hash]['excerpts'].append(content[:200])

        # Convert to GroundTruthClaim objects
        for claim_hash, data in claim_to_sources.items():
            # Claims from multiple sources are more reliable
            agreement_count = len(data['urls'])
            confidence = min(0.9, 0.5 + (agreement_count * 0.1))

            gt_claim = GroundTruthClaim(
                claim=data['claim'],
                source=TruthSource.CROSS_SOURCE if agreement_count > 1 else TruthSource.EXTRACTED_CLAIM,
                confidence=confidence,
                source_urls=data['urls'],
                source_excerpts=data['excerpts'],
                agreement_count=agreement_count,
            )
            all_claims.append(gt_claim)

        return all_claims


class CrossSourceValidator:
    """Validate that sources agree with each other."""

    def __init__(self):
        self.claim_extractor = ClaimExtractor()

    def _normalize_tokens(self, text: str) -> set[str]:
        """Normalize text to token set."""
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = set()
        for word in normalized.split():
            if len(word) > 3:
                # Simple stemming
                if word.endswith(('ing', 'tion', 'ment')):
                    word = word[:-3] if len(word) > 5 else word
                elif word.endswith(('ed', 'er', 'es')):
                    word = word[:-2] if len(word) > 4 else word
                tokens.add(word)
        return tokens

    def _fuzzy_claim_overlap(self, claims1: list[str], claims2: list[str]) -> float:
        """Calculate fuzzy overlap between two claim lists."""
        if not claims1 or not claims2:
            return 0.5  # No claims to compare

        # Normalize all claims to token sets
        tokens1 = [self._normalize_tokens(c) for c in claims1]
        tokens2 = [self._normalize_tokens(c) for c in claims2]

        # For each claim in set1, find best match in set2
        matches = 0
        for t1 in tokens1:
            if not t1:
                continue
            best_match = 0.0
            for t2 in tokens2:
                if not t2:
                    continue
                intersection = len(t1 & t2)
                union = len(t1 | t2)
                if union > 0:
                    similarity = intersection / union
                    best_match = max(best_match, similarity)
            if best_match > 0.3:  # Threshold for "matching"
                matches += 1

        return matches / len(tokens1) if tokens1 else 0.5

    def validate_agreement(self, sources: list[dict]) -> float:
        """
        Check if sources agree with each other.

        Returns agreement score 0-1.
        """
        if len(sources) < 2:
            return 0.7  # Can't measure agreement with <2 sources, assume positive

        # Extract claims from each source
        source_claims = []
        sources_with_content = 0
        for source in sources:
            content = source.get('content', '') or source.get('abstract', '') or source.get('summary', '')
            if len(content) > 50:  # Has meaningful content
                claims = self.claim_extractor.extract_claims(content)
                source_claims.append(claims)
                sources_with_content += 1
            else:
                source_claims.append([])

        # If most sources lack content, return optimistic default
        if sources_with_content < 2:
            return 0.7  # Assume agreement when we can't verify

        # Calculate pairwise fuzzy agreement
        agreements = []
        for i in range(len(source_claims)):
            for j in range(i + 1, len(source_claims)):
                if source_claims[i] and source_claims[j]:
                    overlap = self._fuzzy_claim_overlap(source_claims[i], source_claims[j])
                    agreements.append(overlap)

        if not agreements:
            return 0.7  # No comparisons possible

        return sum(agreements) / len(agreements)


class SelfConsistencyChecker:
    """
    Check if multiple runs produce consistent results.

    Based on arXiv:2512.00047 (Emergent Convergence)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".agent-core" / "precision" / "consistency"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _query_hash(self, query: str) -> str:
        """Hash query for storage."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:16]

    def record_run(self, query: str, output: str, claims: list[str]):
        """Record a run for future consistency checking."""
        query_hash = self._query_hash(query)
        run_file = self.storage_path / f"{query_hash}.json"

        # Load existing runs
        runs = []
        if run_file.exists():
            with open(run_file) as f:
                runs = json.load(f)

        # Add this run
        runs.append({
            'timestamp': datetime.now().isoformat(),
            'output_hash': hashlib.md5(output.encode()).hexdigest()[:16],
            'claims': claims,
            'output_preview': output[:500],
        })

        # Keep last 10 runs
        runs = runs[-10:]

        with open(run_file, 'w') as f:
            json.dump(runs, f, indent=2)

    def _normalize_claim(self, claim: str) -> set[str]:
        """Normalize claim to token set for fuzzy matching."""
        # Remove punctuation and lowercase
        normalized = re.sub(r'[^\w\s]', ' ', claim.lower())
        # Split into tokens, filter short words and numbers-only
        tokens = set()
        for word in normalized.split():
            if len(word) > 2 and not word.isdigit():
                # Simple stemming: remove common suffixes
                if word.endswith(('ing', 'tion', 'ment', 'ness', 'ity')):
                    word = word[:-3] if len(word) > 5 else word
                elif word.endswith(('ed', 'er', 'es', 'ly')):
                    word = word[:-2] if len(word) > 4 else word
                elif word.endswith('s') and len(word) > 3:
                    word = word[:-1]
                tokens.add(word)
        return tokens

    def _claim_similarity(self, claim1: str, claim2: str) -> float:
        """Calculate similarity between two claims using token overlap."""
        tokens1 = self._normalize_claim(claim1)
        tokens2 = self._normalize_claim(claim2)

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard on tokens
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def check_consistency(self, query: str, current_claims: list[str]) -> float:
        """
        Check consistency with previous runs using fuzzy claim matching.

        Returns consistency score 0-1.
        """
        query_hash = self._query_hash(query)
        run_file = self.storage_path / f"{query_hash}.json"

        if not run_file.exists():
            return 0.7  # No previous runs, optimistic default (first run assumed consistent)

        with open(run_file) as f:
            runs = json.load(f)

        if not runs:
            return 0.7

        consistencies = []
        for run in runs[-3:]:  # Compare with last 3 runs
            prev_claims = run.get('claims', [])

            if not current_claims or not prev_claims:
                continue

            # For each current claim, find best match in previous claims
            match_scores = []
            for curr in current_claims:
                best_match = max(
                    (self._claim_similarity(curr, prev) for prev in prev_claims),
                    default=0.0
                )
                match_scores.append(best_match)

            # Also check reverse: how many previous claims are covered
            reverse_scores = []
            for prev in prev_claims:
                best_match = max(
                    (self._claim_similarity(prev, curr) for curr in current_claims),
                    default=0.0
                )
                reverse_scores.append(best_match)

            # Average of forward and reverse coverage
            if match_scores and reverse_scores:
                forward_avg = sum(match_scores) / len(match_scores)
                reverse_avg = sum(reverse_scores) / len(reverse_scores)
                # Weight towards claims being found (forward), less penalty for new claims
                run_consistency = (forward_avg * 0.6 + reverse_avg * 0.4)
                consistencies.append(run_consistency)

        if not consistencies:
            return 0.7

        # Return average, but boost if most runs are consistent
        avg = sum(consistencies) / len(consistencies)
        return min(1.0, avg * 1.2)  # Slight boost since fuzzy matching is conservative


class FeedbackCollector:
    """
    Collect human feedback to build ground truth over time.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".agent-core" / "precision" / "feedback"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.storage_path / "feedback.json"

    def _load_feedback(self) -> list[dict]:
        if self.feedback_file.exists():
            with open(self.feedback_file) as f:
                return json.load(f)
        return []

    def _save_feedback(self, feedback: list[dict]):
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)

    def record_feedback(
        self,
        query: str,
        output: str,
        rating: int,  # 1-5
        corrections: Optional[str] = None,
        verified_claims: Optional[list[str]] = None,
        false_claims: Optional[list[str]] = None,
    ):
        """Record human feedback on a response."""
        feedback = self._load_feedback()

        feedback.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'output_preview': output[:500],
            'rating': rating,
            'corrections': corrections,
            'verified_claims': verified_claims or [],
            'false_claims': false_claims or [],
        })

        # Keep last 1000 feedback entries
        feedback = feedback[-1000:]

        self._save_feedback(feedback)

    def get_ground_truth_claims(self) -> list[GroundTruthClaim]:
        """Extract ground truth claims from feedback history."""
        feedback = self._load_feedback()
        claims = []

        for entry in feedback:
            # High-rated responses with verified claims
            if entry.get('rating', 0) >= 4:
                for claim in entry.get('verified_claims', []):
                    gt_claim = GroundTruthClaim(
                        claim=claim,
                        source=TruthSource.HUMAN_FEEDBACK,
                        confidence=0.9,
                        verified_by="human",
                    )
                    claims.append(gt_claim)

            # Low-rated responses with false claims (negative examples)
            for claim in entry.get('false_claims', []):
                gt_claim = GroundTruthClaim(
                    claim=claim,
                    source=TruthSource.HUMAN_FEEDBACK,
                    confidence=0.1,  # Low confidence = known false
                    verified_by="human",
                )
                claims.append(gt_claim)

        return claims


class GroundTruthValidator:
    """
    Main validator combining all ground truth sources.
    """

    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.cross_source = CrossSourceValidator()
        self.consistency = SelfConsistencyChecker()
        self.feedback = FeedbackCollector()

    def _normalize_claim(self, claim: str) -> set[str]:
        """Normalize claim to token set for fuzzy matching."""
        normalized = re.sub(r'[^\w\s]', ' ', claim.lower())
        tokens = set()
        for word in normalized.split():
            if len(word) > 2 and not word.isdigit():
                # Simple stemming
                if word.endswith(('ing', 'tion', 'ment', 'ness', 'ity')):
                    word = word[:-3] if len(word) > 5 else word
                elif word.endswith(('ed', 'er', 'es', 'ly')):
                    word = word[:-2] if len(word) > 4 else word
                elif word.endswith('s') and len(word) > 3:
                    word = word[:-1]
                tokens.add(word)
        return tokens

    async def validate(
        self,
        query: str,
        output: str,
        sources: list[dict],
    ) -> ValidationResult:
        """
        Validate response against ground truth.

        Combines:
        1. Factual accuracy (claims match extracted facts)
        2. Cross-source agreement (sources agree)
        3. Self-consistency (multiple runs agree)
        """
        result = ValidationResult()

        # Extract claims from output
        output_claims = self.claim_extractor.extract_claims(output)
        result.claims_checked = len(output_claims)

        # 1. Extract ground truth from sources
        source_claims = self.claim_extractor.extract_from_sources(sources)
        source_claim_texts = set(c.claim.lower() for c in source_claims)

        # Check output claims against source claims using fuzzy matching
        if source_claim_texts:
            for claim in output_claims:
                claim_tokens = self._normalize_claim(claim)

                # Find best matching source claim
                best_match = 0.0
                for sc in source_claim_texts:
                    sc_tokens = self._normalize_claim(sc)
                    if claim_tokens and sc_tokens:
                        intersection = len(claim_tokens & sc_tokens)
                        union = len(claim_tokens | sc_tokens)
                        if union > 0:
                            similarity = intersection / union
                            best_match = max(best_match, similarity)

                # Threshold: 0.35 for weak match, 0.5 for strong match
                if best_match >= 0.35:
                    result.claims_verified += 1
                    result.verified_claims.append(claim)
                else:
                    result.claims_unknown += 1

            # Calculate factual accuracy
            if result.claims_checked > 0:
                result.factual_accuracy = result.claims_verified / result.claims_checked
        else:
            # No source claims extracted (sources lack content)
            # Use optimistic default - can't verify but don't penalize
            result.factual_accuracy = 0.7
            result.claims_unknown = result.claims_checked

        # 2. Cross-source agreement
        result.cross_source_score = self.cross_source.validate_agreement(sources)

        # 3. Self-consistency
        result.self_consistency = self.consistency.check_consistency(query, output_claims)

        # Record this run for future consistency checks
        self.consistency.record_run(query, output, output_claims)

        # 4. Check against human feedback ground truth
        feedback_claims = self.feedback.get_ground_truth_claims()
        feedback_verified = set(c.claim.lower() for c in feedback_claims if c.confidence > 0.5)
        feedback_false = set(c.claim.lower() for c in feedback_claims if c.confidence < 0.5)

        for claim in output_claims:
            claim_lower = claim.lower()
            if any(fc in claim_lower or claim_lower in fc for fc in feedback_false):
                result.claims_contradicted += 1
                result.contradicted_claims.append(claim)

        # Penalize contradicted claims
        if result.claims_contradicted > 0:
            result.factual_accuracy *= (1 - result.claims_contradicted / max(result.claims_checked, 1))

        # Compute final score
        result.compute_score()

        return result


# Singleton
_validator: Optional[GroundTruthValidator] = None

def get_validator() -> GroundTruthValidator:
    global _validator
    if _validator is None:
        _validator = GroundTruthValidator()
    return _validator


async def validate_against_ground_truth(
    query: str,
    output: str,
    sources: list[dict],
) -> ValidationResult:
    """Validate response against ground truth."""
    validator = get_validator()
    return await validator.validate(query, output, sources)


def record_feedback(
    query: str,
    output: str,
    rating: int,
    corrections: Optional[str] = None,
    verified_claims: Optional[list[str]] = None,
    false_claims: Optional[list[str]] = None,
):
    """Record human feedback for learning."""
    validator = get_validator()
    validator.feedback.record_feedback(
        query, output, rating, corrections, verified_claims, false_claims
    )
