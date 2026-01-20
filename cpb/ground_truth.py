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

import asyncio
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
        # Weights based on reliability
        weights = {
            'factual_accuracy': 0.4,
            'cross_source': 0.35,
            'self_consistency': 0.25,
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

        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        for sentence in sentences:
            # Skip short sentences
            if len(sentence) < 20:
                continue

            # Check for claim patterns
            for pattern in self.CLAIM_PATTERNS:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                claims.extend(matches)

            # Also extract sentences with numbers (likely factual)
            if re.search(r'\d+(?:\.\d+)?%?', sentence):
                # Clean and add
                clean = sentence.strip()
                if clean and clean not in claims:
                    claims.append(clean)

        return claims[:20]  # Limit to top 20 claims

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

    def validate_agreement(self, sources: list[dict]) -> float:
        """
        Check if sources agree with each other.

        Returns agreement score 0-1.
        """
        if len(sources) < 2:
            return 0.5  # Can't measure agreement with <2 sources

        # Extract claims from each source
        source_claims = []
        for source in sources:
            content = source.get('content', '') or source.get('abstract', '')
            claims = self.claim_extractor.extract_claims(content)
            source_claims.append(set(c.lower() for c in claims))

        # Calculate pairwise agreement
        agreements = []
        for i in range(len(source_claims)):
            for j in range(i + 1, len(source_claims)):
                if source_claims[i] and source_claims[j]:
                    # Jaccard similarity
                    intersection = len(source_claims[i] & source_claims[j])
                    union = len(source_claims[i] | source_claims[j])
                    if union > 0:
                        agreements.append(intersection / union)

        return sum(agreements) / len(agreements) if agreements else 0.5


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

    def check_consistency(self, query: str, current_claims: list[str]) -> float:
        """
        Check consistency with previous runs.

        Returns consistency score 0-1.
        """
        query_hash = self._query_hash(query)
        run_file = self.storage_path / f"{query_hash}.json"

        if not run_file.exists():
            return 0.5  # No previous runs, neutral score

        with open(run_file) as f:
            runs = json.load(f)

        if not runs:
            return 0.5

        # Compare current claims with previous runs
        current_set = set(c.lower() for c in current_claims)

        consistencies = []
        for run in runs[-5:]:  # Compare with last 5 runs
            prev_set = set(c.lower() for c in run.get('claims', []))

            if current_set and prev_set:
                # Jaccard similarity
                intersection = len(current_set & prev_set)
                union = len(current_set | prev_set)
                if union > 0:
                    consistencies.append(intersection / union)

        return sum(consistencies) / len(consistencies) if consistencies else 0.5


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

        # Check output claims against source claims
        for claim in output_claims:
            claim_lower = claim.lower()

            # Check for match (fuzzy)
            matched = False
            for sc in source_claim_texts:
                # Simple containment check (could use embeddings for better matching)
                if claim_lower in sc or sc in claim_lower:
                    matched = True
                    break
                # Check for significant word overlap
                claim_words = set(claim_lower.split())
                sc_words = set(sc.split())
                if len(claim_words & sc_words) / max(len(claim_words), 1) > 0.5:
                    matched = True
                    break

            if matched:
                result.claims_verified += 1
                result.verified_claims.append(claim)
            else:
                result.claims_unknown += 1

        # Calculate factual accuracy
        if result.claims_checked > 0:
            result.factual_accuracy = result.claims_verified / result.claims_checked

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
