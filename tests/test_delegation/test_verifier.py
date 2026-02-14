"""
Tests for delegation.verifier module.

Tests all 4 verification methods and integration points.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from delegation.models import SubTask, VerificationResult, VerificationMethod, TaskProfile
from delegation.verifier import (
    verify_completion,
    feed_to_trust_ledger,
    feed_to_memory_bleed,
    _verify_automated_test,
    _verify_semantic_similarity,
    _verify_human_review,
    _verify_ground_truth,
)


# ============================================================================
# Test Automated Test Verification
# ============================================================================

class TestAutomatedTestVerification:
    """Test automated test verification method."""

    def test_automated_test_pass(self):
        """Automated test should pass when validation function returns True."""
        subtask = SubTask(
            id="test-1",
            description="Run unit tests",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        # Validation function that always passes
        def always_pass(result: str) -> bool:
            return True

        result = verify_completion(subtask, "test output", validation_fn=always_pass)

        assert result.passed is True
        assert result.quality_score == 1.0
        assert result.method == VerificationMethod.AUTOMATED_TEST
        assert "passed" in result.feedback.lower()
        assert result.evidence["test_passed"] is True

    def test_automated_test_fail(self):
        """Automated test should fail when validation function returns False."""
        subtask = SubTask(
            id="test-2",
            description="Run integration tests",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.4,
            estimated_duration=0.3,
            parallel_safe=False
        )

        # Validation function that always fails
        def always_fail(result: str) -> bool:
            return False

        result = verify_completion(subtask, "test output", validation_fn=always_fail)

        assert result.passed is False
        assert result.quality_score == 0.0
        assert result.method == VerificationMethod.AUTOMATED_TEST
        assert "failed" in result.feedback.lower()
        assert result.evidence["test_passed"] is False

    def test_automated_test_missing_validation_fn(self):
        """Should fail gracefully when no validation function provided."""
        subtask = SubTask(
            id="test-3",
            description="Run tests",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        result = verify_completion(subtask, "test output")  # No validation_fn

        assert result.passed is False
        assert result.quality_score == 0.0
        assert "no validation function" in result.feedback.lower()
        assert result.evidence["error"] == "missing_validation_fn"

    def test_automated_test_exception_handling(self):
        """Should handle exceptions from validation function gracefully."""
        subtask = SubTask(
            id="test-4",
            description="Run tests with buggy validator",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        # Validation function that raises exception
        def buggy_validator(result: str) -> bool:
            raise ValueError("Validator bug")

        result = verify_completion(subtask, "test output", validation_fn=buggy_validator)

        assert result.passed is False
        assert result.quality_score == 0.0
        assert "exception" in result.feedback.lower()
        assert result.evidence["error_type"] == "ValueError"


# ============================================================================
# Test Semantic Similarity Verification
# ============================================================================

class TestSemanticSimilarityVerification:
    """Test semantic similarity verification method."""

    def test_semantic_similarity_pass_heuristic(self):
        """Semantic similarity should pass when word overlap >= 0.75 (heuristic fallback)."""
        subtask = SubTask(
            id="sem-1",
            description="Generate summary",
            verification_method=VerificationMethod.SEMANTIC_SIMILARITY,
            estimated_cost=0.5,
            estimated_duration=0.4,
            parallel_safe=True
        )

        # High overlap (same words)
        result_text = "multi agent orchestration framework"
        expected = "multi agent orchestration framework"

        with patch("delegation.verifier.HAS_EMBEDDINGS", False):
            result = verify_completion(subtask, result_text, expected_output=expected)

        assert result.passed is True  # Jaccard similarity = 1.0 >= 0.75
        assert result.quality_score >= 0.75
        assert result.method == VerificationMethod.SEMANTIC_SIMILARITY
        assert "heuristic" in result.feedback.lower()

    def test_semantic_similarity_fail_heuristic(self):
        """Semantic similarity should fail when word overlap < 0.75."""
        subtask = SubTask(
            id="sem-2",
            description="Generate summary",
            verification_method=VerificationMethod.SEMANTIC_SIMILARITY,
            estimated_cost=0.5,
            estimated_duration=0.4,
            parallel_safe=True
        )

        # Low overlap (completely different words)
        result_text = "apple banana cherry"
        expected = "dog cat elephant"

        with patch("delegation.verifier.HAS_EMBEDDINGS", False):
            result = verify_completion(subtask, result_text, expected_output=expected)

        assert result.passed is False  # No word overlap
        assert result.quality_score < 0.75
        assert result.method == VerificationMethod.SEMANTIC_SIMILARITY

    def test_semantic_similarity_missing_expected_output(self):
        """Should fail when no expected output provided."""
        subtask = SubTask(
            id="sem-3",
            description="Generate summary",
            verification_method=VerificationMethod.SEMANTIC_SIMILARITY,
            estimated_cost=0.5,
            estimated_duration=0.4,
            parallel_safe=True
        )

        result = verify_completion(subtask, "some output")  # No expected_output

        assert result.passed is False
        assert result.quality_score == 0.0
        assert "no expected output" in result.feedback.lower()
        assert result.evidence["error"] == "missing_expected_output"

    @patch("delegation.verifier.HAS_EMBEDDINGS", True)
    @patch("delegation.verifier.embed_single")
    @patch("delegation.verifier.cosine_similarity")
    def test_semantic_similarity_with_embeddings(self, mock_cosine, mock_embed):
        """Should use embeddings when available."""
        mock_embed.return_value = [0.1] * 768  # Mock embedding vector
        mock_cosine.return_value = 0.85  # High similarity

        subtask = SubTask(
            id="sem-4",
            description="Generate summary",
            verification_method=VerificationMethod.SEMANTIC_SIMILARITY,
            estimated_cost=0.5,
            estimated_duration=0.4,
            parallel_safe=True
        )

        result = verify_completion(subtask, "result text", expected_output="expected text")

        assert result.passed is True  # 0.85 >= 0.75
        assert result.quality_score == 0.85
        assert "nomic" in result.evidence["method"]
        mock_embed.assert_called()
        mock_cosine.assert_called_once()


# ============================================================================
# Test Human Review Verification
# ============================================================================

class TestHumanReviewVerification:
    """Test human review verification method."""

    def test_human_review_always_requires_manual(self):
        """Human review should always return passed=False (pending review)."""
        subtask = SubTask(
            id="human-1",
            description="Review UX design",
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=0.6,
            estimated_duration=0.5,
            parallel_safe=False
        )

        result = verify_completion(subtask, "UX design output")

        assert result.passed is False  # Never auto-pass
        assert result.quality_score == 0.5  # Neutral pending review
        assert result.method == VerificationMethod.HUMAN_REVIEW
        assert "human review" in result.feedback.lower()
        assert result.evidence["requires_human_review"] is True

    def test_human_review_includes_preview(self):
        """Human review should include preview of result."""
        subtask = SubTask(
            id="human-2",
            description="Review content",
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=0.6,
            estimated_duration=0.5,
            parallel_safe=False
        )

        long_result = "A" * 300  # Long result
        result = verify_completion(subtask, long_result)

        # Should truncate to 200 chars
        assert len(result.evidence["result_preview"]) == 200


# ============================================================================
# Test Ground Truth Verification
# ============================================================================

class TestGroundTruthVerification:
    """Test ground truth verification method."""

    def test_ground_truth_unavailable(self):
        """Should fail gracefully when CPB ground truth unavailable."""
        subtask = SubTask(
            id="gt-1",
            description="Verify factual claim",
            verification_method=VerificationMethod.GROUND_TRUTH,
            estimated_cost=0.7,
            estimated_duration=0.6,
            parallel_safe=True
        )

        with patch("delegation.verifier.HAS_GROUND_TRUTH", False):
            result = verify_completion(subtask, "factual output")

        assert result.passed is False
        assert result.quality_score == 0.0
        assert "unavailable" in result.feedback.lower()
        assert result.evidence["error"] == "missing_cpb_ground_truth"

    @patch("delegation.verifier.HAS_GROUND_TRUTH", True)
    @patch("delegation.verifier.asyncio.run")
    def test_ground_truth_pass(self, mock_asyncio_run):
        """Should pass when ground truth score >= 0.75."""
        # Mock validation result
        mock_validation = MagicMock()
        mock_validation.ground_truth_score = 0.85
        mock_validation.factual_accuracy = 0.9
        mock_validation.claims_verified = 5
        mock_validation.claims_contradicted = 0
        mock_asyncio_run.return_value = mock_validation

        subtask = SubTask(
            id="gt-2",
            description="Verify factual claim",
            verification_method=VerificationMethod.GROUND_TRUTH,
            estimated_cost=0.7,
            estimated_duration=0.6,
            parallel_safe=True,
            metadata={"sources": [{"url": "https://arxiv.org/abs/1234.5678"}]}
        )

        result = verify_completion(subtask, "factual output")

        assert result.passed is True  # 0.85 >= 0.75
        assert result.quality_score == 0.85
        assert result.evidence["ground_truth_score"] == 0.85
        assert result.evidence["claims_verified"] == 5

    @patch("delegation.verifier.HAS_GROUND_TRUTH", True)
    @patch("delegation.verifier.asyncio.run")
    def test_ground_truth_fail(self, mock_asyncio_run):
        """Should fail when ground truth score < 0.75."""
        # Mock validation result
        mock_validation = MagicMock()
        mock_validation.ground_truth_score = 0.5
        mock_validation.factual_accuracy = 0.6
        mock_validation.claims_verified = 2
        mock_validation.claims_contradicted = 3
        mock_asyncio_run.return_value = mock_validation

        subtask = SubTask(
            id="gt-3",
            description="Verify factual claim",
            verification_method=VerificationMethod.GROUND_TRUTH,
            estimated_cost=0.7,
            estimated_duration=0.6,
            parallel_safe=True
        )

        result = verify_completion(subtask, "factual output")

        assert result.passed is False  # 0.5 < 0.75
        assert result.quality_score == 0.5
        assert result.evidence["claims_contradicted"] == 3


# ============================================================================
# Test Verification Metadata
# ============================================================================

class TestVerificationMetadata:
    """Test verification result metadata and timing."""

    def test_duration_tracking(self):
        """Verification should track execution duration."""
        subtask = SubTask(
            id="meta-1",
            description="Test duration tracking",
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        start = time.time()
        result = verify_completion(subtask, "output")
        duration = time.time() - start

        assert "duration_seconds" in result.evidence
        # Duration should be close to actual elapsed time (within 100ms tolerance)
        assert abs(result.evidence["duration_seconds"] - duration) < 0.1

    def test_subtask_id_preserved(self):
        """VerificationResult should preserve subtask ID."""
        subtask = SubTask(
            id="unique-task-id-123",
            description="Test ID preservation",
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        result = verify_completion(subtask, "output")

        assert result.subtask_id == "unique-task-id-123"

    def test_timestamp_recorded(self):
        """Verification should record timestamp."""
        subtask = SubTask(
            id="meta-2",
            description="Test timestamp",
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        before = time.time()
        result = verify_completion(subtask, "output")
        after = time.time()

        assert before <= result.timestamp <= after


# ============================================================================
# Test Integration Points
# ============================================================================

class TestTrustLedgerIntegration:
    """Test trust ledger integration."""

    @patch("delegation.verifier.asyncio.run")
    def test_feed_to_trust_ledger(self, mock_asyncio_run):
        """Should feed verification results to trust ledger."""
        verification = VerificationResult(
            subtask_id="task-1",
            timestamp=time.time(),
            method=VerificationMethod.AUTOMATED_TEST,
            passed=True,
            quality_score=0.9,
            feedback="Test passed",
            evidence={"duration_seconds": 0.5}
        )

        # Mock the async function
        feed_to_trust_ledger(verification, agent_id="agent-1")

        # Should have called asyncio.run
        mock_asyncio_run.assert_called_once()


class TestMemoryBleedIntegration:
    """Test memory bleed integration."""

    @patch("delegation.memory_bleed.write_delegation_outcome")
    def test_feed_to_memory_bleed(self, mock_write):
        """Should feed verification outcomes to supermemory."""
        subtask = SubTask(
            id="task-1",
            description="Test task",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        verification = VerificationResult(
            subtask_id="task-1",
            timestamp=time.time(),
            method=VerificationMethod.AUTOMATED_TEST,
            passed=True,
            quality_score=0.9,
            feedback="Test passed",
            evidence={}
        )

        feed_to_memory_bleed(verification, subtask, "test result")

        # Should have called write_delegation_outcome
        mock_write.assert_called_once()
        args = mock_write.call_args[1]
        assert args["task"] == "Test task"
        assert "Verification: automated_test" in args["outcome"]
        assert "Passed: True" in args["outcome"]


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_verification_method(self):
        """Should handle unknown verification method gracefully."""
        # Create a fake verification method (not in enum)
        subtask = SubTask(
            id="edge-1",
            description="Test unknown method",
            verification_method="fake_method",  # type: ignore
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        result = verify_completion(subtask, "output")

        assert result.passed is False
        assert result.quality_score == 0.0
        assert "unknown" in result.feedback.lower()
        assert result.evidence["error"] == "unsupported_method"

    def test_empty_result_string(self):
        """Should handle empty result strings."""
        subtask = SubTask(
            id="edge-2",
            description="Test empty result",
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        result = verify_completion(subtask, "")  # Empty result

        assert result.method == VerificationMethod.HUMAN_REVIEW
        assert result.evidence["result_preview"] == ""

    def test_quality_score_clamping(self):
        """Quality score should always be in [0.0, 1.0]."""
        subtask = SubTask(
            id="edge-3",
            description="Test score clamping",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        def always_pass(result: str) -> bool:
            return True

        result = verify_completion(subtask, "output", validation_fn=always_pass)

        # Quality score must be in valid range
        assert 0.0 <= result.quality_score <= 1.0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance targets."""

    def test_automated_test_performance(self):
        """Automated test verification should be fast (<100ms)."""
        subtask = SubTask(
            id="perf-1",
            description="Performance test",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        def fast_validator(result: str) -> bool:
            return len(result) > 0

        start = time.time()
        result = verify_completion(subtask, "output", validation_fn=fast_validator)
        duration = (time.time() - start) * 1000  # Convert to ms

        assert duration < 100  # Under 100ms
        assert result.passed is True

    def test_human_review_performance(self):
        """Human review verification should be fast (<10ms)."""
        subtask = SubTask(
            id="perf-2",
            description="Performance test",
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=0.3,
            estimated_duration=0.2,
            parallel_safe=True
        )

        start = time.time()
        result = verify_completion(subtask, "output")
        duration = (time.time() - start) * 1000  # Convert to ms

        assert duration < 10  # Under 10ms
        assert result.passed is False  # Always requires human
