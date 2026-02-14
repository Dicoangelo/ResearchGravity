"""
Result Verifier — Subtask Output Verification

Implements the verification framework from arXiv:2602.11865 Section 4.8.

Verifies subtask outputs using multiple methods:

1. Automated Test: Execute validation function attached to subtask
2. Semantic Similarity: Compare output to expected result via embeddings (threshold 0.75)
3. Human Review: Flag for manual verification (subjective tasks)
4. Ground Truth: Use CPB ground truth system for factual claims

Contract-first principle: All subtasks must be verifiable before delegation.

Usage:
    from delegation.verifier import verify_completion
    from delegation.models import SubTask, VerificationMethod

    subtask = SubTask(
        id="task-1",
        description="Write unit tests",
        verification_method=VerificationMethod.AUTOMATED_TEST,
        ...
    )

    result = verify_completion(
        subtask=subtask,
        result="test code here..."
    )
    print(f"Passed: {result.passed}, Confidence: {result.confidence}")
"""

import time
import asyncio
from typing import Optional, Callable, Dict, Any

from .models import SubTask, VerificationResult, VerificationMethod

# Lazy imports with fallbacks
try:
    from mcp_raw.embeddings import embed_single, cosine_similarity
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

try:
    from cpb.ground_truth import validate_against_ground_truth
    HAS_GROUND_TRUTH = True
except ImportError:
    HAS_GROUND_TRUTH = False


def _run_async(coro):
    """Run a coroutine safely from sync or async context."""
    try:
        asyncio.get_running_loop()
        # Already in async context — run in a new thread to avoid RuntimeError
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=10)
    except RuntimeError:
        # No running loop — safe to use asyncio.run directly
        return asyncio.run(coro)


def verify_completion(
    subtask: SubTask,
    result: str,
    expected_output: Optional[str] = None,
    validation_fn: Optional[Callable[[str], bool]] = None,
) -> VerificationResult:
    """
    Verify subtask completion against its contract.

    Dispatches by verification_method:
    - AUTOMATED_TEST: Execute validation_fn (pass/fail)
    - SEMANTIC_SIMILARITY: Compare result to expected_output (>= 0.75 threshold)
    - HUMAN_REVIEW: Flag for manual check (always requires human)
    - GROUND_TRUTH: Use CPB ground truth for factual validation

    Args:
        subtask: SubTask with verification_method specified
        result: Agent output to verify
        expected_output: Expected output for semantic comparison
        validation_fn: Validation callable for automated tests

    Returns:
        VerificationResult with passed, confidence, method, evidence, duration
    """
    start_time = time.time()
    method = subtask.verification_method

    # Dispatch by verification method
    if method == VerificationMethod.AUTOMATED_TEST:
        verification = _verify_automated_test(subtask, result, validation_fn)
    elif method == VerificationMethod.SEMANTIC_SIMILARITY:
        verification = _verify_semantic_similarity(subtask, result, expected_output)
    elif method == VerificationMethod.HUMAN_REVIEW:
        verification = _verify_human_review(subtask, result)
    elif method == VerificationMethod.GROUND_TRUTH:
        verification = _verify_ground_truth(subtask, result)
    else:
        # Fallback for unknown methods
        verification = VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=method,
            passed=False,
            quality_score=0.0,
            feedback=f"Unknown verification method: {method}",
            evidence={"error": "unsupported_method"}
        )

    # Record duration
    duration = time.time() - start_time
    verification.evidence["duration_seconds"] = round(duration, 3)

    return verification


def _verify_automated_test(
    subtask: SubTask,
    result: str,
    validation_fn: Optional[Callable[[str], bool]],
) -> VerificationResult:
    """
    Execute validation function attached to subtask.

    Args:
        subtask: SubTask being verified
        result: Agent output
        validation_fn: Callable that returns True if result is valid

    Returns:
        VerificationResult with pass/fail from validation_fn
    """
    if validation_fn is None:
        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.AUTOMATED_TEST,
            passed=False,
            quality_score=0.0,
            feedback="No validation function provided for automated test",
            evidence={"error": "missing_validation_fn"}
        )

    try:
        # Execute validation function
        passed = validation_fn(result)
        confidence = 1.0 if passed else 0.0

        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.AUTOMATED_TEST,
            passed=bool(passed),
            quality_score=confidence,
            feedback="Automated test passed" if passed else "Automated test failed",
            evidence={
                "validation_fn": validation_fn.__name__ if hasattr(validation_fn, '__name__') else "anonymous",
                "test_passed": passed
            }
        )
    except Exception as e:
        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.AUTOMATED_TEST,
            passed=False,
            quality_score=0.0,
            feedback=f"Validation function raised exception: {str(e)}",
            evidence={"error": str(e), "error_type": type(e).__name__}
        )


def _verify_semantic_similarity(
    subtask: SubTask,
    result: str,
    expected_output: Optional[str],
) -> VerificationResult:
    """
    Compare result to expected output via semantic embeddings.

    Uses mcp_raw.embeddings pipeline (Nomic 768d vectors).
    Threshold: 0.75 for passing (from acceptance criteria).

    Args:
        subtask: SubTask being verified
        result: Agent output
        expected_output: Expected output for comparison

    Returns:
        VerificationResult with similarity score and pass/fail (>= 0.75)
    """
    if expected_output is None:
        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.SEMANTIC_SIMILARITY,
            passed=False,
            quality_score=0.0,
            feedback="No expected output provided for semantic comparison",
            evidence={"error": "missing_expected_output"}
        )

    if not HAS_EMBEDDINGS:
        # Fallback: simple string overlap heuristic
        result_words = set(result.lower().split())
        expected_words = set(expected_output.lower().split())
        if not expected_words:
            similarity = 0.0
        else:
            overlap = len(result_words & expected_words)
            similarity = overlap / max(len(result_words), len(expected_words))
        similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        passed = similarity >= 0.75
        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.SEMANTIC_SIMILARITY,
            passed=passed,
            quality_score=similarity,
            feedback=f"Heuristic similarity: {similarity:.3f} (embeddings unavailable)",
            evidence={
                "similarity": round(similarity, 3),
                "threshold": 0.75,
                "method": "word_overlap_heuristic"
            }
        )

    try:
        # Embed both texts
        result_emb = embed_single(result, prefix="search_document")
        expected_emb = embed_single(expected_output, prefix="search_document")

        # Compute cosine similarity
        similarity = cosine_similarity(result_emb, expected_emb)
        similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        passed = similarity >= 0.75
        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.SEMANTIC_SIMILARITY,
            passed=passed,
            quality_score=similarity,
            feedback=f"Semantic similarity: {similarity:.3f} ({'PASS' if passed else 'FAIL'}, threshold 0.75)",
            evidence={
                "similarity": round(similarity, 3),
                "threshold": 0.75,
                "method": "nomic_embeddings_768d"
            }
        )
    except Exception as e:
        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.SEMANTIC_SIMILARITY,
            passed=False,
            quality_score=0.0,
            feedback=f"Embedding computation failed: {str(e)}",
            evidence={"error": str(e), "error_type": type(e).__name__}
        )


def _verify_human_review(
    subtask: SubTask,
    result: str,
) -> VerificationResult:
    """
    Flag for manual human review.

    For subjective tasks that require human judgment.
    Always returns passed=False with quality_score=0.5 (pending review).

    Args:
        subtask: SubTask being verified
        result: Agent output

    Returns:
        VerificationResult flagged for human review (not passed)
    """
    return VerificationResult(
        subtask_id=subtask.id,
        timestamp=time.time(),
        method=VerificationMethod.HUMAN_REVIEW,
        passed=False,  # Never auto-pass, always requires human
        quality_score=0.5,  # Neutral pending review
        feedback="Flagged for human review (subjective task)",
        evidence={
            "requires_human_review": True,
            "result_preview": result[:200] if result else ""
        }
    )


def _verify_ground_truth(
    subtask: SubTask,
    result: str,
) -> VerificationResult:
    """
    Verify against CPB ground truth system.

    Uses cpb/ground_truth.py validate_against_ground_truth for factual claims.

    Args:
        subtask: SubTask being verified
        result: Agent output

    Returns:
        VerificationResult with ground truth validation score
    """
    if not HAS_GROUND_TRUTH:
        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.GROUND_TRUTH,
            passed=False,
            quality_score=0.0,
            feedback="CPB ground truth system unavailable",
            evidence={"error": "missing_cpb_ground_truth"}
        )

    try:
        # Extract query from subtask description
        query = subtask.description

        # Extract sources from subtask metadata if available
        sources = subtask.metadata.get("sources", [])

        # Run ground truth validation
        validation_result = _run_async(
            validate_against_ground_truth(
                query=query,
                output=result,
                sources=sources
            )
        )

        # Extract ground truth score
        gt_score = validation_result.ground_truth_score
        gt_score = max(0.0, min(1.0, gt_score))  # Clamp to [0, 1]

        # Pass if ground truth score >= 0.75
        passed = gt_score >= 0.75

        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.GROUND_TRUTH,
            passed=passed,
            quality_score=gt_score,
            feedback=f"Ground truth score: {gt_score:.3f} ({'PASS' if passed else 'FAIL'}, threshold 0.75)",
            evidence={
                "ground_truth_score": round(gt_score, 3),
                "factual_accuracy": round(validation_result.factual_accuracy, 3),
                "claims_verified": validation_result.claims_verified,
                "claims_contradicted": validation_result.claims_contradicted,
                "threshold": 0.75
            }
        )
    except Exception as e:
        return VerificationResult(
            subtask_id=subtask.id,
            timestamp=time.time(),
            method=VerificationMethod.GROUND_TRUTH,
            passed=False,
            quality_score=0.0,
            feedback=f"Ground truth validation failed: {str(e)}",
            evidence={"error": str(e), "error_type": type(e).__name__}
        )


# ============================================================================
# Integration Points
# ============================================================================

def feed_to_trust_ledger(
    verification: VerificationResult,
    agent_id: str,
    trust_ledger: Optional[Any] = None
):
    """
    Feed verification result to trust ledger (US-004).

    Args:
        verification: VerificationResult from verify_completion
        agent_id: Agent ID that produced the result
        trust_ledger: TrustLedger instance (optional, for testing)

    Returns:
        None (side effect: updates trust ledger)
    """
    if trust_ledger is None:
        try:
            from .trust_ledger import TrustLedger
        except ImportError:
            return  # Graceful degradation

    # Run async context manager (safe from any calling context)
    _run_async(_feed_to_trust_ledger_async(verification, agent_id, trust_ledger))


async def _feed_to_trust_ledger_async(verification: VerificationResult, agent_id: str, ledger_class):
    """Async helper for feeding to trust ledger."""
    from .trust_ledger import TrustLedger
    async with TrustLedger() as ledger:
        # Infer task type from verification method
        task_type = f"verification_{verification.method.value}"

        await ledger.record_outcome(
            agent_id=agent_id,
            task_type=task_type,
            success=verification.passed,
            quality_score=verification.quality_score,
            duration=verification.evidence.get("duration_seconds", 0.0)
        )


def feed_to_memory_bleed(
    verification: VerificationResult,
    subtask: SubTask,
    result: str
):
    """
    Feed verification outcome to memory bleed (US-008).

    Args:
        verification: VerificationResult from verify_completion
        subtask: SubTask that was verified
        result: Agent output

    Returns:
        None (side effect: writes to supermemory)
    """
    try:
        from .memory_bleed import write_delegation_outcome
    except ImportError:
        return  # Graceful degradation

    # Write to supermemory reviews table
    write_delegation_outcome(
        task=subtask.description,
        outcome=f"Verification: {verification.method.value} | Passed: {verification.passed} | Score: {verification.quality_score:.3f}\nResult: {result[:500]}"
    )
