#!/usr/bin/env python3
"""
CPB Run Logger - Document all precision runs for research continuity.

Stores all runs regardless of DQ score, organized by quality tier:
- breakthrough/ : DQ >= 0.80 (high quality, verified)
- developing/   : DQ < 0.80 (needs refinement, still valuable data)

Every run is captured because:
1. Compute was used - don't waste the output
2. Low-scoring runs reveal what needs improvement
3. Longitudinal tracking shows system evolution
4. Research continuity across sessions
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


# =============================================================================
# CONFIGURATION
# =============================================================================

RUNS_BASE_DIR = Path.home() / ".agent-core" / "precision" / "runs"
BREAKTHROUGH_DIR = RUNS_BASE_DIR / "breakthrough"  # DQ >= 0.80
DEVELOPING_DIR = RUNS_BASE_DIR / "developing"      # DQ < 0.80

DQ_BREAKTHROUGH_THRESHOLD = 0.80


# =============================================================================
# RUN LOGGER
# =============================================================================

class RunLogger:
    """
    Log all precision runs for research documentation.

    Organizes runs into:
    - breakthrough/ : High-quality runs (DQ >= 0.80)
    - developing/   : Runs needing refinement (DQ < 0.80)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or RUNS_BASE_DIR
        self.breakthrough_dir = self.base_dir / "breakthrough"
        self.developing_dir = self.base_dir / "developing"

        # Ensure directories exist
        self.breakthrough_dir.mkdir(parents=True, exist_ok=True)
        self.developing_dir.mkdir(parents=True, exist_ok=True)

    def log_run(self, result: Any, query: str) -> Dict[str, Any]:
        """
        Log a precision run result.

        Args:
            result: PrecisionResult object
            query: Original query string

        Returns:
            Dict with log metadata (path, tier, run_id)
        """
        # Determine tier
        dq_score = getattr(result, 'dq_score', 0.0)
        is_breakthrough = dq_score >= DQ_BREAKTHROUGH_THRESHOLD
        tier = "breakthrough" if is_breakthrough else "developing"
        target_dir = self.breakthrough_dir if is_breakthrough else self.developing_dir

        # Generate run ID
        timestamp = datetime.now()
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        run_id = f"{timestamp.strftime('%Y%m%d-%H%M%S')}-{query_hash}"

        # Build run document
        run_doc = {
            "run_id": run_id,
            "timestamp": timestamp.isoformat(),
            "tier": tier,
            "dq_score": dq_score,
            "threshold": DQ_BREAKTHROUGH_THRESHOLD,

            # Query info
            "original_query": query,
            "enhanced_query": getattr(result, 'enhanced_query', query),
            "query_was_enhanced": getattr(result, 'query_was_enhanced', False),
            "enhancement_reasoning": getattr(result, 'enhancement_reasoning', ''),
            "query_dimensions": getattr(result, 'query_dimensions', []),
            "follow_up_queries": getattr(result, 'follow_up_queries', []),

            # Output
            "output": getattr(result, 'output', ''),
            "verified": getattr(result, 'verified', False),
            "needs_review": getattr(result, 'needs_review', False),

            # Scores
            "scores": {
                "validity": getattr(result, 'validity', 0.0),
                "specificity": getattr(result, 'specificity', 0.0),
                "correctness": getattr(result, 'correctness', 0.0),
                "ground_truth": getattr(result, 'ground_truth_score', 0.0),
                "factual_accuracy": getattr(result, 'factual_accuracy', 0.0),
                "cross_source": getattr(result, 'cross_source_score', 0.0),
                "self_consistency": getattr(result, 'self_consistency', 0.0),
            },

            # Evidence
            "sources": getattr(result, 'sources', []),
            "citations_found": getattr(result, 'citations_found', 0),
            "citations_verified": getattr(result, 'citations_verified', 0),

            # Claims
            "claims_checked": getattr(result, 'claims_checked', 0),
            "claims_verified": getattr(result, 'claims_verified', 0),
            "claims_contradicted": getattr(result, 'claims_contradicted', 0),
            "verified_claims": getattr(result, 'verified_claims', []),
            "contradicted_claims": getattr(result, 'contradicted_claims', []),

            # Execution
            "execution": {
                "time_ms": getattr(result, 'execution_time_ms', 0),
                "search_time_ms": getattr(result, 'search_time_ms', 0),
                "retry_count": getattr(result, 'retry_count', 0),
                "agent_count": getattr(result, 'agent_count', 7),
                "path": getattr(result, 'path', 'cascade').value if hasattr(getattr(result, 'path', None), 'value') else 'cascade',
                "rg_connection_mode": getattr(result, 'rg_connection_mode', 'unknown'),
            },

            # Search stats
            "search": {
                "tier1_count": getattr(result, 'tier1_count', 0),
                "tier2_count": getattr(result, 'tier2_count', 0),
                "tier3_count": getattr(result, 'tier3_count', 0),
                "total_sources": getattr(result, 'total_sources_found', 0),
            },

            # Refinement history
            "refinement_targets": getattr(result, 'refinement_targets', []),
            "feedback_history": getattr(result, 'feedback_history', []),
            "warnings": getattr(result, 'warnings', []),
        }

        # Save to file
        filename = f"{run_id}.json"
        filepath = target_dir / filename

        with open(filepath, 'w') as f:
            json.dump(run_doc, f, indent=2, default=str)

        return {
            "run_id": run_id,
            "tier": tier,
            "path": str(filepath),
            "dq_score": dq_score,
            "is_breakthrough": is_breakthrough,
        }

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a run by ID."""
        # Check both directories
        for tier_dir in [self.breakthrough_dir, self.developing_dir]:
            filepath = tier_dir / f"{run_id}.json"
            if filepath.exists():
                with open(filepath) as f:
                    return json.load(f)
        return None

    def list_runs(self, tier: Optional[str] = None, limit: int = 20) -> list:
        """List recent runs, optionally filtered by tier."""
        runs = []

        dirs_to_check = []
        if tier == "breakthrough":
            dirs_to_check = [self.breakthrough_dir]
        elif tier == "developing":
            dirs_to_check = [self.developing_dir]
        else:
            dirs_to_check = [self.breakthrough_dir, self.developing_dir]

        for tier_dir in dirs_to_check:
            if tier_dir.exists():
                for filepath in tier_dir.glob("*.json"):
                    try:
                        with open(filepath) as f:
                            run = json.load(f)
                            runs.append({
                                "run_id": run.get("run_id"),
                                "timestamp": run.get("timestamp"),
                                "tier": run.get("tier"),
                                "dq_score": run.get("dq_score"),
                                "query": run.get("original_query", "")[:60] + "...",
                                "verified": run.get("verified", False),
                            })
                    except Exception:
                        pass

        # Sort by timestamp descending
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return runs[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged runs."""
        breakthrough_count = len(list(self.breakthrough_dir.glob("*.json"))) if self.breakthrough_dir.exists() else 0
        developing_count = len(list(self.developing_dir.glob("*.json"))) if self.developing_dir.exists() else 0

        return {
            "total_runs": breakthrough_count + developing_count,
            "breakthrough_runs": breakthrough_count,
            "developing_runs": developing_count,
            "breakthrough_rate": breakthrough_count / max(1, breakthrough_count + developing_count),
            "breakthrough_dir": str(self.breakthrough_dir),
            "developing_dir": str(self.developing_dir),
        }


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_run_logger: Optional[RunLogger] = None


def get_run_logger() -> RunLogger:
    """Get or create the run logger singleton."""
    global _run_logger
    if _run_logger is None:
        _run_logger = RunLogger()
    return _run_logger


def log_run(result: Any, query: str) -> Dict[str, Any]:
    """Log a precision run result."""
    return get_run_logger().log_run(result, query)


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a run by ID."""
    return get_run_logger().get_run(run_id)


def list_runs(tier: Optional[str] = None, limit: int = 20) -> list:
    """List recent runs."""
    return get_run_logger().list_runs(tier, limit)


def get_run_stats() -> Dict[str, Any]:
    """Get run statistics."""
    return get_run_logger().get_stats()


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Show stats
    stats = get_run_stats()
    print("Run Logger Stats:")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Breakthrough: {stats['breakthrough_runs']} ({stats['breakthrough_rate']:.1%})")
    print(f"  Developing: {stats['developing_runs']}")
    print(f"\nDirectories:")
    print(f"  Breakthrough: {stats['breakthrough_dir']}")
    print(f"  Developing: {stats['developing_dir']}")

    # List recent runs
    print("\nRecent runs:")
    for run in list_runs(limit=5):
        icon = "✅" if run['tier'] == 'breakthrough' else "🔄"
        print(f"  {icon} [{run['dq_score']:.2f}] {run['query']}")
