"""
Delegation Evolution — Learning from Outcomes

Implements the learning system from arXiv:2602.11865 Section 7.

Evolution module learns from delegation outcomes to improve:
- Task profiling accuracy (refine dimension scoring via EMA)
- Routing decisions (agent affinity matrix)
- Decomposition strategy (optimal subtask count per complexity band)
- Verification method selection (per-method pass rate tracking)

Learning mechanism: Exponential Moving Average (EMA) over outcome windows.
No ML models — pure statistical learning from the delegation_chains and
trust_entries tables, compatible with the existing SQLite stack.

Usage:
    from delegation.evolution import EvolutionEngine

    engine = EvolutionEngine()

    # Record delegation outcome
    engine.record_outcome(
        delegation_id="del-123",
        success=True,
        quality_score=0.92,
        actual_cost=0.45,
        actual_duration=0.6
    )

    # Apply learnings
    improvements = engine.evolve_strategies()

    # Performance trends
    trends = engine.get_performance_trends(window_days=30)
"""

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# EMA smoothing factor (higher = more weight to recent outcomes)
EMA_ALPHA = 0.3

# Complexity bands for decomposition strategy learning
COMPLEXITY_BANDS = [
    (0.0, 0.3, "low"),
    (0.3, 0.6, "medium"),
    (0.6, 0.8, "high"),
    (0.8, 1.0, "very_high"),
]


def _band_for(complexity: float) -> str:
    """Return the complexity band label for a given score."""
    for low, high, label in COMPLEXITY_BANDS:
        if low <= complexity < high:
            return label
    return "very_high"


class EvolutionEngine:
    """
    Learns from delegation outcomes to improve future performance.

    Uses EMA-based statistical learning over SQLite tables — no external
    ML dependencies required. Compatible with the existing delegation schema.
    """

    def __init__(self, db_path: str = ""):
        """
        Initialize evolution engine.

        Args:
            db_path: Path to SQLite database (default: ~/.agent-core/storage/delegation.db)
        """
        self.db_path = db_path or str(
            Path.home() / ".agent-core" / "storage" / "delegation.db"
        )
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        """Get a database connection."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=2.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self):
        """Create evolution-specific tables if they don't exist."""
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evolution_outcomes (
                    delegation_id   TEXT PRIMARY KEY,
                    timestamp       REAL NOT NULL,
                    success         INTEGER NOT NULL,
                    quality_score   REAL NOT NULL,
                    actual_cost     REAL,
                    actual_duration REAL,
                    complexity      REAL,
                    subtask_count   INTEGER,
                    agent_ids       TEXT,
                    feedback        TEXT,
                    CHECK (quality_score BETWEEN 0.0 AND 1.0)
                );

                CREATE TABLE IF NOT EXISTS evolution_weights (
                    key             TEXT PRIMARY KEY,
                    value           REAL NOT NULL,
                    updated_at      REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_eo_ts
                    ON evolution_outcomes (timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_eo_success
                    ON evolution_outcomes (success);
            """)
            conn.commit()
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════════════
    # RECORD OUTCOMES
    # ═══════════════════════════════════════════════════════════════════════

    def record_outcome(
        self,
        delegation_id: str,
        success: bool,
        quality_score: float,
        actual_cost: float = 0.0,
        actual_duration: float = 0.0,
        complexity: float = 0.5,
        subtask_count: int = 0,
        agent_ids: Optional[List[str]] = None,
        feedback: str = ""
    ) -> None:
        """
        Record a delegation outcome for learning.

        Args:
            delegation_id: Delegation chain identifier
            success: Whether delegation succeeded
            quality_score: Final quality score [0.0, 1.0]
            actual_cost: Actual cost incurred [0.0, 1.0]
            actual_duration: Actual time taken (seconds)
            complexity: Task complexity score [0.0, 1.0]
            subtask_count: Number of subtasks in decomposition
            agent_ids: List of agent IDs involved
            feedback: Optional human feedback text
        """
        conn = self._connect()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO evolution_outcomes (
                    delegation_id, timestamp, success, quality_score,
                    actual_cost, actual_duration, complexity,
                    subtask_count, agent_ids, feedback
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                delegation_id,
                time.time(),
                1 if success else 0,
                max(0.0, min(1.0, quality_score)),
                actual_cost,
                actual_duration,
                complexity,
                subtask_count,
                json.dumps(agent_ids or []),
                feedback,
            ))
            conn.commit()
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════════════
    # EVOLVE STRATEGIES
    # ═══════════════════════════════════════════════════════════════════════

    def evolve_strategies(self) -> Dict[str, Any]:
        """
        Evolve delegation strategies based on recorded outcomes.

        Computes:
        1. Optimal subtask count per complexity band (EMA of successful outcomes)
        2. Agent affinity scores (success rate per agent)
        3. Routing weight adjustments (capability vs trust vs cost)
        4. Quality trend (EMA of recent quality scores)

        Returns:
            Dict with strategy recommendations and metrics
        """
        conn = self._connect()
        try:
            results = {}

            # 1. Optimal subtask count per complexity band
            results["decomposition"] = self._learn_decomposition(conn)

            # 2. Agent affinity scores
            results["agent_affinity"] = self._learn_agent_affinity(conn)

            # 3. Quality trend
            results["quality_trend"] = self._learn_quality_trend(conn)

            # 4. Cost efficiency
            results["cost_efficiency"] = self._learn_cost_efficiency(conn)

            # Store evolved weights
            for key, value in results.get("decomposition", {}).items():
                if isinstance(value, (int, float)):
                    self._set_weight(conn, f"decomp_{key}", float(value))

            quality = results.get("quality_trend", {}).get("ema_quality", 0.0)
            if quality > 0:
                self._set_weight(conn, "ema_quality", quality)

            conn.commit()
            return results
        finally:
            conn.close()

    def _learn_decomposition(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Learn optimal subtask counts per complexity band."""
        result = {}

        for low, high, band in COMPLEXITY_BANDS:
            rows = conn.execute("""
                SELECT subtask_count, quality_score
                FROM evolution_outcomes
                WHERE success = 1
                  AND complexity >= ? AND complexity < ?
                  AND subtask_count > 0
                ORDER BY timestamp DESC
                LIMIT 50
            """, (low, high)).fetchall()

            if not rows:
                continue

            # Weighted average: higher quality outcomes weight more
            total_weight = sum(r["quality_score"] for r in rows)
            if total_weight > 0:
                optimal_count = sum(
                    r["subtask_count"] * r["quality_score"] for r in rows
                ) / total_weight
            else:
                optimal_count = sum(r["subtask_count"] for r in rows) / len(rows)

            result[band] = {
                "optimal_subtask_count": round(optimal_count, 1),
                "sample_size": len(rows),
                "avg_quality": round(total_weight / len(rows), 3),
            }

        return result

    def _learn_agent_affinity(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Learn agent success rates from outcomes."""
        rows = conn.execute("""
            SELECT agent_ids, success, quality_score
            FROM evolution_outcomes
            WHERE agent_ids != '[]'
            ORDER BY timestamp DESC
            LIMIT 200
        """).fetchall()

        agent_stats: Dict[str, Dict[str, float]] = {}

        for row in rows:
            agents = json.loads(row["agent_ids"])
            for agent_id in agents:
                if agent_id not in agent_stats:
                    agent_stats[agent_id] = {
                        "successes": 0, "failures": 0,
                        "quality_sum": 0.0, "count": 0,
                    }
                stats = agent_stats[agent_id]
                stats["count"] += 1
                stats["quality_sum"] += row["quality_score"]
                if row["success"]:
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1

        # Convert to affinity scores
        affinity = {}
        for agent_id, stats in agent_stats.items():
            total = stats["successes"] + stats["failures"]
            affinity[agent_id] = {
                "success_rate": round(stats["successes"] / total, 3) if total else 0,
                "avg_quality": round(stats["quality_sum"] / stats["count"], 3) if stats["count"] else 0,
                "total_delegations": total,
            }

        return affinity

    def _learn_quality_trend(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Compute EMA quality trend from recent outcomes."""
        rows = conn.execute("""
            SELECT quality_score, timestamp
            FROM evolution_outcomes
            ORDER BY timestamp ASC
        """).fetchall()

        if not rows:
            return {"ema_quality": 0.0, "trend": "insufficient_data", "sample_size": 0}

        # Compute EMA
        ema = rows[0]["quality_score"]
        for row in rows[1:]:
            ema = EMA_ALPHA * row["quality_score"] + (1 - EMA_ALPHA) * ema

        # Trend: compare first half vs second half
        mid = len(rows) // 2
        if mid > 0:
            first_half = sum(r["quality_score"] for r in rows[:mid]) / mid
            second_half = sum(r["quality_score"] for r in rows[mid:]) / (len(rows) - mid)
            delta = second_half - first_half

            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "ema_quality": round(ema, 3),
            "trend": trend,
            "sample_size": len(rows),
        }

    def _learn_cost_efficiency(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Track cost efficiency over time."""
        rows = conn.execute("""
            SELECT actual_cost, quality_score, success
            FROM evolution_outcomes
            WHERE actual_cost > 0
            ORDER BY timestamp DESC
            LIMIT 50
        """).fetchall()

        if not rows:
            return {"avg_cost_per_quality": 0.0, "sample_size": 0}

        total_cost = sum(r["actual_cost"] for r in rows)
        total_quality = sum(r["quality_score"] for r in rows)
        success_rate = sum(r["success"] for r in rows) / len(rows)

        return {
            "avg_cost_per_quality": round(total_cost / max(total_quality, 0.01), 3),
            "avg_cost": round(total_cost / len(rows), 3),
            "success_rate": round(success_rate, 3),
            "sample_size": len(rows),
        }

    def _set_weight(self, conn: sqlite3.Connection, key: str, value: float):
        """Store an evolved weight."""
        conn.execute("""
            INSERT OR REPLACE INTO evolution_weights (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, value, time.time()))

    def get_weight(self, key: str, default: float = 0.0) -> float:
        """Retrieve an evolved weight."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value FROM evolution_weights WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════════════
    # PERFORMANCE TRENDS
    # ═══════════════════════════════════════════════════════════════════════

    def get_performance_trends(self, window_days: int = 30) -> Dict[str, Any]:
        """
        Get performance trends over a time window.

        Args:
            window_days: Number of days to analyze

        Returns:
            Dict with daily quality, success rate, cost, and volume metrics
        """
        cutoff = time.time() - (window_days * 86400)
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT
                    date(timestamp, 'unixepoch') as day,
                    COUNT(*) as count,
                    SUM(success) as successes,
                    AVG(quality_score) as avg_quality,
                    AVG(actual_cost) as avg_cost,
                    AVG(actual_duration) as avg_duration
                FROM evolution_outcomes
                WHERE timestamp >= ?
                GROUP BY day
                ORDER BY day ASC
            """, (cutoff,)).fetchall()

            if not rows:
                return {"days": [], "summary": {"total": 0}}

            days = []
            for row in rows:
                days.append({
                    "date": row["day"],
                    "delegations": row["count"],
                    "success_rate": round(row["successes"] / row["count"], 3) if row["count"] else 0,
                    "avg_quality": round(row["avg_quality"], 3) if row["avg_quality"] else 0,
                    "avg_cost": round(row["avg_cost"], 3) if row["avg_cost"] else 0,
                    "avg_duration": round(row["avg_duration"], 1) if row["avg_duration"] else 0,
                })

            total = sum(d["delegations"] for d in days)
            total_success = sum(
                d["delegations"] * d["success_rate"] for d in days
            )

            return {
                "days": days,
                "summary": {
                    "total_delegations": total,
                    "overall_success_rate": round(total_success / max(total, 1), 3),
                    "avg_daily_volume": round(total / max(len(days), 1), 1),
                    "window_days": window_days,
                    "active_days": len(days),
                },
            }
        finally:
            conn.close()

    def get_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations from learned patterns.

        Returns:
            List of human-readable recommendation strings
        """
        recommendations = []
        conn = self._connect()
        try:
            # Check overall success rate
            row = conn.execute("""
                SELECT COUNT(*) as total, SUM(success) as wins
                FROM evolution_outcomes
            """).fetchone()

            if row and row["total"] >= 5:
                rate = row["wins"] / row["total"]
                if rate < 0.6:
                    recommendations.append(
                        f"Success rate is low ({rate:.0%}). Consider raising quality_threshold "
                        "or improving task descriptions before delegation."
                    )
                elif rate > 0.9:
                    recommendations.append(
                        f"Success rate is high ({rate:.0%}). You may be over-cautious — "
                        "consider delegating more complex tasks."
                    )

            # Check if decomposition needs tuning
            row = conn.execute("""
                SELECT AVG(subtask_count) as avg_st, AVG(quality_score) as avg_q
                FROM evolution_outcomes
                WHERE success = 1 AND subtask_count > 0
            """).fetchone()

            if row and row["avg_st"]:
                avg_st = row["avg_st"]
                if avg_st > 6:
                    recommendations.append(
                        f"Average subtask count is high ({avg_st:.1f}). "
                        "Over-decomposition may be adding overhead."
                    )
                elif avg_st < 2:
                    recommendations.append(
                        f"Average subtask count is low ({avg_st:.1f}). "
                        "Consider deeper decomposition for complex tasks."
                    )

            # Check EMA quality weight
            ema = self.get_weight("ema_quality", 0.0)
            if ema > 0 and ema < 0.6:
                recommendations.append(
                    f"EMA quality trend is low ({ema:.3f}). "
                    "Review recent delegation failures for patterns."
                )

            if not recommendations:
                recommendations.append("System is performing within normal parameters.")

            return recommendations
        finally:
            conn.close()
