"""
Coherence Engine — FSRS Insight Resurfacing

Free Spaced Repetition Scheduler adapted for cognitive insights.
Instead of flashcards, we schedule coherence moments and insights
for resurfacing at optimal intervals.

The FSRS algorithm determines when to re-show an insight based on:
  - Retrievability: probability the user still remembers (0-1)
  - Stability: days until retrievability drops to 0.9
  - Difficulty: inherent complexity of the insight (0-1)
  - Rating history: user feedback on each review

Sources:
  - FSRS Algorithm: https://github.com/open-spaced-repetition/fsrs4anki
  - Adapted for cognitive insights (not flashcards)
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("coherence.fsrs")


@dataclass
class InsightCard:
    """A schedulable insight derived from a coherence moment."""
    insight_id: str
    moment_id: str
    retrievability: float = 1.0   # Probability of recall (0-1)
    stability: float = 1.0        # Days until R drops to 0.9
    difficulty: float = 0.5       # Inherent complexity (0-1)
    last_review: Optional[datetime] = None
    next_review: Optional[datetime] = None
    review_count: int = 0
    rating_history: List[Dict] = field(default_factory=list)


# FSRS parameters (simplified from the 17-parameter model)
# These are calibrated for cognitive insight review, not flashcards.
INITIAL_STABILITY = {
    1: 0.5,   # forgot → re-show quickly
    2: 1.0,   # hard → 1 day
    3: 3.0,   # good → 3 days
    4: 7.0,   # easy → 1 week
}
STABILITY_GROWTH = 1.5       # How much stability grows per successful review
DIFFICULTY_DECAY = 0.1       # How much difficulty affects stability growth
MIN_STABILITY = 0.25         # Minimum 6 hours between reviews
MAX_STABILITY = 365.0        # Maximum 1 year between reviews
RETRIEVABILITY_TARGET = 0.9  # Re-show when R drops below this


def compute_retrievability(stability: float, elapsed_days: float) -> float:
    """
    Compute current retrievability using the forgetting curve.

    R(t) = (1 + t / (9 * S))^(-1)

    Where S = stability (in days), t = elapsed time (in days).
    This is the FSRS power-law forgetting curve.
    """
    if stability <= 0 or elapsed_days < 0:
        return 1.0
    return (1 + elapsed_days / (9 * stability)) ** (-1)


def schedule_next_review(
    stability: float,
    difficulty: float,
    rating: int,
    review_count: int = 0,
) -> tuple:
    """
    FSRS scheduling algorithm (simplified).

    Args:
        stability: Current stability in days
        difficulty: Difficulty factor (0-1)
        rating: User rating (1=forgot, 2=hard, 3=good, 4=easy)
        review_count: Number of previous reviews

    Returns:
        (new_stability, new_difficulty, interval_days)
    """
    if review_count == 0:
        # First review: use initial stability lookup
        new_stability = INITIAL_STABILITY.get(rating, 3.0)
        new_difficulty = max(0.0, min(1.0, difficulty + (3 - rating) * 0.1))
        return new_stability, new_difficulty, new_stability

    if rating == 1:
        # Forgot: reset stability significantly
        new_stability = max(MIN_STABILITY, stability * 0.2)
        new_difficulty = min(1.0, difficulty + 0.1)
    elif rating == 2:
        # Hard: slight growth
        growth = STABILITY_GROWTH * (1.0 - difficulty * DIFFICULTY_DECAY)
        new_stability = max(MIN_STABILITY, stability * growth * 0.8)
        new_difficulty = min(1.0, difficulty + 0.05)
    elif rating == 3:
        # Good: normal growth
        growth = STABILITY_GROWTH * (1.0 - difficulty * DIFFICULTY_DECAY)
        new_stability = stability * growth
        new_difficulty = max(0.0, difficulty - 0.02)
    else:
        # Easy: accelerated growth
        growth = STABILITY_GROWTH * (1.0 - difficulty * DIFFICULTY_DECAY) * 1.3
        new_stability = stability * growth
        new_difficulty = max(0.0, difficulty - 0.05)

    new_stability = max(MIN_STABILITY, min(MAX_STABILITY, new_stability))
    interval_days = new_stability * RETRIEVABILITY_TARGET

    return new_stability, new_difficulty, interval_days


class InsightScheduler:
    """
    Manages the FSRS review schedule for coherence insights.

    Creates InsightCards from coherence moments, schedules reviews,
    processes user ratings, and surfaces due insights.
    """

    def __init__(self, pool=None):
        self._pool = pool

    async def ensure_schema(self):
        """Create the insight_schedule table if it doesn't exist."""
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS insight_schedule (
                    insight_id TEXT PRIMARY KEY,
                    moment_id TEXT,
                    retrievability REAL DEFAULT 1.0,
                    stability REAL DEFAULT 1.0,
                    difficulty REAL DEFAULT 0.5,
                    last_review TIMESTAMPTZ,
                    next_review TIMESTAMPTZ,
                    review_count INT DEFAULT 0,
                    rating_history JSONB DEFAULT '[]',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_insight_next_review
                ON insight_schedule(next_review)
                WHERE next_review IS NOT NULL
            """)

    async def schedule_moment(self, moment_id: str, confidence: float = 0.8):
        """
        Create an insight card from a coherence moment.

        Higher confidence moments get lower initial difficulty
        (they're clearer, so easier to remember).
        """
        if not self._pool:
            return

        import hashlib
        insight_id = f"ins-{hashlib.sha256(moment_id.encode()).hexdigest()[:12]}"
        difficulty = max(0.1, 1.0 - confidence)

        # First review: tomorrow
        next_review = datetime.now(timezone.utc) + timedelta(days=1)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO insight_schedule
                   (insight_id, moment_id, difficulty, next_review)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (insight_id) DO NOTHING""",
                insight_id, moment_id, difficulty, next_review,
            )

    async def get_due_insights(self, limit: int = 10) -> List[Dict]:
        """Get insights that are due for review."""
        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT ins.*, cm.description, cm.coherence_type,
                          cm.platforms, cm.confidence as moment_confidence
                   FROM insight_schedule ins
                   JOIN coherence_moments cm ON ins.moment_id = cm.moment_id
                   WHERE ins.next_review <= NOW()
                   ORDER BY ins.next_review ASC
                   LIMIT $1""",
                limit,
            )

        return [dict(r) for r in rows]

    async def record_review(self, insight_id: str, rating: int):
        """
        Record a user's review of an insight.

        Rating: 1=forgot, 2=hard, 3=good, 4=easy
        """
        if not self._pool or rating not in (1, 2, 3, 4):
            return

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM insight_schedule WHERE insight_id = $1",
                insight_id,
            )
            if not row:
                return

            stability = row["stability"]
            difficulty = row["difficulty"]
            review_count = row["review_count"]

            new_stability, new_difficulty, interval_days = schedule_next_review(
                stability, difficulty, rating, review_count,
            )

            now = datetime.now(timezone.utc)
            next_review = now + timedelta(days=interval_days)
            new_retrievability = 1.0  # Just reviewed

            # Append to rating history
            import json
            history = json.loads(row["rating_history"] or "[]")
            history.append({"rating": rating, "timestamp": now.isoformat()})

            await conn.execute(
                """UPDATE insight_schedule SET
                       stability = $2,
                       difficulty = $3,
                       retrievability = $4,
                       last_review = $5,
                       next_review = $6,
                       review_count = review_count + 1,
                       rating_history = $7::jsonb
                   WHERE insight_id = $1""",
                insight_id, new_stability, new_difficulty,
                new_retrievability, now, next_review, json.dumps(history),
            )

            log.info(
                f"Review recorded: {insight_id} | rating={rating} "
                f"| stability={new_stability:.1f}d | next={next_review.strftime('%Y-%m-%d')}"
            )

    async def schedule_new_moments(self, min_confidence: float = 0.82):
        """
        Auto-schedule unscheduled high-confidence coherence moments.

        Called by the consolidation daemon.
        """
        if not self._pool:
            return 0

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT cm.moment_id, cm.confidence
                   FROM coherence_moments cm
                   WHERE cm.confidence >= $1
                     AND NOT EXISTS (
                         SELECT 1 FROM insight_schedule ins
                         WHERE ins.moment_id = cm.moment_id
                     )
                   ORDER BY cm.confidence DESC
                   LIMIT 50""",
                min_confidence,
            )

        scheduled = 0
        for row in rows:
            await self.schedule_moment(row["moment_id"], row["confidence"])
            scheduled += 1

        if scheduled:
            log.info(f"Scheduled {scheduled} new insights for review")
        return scheduled
