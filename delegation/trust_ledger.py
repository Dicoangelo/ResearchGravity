"""
Trust Ledger — Agent Trust Score Tracking with Bayesian Updates

Implements the trust system from arXiv:2602.11865 Section 4.6.

Uses Bayesian inference with Beta distribution for trust score updates:
- Prior: Beta(α=1, β=1) for new agents → E[Beta] = 0.5 (uninformative prior)
- Update: α = successes + 1, β = failures + 1
- Trust score: E[Beta] = α/(α+β)

Mathematical elegance: The Beta distribution is the conjugate prior for the
Bernoulli distribution, making Bayesian updates analytically tractable.

Examples:
- New agent: Beta(1,1) → trust = 0.5 (maximum uncertainty)
- After 10 successes, 0 failures: Beta(11,1) → trust = 0.917
- After 5 successes, 5 failures: Beta(6,6) → trust = 0.5
- After 8 successes, 2 failures: Beta(9,3) → trust = 0.75

Trust scores range [0.0, 1.0]:
- 0.0-0.3: Low trust (needs supervision)
- 0.3-0.7: Medium trust (standard delegation)
- 0.7-1.0: High trust (autonomous operation)

Decay function: trust_score *= 0.95 for entries not updated in 7+ days.

Usage:
    from delegation.trust_ledger import TrustLedger

    async with TrustLedger() as ledger:
        await ledger.record_outcome(
            agent_id="agent-1",
            task_type="code_generation",
            success=True,
            quality=0.9,
            duration=120.5
        )
        score = await ledger.get_trust_score("agent-1", "code_generation")
        top_agents = await ledger.get_top_agents("code_generation", limit=5)
"""

import aiosqlite
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AgentTrustScore:
    """Agent trust score with Bayesian statistics"""
    agent_id: str
    task_type: str
    success_count: int
    failure_count: int
    avg_quality: float
    avg_duration: float
    trust_score: float
    last_updated: str


class TrustLedger:
    """
    Persistent trust ledger with Bayesian trust score updates.

    Stores agent performance per task type using Beta distribution for trust.
    Database stored at ~/.agent-core/storage/trust_ledger.db

    Trust calculation (Bayesian):
    - α (alpha) = success_count + 1
    - β (beta) = failure_count + 1
    - trust_score = α / (α + β)

    Decay: trust_score *= 0.95 for entries not updated in 7+ days (applied on query)
    """

    DECAY_DAYS = 7
    DECAY_FACTOR = 0.95
    DB_PATH = Path.home() / ".agent-core" / "storage" / "trust_ledger.db"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize trust ledger.

        Args:
            db_path: Optional custom path to SQLite database
        """
        self.db_path = Path(db_path) if db_path else self.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: Optional[aiosqlite.Connection] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_db()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _init_db(self):
        """Initialize database and create schema"""
        self._db = await aiosqlite.connect(str(self.db_path))

        # Create schema
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trust_entries (
                agent_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_quality REAL DEFAULT 0.0,
                avg_duration REAL DEFAULT 0.0,
                trust_score REAL DEFAULT 0.5,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (agent_id, task_type),
                CHECK (success_count >= 0),
                CHECK (failure_count >= 0),
                CHECK (avg_quality BETWEEN 0.0 AND 1.0),
                CHECK (avg_duration >= 0.0),
                CHECK (trust_score BETWEEN 0.0 AND 1.0)
            )
        """)

        # Create indexes for fast queries
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_trust_task_type
            ON trust_entries(task_type, trust_score DESC)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_trust_agent
            ON trust_entries(agent_id)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_trust_updated
            ON trust_entries(last_updated)
        """)

        await self._db.commit()

    async def close(self):
        """Close database connection"""
        if self._db:
            await self._db.close()
            self._db = None

    async def record_outcome(
        self,
        agent_id: str,
        task_type: str,
        success: bool,
        quality: float,
        duration: float
    ) -> float:
        """
        Record task outcome and update trust score using Bayesian inference.

        Args:
            agent_id: Agent identifier
            task_type: Task type for capability-specific trust
            success: Whether task completed successfully
            quality: Quality score [0.0, 1.0]
            duration: Task duration in seconds

        Returns:
            Updated trust score [0.0, 1.0]
        """
        if not 0.0 <= quality <= 1.0:
            raise ValueError(f"quality must be in [0.0, 1.0], got {quality}")
        if duration < 0.0:
            raise ValueError(f"duration must be >= 0.0, got {duration}")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # Get current entry or initialize with uninformative prior
        cursor = await self._db.execute(
            "SELECT success_count, failure_count, avg_quality, avg_duration FROM trust_entries WHERE agent_id = ? AND task_type = ?",
            (agent_id, task_type)
        )
        row = await cursor.fetchone()

        if row:
            # Update existing entry
            success_count, failure_count, avg_quality, avg_duration = row
            total_tasks = success_count + failure_count

            # Update counts
            if success:
                success_count += 1
            else:
                failure_count += 1

            # Update running averages
            new_total = total_tasks + 1
            avg_quality = (avg_quality * total_tasks + quality) / new_total
            avg_duration = (avg_duration * total_tasks + duration) / new_total
        else:
            # New agent: initialize with uninformative prior Beta(1,1)
            success_count = 1 if success else 0
            failure_count = 0 if success else 1
            avg_quality = quality
            avg_duration = duration

        # Bayesian trust score calculation
        # Beta distribution: α = successes + 1, β = failures + 1
        # E[Beta] = α / (α + β)
        alpha = success_count + 1
        beta = failure_count + 1
        trust_score = alpha / (alpha + beta)

        # Upsert the entry
        await self._db.execute("""
            INSERT INTO trust_entries (agent_id, task_type, success_count, failure_count, avg_quality, avg_duration, trust_score, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, task_type) DO UPDATE SET
                success_count = excluded.success_count,
                failure_count = excluded.failure_count,
                avg_quality = excluded.avg_quality,
                avg_duration = excluded.avg_duration,
                trust_score = excluded.trust_score,
                last_updated = excluded.last_updated
        """, (agent_id, task_type, success_count, failure_count, avg_quality, avg_duration, trust_score, timestamp))

        await self._db.commit()

        return trust_score

    async def get_trust_score(self, agent_id: str, task_type: str) -> float:
        """
        Get current trust score for an agent on a specific task type.

        Applies time decay: trust_score *= 0.95 for entries not updated in 7+ days.

        Args:
            agent_id: Agent identifier
            task_type: Task type

        Returns:
            Trust score [0.0, 1.0], or 0.5 for new agents (uninformative prior)
        """
        cursor = await self._db.execute(
            "SELECT trust_score, last_updated FROM trust_entries WHERE agent_id = ? AND task_type = ?",
            (agent_id, task_type)
        )
        row = await cursor.fetchone()

        if not row:
            # New agent: return uninformative prior (maximum uncertainty)
            return 0.5

        trust_score, last_updated = row

        # Apply time decay
        last_updated_time = time.mktime(time.strptime(last_updated, "%Y-%m-%d %H:%M:%S"))
        days_since_update = (time.time() - last_updated_time) / (24 * 3600)

        if days_since_update >= self.DECAY_DAYS:
            # Apply decay: trust_score *= 0.95 for stale entries
            trust_score *= self.DECAY_FACTOR
            trust_score = max(0.0, min(1.0, trust_score))  # Clamp to [0.0, 1.0]

        return trust_score

    async def get_top_agents(self, task_type: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-performing agents for a specific task type.

        Applies time decay to all entries before ranking.

        Args:
            task_type: Task type
            limit: Maximum number of agents to return

        Returns:
            List of (agent_id, trust_score) tuples, sorted by trust_score descending
        """
        cursor = await self._db.execute(
            "SELECT agent_id, trust_score, last_updated FROM trust_entries WHERE task_type = ? ORDER BY trust_score DESC",
            (task_type,)
        )
        rows = await cursor.fetchall()

        # Apply time decay to all entries
        current_time = time.time()
        decayed_scores = []

        for agent_id, trust_score, last_updated in rows:
            last_updated_time = time.mktime(time.strptime(last_updated, "%Y-%m-%d %H:%M:%S"))
            days_since_update = (current_time - last_updated_time) / (24 * 3600)

            if days_since_update >= self.DECAY_DAYS:
                trust_score *= self.DECAY_FACTOR
                trust_score = max(0.0, min(1.0, trust_score))

            decayed_scores.append((agent_id, trust_score))

        # Re-sort after decay and limit results
        decayed_scores.sort(key=lambda x: x[1], reverse=True)
        return decayed_scores[:limit]

    async def get_agent_stats(self, agent_id: str, task_type: str) -> Optional[AgentTrustScore]:
        """
        Get detailed statistics for an agent on a specific task type.

        Args:
            agent_id: Agent identifier
            task_type: Task type

        Returns:
            AgentTrustScore object or None if agent has no history
        """
        cursor = await self._db.execute(
            """SELECT agent_id, task_type, success_count, failure_count, avg_quality, avg_duration, trust_score, last_updated
               FROM trust_entries WHERE agent_id = ? AND task_type = ?""",
            (agent_id, task_type)
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return AgentTrustScore(
            agent_id=row[0],
            task_type=row[1],
            success_count=row[2],
            failure_count=row[3],
            avg_quality=row[4],
            avg_duration=row[5],
            trust_score=row[6],
            last_updated=row[7]
        )
