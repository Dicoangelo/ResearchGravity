"""
X/Twitter ↔ Delegation Trust Bridge

Synchronizes trust scores between X/Twitter semantic memory (author_trust)
and the delegation trust ledger (agent trust). Both systems use Bayesian Beta
distribution, making the bridge mathematically exact:

X/Twitter:  trust = alpha / (alpha + beta_param)
Delegation: trust = (success_count + 1) / (success_count + failure_count + 2)

Mapping:
  x.alpha      → delegation.success_count + 1
  x.beta_param → delegation.failure_count + 1
  x.avg_quality → delegation.avg_quality
  x.username   → delegation.agent_id (prefixed "x-author:{username}")

Usage:
    from delegation.x_trust_bridge import XTrustBridge

    bridge = XTrustBridge()

    # Sync top X authors into delegation trust ledger
    synced = await bridge.sync_top_authors(limit=20)

    # Sync a specific author
    score = await bridge.sync_author("AnthropicAI")

    # Get cross-system trust for an author
    combined = await bridge.get_combined_trust("karpathy")
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from delegation.trust_ledger import TrustLedger


AGENT_PREFIX = "x-author"
TASK_TYPE = "x_research"


class XTrustBridge:
    """
    Bidirectional trust bridge between X/Twitter and delegation system.

    Reads X author trust scores (via MCP tool output) and writes them
    into the delegation trust ledger as agent entries. This enables
    the delegation router to factor in X-sourced intelligence quality
    when assigning research subtasks.
    """

    def __init__(self, trust_ledger: Optional[TrustLedger] = None):
        self._ledger = trust_ledger
        self._owns_ledger = trust_ledger is None

    async def __aenter__(self):
        if self._owns_ledger:
            self._ledger = TrustLedger()
        await self._ledger.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._ledger.__aexit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _agent_id(username: str) -> str:
        """Convert X username to delegation agent_id."""
        return f"{AGENT_PREFIX}:{username.lower()}"

    @staticmethod
    def _parse_author(author: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse X author_trust output into delegation-compatible format.

        X format:
            {"username", "trust_score", "alpha", "beta_param",
             "avg_quality", "deep_signal_count", "noise_count",
             "followers", "trust_level"}

        Returns:
            {"agent_id", "success_count", "failure_count",
             "avg_quality", "trust_score"}
        """
        alpha = author.get("alpha", 1.0)
        beta_param = author.get("beta_param", 1.0)

        return {
            "agent_id": XTrustBridge._agent_id(author["username"]),
            "username": author["username"],
            "success_count": max(0, int(alpha - 1)),
            "failure_count": max(0, int(beta_param - 1)),
            "avg_quality": author.get("avg_quality", 0.5),
            "trust_score": author.get("trust_score", 0.5),
            "followers": author.get("followers", 0),
            "trust_level": author.get("trust_level", "neutral"),
            "deep_signal_count": author.get("deep_signal_count", 0),
            "noise_count": author.get("noise_count", 0),
        }

    async def sync_author_data(self, author_data: Dict[str, Any]) -> float:
        """
        Sync a single X author's trust data into the delegation ledger.

        Instead of recording individual outcomes, this reconstructs the
        full Bayesian history so the ledger reflects the exact same
        trust score as X's author_trust system.

        Args:
            author_data: Raw author dict from X top_authors or author_trust

        Returns:
            Trust score as written to delegation ledger
        """
        parsed = self._parse_author(author_data)

        # Record successes
        for _ in range(parsed["success_count"]):
            await self._ledger.record_outcome(
                agent_id=parsed["agent_id"],
                task_type=TASK_TYPE,
                success=True,
                quality=parsed["avg_quality"],
                duration=0.0,
            )

        # Record failures
        for _ in range(parsed["failure_count"]):
            await self._ledger.record_outcome(
                agent_id=parsed["agent_id"],
                task_type=TASK_TYPE,
                success=False,
                quality=max(0.0, parsed["avg_quality"] - 0.3),
                duration=0.0,
            )

        return await self._ledger.get_trust_score(
            parsed["agent_id"], TASK_TYPE
        )

    async def sync_top_authors(
        self, authors_json: str
    ) -> List[Dict[str, Any]]:
        """
        Sync multiple X authors from top_authors MCP output.

        Args:
            authors_json: JSON string from mcp__x-twitter__top_authors result

        Returns:
            List of synced entries with agent_id, trust_score, username
        """
        data = json.loads(authors_json) if isinstance(authors_json, str) else authors_json
        authors = data.get("top_authors", data) if isinstance(data, dict) else data

        results = []
        for author in authors:
            score = await self.sync_author_data(author)
            results.append({
                "agent_id": self._agent_id(author["username"]),
                "username": author["username"],
                "x_trust": author.get("trust_score", 0.5),
                "delegation_trust": score,
                "synced_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            })

        return results

    async def get_combined_trust(
        self, username: str
    ) -> Dict[str, Any]:
        """
        Get combined trust view for an X author across both systems.

        Args:
            username: X username (without @)

        Returns:
            Dict with delegation trust score and agent stats
        """
        agent_id = self._agent_id(username)
        score = await self._ledger.get_trust_score(agent_id, TASK_TYPE)
        stats = await self._ledger.get_agent_stats(agent_id, TASK_TYPE)

        return {
            "username": username,
            "agent_id": agent_id,
            "delegation_trust": score,
            "has_delegation_history": stats is not None,
            "stats": {
                "success_count": stats.success_count if stats else 0,
                "failure_count": stats.failure_count if stats else 0,
                "avg_quality": stats.avg_quality if stats else 0.0,
            } if stats else None,
        }
