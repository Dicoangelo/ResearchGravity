#!/usr/bin/env python3
"""
Pack Selector - Intelligent pack selection using DQ scoring and ACE consensus.

Implements multi-layer intelligence:
- Layer 1: DQ Scoring (Validity + Specificity + Correctness)
- Layer 2: ACE Consensus (Multi-agent voting)
- Layer 3: Optimization (Token budget management)

Usage:
  python3 select_packs.py --context "debugging React performance"
  python3 select_packs.py --context "multi-agent orchestration" --budget 50000
  python3 select_packs.py --auto  # Auto-detect from current directory
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


AGENT_CORE = Path.home() / ".agent-core"
PACK_DIR = AGENT_CORE / "context-packs"


class DQScorer:
    """DQ Scoring: Validity (40%) + Specificity (30%) + Correctness (30%)"""

    def score(self, pack: Dict, context: str) -> float:
        """Calculate DQ score for pack given context"""

        validity = self._score_validity(pack, context)
        specificity = self._score_specificity(pack, context)
        correctness = self._score_correctness(pack)

        dq_score = (validity * 0.4) + (specificity * 0.3) + (correctness * 0.3)

        return dq_score

    def _score_validity(self, pack: Dict, context: str) -> float:
        """How relevant is this pack to the context?"""

        # Keyword matching
        context_lower = context.lower()
        keywords = pack['content'].get('keywords', [])

        matches = sum(1 for kw in keywords if kw.lower() in context_lower)
        keyword_score = min(matches / len(keywords) if keywords else 0, 1.0)

        # Paper relevance (if papers mentioned in context)
        papers = pack['content'].get('papers', [])
        paper_ids = [p.get('arxiv_id', '') for p in papers]
        paper_matches = sum(1 for pid in paper_ids if pid in context)
        paper_score = min(paper_matches / len(papers) if papers else 0, 1.0)

        # Base metadata score
        base_validity = pack.get('dq_metadata', {}).get('base_validity', 0.5)

        # Weighted average
        validity = (
            keyword_score * 0.5 +
            paper_score * 0.2 +
            base_validity * 0.3
        )

        return validity

    def _score_specificity(self, pack: Dict, context: str) -> float:
        """How targeted is this pack (vs generic)?"""

        keywords = pack['content'].get('keywords', [])

        # More keywords = more specific
        specificity_from_keywords = min(len(keywords) / 10, 1.0)

        # More papers = more specific
        papers = pack['content'].get('papers', [])
        specificity_from_papers = min(len(papers) / 5, 1.0)

        # Pack type specificity
        pack_type = pack.get('type', 'domain')
        type_specificity = {
            'project': 0.9,  # Very specific
            'pattern': 0.8,
            'domain': 0.7,
            'paper': 0.6
        }.get(pack_type, 0.5)

        # Base metadata score
        base_specificity = pack.get('dq_metadata', {}).get('base_specificity', 0.5)

        # Weighted average
        specificity = (
            specificity_from_keywords * 0.3 +
            specificity_from_papers * 0.2 +
            type_specificity * 0.2 +
            base_specificity * 0.3
        )

        return specificity

    def _score_correctness(self, pack: Dict) -> float:
        """How up-to-date and accurate is this pack?"""

        # Recency score
        created = datetime.fromisoformat(pack['created'].replace('Z', '+00:00'))
        age_days = (datetime.now(created.tzinfo) - created).days
        recency_score = max(1.0 - (age_days / 365), 0.3)  # Decay over a year

        # Usage-based correctness
        usage_stats = pack.get('usage_stats', {})
        times_selected = usage_stats.get('times_selected', 0)
        avg_relevance = usage_stats.get('avg_session_relevance', 0.0)

        # If pack has been used, factor in feedback
        if times_selected > 0:
            usage_score = min(avg_relevance, 1.0)
        else:
            usage_score = pack.get('dq_metadata', {}).get('base_correctness', 0.5)

        # Weighted average
        correctness = (recency_score * 0.4) + (usage_score * 0.6)

        return correctness


class ACEConsensus:
    """ACE: Adaptive Consensus Engine with multi-agent voting"""

    def __init__(self):
        self.agents = {
            'relevance_agent': self._relevance_agent,
            'cost_agent': self._cost_agent,
            'recency_agent': self._recency_agent,
            'pattern_agent': self._pattern_agent,
            'quality_agent': self._quality_agent
        }

    def select(
        self,
        scored_packs: Dict[str, float],
        context: str,
        pack_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Multi-agent consensus on pack selection"""

        # Each agent votes
        votes = {}
        for agent_name, agent_func in self.agents.items():
            votes[agent_name] = agent_func(scored_packs, context, pack_data)

        # Adaptive weights based on context
        weights = self._adaptive_weights(context)

        # Weighted consensus
        consensus_scores = {}
        for pack_id in scored_packs.keys():
            weighted_score = sum(
                votes[agent][pack_id] * weights[agent]
                for agent in self.agents.keys()
            )
            consensus_scores[pack_id] = weighted_score

        return consensus_scores

    def _relevance_agent(
        self,
        scored_packs: Dict[str, float],
        context: str,
        pack_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Agent focused on semantic relevance"""
        # Use DQ scores as base
        return scored_packs.copy()

    def _cost_agent(
        self,
        scored_packs: Dict[str, float],
        context: str,
        pack_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Agent focused on token efficiency"""
        scores = {}
        for pack_id, dq_score in scored_packs.items():
            pack = pack_data[pack_id]
            size = pack.get('size_tokens', 1000)

            # Prefer smaller packs for efficiency
            efficiency = 1.0 / (1.0 + (size / 1000))

            # Balance: relevance vs cost
            scores[pack_id] = dq_score * 0.6 + efficiency * 0.4

        return scores

    def _recency_agent(
        self,
        scored_packs: Dict[str, float],
        context: str,
        pack_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Agent focused on freshness"""
        scores = {}
        for pack_id, dq_score in scored_packs.items():
            pack = pack_data[pack_id]
            created = datetime.fromisoformat(pack['created'].replace('Z', '+00:00'))
            age_days = (datetime.now(created.tzinfo) - created).days

            # Prefer recent packs
            recency = max(1.0 - (age_days / 180), 0.2)  # 6 month decay

            scores[pack_id] = dq_score * 0.5 + recency * 0.5

        return scores

    def _pattern_agent(
        self,
        scored_packs: Dict[str, float],
        context: str,
        pack_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Agent focused on workflow patterns"""
        scores = {}

        # Detect patterns in context
        is_debugging = any(w in context.lower() for w in ['debug', 'error', 'bug', 'fix'])
        is_architecture = any(w in context.lower() for w in ['architecture', 'design', 'system'])
        is_optimization = any(w in context.lower() for w in ['optimize', 'performance', 'speed'])

        for pack_id, dq_score in scored_packs.items():
            pack = pack_data[pack_id]
            pack_type = pack.get('type', '')

            # Boost pattern-matching packs
            boost = 1.0
            if is_debugging and pack_type == 'pattern':
                boost = 1.3
            elif is_architecture and pack_type == 'domain':
                boost = 1.2
            elif is_optimization and 'optimization' in pack_id:
                boost = 1.25

            scores[pack_id] = min(dq_score * boost, 1.0)

        return scores

    def _quality_agent(
        self,
        scored_packs: Dict[str, float],
        context: str,
        pack_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Agent focused on proven quality"""
        scores = {}
        for pack_id, dq_score in scored_packs.items():
            pack = pack_data[pack_id]
            usage_stats = pack.get('usage_stats', {})

            times_selected = usage_stats.get('times_selected', 0)
            avg_relevance = usage_stats.get('avg_session_relevance', 0.5)

            # Boost packs with good track record
            if times_selected > 5:
                quality_boost = avg_relevance * 0.2
            else:
                quality_boost = 0

            scores[pack_id] = min(dq_score + quality_boost, 1.0)

        return scores

    def _adaptive_weights(self, context: str) -> Dict[str, float]:
        """Adaptive agent weights based on context"""

        # Default weights
        weights = {
            'relevance_agent': 0.3,
            'cost_agent': 0.2,
            'recency_agent': 0.15,
            'pattern_agent': 0.2,
            'quality_agent': 0.15
        }

        # Adjust based on context signals
        context_lower = context.lower()

        # If context mentions "latest" or "new", boost recency
        if any(w in context_lower for w in ['latest', 'new', 'recent', '2026']):
            weights['recency_agent'] += 0.1
            weights['quality_agent'] -= 0.1

        # If context is about optimization/cost, boost cost agent
        if any(w in context_lower for w in ['optimize', 'cost', 'efficient', 'token']):
            weights['cost_agent'] += 0.1
            weights['relevance_agent'] -= 0.1

        # Normalize to sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        return weights


class PackSelector:
    """Main pack selection orchestrator"""

    def __init__(self):
        self.pack_dir = PACK_DIR
        self.registry_path = self.pack_dir / "registry.json"
        self.dq_scorer = DQScorer()
        self.ace_consensus = ACEConsensus()

        self.registry = self._load_registry()
        self.pack_cache = {}

    def _load_registry(self) -> Dict:
        """Load pack registry"""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"packs": {}}

    def _load_pack(self, pack_id: str) -> Optional[Dict]:
        """Load full pack data"""
        if pack_id in self.pack_cache:
            return self.pack_cache[pack_id]

        pack_info = self.registry['packs'].get(pack_id)
        if not pack_info:
            return None

        pack_file = self.pack_dir / pack_info['file']
        if not pack_file.exists():
            return None

        with open(pack_file) as f:
            pack_data = json.load(f)

        self.pack_cache[pack_id] = pack_data
        return pack_data

    def select_packs(
        self,
        context: str,
        token_budget: int = 50000,
        min_packs: int = 1,
        max_packs: int = 5
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select optimal packs using DQ + ACE

        Returns:
            (selected_pack_ids, metadata)
        """

        # Load all packs
        all_packs = {}
        for pack_id in self.registry['packs'].keys():
            pack = self._load_pack(pack_id)
            if pack:
                all_packs[pack_id] = pack

        if not all_packs:
            return [], {"error": "No packs available"}

        # Phase 1: DQ Scoring
        dq_scores = {}
        for pack_id, pack in all_packs.items():
            dq_scores[pack_id] = self.dq_scorer.score(pack, context)

        # Phase 2: ACE Consensus
        consensus_scores = self.ace_consensus.select(dq_scores, context, all_packs)

        # Phase 3: Greedy knapsack selection within budget
        selected = []
        tokens_used = 0

        # Sort by consensus score
        sorted_packs = sorted(
            consensus_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for pack_id, score in sorted_packs:
            pack = all_packs[pack_id]
            pack_size = pack['size_tokens']

            # Check budget
            if tokens_used + pack_size <= token_budget:
                selected.append(pack_id)
                tokens_used += pack_size

                # Check max packs
                if len(selected) >= max_packs:
                    break

        # Ensure minimum packs
        if len(selected) < min_packs and sorted_packs:
            # Force add highest scoring packs even if over budget
            for pack_id, score in sorted_packs[:min_packs]:
                if pack_id not in selected:
                    selected.append(pack_id)
                    tokens_used += all_packs[pack_id]['size_tokens']

        # Metadata
        metadata = {
            "context": context,
            "token_budget": token_budget,
            "tokens_used": tokens_used,
            "tokens_saved": token_budget - tokens_used,
            "num_packs": len(selected),
            "dq_scores": {pid: dq_scores.get(pid, 0) for pid in selected},
            "consensus_scores": {pid: consensus_scores.get(pid, 0) for pid in selected},
            "selection_time_ms": 0,  # TODO: measure
            "ace_weights": self.ace_consensus._adaptive_weights(context)
        }

        return selected, metadata

    def auto_detect_context(self) -> str:
        """Auto-detect context from current directory and recent git history"""

        cwd = Path.cwd()
        context_parts = []

        # Project detection
        if "OS-App" in str(cwd):
            context_parts.append("OS-App React Vite agentic kernel")
        elif "CareerCoach" in str(cwd):
            context_parts.append("CareerCoach Next.js AI agents")
        elif "researchgravity" in str(cwd):
            context_parts.append("research tracking Python")

        # Try to read recent git log
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'log', '-5', '--oneline'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                context_parts.append(result.stdout)
        except:
            pass

        # Check for error patterns in git status
        try:
            result = subprocess.run(
                ['git', 'diff', '--stat'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                context_parts.append(result.stdout[:200])
        except:
            pass

        return " ".join(context_parts)

    def format_output(
        self,
        selected_packs: List[str],
        metadata: Dict[str, Any],
        format_type: str = "markdown"
    ) -> str:
        """Format selection output"""

        if format_type == "json":
            return json.dumps({
                "selected_packs": selected_packs,
                "metadata": metadata
            }, indent=2)

        # Markdown format
        lines = []
        lines.append("# Pack Selection Results\n")
        lines.append(f"**Context:** {metadata['context']}\n")
        lines.append(f"**Selected:** {metadata['num_packs']} packs ({metadata['tokens_used']} tokens)\n")
        lines.append(f"**Budget:** {metadata['token_budget']} tokens\n")
        lines.append(f"**Saved:** {metadata['tokens_saved']} tokens\n")
        lines.append("\n## Selected Packs\n")

        for pack_id in selected_packs:
            dq = metadata['dq_scores'].get(pack_id, 0)
            consensus = metadata['consensus_scores'].get(pack_id, 0)
            pack = self._load_pack(pack_id)

            lines.append(f"### {pack_id}")
            lines.append(f"- Type: {pack['type']}")
            lines.append(f"- Size: {pack['size_tokens']} tokens")
            lines.append(f"- DQ Score: {dq:.3f}")
            lines.append(f"- Consensus Score: {consensus:.3f}")
            lines.append(f"- Keywords: {', '.join(pack['content'].get('keywords', []))}\n")

        lines.append("\n## ACE Agent Weights\n")
        for agent, weight in metadata['ace_weights'].items():
            lines.append(f"- {agent}: {weight:.2f}")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Select optimal context packs using DQ + ACE"
    )

    parser.add_argument(
        '--context',
        help="Context description for pack selection"
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help="Auto-detect context from current directory"
    )

    parser.add_argument(
        '--budget',
        type=int,
        default=50000,
        help="Token budget (default: 50000)"
    )

    parser.add_argument(
        '--min-packs',
        type=int,
        default=1,
        help="Minimum packs to select"
    )

    parser.add_argument(
        '--max-packs',
        type=int,
        default=5,
        help="Maximum packs to select"
    )

    parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help="Output format"
    )

    args = parser.parse_args()

    selector = PackSelector()

    # Determine context
    if args.auto:
        context = selector.auto_detect_context()
        print(f"Auto-detected context: {context}\n")
    elif args.context:
        context = args.context
    else:
        print("Error: Provide --context or use --auto")
        return

    # Select packs
    selected, metadata = selector.select_packs(
        context=context,
        token_budget=args.budget,
        min_packs=args.min_packs,
        max_packs=args.max_packs
    )

    # Output
    output = selector.format_output(selected, metadata, format_type=args.format)
    print(output)


if __name__ == '__main__':
    main()
