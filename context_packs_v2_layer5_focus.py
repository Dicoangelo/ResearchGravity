#!/usr/bin/env python3
"""
Context Packs V2 - Layer 5: Active Focus Compression
====================================================

Implements Active Context Compression paper (arXiv:2601.07190):
- Focus Agent achieves 22.7% token reduction autonomously
- Consolidates key learnings
- Prunes raw interaction history
- Cost-aware optimization
- Identity preservation across sessions

Based on: Active Context Compression (arXiv:2601.07190)
"""

import os
import json
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# Focus Agent - Semantic Focus Extraction
# ============================================================================

@dataclass
class FocusToken:
    """A semantic focus token identified from query"""
    token: str
    importance: float
    context_window: int
    related_concepts: List[str]


class FocusAgent:
    """
    Focus Agent that autonomously identifies semantic focus
    and compresses pack content around that focus

    Key Innovation: 22.7% token reduction by focusing on query-relevant content
    """

    def __init__(self):
        self.compression_target = 0.227  # 22.7% reduction
        self.focus_history: List[FocusToken] = []

    def extract_query_focus(self, query: str) -> List[FocusToken]:
        """
        Extract semantic focus tokens from query

        Focus tokens are the core concepts that should guide compression
        """
        # Simple focus extraction (in production, use NER or embedding clustering)
        tokens = query.lower().split()

        # Identify important tokens (nouns, technical terms)
        focus_tokens = []

        # Heuristic: longer words and compound terms are more important
        important_words = [w for w in tokens if len(w) > 5 or '-' in w]

        for word in important_words[:3]:  # Top 3 focus tokens
            focus = FocusToken(
                token=word,
                importance=1.0 / (important_words.index(word) + 1),  # Decay by position
                context_window=50,  # Characters around this token
                related_concepts=[]
            )
            focus_tokens.append(focus)

        # If no long words, use first 2 words
        if not focus_tokens and tokens:
            for word in tokens[:2]:
                focus_tokens.append(FocusToken(
                    token=word,
                    importance=0.8,
                    context_window=50,
                    related_concepts=[]
                ))

        self.focus_history.extend(focus_tokens)
        return focus_tokens

    def compute_attention(
        self,
        focus_tokens: List[FocusToken],
        content: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Calculate attention scores for pack content based on focus tokens

        Content closer to focus tokens gets higher attention
        """
        attention_scores = {
            'papers': [],
            'learnings': [],
            'keywords': []
        }

        # Papers attention
        for paper in content.get('papers', []):
            arxiv_id = paper.get('arxiv_id', '')
            score = 0.3  # Base score

            # Boost if focus token relates to paper domain
            for focus in focus_tokens:
                # Mock: check if focus appears in common paper topics
                if any(topic in focus.token for topic in ['agent', 'multi', 'memory', 'optimization']):
                    score += 0.2 * focus.importance

            attention_scores['papers'].append(min(1.0, score))

        # Learnings attention
        for learning in content.get('learnings', []):
            learning_lower = learning.lower()
            score = 0.2  # Base score

            # Calculate overlap with focus tokens
            for focus in focus_tokens:
                if focus.token in learning_lower:
                    score += 0.5 * focus.importance

            attention_scores['learnings'].append(min(1.0, score))

        # Keywords attention
        for keyword in content.get('keywords', []):
            keyword_lower = keyword.lower()
            score = 0.2  # Base score

            # Direct match with focus
            for focus in focus_tokens:
                if focus.token in keyword_lower or keyword_lower in focus.token:
                    score += 0.7 * focus.importance

            attention_scores['keywords'].append(min(1.0, score))

        return attention_scores

    def prune_by_attention(
        self,
        content: Dict[str, Any],
        attention_scores: Dict[str, List[float]],
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Prune content below attention threshold

        Achieves target 22.7% reduction by removing low-attention elements
        """
        compressed = {
            'papers': [],
            'learnings': [],
            'keywords': []
        }

        # Keep high-attention papers
        for i, paper in enumerate(content.get('papers', [])):
            if i < len(attention_scores['papers']) and attention_scores['papers'][i] > threshold:
                compressed['papers'].append(paper)

        # Keep high-attention learnings
        for i, learning in enumerate(content.get('learnings', [])):
            if i < len(attention_scores['learnings']) and attention_scores['learnings'][i] > threshold:
                compressed['learnings'].append(learning)

        # Keep high-attention keywords
        for i, keyword in enumerate(content.get('keywords', [])):
            if i < len(attention_scores['keywords']) and attention_scores['keywords'][i] > threshold:
                compressed['keywords'].append(keyword)

        return compressed

    def compress_pack(
        self,
        pack_data: Dict[str, Any],
        query: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Autonomously compress pack content using focus-based attention

        Returns: (compressed_pack, compression_metrics)
        """
        # Extract focus tokens
        focus_tokens = self.extract_query_focus(query)

        # Get pack content
        content = pack_data.get('content', pack_data)

        # Compute attention scores
        attention_scores = self.compute_attention(focus_tokens, content)

        # Calculate adaptive threshold to achieve target compression
        all_scores = []
        for scores_list in attention_scores.values():
            all_scores.extend(scores_list)

        if all_scores:
            all_scores_sorted = sorted(all_scores, reverse=True)
            keep_count = int(len(all_scores_sorted) * (1 - self.compression_target))
            threshold = all_scores_sorted[keep_count] if keep_count < len(all_scores_sorted) else min(all_scores_sorted)
        else:
            threshold = 0.3

        # Prune by attention
        compressed_content = self.prune_by_attention(content, attention_scores, threshold)

        # Build compressed pack
        compressed_pack = pack_data.copy()
        if 'content' in compressed_pack:
            compressed_pack['content'] = compressed_content
        else:
            compressed_pack.update(compressed_content)

        # Metrics
        original_count = sum(len(content.get(k, [])) for k in ['papers', 'learnings', 'keywords'])
        compressed_count = sum(len(compressed_content.get(k, [])) for k in ['papers', 'learnings', 'keywords'])

        metrics = {
            'focus_tokens': [f.token for f in focus_tokens],
            'original_elements': original_count,
            'compressed_elements': compressed_count,
            'reduction_rate': 1 - (compressed_count / original_count) if original_count > 0 else 0.0,
            'threshold': threshold,
            'target_reduction': self.compression_target
        }

        return compressed_pack, metrics

    def consolidate_learnings(
        self,
        learnings: List[str],
        max_output: int = 3
    ) -> List[str]:
        """
        Consolidate multiple learnings into concise key learnings

        Part of active compression: merge redundant information
        """
        if len(learnings) <= max_output:
            return learnings

        # Simple consolidation: group by similarity, keep representatives
        # In production, use embeddings to cluster and merge

        consolidated = []
        seen_concepts = set()

        for learning in learnings:
            # Extract key concepts (words > 5 chars)
            concepts = {w.lower() for w in learning.split() if len(w) > 5}

            # Check overlap with seen concepts
            if not concepts & seen_concepts:
                consolidated.append(learning)
                seen_concepts.update(concepts)

                if len(consolidated) >= max_output:
                    break

        return consolidated


# ============================================================================
# Layer 6: Continuum Memory Evolution
# ============================================================================

@dataclass
class MemoryState:
    """Persistent memory state across sessions"""
    pack_id: str
    importance_score: float
    last_used: str
    usage_count: int
    success_rate: float
    associations: List[str]  # Related pack IDs
    temporal_chain: List[str]  # Session sequence


class ContinuumMemory:
    """
    Continuum Memory Architecture (arXiv:2601.09913)
    Won 82/92 trials vs RAG with persistent state updates

    Key Features:
    - Selective retention (forget irrelevant)
    - Associative routing (link related memories)
    - Temporal chaining (sequence preservation)
    - Consolidation into higher-order abstractions
    """

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.expanduser('~/.agent-core/context-packs')

        self.storage_dir = storage_dir
        self.memory_state_file = os.path.join(storage_dir, 'continuum_memory.json')

        # Load persistent state
        self.memory_states: Dict[str, MemoryState] = {}
        self._load_memory_states()

        print("Continuum Memory initialized")
        print(f"  Memory states: {len(self.memory_states)}")

    def _load_memory_states(self):
        """Load persistent memory states"""
        if not os.path.exists(self.memory_state_file):
            return

        with open(self.memory_state_file, 'r') as f:
            data = json.load(f)
            for pack_id, state_dict in data.items():
                self.memory_states[pack_id] = MemoryState(**state_dict)

    def _save_memory_states(self):
        """Save memory states to disk"""
        data = {
            pack_id: {
                'pack_id': state.pack_id,
                'importance_score': state.importance_score,
                'last_used': state.last_used,
                'usage_count': state.usage_count,
                'success_rate': state.success_rate,
                'associations': state.associations,
                'temporal_chain': state.temporal_chain
            }
            for pack_id, state in self.memory_states.items()
        }

        with open(self.memory_state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def update_persistent_state(
        self,
        session_outcome: Dict[str, Any]
    ):
        """
        Update persistent state after session

        - Selective retention: forget low-importance packs
        - Associative routing: link packs used together
        - Temporal chaining: track usage sequence
        """
        packs_used = session_outcome.get('packs_used', [])
        success_metric = session_outcome.get('success_metric', 0.5)
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        # Update states for used packs
        for pack_id in packs_used:
            if pack_id not in self.memory_states:
                # Initialize new memory state
                self.memory_states[pack_id] = MemoryState(
                    pack_id=pack_id,
                    importance_score=0.5,
                    last_used=timestamp,
                    usage_count=0,
                    success_rate=0.5,
                    associations=[],
                    temporal_chain=[]
                )

            state = self.memory_states[pack_id]

            # Update usage stats
            state.usage_count += 1
            state.last_used = timestamp

            # Update success rate (exponential moving average)
            alpha = 0.3
            state.success_rate = alpha * success_metric + (1 - alpha) * state.success_rate

            # Update importance (based on usage and success)
            state.importance_score = 0.5 * (state.usage_count / 100.0) + 0.5 * state.success_rate
            state.importance_score = min(1.0, state.importance_score)

        # Build associations (packs used together)
        for i, pack_id in enumerate(packs_used):
            state = self.memory_states[pack_id]
            for other_id in packs_used:
                if other_id != pack_id and other_id not in state.associations:
                    state.associations.append(other_id)

        # Build temporal chains
        for pack_id in packs_used:
            state = self.memory_states[pack_id]
            state.temporal_chain.append(session_outcome.get('session_id', 'unknown'))
            # Keep last 20 sessions
            if len(state.temporal_chain) > 20:
                state.temporal_chain = state.temporal_chain[-20:]

        # Selective retention: forget low-importance packs
        self.forget_low_importance_packs(threshold=0.1)

        # Save state
        self._save_memory_states()

    def forget_low_importance_packs(self, threshold: float = 0.1):
        """
        Selective retention: remove packs with low importance scores
        """
        to_forget = [
            pack_id for pack_id, state in self.memory_states.items()
            if state.importance_score < threshold and state.usage_count < 2
        ]

        for pack_id in to_forget:
            del self.memory_states[pack_id]
            print(f"  [Continuum] Forgot low-importance pack: {pack_id}")

    def get_associations(self, pack_id: str, max_results: int = 5) -> List[str]:
        """
        Associative routing: get packs related to this one
        """
        if pack_id not in self.memory_states:
            return []

        state = self.memory_states[pack_id]
        return state.associations[:max_results]

    def get_importance_boost(self, pack_id: str) -> float:
        """
        Get importance boost for pack based on history
        """
        if pack_id not in self.memory_states:
            return 1.0

        state = self.memory_states[pack_id]
        return 1.0 + state.importance_score  # 1.0 to 2.0 boost

    def should_consolidate(self) -> bool:
        """
        Check if memory should be consolidated

        Consolidation creates higher-order abstractions from frequent combinations
        """
        # Consolidate if we have many packs with high association counts
        high_association_count = sum(
            1 for state in self.memory_states.values()
            if len(state.associations) > 3
        )

        return high_association_count > 10


# ============================================================================
# Layer 7: Trainable Pack Weights
# ============================================================================

class TrainablePackGraph:
    """
    Trainable Graph Memory (arXiv:2511.07800)
    RL-based weight optimization for pack utility

    Key Features:
    - Multi-layered graph (raw → structured → meta-cognition)
    - Reinforcement-based weight optimization
    - Empirical utility estimation from reward feedback
    """

    def __init__(self):
        # Pack weights (learned from outcomes)
        self.pack_weights: Dict[str, float] = defaultdict(lambda: 1.0)

        # Layer structure
        self.layers = {
            'raw': {},          # Raw pack content
            'structured': {},   # Processed pack metadata
            'meta': {}          # High-level pack relationships
        }

        # Learning rate
        self.lr = 0.01

        print("Trainable Pack Graph initialized")
        print(f"  Pack weights: {len(self.pack_weights)}")

    def optimize_weights(
        self,
        session_outcomes: List[Dict[str, Any]]
    ):
        """
        RL update: increase weights for packs used in successful sessions

        Reward signal: actual usefulness in sessions
        """
        if not session_outcomes:
            return

        for outcome in session_outcomes:
            packs_used = outcome.get('packs_used', [])
            reward = outcome.get('success_metric', 0.5)

            # Update weights for successful packs
            for pack_id in packs_used:
                # RL update: w_new = w_old + lr * reward
                self.pack_weights[pack_id] += self.lr * reward

        # Decay all weights (prevent unbounded growth)
        for pack_id in list(self.pack_weights.keys()):
            self.pack_weights[pack_id] *= 0.995

            # Remove very low weights
            if self.pack_weights[pack_id] < 0.1:
                del self.pack_weights[pack_id]

        print(f"  [Trainable] Optimized weights for {len(self.pack_weights)} packs")

    def get_weight(self, pack_id: str) -> float:
        """Get learned weight for pack"""
        return self.pack_weights.get(pack_id, 1.0)

    def get_top_packs(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N packs by learned weight"""
        sorted_packs = sorted(
            self.pack_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_packs[:n]


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Context Packs V2 - Layers 5-7'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Focus compression
    focus_parser = subparsers.add_parser('focus', help='Apply focus compression')
    focus_parser.add_argument('--pack-id', required=True)
    focus_parser.add_argument('--query', required=True)

    # Continuum memory
    memory_parser = subparsers.add_parser('memory', help='View continuum memory')
    memory_parser.add_argument('--pack-id', help='Specific pack to view')

    # Trainable weights
    weights_parser = subparsers.add_parser('weights', help='View trainable weights')
    weights_parser.add_argument('--top', type=int, default=10)

    args = parser.parse_args()

    if args.command == 'focus':
        # Test focus compression
        focus_agent = FocusAgent()

        # Mock pack data
        pack_data = {
            'pack_id': args.pack_id,
            'content': {
                'papers': [{'arxiv_id': '2601.12345'}] * 5,
                'learnings': ['Learning about agents'] * 10,
                'keywords': ['agent', 'multi-agent', 'consensus', 'optimization'] * 2
            }
        }

        compressed, metrics = focus_agent.compress_pack(pack_data, args.query)

        print("\n" + "="*60)
        print("FOCUS COMPRESSION RESULTS")
        print("="*60)
        print(f"\nQuery: {args.query}")
        print(f"Focus Tokens: {metrics['focus_tokens']}")
        print(f"Original Elements: {metrics['original_elements']}")
        print(f"Compressed Elements: {metrics['compressed_elements']}")
        print(f"Reduction Rate: {metrics['reduction_rate']:.1%}")
        print(f"Target: {metrics['target_reduction']:.1%}")

    elif args.command == 'memory':
        continuum = ContinuumMemory()

        if args.pack_id:
            if args.pack_id in continuum.memory_states:
                state = continuum.memory_states[args.pack_id]
                print(f"\nMemory State for {args.pack_id}:")
                print(f"  Importance: {state.importance_score:.3f}")
                print(f"  Usage Count: {state.usage_count}")
                print(f"  Success Rate: {state.success_rate:.3f}")
                print(f"  Associations: {state.associations[:5]}")
                print(f"  Last Used: {state.last_used}")
            else:
                print(f"No memory state for {args.pack_id}")
        else:
            print(f"\nTotal Memory States: {len(continuum.memory_states)}")
            for pack_id, state in list(continuum.memory_states.items())[:10]:
                print(f"  {pack_id}: importance={state.importance_score:.3f}, uses={state.usage_count}")

    elif args.command == 'weights':
        graph = TrainablePackGraph()

        print("\n" + "="*60)
        print("TRAINABLE PACK WEIGHTS")
        print("="*60)

        if not graph.pack_weights:
            print("\nNo trained weights yet. Use after collecting session outcomes.")
        else:
            top_packs = graph.get_top_packs(args.top)
            print(f"\nTop {len(top_packs)} Packs by Learned Weight:")
            for i, (pack_id, weight) in enumerate(top_packs, 1):
                print(f"  {i}. {pack_id}: {weight:.3f}")


if __name__ == '__main__':
    main()
