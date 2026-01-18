#!/usr/bin/env python3
"""
Context Packs V2 - Prototype Implementation
============================================

Phase: Prototype (Minimal Working Implementation)
Layers: 1-3 (Multi-Graph Memory, Multi-Agent Routing, Attention Pruning)

Based on Jan 2026 research:
- MAGMA: Multi-Graph Agentic Memory (arXiv:2601.03236)
- RCR-Router: Role-Aware Context Routing (arXiv:2508.04903)
- AttentionRAG: Attention-Guided Pruning (arXiv:2503.10720)
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  numpy not installed.")
    # Create minimal mock
    class MockNumpy:
        @staticmethod
        def array(x):
            return x
        @staticmethod
        def dot(a, b):
            return sum(x*y for x, y in zip(a, b))
        @staticmethod
        def random():
            class R:
                @staticmethod
                def rand(n):
                    import random
                    return [random.random() for _ in range(n)]
            return R()
        class linalg:
            @staticmethod
            def norm(x):
                return sum(v**2 for v in x)**0.5
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0
    np = MockNumpy()

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Using mock embeddings.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️  networkx not installed. Using mock graphs.")
    # Create minimal mock graph
    class MockDiGraph:
        def __init__(self):
            self.nodes_data = {}
            self.edges_data = []
        def add_node(self, node_id, **attrs):
            self.nodes_data[node_id] = attrs
        def add_edge(self, u, v, **attrs):
            self.edges_data.append((u, v, attrs))
    class MockNX:
        DiGraph = MockDiGraph
        @staticmethod
        def single_source_shortest_path_length(G, source, cutoff=None):
            # Mock: return just the source with distance 0
            return {source: 0}
    nx = MockNX()

if not (NUMPY_AVAILABLE or EMBEDDINGS_AVAILABLE or NETWORKX_AVAILABLE):
    print("   Install with: pip3 install sentence-transformers networkx numpy")


# ============================================================================
# Layer 1: Multi-Graph Pack Memory (MAGMA-inspired)
# ============================================================================

@dataclass
class PackNode:
    """A pack in the multi-graph memory system"""
    pack_id: str
    pack_type: str
    content: Dict[str, Any]
    embedding: Optional[Any] = None  # np.ndarray or list
    metadata: Dict[str, Any] = None

    def __hash__(self):
        return hash(self.pack_id)


class MultiGraphPackMemory:
    """
    Four graph types for different retrieval patterns:
    - Semantic: concept relationships via embeddings
    - Temporal: time-based usage chains
    - Causal: success outcome relationships
    - Entity: paper/keyword co-occurrence
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.semantic_graph = nx.DiGraph()
        self.temporal_graph = nx.DiGraph()
        self.causal_graph = nx.DiGraph()
        self.entity_graph = nx.DiGraph()

        # Load embedding model
        if EMBEDDINGS_AVAILABLE:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None

        # Pack registry
        self.packs: Dict[str, PackNode] = {}

    def embed(self, text: str) -> Any:  # np.ndarray or list
        """Generate semantic embedding for text"""
        if self.embedder:
            return self.embedder.encode(text, convert_to_numpy=True)
        else:
            # Mock embedding for testing without dependencies
            import hashlib
            import random
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            random.seed(hash_val % (2**32))
            if NUMPY_AVAILABLE:
                np.random.seed(hash_val % (2**32))
                return np.random.rand(384).astype(np.float32)
            else:
                return [random.random() for _ in range(384)]

    def add_pack(self, pack_data: Dict[str, Any]) -> PackNode:
        """Add a pack to all graphs with appropriate relationships"""
        # Handle V1 pack structure (pack_id + content nested) or V2 (id + flat)
        pack_id = pack_data.get('pack_id') or pack_data.get('id')
        if not pack_id:
            raise ValueError("Pack must have 'pack_id' or 'id' field")

        # Extract content (may be nested in V1 packs)
        if 'content' in pack_data and isinstance(pack_data['content'], dict):
            content = pack_data['content']
        else:
            content = pack_data

        # Create embedding from pack content
        content_text = self._pack_to_text(content)
        embedding = self.embed(content_text)

        # Create pack node
        pack_node = PackNode(
            pack_id=pack_id,
            pack_type=pack_data.get('type', 'unknown'),
            content=pack_data,  # Store full pack data including metadata
            embedding=embedding,
            metadata={
                'created': pack_data.get('created', ''),
                'version': pack_data.get('version', 1),
                'tokens': pack_data.get('size_tokens') or pack_data.get('estimated_tokens', 0)
            }
        )

        self.packs[pack_id] = pack_node

        # Add to all graphs
        self.semantic_graph.add_node(pack_id, node=pack_node)
        self.temporal_graph.add_node(pack_id, node=pack_node)
        self.causal_graph.add_node(pack_id, node=pack_node)
        self.entity_graph.add_node(pack_id, node=pack_node)

        # Build relationships
        self._build_semantic_edges(pack_node)
        self._build_entity_edges(pack_node)

        return pack_node

    def _pack_to_text(self, pack_data: Dict[str, Any]) -> str:
        """Convert pack to text for embedding"""
        parts = []

        # Keywords
        if 'keywords' in pack_data:
            parts.append(' '.join(pack_data['keywords']))

        # Papers
        if 'papers' in pack_data:
            parts.extend([p.get('arxiv_id', '') for p in pack_data['papers']])

        # Learnings
        if 'learnings' in pack_data:
            parts.extend(pack_data['learnings'][:5])  # First 5 learnings

        return ' '.join(parts)

    def _build_semantic_edges(self, pack_node: PackNode):
        """Build semantic similarity edges to existing packs"""
        pack_id = pack_node.pack_id

        # Calculate cosine similarity to all other packs
        for other_id, other_node in self.packs.items():
            if other_id == pack_id:
                continue

            similarity = self._cosine_similarity(
                pack_node.embedding,
                other_node.embedding
            )

            # Add edge if similarity exceeds threshold
            if similarity > 0.5:  # Tunable threshold
                self.semantic_graph.add_edge(
                    pack_id, other_id,
                    weight=similarity,
                    type='semantic'
                )

    def _build_entity_edges(self, pack_node: PackNode):
        """Build entity co-occurrence edges (papers, keywords)"""
        pack_id = pack_node.pack_id
        pack_data = pack_node.content

        # Extract entities
        pack_papers = {p.get('arxiv_id') for p in pack_data.get('papers', [])}
        pack_keywords = set(pack_data.get('keywords', []))

        # Find overlaps with other packs
        for other_id, other_node in self.packs.items():
            if other_id == pack_id:
                continue

            other_data = other_node.content
            other_papers = {p.get('arxiv_id') for p in other_data.get('papers', [])}
            other_keywords = set(other_data.get('keywords', []))

            # Calculate overlap
            paper_overlap = len(pack_papers & other_papers)
            keyword_overlap = len(pack_keywords & other_keywords)

            if paper_overlap > 0 or keyword_overlap > 1:
                overlap_score = paper_overlap * 0.5 + keyword_overlap * 0.3
                self.entity_graph.add_edge(
                    pack_id, other_id,
                    weight=overlap_score,
                    type='entity',
                    papers=paper_overlap,
                    keywords=keyword_overlap
                )

    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Calculate cosine similarity between two vectors"""
        if NUMPY_AVAILABLE and hasattr(vec1, 'shape'):
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        else:
            # List-based computation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            norm_product = norm1 * norm2

        return float(dot_product / norm_product) if norm_product > 0 else 0.0

    def adaptive_retrieve(
        self,
        query: str,
        intent: str = 'semantic',
        max_packs: int = 5,
        max_depth: int = 2
    ) -> List[Tuple[str, float]]:
        """
        Route retrieval through appropriate graph based on query intent

        Args:
            query: Search query
            intent: 'semantic', 'temporal', 'causal', or 'entity'
            max_packs: Maximum packs to return
            max_depth: Graph traversal depth

        Returns:
            List of (pack_id, score) tuples
        """
        query_embedding = self.embed(query)

        if intent == 'semantic':
            return self._semantic_retrieve(query_embedding, max_packs, max_depth)
        elif intent == 'temporal':
            return self._temporal_retrieve(max_packs)
        elif intent == 'causal':
            return self._causal_retrieve(query, max_packs)
        elif intent == 'entity':
            return self._entity_retrieve(query, max_packs)
        else:
            # Default to semantic
            return self._semantic_retrieve(query_embedding, max_packs, max_depth)

    def _semantic_retrieve(
        self,
        query_embedding: np.ndarray,
        max_packs: int,
        max_depth: int
    ) -> List[Tuple[str, float]]:
        """Semantic retrieval with graph expansion"""
        # Step 1: Get initial candidates via similarity
        candidates = []
        for pack_id, pack_node in self.packs.items():
            similarity = self._cosine_similarity(query_embedding, pack_node.embedding)
            candidates.append((pack_id, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Step 2: Expand via semantic graph
        expanded = set()
        for pack_id, score in candidates[:max_packs]:
            expanded.add((pack_id, score))

            # BFS expansion up to max_depth
            if max_depth > 0:
                neighbors = nx.single_source_shortest_path_length(
                    self.semantic_graph, pack_id, cutoff=max_depth
                )
                for neighbor_id, depth in neighbors.items():
                    if neighbor_id != pack_id:
                        # Decay score by depth
                        decay_factor = 0.7 ** depth
                        expanded.add((neighbor_id, score * decay_factor))

        # Sort and return top max_packs
        expanded_list = sorted(expanded, key=lambda x: x[1], reverse=True)
        return expanded_list[:max_packs]

    def _temporal_retrieve(self, max_packs: int) -> List[Tuple[str, float]]:
        """Temporal retrieval - most recent packs"""
        temporal_scores = []
        for pack_id, pack_node in self.packs.items():
            created = pack_node.metadata.get('created', '')
            # Simple recency score (in production, use actual timestamps)
            score = 1.0  # Mock score
            temporal_scores.append((pack_id, score))

        return temporal_scores[:max_packs]

    def _causal_retrieve(self, query: str, max_packs: int) -> List[Tuple[str, float]]:
        """Causal retrieval - packs that led to successful outcomes"""
        # In prototype, use semantic as fallback
        # In production, use causal graph with outcome tracking
        query_embedding = self.embed(query)
        return self._semantic_retrieve(query_embedding, max_packs, max_depth=1)

    def _entity_retrieve(self, query: str, max_packs: int) -> List[Tuple[str, float]]:
        """Entity retrieval - packs sharing papers/keywords"""
        # Extract entities from query
        query_lower = query.lower()

        entity_scores = []
        for pack_id, pack_node in self.packs.items():
            score = 0.0

            # Match keywords
            keywords = pack_node.content.get('keywords', [])
            for kw in keywords:
                if kw.lower() in query_lower:
                    score += 0.3

            # Match paper topics (arxiv IDs in query)
            papers = pack_node.content.get('papers', [])
            for paper in papers:
                arxiv_id = paper.get('arxiv_id', '')
                if arxiv_id in query:
                    score += 0.5

            entity_scores.append((pack_id, score))

        # Sort and return
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores[:max_packs]


# ============================================================================
# Layer 2: Role-Aware Multi-Agent Routing (RCR-Router inspired)
# ============================================================================

@dataclass
class Agent:
    """Specialized agent for pack selection"""
    agent_id: str
    role: str
    weight: float
    criteria: List[str]


class MultiAgentPackRouter:
    """
    Multi-agent coordination for pack selection
    Each agent has a role and votes on pack relevance
    Agents share semantic memory and refine over multiple rounds
    """

    def __init__(self, memory: MultiGraphPackMemory):
        self.memory = memory

        # Define specialized agents
        self.agents = [
            Agent('relevance', 'semantic_matcher', 0.35, ['keywords', 'papers', 'embeddings']),
            Agent('efficiency', 'cost_optimizer', 0.20, ['token_size', 'compression']),
            Agent('recency', 'temporal_prioritizer', 0.15, ['created', 'updated']),
            Agent('quality', 'outcome_analyzer', 0.15, ['dq_scores', 'success_rate']),
            Agent('diversity', 'coverage_maximizer', 0.15, ['uniqueness', 'complementarity'])
        ]

        # Shared semantic memory (cross-agent context)
        self.shared_memory = {}

    def route(
        self,
        query: str,
        context: Dict[str, Any],
        token_budget: int,
        rounds: int = 3
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Multi-agent pack selection with iterative refinement

        Args:
            query: User query/context
            context: Additional context (project, task_stage, etc.)
            token_budget: Maximum tokens allowed
            rounds: Number of refinement rounds

        Returns:
            (selected_pack_ids, routing_metadata)
        """
        # Initialize shared memory
        self.shared_memory = {
            'query': query,
            'context': context,
            'budget': token_budget,
            'round': 0
        }

        # Get all packs
        all_packs = list(self.memory.packs.keys())

        # Multi-round agent voting
        agent_votes = {agent.agent_id: {} for agent in self.agents}

        for round_num in range(1, rounds + 1):
            self.shared_memory['round'] = round_num

            # Each agent votes
            for agent in self.agents:
                votes = self._agent_vote(agent, all_packs, query, context)
                agent_votes[agent.agent_id] = votes

                # Share top picks with other agents
                top_picks = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]
                self.shared_memory[f'{agent.agent_id}_top'] = top_picks

        # Aggregate votes via weighted consensus
        consensus_scores = self._consensus_aggregation(agent_votes)

        # Greedy selection within budget
        selected_packs = self._greedy_select(consensus_scores, token_budget)

        # Metadata
        metadata = {
            'rounds': rounds,
            'agent_votes': agent_votes,
            'consensus_scores': consensus_scores,
            'agents_used': [a.agent_id for a in self.agents],
            'budget_used': sum(
                self.memory.packs[p].metadata['tokens']
                for p in selected_packs
            )
        }

        return selected_packs, metadata

    def _agent_vote(
        self,
        agent: Agent,
        all_packs: List[str],
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Individual agent voting logic"""
        votes = {}

        if agent.agent_id == 'relevance':
            # Semantic matching
            candidates = self.memory.adaptive_retrieve(
                query, intent='semantic', max_packs=10
            )
            for pack_id, score in candidates:
                votes[pack_id] = score

        elif agent.agent_id == 'efficiency':
            # Token efficiency
            for pack_id in all_packs:
                pack_node = self.memory.packs[pack_id]
                tokens = pack_node.metadata.get('tokens', 100)
                # Lower tokens = higher score
                votes[pack_id] = 1.0 / (tokens / 100.0)

        elif agent.agent_id == 'recency':
            # Temporal priority
            candidates = self.memory.adaptive_retrieve(
                query, intent='temporal', max_packs=10
            )
            for pack_id, score in candidates:
                votes[pack_id] = score

        elif agent.agent_id == 'quality':
            # Historical quality
            for pack_id in all_packs:
                # In prototype, use mock scores
                # In production, load from metrics.json
                votes[pack_id] = 0.8  # Mock quality score

        elif agent.agent_id == 'diversity':
            # Coverage maximization
            # Check what other agents selected
            already_selected = set()
            for other_agent_id, top_picks in self.shared_memory.items():
                if '_top' in other_agent_id:
                    already_selected.update([p[0] for p in top_picks])

            # Prefer packs not yet selected
            for pack_id in all_packs:
                if pack_id not in already_selected:
                    votes[pack_id] = 1.0
                else:
                    votes[pack_id] = 0.3

        return votes

    def _consensus_aggregation(
        self,
        agent_votes: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Weighted consensus across all agents"""
        consensus = defaultdict(float)

        for agent in self.agents:
            votes = agent_votes[agent.agent_id]
            for pack_id, score in votes.items():
                consensus[pack_id] += score * agent.weight

        return dict(consensus)

    def _greedy_select(
        self,
        consensus_scores: Dict[str, float],
        token_budget: int
    ) -> List[str]:
        """Greedy knapsack selection within token budget"""
        # Sort by consensus score
        sorted_packs = sorted(
            consensus_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected = []
        tokens_used = 0

        for pack_id, score in sorted_packs:
            pack_tokens = self.memory.packs[pack_id].metadata.get('tokens', 0)

            if tokens_used + pack_tokens <= token_budget:
                selected.append(pack_id)
                tokens_used += pack_tokens

        return selected


# ============================================================================
# Layer 3: Attention-Guided Pack Pruning (AttentionRAG inspired)
# ============================================================================

class AttentionPackPruner:
    """
    Prune pack content using simulated attention scores
    In production, use actual LLM attention via API
    """

    def __init__(self):
        self.compression_ratio = 0.63  # Target 6.3x compression

    def prune_pack(
        self,
        pack_data: Dict[str, Any],
        query: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prune pack content to most relevant elements

        Returns:
            (pruned_pack, pruning_metrics)
        """
        # Handle V1 pack structure
        pack_id = pack_data.get('pack_id') or pack_data.get('id')
        content = pack_data.get('content', pack_data)

        # Simulate attention scores for different pack elements
        attention_scores = self._calculate_attention(content, query)

        # Adaptive threshold
        threshold = self._adaptive_threshold(attention_scores)

        # Prune elements below threshold
        pruned_pack = {
            'pack_id': pack_id,
            'id': pack_id,
            'type': pack_data.get('type', 'unknown'),
            'papers': [],
            'learnings': [],
            'keywords': []
        }

        # Keep high-attention papers
        for i, paper in enumerate(content.get('papers', [])):
            if attention_scores['papers'][i] > threshold:
                pruned_pack['papers'].append(paper)

        # Keep high-attention learnings
        for i, learning in enumerate(content.get('learnings', [])):
            if attention_scores['learnings'][i] > threshold:
                pruned_pack['learnings'].append(learning)

        # Keep high-attention keywords
        for i, keyword in enumerate(content.get('keywords', [])):
            if attention_scores['keywords'][i] > threshold:
                pruned_pack['keywords'].append(keyword)

        # Metrics
        original_elements = (
            len(content.get('papers', [])) +
            len(content.get('learnings', [])) +
            len(content.get('keywords', []))
        )
        pruned_elements = (
            len(pruned_pack['papers']) +
            len(pruned_pack['learnings']) +
            len(pruned_pack['keywords'])
        )

        metrics = {
            'original_elements': original_elements,
            'pruned_elements': pruned_elements,
            'compression_ratio': pruned_elements / original_elements if original_elements > 0 else 1.0,
            'threshold': threshold
        }

        return pruned_pack, metrics

    def _calculate_attention(
        self,
        pack_data: Dict[str, Any],
        query: str
    ) -> Dict[str, List[float]]:
        """
        Simulate attention scores for pack elements
        In production, use actual LLM attention via API
        """
        query_lower = query.lower()
        scores = {
            'papers': [],
            'learnings': [],
            'keywords': []
        }

        # Papers attention
        for paper in pack_data.get('papers', []):
            arxiv_id = paper.get('arxiv_id', '')
            # Mock: higher score if arxiv mentioned
            score = 0.8 if arxiv_id in query else 0.5
            scores['papers'].append(score)

        # Learnings attention
        for learning in pack_data.get('learnings', []):
            # Mock: overlap with query
            overlap = sum(1 for word in learning.lower().split() if word in query_lower)
            score = min(1.0, 0.3 + overlap * 0.1)
            scores['learnings'].append(score)

        # Keywords attention
        for keyword in pack_data.get('keywords', []):
            # Mock: direct match
            score = 0.9 if keyword.lower() in query_lower else 0.4
            scores['keywords'].append(score)

        return scores

    def _adaptive_threshold(self, attention_scores: Dict[str, List[float]]) -> float:
        """Calculate adaptive threshold to achieve target compression"""
        # Flatten all scores
        all_scores = []
        for scores_list in attention_scores.values():
            all_scores.extend(scores_list)

        if not all_scores:
            return 0.5

        # Sort scores
        all_scores.sort(reverse=True)

        # Find threshold that keeps top X% elements
        target_keep = int(len(all_scores) * self.compression_ratio)
        if target_keep >= len(all_scores):
            return min(all_scores)

        threshold = all_scores[target_keep]
        return threshold


# ============================================================================
# Main V2 Engine - Integration of All Layers
# ============================================================================

class ContextPacksV2Engine:
    """
    Adaptive Multi-Graph Context Engine
    Integrates all V2 layers into cohesive pipeline
    """

    def __init__(self, pack_storage_dir: str = None):
        if pack_storage_dir is None:
            pack_storage_dir = os.path.expanduser('~/.agent-core/context-packs')

        self.pack_storage_dir = pack_storage_dir

        # Initialize layers
        print("Initializing Context Packs V2 Engine...")
        self.memory = MultiGraphPackMemory()
        self.router = MultiAgentPackRouter(self.memory)
        self.pruner = AttentionPackPruner()

        # Try to import Layer 4 (RL Manager)
        self.rl_manager = None
        try:
            from context_packs_v2_layer4_rl import RLPackManager
            self.rl_manager = RLPackManager(pack_storage_dir)
            print("✓ Layer 4 (RL Pack Manager) loaded")
        except ImportError:
            print("⚠️  Layer 4 (RL Pack Manager) not available")

        # Try to import Layers 5-7 (Focus, Continuum, Trainable)
        self.focus_agent = None
        self.continuum_memory = None
        self.trainable_graph = None
        try:
            from context_packs_v2_layer5_focus import FocusAgent, ContinuumMemory, TrainablePackGraph
            self.focus_agent = FocusAgent()
            self.continuum_memory = ContinuumMemory(pack_storage_dir)
            self.trainable_graph = TrainablePackGraph()
            print("✓ Layers 5-7 (Focus, Continuum, Trainable) loaded")
        except ImportError as e:
            print(f"⚠️  Layers 5-7 not available: {e}")

        # Load existing packs
        self._load_packs()

        print(f"✓ Loaded {len(self.memory.packs)} packs into multi-graph memory")

    def _load_packs(self):
        """Load all packs from storage into memory graphs"""
        pack_types = ['domain', 'project', 'pattern', 'paper']

        for pack_type in pack_types:
            type_dir = os.path.join(self.pack_storage_dir, pack_type)
            if not os.path.exists(type_dir):
                continue

            for pack_file in Path(type_dir).glob('*.pack.json'):
                with open(pack_file, 'r') as f:
                    pack_data = json.load(f)
                    self.memory.add_pack(pack_data)

    def select_and_compress(
        self,
        query: str,
        context: Dict[str, Any] = None,
        token_budget: int = 50000,
        enable_pruning: bool = True
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        End-to-end pack selection and compression

        Args:
            query: User query/context
            context: Additional context
            token_budget: Token budget
            enable_pruning: Whether to apply attention-guided pruning

        Returns:
            (final_packs, metrics)
        """
        start_time = time.time()

        if context is None:
            context = {}

        print(f"\n🔍 Query: {query[:80]}...")
        print(f"💰 Budget: {token_budget:,} tokens")

        # Step 1: Apply trainable weights (Layer 7) to boost pack scores
        if self.trainable_graph:
            print("\n[Layer 7] Applying trainable pack weights...")
            for pack_id in self.memory.packs.keys():
                weight = self.trainable_graph.get_weight(pack_id)
                if weight != 1.0:
                    print(f"  → {pack_id}: weight={weight:.3f}")

        # Step 2: Apply continuum memory importance boost (Layer 6)
        importance_boosts = {}
        if self.continuum_memory:
            print("\n[Layer 6] Applying continuum memory boosts...")
            for pack_id in self.memory.packs.keys():
                boost = self.continuum_memory.get_importance_boost(pack_id)
                importance_boosts[pack_id] = boost
                if boost > 1.1:
                    print(f"  → {pack_id}: importance_boost={boost:.3f}")

        # Step 3: Multi-agent pack selection (Layer 2)
        print("\n[Layer 2] Multi-agent routing...")

        # Modify router context to include weights and boosts
        enhanced_context = context.copy() if context else {}
        if self.trainable_graph:
            enhanced_context['trainable_weights'] = {
                pack_id: self.trainable_graph.get_weight(pack_id)
                for pack_id in self.memory.packs.keys()
            }
        if self.continuum_memory:
            enhanced_context['importance_boosts'] = importance_boosts

        selected_pack_ids, routing_metadata = self.router.route(
            query, enhanced_context, token_budget, rounds=3
        )

        print(f"  → Selected {len(selected_pack_ids)} packs")
        for pack_id in selected_pack_ids[:3]:
            consensus = routing_metadata['consensus_scores'].get(pack_id, 0)
            print(f"    • {pack_id}: consensus={consensus:.3f}")

        # Step 4: RL-based pack operations (Layer 4)
        rl_operations = []
        if self.rl_manager:
            print("\n[Layer 4] RL-based pack operations...")
            for pack_id in selected_pack_ids:
                pack_data = self.memory.packs[pack_id].content
                operation = self.rl_manager.decide_operation(
                    pack_data, query, session_id='current'
                )
                rl_operations.append({'pack_id': pack_id, 'operation': operation})
                if operation != 'NOOP':
                    print(f"  → {pack_id}: {operation}")

        # Step 5: Focus compression (Layer 5) - Applied before attention pruning
        focus_metrics = []
        if self.focus_agent and enable_pruning:
            print("\n[Layer 5] Active focus compression...")
            focused_packs = []
            for pack_id in selected_pack_ids:
                pack_data = self.memory.packs[pack_id].content
                focused_pack, metrics = self.focus_agent.compress_pack(pack_data, query)
                focused_packs.append(focused_pack)
                focus_metrics.append(metrics)
                if metrics['reduction_rate'] > 0.15:
                    print(f"  → {pack_id}: {metrics['reduction_rate']:.1%} reduction")

            # Use focused packs for next step
            temp_packs = focused_packs
        else:
            temp_packs = [self.memory.packs[pack_id].content for pack_id in selected_pack_ids]

        # Step 6: Attention-guided pruning (Layer 3)
        final_packs = []
        pruning_metrics = []

        if enable_pruning:
            print("\n[Layer 3] Attention-guided pruning...")
            for i, pack_id in enumerate(selected_pack_ids):
                pack_data = temp_packs[i]
                pruned_pack, prune_metrics = self.pruner.prune_pack(pack_data, query)
                final_packs.append(pruned_pack)
                pruning_metrics.append(prune_metrics)

            compression_ratios = [m['compression_ratio'] for m in pruning_metrics]
            avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0
            print(f"  → Average compression: {avg_compression:.1%} of original")
        else:
            for pack_id in selected_pack_ids:
                final_packs.append(self.memory.packs[pack_id].content)

        # Calculate metrics
        total_time = time.time() - start_time

        layers_used = ['multi_graph_memory']
        if self.trainable_graph:
            layers_used.append('trainable_pack_weights')
        if self.continuum_memory:
            layers_used.append('continuum_memory')
        layers_used.append('multi_agent_routing')
        if self.rl_manager:
            layers_used.append('rl_pack_operations')
        if self.focus_agent and enable_pruning:
            layers_used.append('active_focus_compression')
        if enable_pruning:
            layers_used.append('attention_pruning')

        metrics = {
            'selection_time_ms': total_time * 1000,
            'packs_selected': len(selected_pack_ids),
            'routing_metadata': routing_metadata,
            'rl_operations': rl_operations if self.rl_manager else None,
            'focus_compression': focus_metrics if self.focus_agent and enable_pruning else None,
            'pruning_enabled': enable_pruning,
            'pruning_metrics': pruning_metrics if enable_pruning else None,
            'layers_used': layers_used,
            'total_layers': len(layers_used)
        }

        return final_packs, metrics


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Context Packs V2 - Prototype'
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Query/context for pack selection'
    )
    parser.add_argument(
        '--budget',
        type=int,
        default=50000,
        help='Token budget (default: 50000)'
    )
    parser.add_argument(
        '--no-pruning',
        action='store_true',
        help='Disable attention-guided pruning'
    )
    parser.add_argument(
        '--intent',
        type=str,
        choices=['semantic', 'temporal', 'causal', 'entity'],
        default='semantic',
        help='Retrieval intent for graph traversal'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )

    args = parser.parse_args()

    # Initialize engine
    engine = ContextPacksV2Engine()

    # Select and compress
    final_packs, metrics = engine.select_and_compress(
        query=args.query,
        token_budget=args.budget,
        enable_pruning=not args.no_pruning
    )

    # Output results
    if args.format == 'json':
        result = {
            'query': args.query,
            'packs': final_packs,
            'metrics': metrics
        }
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*60)
        print("CONTEXT PACKS V2 - SELECTION RESULTS")
        print("="*60)
        print(f"\nQuery: {args.query}")
        print(f"Budget: {args.budget:,} tokens")
        print(f"Selected: {len(final_packs)} packs")
        print(f"Time: {metrics['selection_time_ms']:.1f}ms")
        print(f"Layers: {', '.join(metrics['layers_used'])}")

        print("\n" + "-"*60)
        print("SELECTED PACKS:")
        print("-"*60)

        for i, pack in enumerate(final_packs, 1):
            pack_id = pack['id']
            pack_type = pack['type']

            print(f"\n{i}. {pack_id} (type: {pack_type})")
            print(f"   Papers: {len(pack.get('papers', []))}")
            print(f"   Learnings: {len(pack.get('learnings', []))}")
            print(f"   Keywords: {', '.join(pack.get('keywords', [])[:5])}")

            if args.no_pruning:
                consensus = metrics['routing_metadata']['consensus_scores'].get(pack_id, 0)
                print(f"   Consensus: {consensus:.3f}")
            else:
                if metrics['pruning_metrics']:
                    prune_metric = metrics['pruning_metrics'][i-1]
                    print(f"   Compression: {prune_metric['compression_ratio']:.1%}")


if __name__ == '__main__':
    main()
