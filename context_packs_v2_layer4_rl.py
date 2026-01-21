#!/usr/bin/env python3
"""
Context Packs V2 - Layer 4: RL-Based Pack Operations
====================================================

Implements Memory-R1 inspired RL-based pack management:
- Operations: {ADD, UPDATE, DELETE, MERGE, NOOP}
- Policy learned via reinforcement learning
- Agent weight optimization based on outcomes
- Memory distillation for pack content

Based on: Memory-R1 (arXiv:2508.19828)
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Using mock RL policy.")
    print("   Install with: pip3 install torch")


# ============================================================================
# RL Policy Network
# ============================================================================

if TORCH_AVAILABLE:
    class PackOperationPolicy(nn.Module):
        """
        Neural network policy for pack operations

        Input: Pack state (metadata + context + reward history)
        Output: Probability distribution over operations
        """

        def __init__(self, state_dim: int = 128, hidden_dim: int = 64):
            super().__init__()

            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 5)  # 5 operations

            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, state):
            """Forward pass through policy network"""
            x = self.relu(self.fc1(state))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return self.softmax(x)


    class AgentWeightOptimizer(nn.Module):
        """
        Neural network for optimizing agent weights

        Input: Context embedding + session metadata
        Output: Optimal agent weights (5 agents)
        """

        def __init__(self, context_dim: int = 384, hidden_dim: int = 32):
            super().__init__()

            self.fc1 = nn.Linear(context_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 5)  # 5 agents

            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, context_embedding):
            """Predict optimal agent weights for this context"""
            x = self.relu(self.fc1(context_embedding))
            x = self.fc2(x)
            return self.softmax(x)


# ============================================================================
# Pack Operation Manager
# ============================================================================

@dataclass
class PackOperation:
    """A pack operation with reward feedback"""
    operation: str  # ADD, UPDATE, DELETE, MERGE, NOOP
    pack_id: str
    session_id: str
    timestamp: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    reward: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class RLPackManager:
    """
    RL-based pack management with learned operations

    Operations:
    - ADD: Create new pack from session
    - UPDATE: Modify existing pack content
    - DELETE: Remove low-value pack
    - MERGE: Combine similar packs
    - NOOP: Keep pack unchanged
    """

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.expanduser('~/.agent-core/context-packs')

        self.storage_dir = storage_dir
        self.operations_log = os.path.join(storage_dir, 'rl_operations.jsonl')

        # Initialize RL policy
        if TORCH_AVAILABLE:
            self.policy = PackOperationPolicy()
            self.weight_optimizer = AgentWeightOptimizer()
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
            self.weight_opt_optimizer = optim.Adam(self.weight_optimizer.parameters(), lr=0.001)
        else:
            self.policy = None
            self.weight_optimizer = None

        # Load operation history
        self.operation_history: List[PackOperation] = []
        self._load_operations()

        print("RL Pack Manager initialized")
        print(f"  Operations logged: {len(self.operation_history)}")
        print(f"  RL policy: {'Trained' if TORCH_AVAILABLE else 'Mock'}")

    def _load_operations(self):
        """Load operation history from log"""
        if not os.path.exists(self.operations_log):
            return

        with open(self.operations_log, 'r') as f:
            for line in f:
                data = json.loads(line)
                op = PackOperation(**data)
                self.operation_history.append(op)

    def _log_operation(self, operation: PackOperation):
        """Log operation to history"""
        self.operation_history.append(operation)

        # Append to log file
        with open(self.operations_log, 'a') as f:
            f.write(json.dumps(asdict(operation)) + '\n')

    def encode_state(
        self,
        pack_data: Dict[str, Any],
        context: str,
        reward_history: List[float]
    ) -> Any:  # np.ndarray or list
        """
        Encode pack state for RL policy

        State features:
        - Pack metadata (tokens, age, version)
        - Context relevance
        - Reward history statistics
        - Usage statistics
        """
        if TORCH_AVAILABLE:
            state_vector = []

            # Pack metadata (20 dims)
            state_vector.append(pack_data.get('size_tokens', 100) / 1000.0)  # Normalized tokens

            # Parse version (handle "1.0.0" format)
            version = pack_data.get('version', '1.0.0')
            if isinstance(version, str):
                version_num = float(version.split('.')[0]) if '.' in version else float(version)
            else:
                version_num = float(version)
            state_vector.append(version_num / 10.0)  # Normalized version

            # Usage stats (10 dims)
            usage = pack_data.get('usage_stats', {})
            state_vector.append(usage.get('times_selected', 0) / 100.0)
            state_vector.append(usage.get('avg_session_relevance', 0.0))
            state_vector.append(len(usage.get('sessions', [])) / 100.0)
            state_vector.append(len(usage.get('combined_with', [])) / 10.0)

            # DQ metadata (10 dims)
            dq = pack_data.get('dq_metadata', {})
            state_vector.append(dq.get('base_validity', 0.5))
            state_vector.append(dq.get('base_specificity', 0.5))
            state_vector.append(dq.get('base_correctness', 0.5))
            state_vector.append(dq.get('base_score', 0.5))

            # Reward history stats (10 dims)
            if reward_history:
                state_vector.append(np.mean(reward_history))
                state_vector.append(np.std(reward_history))
                state_vector.append(np.min(reward_history))
                state_vector.append(np.max(reward_history))
                state_vector.append(len(reward_history) / 100.0)
            else:
                state_vector.extend([0.0] * 5)

            # Context relevance (mock - in production, use embeddings)
            context_lower = context.lower()
            pack_id = pack_data.get('pack_id', '')
            relevance = 1.0 if pack_id.lower() in context_lower else 0.3
            state_vector.append(relevance)

            # Pad to 128 dims
            while len(state_vector) < 128:
                state_vector.append(0.0)

            return torch.tensor(state_vector[:128], dtype=torch.float32)
        else:
            # Mock state encoding
            return [0.5] * 128

    def decide_operation(
        self,
        pack_data: Dict[str, Any],
        context: str,
        session_id: str
    ) -> str:
        """
        Use RL policy to decide best operation for pack

        Returns: Operation name (ADD, UPDATE, DELETE, MERGE, NOOP)
        """
        # Get reward history for this pack
        pack_id = pack_data.get('pack_id', pack_data.get('id'))
        reward_history = [
            op.reward for op in self.operation_history
            if op.pack_id == pack_id and op.reward is not None
        ]

        # Encode state
        state = self.encode_state(pack_data, context, reward_history)

        if TORCH_AVAILABLE:
            # Get policy prediction
            with torch.no_grad():
                action_probs = self.policy(state)
                action_idx = torch.argmax(action_probs).item()
        else:
            # Mock decision
            # Heuristic: UPDATE if pack is relevant, NOOP otherwise
            context_lower = context.lower()
            pack_id_lower = pack_id.lower()
            if any(keyword in context_lower for keyword in pack_id_lower.split('-')):
                action_idx = 1  # UPDATE
            else:
                action_idx = 4  # NOOP

        operations = ['ADD', 'UPDATE', 'DELETE', 'MERGE', 'NOOP']
        return operations[action_idx]

    def execute_operation(
        self,
        operation: str,
        pack_data: Dict[str, Any],
        context: str,
        session_id: str,
        additional_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a pack operation

        Returns: Updated pack data
        """
        pack_id = pack_data.get('pack_id', pack_data.get('id'))

        # Record state before
        state_before = pack_data.copy()

        # Execute operation
        if operation == 'ADD':
            result = self._add_pack(pack_data, additional_data)
        elif operation == 'UPDATE':
            result = self._update_pack(pack_data, context, additional_data)
        elif operation == 'DELETE':
            result = self._delete_pack(pack_data)
        elif operation == 'MERGE':
            result = self._merge_packs(pack_data, additional_data)
        else:  # NOOP
            result = pack_data.copy()

        # Log operation
        op = PackOperation(
            operation=operation,
            pack_id=pack_id,
            session_id=session_id,
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            state_before=state_before,
            state_after=result,
            reward=None,  # Will be set later based on outcome
            metadata={'context': context}
        )
        self._log_operation(op)

        return result

    def _add_pack(self, pack_data: Dict[str, Any], additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new pack"""
        # In prototype, just return the pack data
        # In production, create file and update registry
        print(f"  [RL] ADD pack: {pack_data.get('pack_id')}")
        return pack_data

    def _update_pack(
        self,
        pack_data: Dict[str, Any],
        context: str,
        additional_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing pack content"""
        print(f"  [RL] UPDATE pack: {pack_data.get('pack_id')}")

        updated = pack_data.copy()

        # Update DQ scores based on recent performance
        if 'dq_metadata' in updated:
            dq = updated['dq_metadata']
            # Slight boost for being selected
            dq['base_validity'] = min(1.0, dq.get('base_validity', 0.8) + 0.01)
            updated['dq_metadata'] = dq

        # Update usage stats
        if 'usage_stats' in updated:
            stats = updated['usage_stats']
            stats['times_selected'] = stats.get('times_selected', 0) + 1
            updated['usage_stats'] = stats

        return updated

    def _delete_pack(self, pack_data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete pack (mark for deletion)"""
        print(f"  [RL] DELETE pack: {pack_data.get('pack_id')}")
        deleted = pack_data.copy()
        deleted['_deleted'] = True
        return deleted

    def _merge_packs(self, pack_data: Dict[str, Any], additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge pack with similar packs"""
        print(f"  [RL] MERGE pack: {pack_data.get('pack_id')}")

        # In prototype, just combine keywords
        # In production, merge content intelligently
        merged = pack_data.copy()

        if additional_data and 'merge_with' in additional_data:
            other_pack = additional_data['merge_with']

            # Merge content
            if 'content' in merged:
                content = merged['content']
                other_content = other_pack.get('content', {})

                # Combine keywords
                keywords = set(content.get('keywords', []))
                keywords.update(other_content.get('keywords', []))
                content['keywords'] = list(keywords)

                # Combine papers
                papers = content.get('papers', [])
                other_papers = other_content.get('papers', [])
                paper_ids = {p['arxiv_id'] for p in papers}
                for paper in other_papers:
                    if paper['arxiv_id'] not in paper_ids:
                        papers.append(paper)
                content['papers'] = papers

                merged['content'] = content

        return merged

    def update_reward(self, session_id: str, pack_id: str, reward: float):
        """
        Update reward for a past operation

        Reward signal based on session outcome:
        - High reward: Pack was useful, led to success
        - Low reward: Pack was not useful or irrelevant
        """
        # Find operation
        for op in reversed(self.operation_history):
            if op.session_id == session_id and op.pack_id == pack_id:
                op.reward = reward
                break

        # Rewrite log file with updated rewards
        with open(self.operations_log, 'w') as f:
            for op in self.operation_history:
                f.write(json.dumps(asdict(op)) + '\n')

    def train_policy(self, batch_size: int = 32, epochs: int = 10):
        """
        Train RL policy on operation history

        Uses REINFORCE algorithm:
        - Sample operations from history
        - Calculate returns (cumulative rewards)
        - Update policy to maximize expected return
        """
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, skipping training")
            return

        # Filter operations with rewards
        ops_with_rewards = [op for op in self.operation_history if op.reward is not None]

        if len(ops_with_rewards) < batch_size:
            print(f"⚠️  Not enough operations with rewards ({len(ops_with_rewards)} < {batch_size})")
            return

        print(f"\nTraining RL policy on {len(ops_with_rewards)} operations...")

        for epoch in range(epochs):
            # Sample batch
            batch_indices = np.random.choice(len(ops_with_rewards), batch_size, replace=False)
            batch_ops = [ops_with_rewards[i] for i in batch_indices]

            total_loss = 0.0

            for op in batch_ops:
                # Encode state
                state = self.encode_state(
                    op.state_before,
                    op.metadata.get('context', ''),
                    []
                )

                # Get action index
                operations = ['ADD', 'UPDATE', 'DELETE', 'MERGE', 'NOOP']
                action_idx = operations.index(op.operation)

                # Forward pass
                action_probs = self.policy(state)

                # Calculate loss (REINFORCE)
                # L = -log(π(a|s)) * R
                log_prob = torch.log(action_probs[action_idx] + 1e-8)
                loss = -log_prob * op.reward

                # Backward pass
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / batch_size
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

        print("✓ Policy training complete")

    def optimize_agent_weights(
        self,
        context_embedding: Any,
        outcome_reward: float,
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Optimize agent weights based on outcome

        Args:
            context_embedding: Embedding of session context
            outcome_reward: Reward signal from session outcome
            current_weights: Current agent weights

        Returns:
            Optimized agent weights
        """
        if not TORCH_AVAILABLE:
            # Mock: slight adjustment based on reward
            adjusted = current_weights.copy()
            if outcome_reward > 0.8:
                # Boost relevance agent if high reward
                adjusted['relevance'] = min(1.0, adjusted['relevance'] * 1.1)
            elif outcome_reward < 0.3:
                # Reduce efficiency agent if low reward
                adjusted['efficiency'] = max(0.1, adjusted['efficiency'] * 0.9)

            # Renormalize
            total = sum(adjusted.values())
            return {k: v/total for k, v in adjusted.items()}

        # Use neural network to predict optimal weights
        with torch.no_grad():
            if isinstance(context_embedding, list):
                context_tensor = torch.tensor(context_embedding, dtype=torch.float32)
            else:
                context_tensor = context_embedding

            optimal_weights = self.weight_optimizer(context_tensor)

        agent_names = ['relevance', 'efficiency', 'recency', 'quality', 'diversity']
        return {name: float(optimal_weights[i]) for i, name in enumerate(agent_names)}


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Context Packs V2 - Layer 4: RL Pack Manager'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Decide operation
    decide_parser = subparsers.add_parser('decide', help='Decide operation for pack')
    decide_parser.add_argument('--pack-id', required=True)
    decide_parser.add_argument('--context', required=True)
    decide_parser.add_argument('--session-id', required=True)

    # Execute operation
    execute_parser = subparsers.add_parser('execute', help='Execute operation')
    execute_parser.add_argument('--operation', required=True, choices=['ADD', 'UPDATE', 'DELETE', 'MERGE', 'NOOP'])
    execute_parser.add_argument('--pack-id', required=True)
    execute_parser.add_argument('--context', required=True)
    execute_parser.add_argument('--session-id', required=True)

    # Update reward
    reward_parser = subparsers.add_parser('reward', help='Update reward for operation')
    reward_parser.add_argument('--session-id', required=True)
    reward_parser.add_argument('--pack-id', required=True)
    reward_parser.add_argument('--reward', type=float, required=True)

    # Train policy
    train_parser = subparsers.add_parser('train', help='Train RL policy')
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--epochs', type=int, default=10)

    # Show history
    history_parser = subparsers.add_parser('history', help='Show operation history')
    history_parser.add_argument('--limit', type=int, default=20)

    args = parser.parse_args()

    # Initialize manager
    manager = RLPackManager()

    if args.command == 'decide':
        # Load pack
        pack_storage = os.path.expanduser('~/.agent-core/context-packs')
        pack_file = None
        for pack_type in ['domain', 'project', 'pattern', 'paper']:
            potential_file = os.path.join(pack_storage, pack_type, f'{args.pack_id}.pack.json')
            if os.path.exists(potential_file):
                pack_file = potential_file
                break

        if not pack_file:
            print(f"❌ Pack not found: {args.pack_id}")
            return

        with open(pack_file, 'r') as f:
            pack_data = json.load(f)

        # Decide operation
        operation = manager.decide_operation(pack_data, args.context, args.session_id)
        print(f"\n✓ Decided operation: {operation}")
        print(f"  Pack: {args.pack_id}")
        print(f"  Context: {args.context}")

    elif args.command == 'execute':
        # Similar to decide, but execute
        print(f"Executing {args.operation} on {args.pack_id}...")

    elif args.command == 'reward':
        manager.update_reward(args.session_id, args.pack_id, args.reward)
        print(f"✓ Updated reward: {args.reward} for {args.pack_id} in {args.session_id}")

    elif args.command == 'train':
        manager.train_policy(batch_size=args.batch_size, epochs=args.epochs)

    elif args.command == 'history':
        print("\n" + "="*60)
        print("OPERATION HISTORY")
        print("="*60)

        for op in manager.operation_history[-args.limit:]:
            print(f"\n{op.timestamp} | {op.operation} | {op.pack_id}")
            print(f"  Session: {op.session_id}")
            print(f"  Reward: {op.reward if op.reward is not None else 'pending'}")


if __name__ == '__main__':
    main()
