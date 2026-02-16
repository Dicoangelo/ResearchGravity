# Context Packs V2 - API Reference

**Version:** 2.0.0
**Date:** January 2026

---

## Overview

This document provides detailed API documentation for all Context Packs V2 components.

**System Components:**
- [ContextPacksV2Engine](#contextpacksv2engine) - Main V2 engine (Layers 1-3 + integration)
- [MultiGraphPackMemory](#multigraphpackmemory) - Layer 1: Multi-graph memory
- [MultiAgentPackRouter](#multiagentpackrouter) - Layer 2: Multi-agent routing
- [AttentionPackPruner](#attentionpackpruner) - Layer 3: Attention pruning
- [RLPackManager](#rlpackmanager) - Layer 4: RL pack operations
- [FocusAgent](#focusagent) - Layer 5: Active focus compression
- [ContinuumMemory](#continuummemory) - Layer 6: Continuum memory evolution
- [TrainablePackGraph](#trainablepackgraph) - Layer 7: Trainable pack weights
- [PackSelectorV2Integrated](#packselectorv2integrated) - Production selector with V1/V2 routing

---

## ContextPacksV2Engine

Main V2 engine that integrates all 7 layers into a cohesive pipeline.

**Location:** `context_packs_v2_prototype.py`

### Constructor

```python
ContextPacksV2Engine(
    pack_dir: str = "~/.agent-core/context-packs",
    embedding_model: str = "all-MiniLM-L6-v2"
)
```

**Parameters:**
- `pack_dir` (str): Directory containing context packs. Default: `~/.agent-core/context-packs`
- `embedding_model` (str): Sentence-transformers model name. Default: `all-MiniLM-L6-v2`

**Returns:** ContextPacksV2Engine instance

**Example:**
```python
from context_packs_v2_prototype import ContextPacksV2Engine

# Initialize with defaults
engine = ContextPacksV2Engine()

# Initialize with custom pack directory
engine = ContextPacksV2Engine(pack_dir="/custom/path/to/packs")
```

---

### select_and_compress()

Select and compress packs using all 7 layers.

```python
select_and_compress(
    query: str,
    context: Dict[str, Any] = None,
    token_budget: int = 50000,
    enable_pruning: bool = True
) -> Tuple[List[Dict], Dict[str, Any]]
```

**Parameters:**
- `query` (str): Query string for pack selection
- `context` (dict): Optional context dictionary with additional information
- `token_budget` (int): Maximum tokens to use. Default: 50000
- `enable_pruning` (bool): Whether to enable attention pruning. Default: True

**Returns:**
- `Tuple[List[Dict], Dict[str, Any]]`:
  - `packs` (List[Dict]): Selected and compressed packs
  - `metrics` (Dict): Selection metrics including:
    - `selection_time_ms` (float): Total selection time in milliseconds
    - `packs_selected` (int): Number of packs selected
    - `total_layers` (int): Number of layers used (should be 7)
    - `layers_used` (List[str]): Names of layers that executed
    - `routing_metadata` (Dict): Multi-agent routing details
    - `rl_operations` (List[Dict]): RL operations applied
    - `focus_compression` (List[Dict]): Focus compression results
    - `pruning_metrics` (Dict): Attention pruning stats

**Example:**
```python
# Basic usage
packs, metrics = engine.select_and_compress(
    query="debugging multi-agent consensus mechanisms",
    token_budget=50000
)

print(f"Selected {len(packs)} packs in {metrics['selection_time_ms']:.1f}ms")
print(f"Layers: {metrics['layers_used']}")

# With context
packs, metrics = engine.select_and_compress(
    query="React performance optimization",
    context={"project": "OS-App", "framework": "React 19"},
    token_budget=30000,
    enable_pruning=True
)

# Process selected packs
for pack in packs:
    print(f"Pack: {pack['pack_id']}")
    print(f"  Type: {pack['type']}")
    print(f"  Size: {pack['size_tokens']} tokens")
    print(f"  Papers: {len(pack['content']['papers'])}")
```

---

## MultiGraphPackMemory

Layer 1: Multi-graph memory with 4 graph types and semantic embeddings.

**Location:** `context_packs_v2_prototype.py`

### Constructor

```python
MultiGraphPackMemory(embedding_model: str = "all-MiniLM-L6-v2")
```

**Parameters:**
- `embedding_model` (str): Sentence-transformers model. Default: `all-MiniLM-L6-v2`

**Attributes:**
- `semantic_graph` (nx.DiGraph): Semantic similarity graph
- `temporal_graph` (nx.DiGraph): Temporal recency graph
- `causal_graph` (nx.DiGraph): Causal dependency graph
- `entity_graph` (nx.DiGraph): Entity co-occurrence graph
- `embedder` (SentenceTransformer): Embedding model
- `packs` (Dict): Loaded packs by ID

---

### load_packs()

Load all packs from storage into memory graphs.

```python
load_packs(pack_dir: str) -> int
```

**Parameters:**
- `pack_dir` (str): Directory containing pack files

**Returns:**
- `int`: Number of packs loaded

**Example:**
```python
memory = MultiGraphPackMemory()
num_packs = memory.load_packs("~/.agent-core/context-packs")
print(f"Loaded {num_packs} packs into 4 graphs")
```

---

### retrieve_by_intent()

Retrieve packs based on query intent and graph type.

```python
retrieve_by_intent(
    query: str,
    top_k: int = 10
) -> Tuple[List[str], str]
```

**Parameters:**
- `query` (str): Query string
- `top_k` (int): Maximum packs to retrieve. Default: 10

**Returns:**
- `Tuple[List[str], str]`:
  - `pack_ids` (List[str]): List of retrieved pack IDs
  - `graph_used` (str): Which graph was used (semantic/temporal/causal/entity)

**Example:**
```python
# Retrieve relevant packs
pack_ids, graph = memory.retrieve_by_intent("debugging React hooks", top_k=5)
print(f"Retrieved {len(pack_ids)} packs using {graph} graph")
```

---

## MultiAgentPackRouter

Layer 2: Multi-agent routing with 5 specialized agents and 3-round consensus.

**Location:** `context_packs_v2_prototype.py`

### Constructor

```python
MultiAgentPackRouter(
    agent_weights: Dict[str, float] = None
)
```

**Parameters:**
- `agent_weights` (Dict[str, float]): Optional custom agent weights. Default:
  - `relevance`: 0.35
  - `efficiency`: 0.20
  - `recency`: 0.15
  - `quality`: 0.15
  - `diversity`: 0.15

**Example:**
```python
# Default weights
router = MultiAgentPackRouter()

# Custom weights (emphasize relevance and quality)
router = MultiAgentPackRouter({
    'relevance': 0.40,
    'efficiency': 0.15,
    'recency': 0.10,
    'quality': 0.25,
    'diversity': 0.10
})
```

---

### route()

Route packs through multi-agent consensus over multiple rounds.

```python
route(
    query: str,
    context: Dict[str, Any],
    candidate_pack_ids: List[str],
    token_budget: int,
    rounds: int = 3
) -> Tuple[List[str], Dict[str, Any]]
```

**Parameters:**
- `query` (str): Query string
- `context` (Dict): Context dictionary
- `candidate_pack_ids` (List[str]): Candidate pack IDs to route
- `token_budget` (int): Token budget
- `rounds` (int): Number of consensus rounds. Default: 3

**Returns:**
- `Tuple[List[str], Dict[str, Any]]`:
  - `selected_ids` (List[str]): Selected pack IDs
  - `metadata` (Dict): Routing metadata with agent votes and consensus scores

**Example:**
```python
router = MultiAgentPackRouter()

selected, metadata = router.route(
    query="multi-agent orchestration",
    context={"session_type": "research"},
    candidate_pack_ids=["pack-1", "pack-2", "pack-3"],
    token_budget=50000,
    rounds=3
)

print(f"Selected {len(selected)} packs after {metadata['rounds']} rounds")
print(f"Consensus scores: {metadata['consensus_scores']}")
```

---

## AttentionPackPruner

Layer 3: Attention-guided element-level pruning for 6.3x compression.

**Location:** `context_packs_v2_prototype.py`

### Constructor

```python
AttentionPackPruner(
    embedding_model: str = "all-MiniLM-L6-v2",
    target_retention: float = 0.63
)
```

**Parameters:**
- `embedding_model` (str): Sentence-transformers model. Default: `all-MiniLM-L6-v2`
- `target_retention` (float): Target retention rate. Default: 0.63 (63%)

---

### prune_pack()

Prune pack elements based on attention scores.

```python
prune_pack(
    pack_data: Dict[str, Any],
    query: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

**Parameters:**
- `pack_data` (Dict): Pack data to prune
- `query` (str): Query for attention scoring

**Returns:**
- `Tuple[Dict[str, Any], Dict[str, Any]]`:
  - `pruned_pack` (Dict): Pruned pack data
  - `metrics` (Dict): Pruning metrics (elements before/after, retention rate)

**Example:**
```python
pruner = AttentionPackPruner()

pruned, metrics = pruner.prune_pack(
    pack_data=pack,
    query="debugging React performance"
)

print(f"Papers: {metrics['papers_before']} → {metrics['papers_after']}")
print(f"Learnings: {metrics['learnings_before']} → {metrics['learnings_after']}")
print(f"Retention: {metrics['retention_rate']:.1%}")
```

---

## RLPackManager

Layer 4: RL-based pack operations with neural network policy.

**Location:** `context_packs_v2_layer4_rl.py`

### Constructor

```python
RLPackManager(
    storage_path: str = "~/.agent-core/context-packs/rl_operations.jsonl"
)
```

**Parameters:**
- `storage_path` (str): Path to store RL operations. Default: `~/.agent-core/context-packs/rl_operations.jsonl`

**Attributes:**
- `policy` (PackOperationPolicy): Neural network policy
- `operations` (List[str]): Available operations: ADD, UPDATE, DELETE, MERGE, NOOP

---

### decide_operation()

Decide which operation to apply to a pack.

```python
decide_operation(
    pack_data: Dict[str, Any],
    context: Dict[str, Any],
    session_id: str
) -> Tuple[str, Dict[str, Any]]
```

**Parameters:**
- `pack_data` (Dict): Pack data
- `context` (Dict): Session context
- `session_id` (str): Unique session identifier

**Returns:**
- `Tuple[str, Dict[str, Any]]`:
  - `operation` (str): Operation to perform (ADD/UPDATE/DELETE/MERGE/NOOP)
  - `metadata` (Dict): Decision metadata

**Example:**
```python
manager = RLPackManager()

operation, metadata = manager.decide_operation(
    pack_data=pack,
    context={"query": "debugging", "session_type": "development"},
    session_id="session-20260118-001"
)

print(f"Operation: {operation}")
print(f"Confidence: {metadata['confidence']:.2f}")

# Apply operation
if operation == "ADD":
    # Add pack to selection
elif operation == "UPDATE":
    # Update pack content
elif operation == "DELETE":
    # Remove pack
elif operation == "MERGE":
    # Merge with another pack
# else NOOP - no action
```

---

### update_reward()

Update reward for a pack operation after session completion.

```python
update_reward(
    session_id: str,
    pack_id: str,
    reward: float
) -> bool
```

**Parameters:**
- `session_id` (str): Session identifier
- `pack_id` (str): Pack identifier
- `reward` (float): Reward value (0.0-1.0, where 1.0 is best)

**Returns:**
- `bool`: True if reward updated successfully

**Example:**
```python
# After successful session
manager.update_reward(
    session_id="session-20260118-001",
    pack_id="multi-agent-orchestration",
    reward=0.9  # High reward for helpful pack
)

# After poor session
manager.update_reward(
    session_id="session-20260118-002",
    pack_id="irrelevant-pack",
    reward=0.2  # Low reward for unhelpful pack
)
```

---

### train_policy()

Train RL policy on collected session outcomes.

```python
train_policy(
    batch_size: int = 32,
    epochs: int = 10
) -> Dict[str, Any]
```

**Parameters:**
- `batch_size` (int): Training batch size. Default: 32
- `epochs` (int): Number of training epochs. Default: 10

**Returns:**
- `Dict[str, Any]`: Training results with loss history

**Example:**
```python
# After collecting 50+ session outcomes
results = manager.train_policy(batch_size=32, epochs=10)

print(f"Training complete!")
print(f"  Samples: {results['samples_used']}")
print(f"  Final loss: {results['final_loss']:.4f}")
print(f"  Epochs: {results['epochs']}")
```

---

### get_history()

Get recent operation history.

```python
get_history(limit: int = 20) -> List[Dict[str, Any]]
```

**Parameters:**
- `limit` (int): Maximum operations to return. Default: 20

**Returns:**
- `List[Dict[str, Any]]`: Recent operations with metadata

**Example:**
```python
history = manager.get_history(limit=10)
for op in history:
    print(f"{op['timestamp']}: {op['operation']} on {op['pack_id']}")
    print(f"  Reward: {op.get('reward', 'pending')}")
```

---

## FocusAgent

Layer 5: Active focus compression for 22.7% autonomous reduction.

**Location:** `context_packs_v2_layer5_focus.py`

### Constructor

```python
FocusAgent(
    embedding_model: str = "all-MiniLM-L6-v2",
    target_reduction: float = 0.227
)
```

**Parameters:**
- `embedding_model` (str): Sentence-transformers model. Default: `all-MiniLM-L6-v2`
- `target_reduction` (float): Target reduction rate. Default: 0.227 (22.7%)

---

### compress_pack()

Compress pack using focus extraction and attention pruning.

```python
compress_pack(
    pack_data: Dict[str, Any],
    query: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

**Parameters:**
- `pack_data` (Dict): Pack to compress
- `query` (str): Query for focus extraction

**Returns:**
- `Tuple[Dict[str, Any], Dict[str, Any]]`:
  - `compressed_pack` (Dict): Compressed pack
  - `metrics` (Dict): Compression metrics

**Example:**
```python
focus = FocusAgent()

compressed, metrics = focus.compress_pack(
    pack_data=pack,
    query="multi-agent consensus mechanisms"
)

print(f"Original size: {metrics['original_tokens']} tokens")
print(f"Compressed size: {metrics['compressed_tokens']} tokens")
print(f"Reduction: {metrics['reduction_rate']:.1%}")
```

---

## ContinuumMemory

Layer 6: Persistent memory evolution across sessions.

**Location:** `context_packs_v2_layer5_focus.py`

### Constructor

```python
ContinuumMemory(
    storage_path: str = "~/.agent-core/context-packs/continuum_memory.json"
)
```

**Parameters:**
- `storage_path` (str): Path to continuum memory storage. Default: `~/.agent-core/context-packs/continuum_memory.json`

**Attributes:**
- `state` (Dict): Current memory state
- `retention_threshold` (float): Importance threshold for retention (0.3)

---

### update_persistent_state()

Update continuum memory with session outcome.

```python
update_persistent_state(
    session_outcome: Dict[str, Any]
) -> Dict[str, Any]
```

**Parameters:**
- `session_outcome` (Dict): Session outcome with:
  - `session_id` (str): Session identifier
  - `packs_used` (List[str]): Pack IDs used
  - `success_metric` (float): Success metric (0.0-1.0)
  - `context` (str): Session context

**Returns:**
- `Dict[str, Any]`: Updated state metadata

**Example:**
```python
continuum = ContinuumMemory()

outcome = {
    'session_id': 'session-20260118-001',
    'packs_used': ['multi-agent-orchestration', 'debugging-patterns'],
    'success_metric': 0.9,
    'context': 'debugging multi-agent consensus'
}

metadata = continuum.update_persistent_state(outcome)
print(f"State updated: {metadata['timestamp']}")
print(f"Total memories: {metadata['total_memories']}")
```

---

### get_pack_state()

Get persistent state for a specific pack.

```python
get_pack_state(pack_id: str) -> Optional[Dict[str, Any]]
```

**Parameters:**
- `pack_id` (str): Pack identifier

**Returns:**
- `Optional[Dict[str, Any]]`: Pack state or None if not found

**Example:**
```python
state = continuum.get_pack_state("multi-agent-orchestration")
if state:
    print(f"Access count: {state['access_count']}")
    print(f"Average success: {state['avg_success']:.2f}")
    print(f"Last used: {state['last_accessed']}")
    print(f"Importance: {state['importance_score']:.2f}")
```

---

### get_memory_summary()

Get summary of continuum memory state.

```python
get_memory_summary() -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Memory summary with stats

**Example:**
```python
summary = continuum.get_memory_summary()
print(f"Total packs tracked: {summary['total_packs']}")
print(f"High importance: {summary['high_importance_count']}")
print(f"Associations: {summary['total_associations']}")
```

---

## TrainablePackGraph

Layer 7: RL-optimized trainable pack weights.

**Location:** `context_packs_v2_layer5_focus.py`

### Constructor

```python
TrainablePackGraph(
    learning_rate: float = 0.01,
    weight_decay: float = 0.995
)
```

**Parameters:**
- `learning_rate` (float): Learning rate for weight updates. Default: 0.01
- `weight_decay` (float): Weight decay per update. Default: 0.995

**Attributes:**
- `weights` (Dict[str, float]): Pack weights (pack_id → weight)

---

### optimize_weights()

Optimize pack weights based on session outcomes.

```python
optimize_weights(
    session_outcomes: List[Dict[str, Any]]
) -> Dict[str, Any]
```

**Parameters:**
- `session_outcomes` (List[Dict]): List of session outcomes, each with:
  - `session_id` (str): Session identifier
  - `packs_used` (List[str]): Pack IDs used
  - `success_metric` (float): Success metric (0.0-1.0)

**Returns:**
- `Dict[str, Any]`: Optimization results

**Example:**
```python
trainable = TrainablePackGraph()

outcomes = [
    {
        'session_id': 'session-001',
        'packs_used': ['multi-agent-orchestration'],
        'success_metric': 0.9
    },
    {
        'session_id': 'session-002',
        'packs_used': ['debugging-patterns'],
        'success_metric': 0.7
    }
]

results = trainable.optimize_weights(outcomes)
print(f"Weights updated: {results['weights_updated']}")
print(f"Average improvement: {results['avg_improvement']:.2%}")
```

---

### get_pack_weight()

Get weight for a specific pack.

```python
get_pack_weight(pack_id: str) -> float
```

**Parameters:**
- `pack_id` (str): Pack identifier

**Returns:**
- `float`: Pack weight (default: 1.0 for new packs)

**Example:**
```python
weight = trainable.get_pack_weight("multi-agent-orchestration")
print(f"Pack weight: {weight:.3f}")
```

---

### get_top_packs()

Get top-weighted packs.

```python
get_top_packs(top_k: int = 10) -> List[Tuple[str, float]]
```

**Parameters:**
- `top_k` (int): Number of top packs to return. Default: 10

**Returns:**
- `List[Tuple[str, float]]`: List of (pack_id, weight) tuples, sorted by weight descending

**Example:**
```python
top_packs = trainable.get_top_packs(top_k=5)
for pack_id, weight in top_packs:
    print(f"{pack_id}: {weight:.3f}")
```

---

## PackSelectorV2Integrated

Production selector with automatic V1/V2 routing and fallback.

**Location:** `select_packs_v2_integrated.py`

### Constructor

```python
PackSelectorV2Integrated(force_v1: bool = False)
```

**Parameters:**
- `force_v1` (bool): Force V1 engine even if V2 available. Default: False

**Example:**
```python
# Auto-select best engine (V2 if available)
selector = PackSelectorV2Integrated()

# Force V1 engine
selector = PackSelectorV2Integrated(force_v1=True)
```

---

### select_packs()

Select packs using V2 (if available) or V1 (fallback).

```python
select_packs(
    context: str = None,
    token_budget: int = 50000,
    min_packs: int = 1,
    max_packs: int = 5,
    enable_pruning: bool = True
) -> Tuple[List[Dict], Dict[str, Any]]
```

**Parameters:**
- `context` (str): Query context (if None, auto-detects from directory)
- `token_budget` (int): Token budget. Default: 50000
- `min_packs` (int): Minimum packs (V1 only). Default: 1
- `max_packs` (int): Maximum packs (V1 only). Default: 5
- `enable_pruning` (bool): Enable pruning (V2 only). Default: True

**Returns:**
- `Tuple[List[Dict], Dict[str, Any]]`:
  - `packs` (List[Dict]): Selected packs
  - `metadata` (Dict): Selection metadata with engine type, layers, timing

**Example:**
```python
selector = PackSelectorV2Integrated()

# Basic selection
packs, metadata = selector.select_packs(
    context="debugging React performance",
    token_budget=50000
)

print(f"Engine: {metadata['engine']}")  # v2 or v1
print(f"Layers: {metadata['layers']}")  # 7 for V2, 2 for V1
print(f"Time: {metadata['selection_time_ms']:.1f}ms")
print(f"Packs: {metadata['packs_selected']}")

# Auto-detect context
packs, metadata = selector.select_packs()  # Uses directory context

# V2 with pruning disabled
packs, metadata = selector.select_packs(
    context="your query",
    enable_pruning=False  # Faster selection
)
```

---

## CLI Usage

### Main Selector

```bash
# V2 (default)
python3 select-packs --context "your query" --budget 50000

# Auto-detect
python3 select-packs --auto

# Force V1
python3 select-packs --context "your query" --v1

# JSON output
python3 select-packs --context "your query" --format json

# Disable pruning (V2)
python3 select-packs --context "your query" --no-pruning
```

### Direct V2 Access

```bash
# V2 prototype CLI
python3 v2 --query "your query" --budget 50000 --format text
```

### Layer 4: RL Pack Manager

```bash
# Decide operation
python3 context_packs_v2_layer4_rl.py decide \
  --pack-id multi-agent-orchestration \
  --context "debugging" \
  --session-id session-123

# Update reward
python3 context_packs_v2_layer4_rl.py reward \
  --session-id session-123 \
  --pack-id multi-agent-orchestration \
  --reward 0.9

# Train policy
python3 context_packs_v2_layer4_rl.py train \
  --batch-size 32 \
  --epochs 10

# View history
python3 context_packs_v2_layer4_rl.py history --limit 20
```

### Layers 5-7: Focus/Continuum/Trainable

```bash
# Focus compression
python3 context_packs_v2_layer5_focus.py focus \
  --pack-id multi-agent-orchestration \
  --query "multi-agent consensus"

# View continuum memory
python3 context_packs_v2_layer5_focus.py memory
python3 context_packs_v2_layer5_focus.py memory --pack-id multi-agent-orchestration

# View trainable weights
python3 context_packs_v2_layer5_focus.py weights --top 10
```

---

## Integration Examples

### Example 1: Complete Session Workflow

```python
from context_packs_v2_prototype import ContextPacksV2Engine
from context_packs_v2_layer4_rl import RLPackManager
from context_packs_v2_layer5_focus import ContinuumMemory, TrainablePackGraph

# Initialize components
engine = ContextPacksV2Engine()
rl_manager = RLPackManager()
continuum = ContinuumMemory()
trainable = TrainablePackGraph()

# 1. Select packs
packs, metrics = engine.select_and_compress(
    query="debugging multi-agent consensus",
    token_budget=50000
)

print(f"Selected {len(packs)} packs in {metrics['selection_time_ms']:.1f}ms")

# 2. Use packs in session
# ... your work here ...

# 3. Record session outcome
session_outcome = {
    'session_id': 'session-20260118-001',
    'packs_used': [pack['pack_id'] for pack in packs],
    'success_metric': 0.9,  # 0.0-1.0
    'context': 'debugging multi-agent consensus'
}

# 4. Update RL rewards
for pack_id in session_outcome['packs_used']:
    rl_manager.update_reward(
        session_id=session_outcome['session_id'],
        pack_id=pack_id,
        reward=session_outcome['success_metric']
    )

# 5. Update continuum memory
continuum.update_persistent_state(session_outcome)

# 6. Update trainable weights
trainable.optimize_weights([session_outcome])

print("Session recorded and learning updated!")
```

---

### Example 2: Training After 50+ Sessions

```python
from context_packs_v2_layer4_rl import RLPackManager
from context_packs_v2_layer5_focus import TrainablePackGraph

# Initialize managers
rl_manager = RLPackManager()
trainable = TrainablePackGraph()

# Train RL policy
print("Training RL policy...")
results = rl_manager.train_policy(batch_size=32, epochs=10)
print(f"  Samples: {results['samples_used']}")
print(f"  Final loss: {results['final_loss']:.4f}")

# View top weighted packs
print("\nTop weighted packs:")
top_packs = trainable.get_top_packs(top_k=10)
for pack_id, weight in top_packs:
    print(f"  {pack_id}: {weight:.3f}")
```

---

### Example 3: Custom Agent Weights

```python
from context_packs_v2_prototype import (
    ContextPacksV2Engine,
    MultiAgentPackRouter
)

# Create custom router emphasizing quality and relevance
custom_router = MultiAgentPackRouter({
    'relevance': 0.40,   # Increased
    'efficiency': 0.10,  # Decreased
    'recency': 0.10,     # Decreased
    'quality': 0.30,     # Increased
    'diversity': 0.10    # Decreased
})

# Initialize engine
engine = ContextPacksV2Engine()

# Replace default router
engine.router = custom_router

# Use as normal
packs, metrics = engine.select_and_compress(
    query="high quality research on multi-agent systems",
    token_budget=50000
)
```

---

### Example 4: Monitoring System Health

```python
from context_packs_v2_layer4_rl import RLPackManager
from context_packs_v2_layer5_focus import ContinuumMemory, TrainablePackGraph

rl_manager = RLPackManager()
continuum = ContinuumMemory()
trainable = TrainablePackGraph()

# Check RL history
print("Recent RL Operations:")
history = rl_manager.get_history(limit=5)
for op in history:
    print(f"  {op['timestamp']}: {op['operation']} on {op['pack_id']}")

# Check continuum memory
print("\nContinuum Memory:")
summary = continuum.get_memory_summary()
print(f"  Total packs tracked: {summary['total_packs']}")
print(f"  High importance: {summary['high_importance_count']}")

# Check top weights
print("\nTop Weighted Packs:")
top = trainable.get_top_packs(top_k=3)
for pack_id, weight in top:
    print(f"  {pack_id}: {weight:.3f}")
```

---

## Data Formats

### Pack Format

```json
{
  "pack_id": "multi-agent-orchestration",
  "type": "domain",
  "version": "1.0.0",
  "created_at": "2026-01-15T10:30:00",
  "size_tokens": 112,
  "content": {
    "papers": [
      {
        "arxiv_id": "2511.15755",
        "title": "MyAntFarm.ai: Multi-Agent Consensus",
        "relevance": "high"
      }
    ],
    "learnings": [
      {
        "text": "DQ Scoring: validity 40% + specificity 30% + correctness 30%",
        "session_id": "session-001",
        "date": "2026-01-15"
      }
    ],
    "keywords": [
      "multi-agent",
      "consensus",
      "voting",
      "orchestration",
      "dq-scoring"
    ]
  }
}
```

---

### Session Outcome Format

```python
session_outcome = {
    'session_id': 'session-20260118-001',
    'packs_used': [
        'multi-agent-orchestration',
        'debugging-patterns'
    ],
    'success_metric': 0.9,  # 0.0-1.0 (task completion/user satisfaction)
    'context': 'debugging multi-agent consensus system',
    'timestamp': '2026-01-18T14:30:00'
}
```

---

### Metrics Format

```python
metrics = {
    'selection_time_ms': 159.8,
    'packs_selected': 4,
    'budget_used': 274,
    'total_layers': 7,
    'layers_used': [
        'multi_graph_memory',
        'trainable_pack_weights',
        'continuum_memory',
        'multi_agent_routing',
        'rl_pack_operations',
        'active_focus_compression',
        'attention_pruning'
    ],
    'routing_metadata': {
        'rounds': 3,
        'consensus_scores': {...}
    },
    'rl_operations': [
        {'pack_id': 'pack-1', 'operation': 'ADD'},
        {'pack_id': 'pack-2', 'operation': 'UPDATE'}
    ],
    'focus_compression': [
        {'pack_id': 'pack-1', 'reduction': 0.227}
    ],
    'pruning_metrics': {
        'retention_rate': 0.63,
        'elements_pruned': 45
    }
}
```

---

## Error Handling

### Common Exceptions

#### ImportError: sentence-transformers
**Cause:** sentence-transformers not installed
**Solution:**
```bash
pip3 install sentence-transformers --break-system-packages
```

#### FileNotFoundError: Pack not found
**Cause:** Pack file missing from storage
**Solution:**
```python
# Check pack directory
import os
pack_dir = os.path.expanduser("~/.agent-core/context-packs")
print(os.listdir(pack_dir))

# Rebuild packs if needed
# python3 build_packs.py --source sessions --topic "your-topic"
```

#### ValueError: Invalid pack format
**Cause:** Pack missing required fields
**Solution:**
```python
# Ensure pack has required fields
required_fields = ['pack_id', 'type', 'content', 'size_tokens']
for field in required_fields:
    if field not in pack_data:
        raise ValueError(f"Pack missing required field: {field}")
```

#### RuntimeError: No packs loaded
**Cause:** Pack directory empty or invalid
**Solution:**
```python
# Check if packs exist
memory = MultiGraphPackMemory()
num_packs = memory.load_packs("~/.agent-core/context-packs")
if num_packs == 0:
    print("No packs found! Build packs first.")
```

---

## Performance Optimization

### Tip 1: Cache Embeddings

First run loads the embedding model (~90MB download). Subsequent runs use cached model.

```python
# First run: slow (downloads model)
engine = ContextPacksV2Engine()  # ~2-3 seconds

# Subsequent runs: fast (uses cache)
engine = ContextPacksV2Engine()  # ~100-400ms
```

---

### Tip 2: Disable Pruning for Speed

```python
# With pruning (more compression, slower)
packs, metrics = engine.select_and_compress(
    query="...",
    enable_pruning=True  # ~300-400ms
)

# Without pruning (faster)
packs, metrics = engine.select_and_compress(
    query="...",
    enable_pruning=False  # ~100-200ms
)
```

---

### Tip 3: Adjust Token Budget

```python
# Large budget (more packs, slower)
packs, metrics = engine.select_and_compress(
    query="...",
    token_budget=100000  # ~400-500ms
)

# Small budget (fewer packs, faster)
packs, metrics = engine.select_and_compress(
    query="...",
    token_budget=10000  # ~100-200ms
)
```

---

## Version History

### V2.0.0 (2026-01-18)
- Initial production release
- 7 layers operational
- Real semantic embeddings
- RL-based operations
- Dual compression (Focus + Attention)
- Persistent evolution
- V1 fallback support

---

## Support

### Getting Help

```bash
# View help
python3 select-packs --help
python3 v2 --help
python3 context_packs_v2_layer4_rl.py --help
python3 context_packs_v2_layer5_focus.py --help
```

### Documentation

- **Quick Start:** [QUICK_START.md](QUICK_START.md)
- **User Guide:** [USER_GUIDE.md](USER_GUIDE.md)
- **Complete System:** [CONTEXT_PACKS_V2_COMPLETE.md](CONTEXT_PACKS_V2_COMPLETE.md)
- **Deployment:** [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md)

### File Locations

```bash
# Implementation
~/researchgravity/context_packs_v2_*.py

# Documentation
~/researchgravity/CONTEXT_PACKS_V2_*.md

# Data storage
~/.agent-core/context-packs/
```

---

**API Reference Complete**

**Context Packs V2** - World-class context management, backed by science.

**Version:** 2.0.0
**Status:** Production-Ready
**Created:** January 2026
