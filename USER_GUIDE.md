# Context Packs V2 - Complete User Guide

**Version:** 2.0.0
**Last Updated:** January 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Layer-by-Layer Guide](#layer-by-layer-guide)
6. [Python API](#python-api)
7. [Training & Optimization](#training--optimization)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Introduction

Context Packs V2 is a 7-layer intelligent context management system built on January 2026 research. It uses real semantic embeddings, multi-graph memory, multi-agent routing, and RL-based learning to select optimal context packs.

### When to Use V2

- **Multi-agent systems** - Best semantic understanding
- **Complex queries** - 7 layers provide deep analysis
- **Learning over time** - System improves with use
- **High accuracy needs** - Real embeddings vs keywords

### When to Use V1

- **Simple queries** - Faster with keyword matching
- **Limited resources** - Lower memory footprint
- **Quick lookups** - No model loading time

---

## Installation

See [QUICK_START.md](QUICK_START.md) for detailed installation.

**Quick install:**
```bash
pip3 install sentence-transformers networkx numpy torch --break-system-packages
cd ~/researchgravity && ./deploy_v2.sh
```

---

## Basic Usage

### Command Line

```bash
# Standard selection
python3 select-packs --context "your query" --budget 50000

# Auto-detect
python3 select-packs --auto

# JSON output
python3 select-packs --context "..." --format json

# Force V1
python3 select-packs --context "..." --v1
```

### Python Integration

```python
from context_packs_v2_prototype import ContextPacksV2Engine

engine = ContextPacksV2Engine()
packs, metrics = engine.select_and_compress(
    query="your query",
    token_budget=50000
)
```

---

## Advanced Features

### Layer 4: RL Pack Manager

Train the system to learn optimal pack operations:

```bash
# After sessions, update rewards
python3 context_packs_v2_layer4_rl.py reward \\
  --session-id session-123 \\
  --pack-id pack-name \\
  --reward 0.9

# Train after 50+ sessions
python3 context_packs_v2_layer4_rl.py train \\
  --batch-size 32 --epochs 10
```

### Layers 5-7: Memory & Weights

```bash
# Focus compression
python3 context_packs_v2_layer5_focus.py focus \\
  --pack-id pack-name --query "..."

# View continuum memory
python3 context_packs_v2_layer5_focus.py memory

# View trainable weights
python3 context_packs_v2_layer5_focus.py weights --top 10
```

---

## Layer-by-Layer Guide

### Layer 1: Multi-Graph Memory
- **Purpose:** Store packs in 4 graph types
- **Graphs:** Semantic, temporal, causal, entity
- **Usage:** Automatic (runs on every query)

### Layer 2: Multi-Agent Routing
- **Purpose:** 5 agents vote over 3 rounds
- **Agents:** Relevance, efficiency, recency, quality, diversity
- **Usage:** Automatic (core selection engine)

### Layer 3: Attention Pruning
- **Purpose:** Element-level compression
- **Target:** 6.3x compression
- **Usage:** Enabled by default (use --no-pruning to disable)

### Layer 4: RL Pack Operations
- **Purpose:** Learn optimal operations
- **Operations:** ADD, UPDATE, DELETE, MERGE, NOOP
- **Usage:** Record rewards, train policy

### Layer 5: Focus Compression
- **Purpose:** 22.7% autonomous reduction
- **Method:** Semantic focus extraction
- **Usage:** Automatic when pruning enabled

### Layer 6: Continuum Memory
- **Purpose:** Persistent evolution
- **Features:** Selective retention, associations
- **Usage:** Updates automatically per session

### Layer 7: Trainable Weights
- **Purpose:** RL-optimized pack selection
- **Method:** Weight optimization from outcomes
- **Usage:** Trains automatically with Layer 4

---

## Troubleshooting

### V2 Not Loading
```bash
python3 -c "import sentence_transformers, networkx, numpy"
# If fails: pip3 install sentence-transformers networkx numpy --break-system-packages
```

### Slow Selection
- First run: Model loading (normal)
- Persistent: Use --no-pruning or --v1

### No Packs Found
```bash
ls ~/.agent-core/context-packs/*/
python3 build_packs.py --source sessions --topic "..." --since 14
```

---

## Best Practices

1. **Use V2 for important queries** - Better accuracy
2. **Record session outcomes** - Enables learning
3. **Train after 50+ sessions** - Optimize performance
4. **Use --auto for projects** - Context detection
5. **Monitor selection times** - Tune if needed

---

See [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation.
