# Context Packs V2 - Quick Start Guide

**Get started with Context Packs V2 in 5 minutes!**

---

## Prerequisites

- Python 3.8+ installed
- Basic terminal knowledge
- ~/researchgravity directory with V2 files

---

## Step 1: Install Dependencies (2 minutes)

```bash
# Install required packages
pip3 install sentence-transformers networkx numpy torch --break-system-packages

# Verify installation
python3 -c "import sentence_transformers, networkx, numpy; print('✓ All dependencies installed')"
```

**What this does:**
- `sentence-transformers` - Real semantic embeddings
- `networkx` - Graph algorithms for 4-graph architecture
- `numpy` - Numerical operations
- `torch` - RL training (optional but recommended)

---

## Step 2: Deploy V2 System (1 minute)

```bash
# Navigate to researchgravity
cd ~/researchgravity

# Run deployment script
./deploy_v2.sh
```

**What this does:**
- Backs up V1 system
- Deploys all V2 files
- Creates convenience symlinks
- Initializes storage
- Runs test to verify

**Expected output:**
```
============================================================
Context Packs V2 - Production Deployment
============================================================

[1/7] Checking dependencies...
✓ All dependencies present

[2/7] Backing up V1 system...
✓ V1 system backed up

[3/7] Deploying V2 files...
✓ All V2 files present
✓ V2 files deployed

[4/7] Creating convenience symlinks...
✓ Created: select-packs
✓ Created: v2

[5/7] Testing V2 engine...
✓ V2 engine operational
   Layers: 7
   Selection time: ~100-400ms

[6/7] Initializing V2 storage...
✓ Created: rl_operations.jsonl
✓ Created: continuum_memory.json

[7/7] Deployment complete!
✅ Context Packs V2 - Production Deployment Complete
```

---

## Step 3: Your First Query (1 minute)

```bash
# Basic pack selection
python3 select-packs \
  --context "debugging multi-agent systems" \
  --budget 50000

# Output shows:
# - Engine: V2
# - Layers: 7
# - Selection time: ~100-400ms
# - Packs selected with details
```

**What this does:**
- Initializes V2 engine (all 7 layers)
- Loads semantic embeddings model
- Builds 4-graph memory from existing packs
- Runs 5 agents over 3 rounds
- Applies RL operations
- Compresses with Focus + Attention
- Returns optimal pack selection

---

## Step 4: Auto-Detect Context (30 seconds)

```bash
# Navigate to any project
cd ~/OS-App  # or your project directory

# Auto-detect context from directory
python3 ~/researchgravity/select-packs --auto --budget 30000
```

**What this does:**
- Detects context from:
  - Directory name
  - Recent git log (if git repo)
  - package.json (if exists)
- Automatically selects relevant packs

---

## Step 5: Explore Features (1 minute)

### View Help

```bash
# Main selector help
python3 select-packs --help

# Direct V2 access help
python3 v2 --help

# RL Manager help
python3 context_packs_v2_layer4_rl.py --help
```

### Try Different Options

```bash
# JSON output (for integration)
python3 select-packs --context "your query" --format json

# Force V1 engine (2 layers, faster)
python3 select-packs --context "simple query" --v1

# Disable pruning (faster selection)
python3 select-packs --context "your query" --no-pruning

# Larger budget
python3 select-packs --context "complex query" --budget 100000
```

---

## Common Use Cases

### Use Case 1: Project-Specific Selection

```bash
# For OS-App project
cd ~/OS-App
python3 ~/researchgravity/select-packs --auto

# For debugging work
python3 ~/researchgravity/select-packs \
  --context "debugging React performance issues" \
  --budget 50000
```

### Use Case 2: Research Paper Context

```bash
python3 select-packs \
  --context "multi-agent consensus arXiv:2508.04903" \
  --budget 30000
```

### Use Case 3: Integration with Scripts

```bash
# Get JSON output
RESULT=$(python3 select-packs --context "your query" --format json)

# Extract pack IDs
echo "$RESULT" | jq -r '.packs[].pack_id'
```

---

## Understanding Output

### Text Output Format

```
============================================================
PACK SELECTION RESULTS (V2)
============================================================

Engine: V2                    # Using 7-layer V2 engine
Layers: 7                     # All layers active
Selected: 4 packs             # Number of packs chosen
Budget Used: 274 tokens       # Tokens consumed
Time: 159.8ms                 # Selection time
Layers Active:                # Which layers executed
  - multi_graph_memory
  - trainable_pack_weights
  - continuum_memory
  - multi_agent_routing
  - rl_pack_operations
  - active_focus_compression
  - attention_pruning

------------------------------------------------------------
SELECTED PACKS:
------------------------------------------------------------

1. multi-agent-orchestration (type: domain)
   Size: 112 tokens
   Papers: 2511.15755, 2508.17536, 2505.19591, 2505.13516
   Keywords: multi-agent, consensus, voting, orchestration, ace, dq-scoring

2. agentic-memory (type: domain)
   Size: 93 tokens
   Papers: 2502.12110, 2504.19413, 2601.01885
   Keywords: memory, zettelkasten, a-mem, mem0, agemem, retention

...
```

### JSON Output Format

```json
{
  "packs": [
    {
      "pack_id": "multi-agent-orchestration",
      "type": "domain",
      "size_tokens": 112,
      "content": {
        "papers": [...],
        "learnings": [...],
        "keywords": [...]
      }
    }
  ],
  "metadata": {
    "engine": "v2",
    "layers": 7,
    "selection_time_ms": 159.8,
    "packs_selected": 4,
    "budget_used": 274,
    "v2_metrics": {
      "layers_used": [...],
      "routing_metadata": {...},
      "rl_operations": [...],
      "focus_compression": [...],
      "pruning_metrics": [...]
    }
  }
}
```

---

## Next Steps

### Build Your Own Packs

```bash
# From recent sessions
python3 build_packs.py --source sessions --topic "your-topic" --since 14

# From learnings
python3 build_packs.py --source learnings --cluster-by topic

# Manual creation
python3 build_packs.py --create \
  --type domain \
  --name "quantum-computing" \
  --papers "2601.12345,2601.23456" \
  --keywords "quantum,qubits,entanglement"
```

### Train the System

After using V2 for 50+ sessions, train the RL policy:

```bash
# Update rewards for sessions (do this after each session)
python3 context_packs_v2_layer4_rl.py reward \
  --session-id session-123 \
  --pack-id multi-agent-orchestration \
  --reward 0.9

# Train policy
python3 context_packs_v2_layer4_rl.py train \
  --batch-size 32 \
  --epochs 10

# View results
python3 context_packs_v2_layer5_focus.py weights --top 10
```

### Learn More

- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete guide with all features
- **[API_REFERENCE.md](API_REFERENCE.md)** - API documentation
- **[CONTEXT_PACKS_V2_COMPLETE.md](CONTEXT_PACKS_V2_COMPLETE.md)** - System architecture

---

## Troubleshooting

### Problem: "V2 engine not available"

**Solution:**
```bash
# Check dependencies
python3 -c "import sentence_transformers, networkx, numpy"

# If missing, reinstall
pip3 install sentence-transformers networkx numpy --break-system-packages
```

### Problem: "Slow first run"

**Solution:**
- First run downloads embeddings model (~90MB)
- Subsequent runs are much faster (model cached)
- This is normal!

### Problem: "No packs found"

**Solution:**
```bash
# Check pack storage
ls -la ~/.agent-core/context-packs/*/

# If empty, build packs
python3 build_packs.py --source sessions --topic "your-topic" --since 14
```

### Problem: "Selection takes >1 second"

**Solution:**
```bash
# Disable pruning
python3 select-packs --context "..." --no-pruning

# Or use V1 for simple queries
python3 select-packs --context "..." --v1
```

---

## Quick Reference Card

```bash
# BASIC USAGE
python3 select-packs --context "your query" --budget 50000
python3 select-packs --auto

# OPTIONS
--context "..."    # Query for pack selection
--auto             # Auto-detect from current directory
--budget N         # Token budget (default: 50000)
--format text|json # Output format (default: text)
--v1               # Force V1 engine (2 layers)
--no-pruning       # Disable compression (faster)

# ADVANCED TOOLS
python3 v2 --query "..." --budget 50000                    # Direct V2 access
python3 context_packs_v2_layer4_rl.py decide --pack-id ... # RL decision
python3 context_packs_v2_layer5_focus.py memory            # View memory
python3 context_packs_v2_layer5_focus.py weights --top 10  # View weights

# HELP
python3 select-packs --help
python3 v2 --help
cat ~/researchgravity/USER_GUIDE.md
```

---

## Summary

**You're now ready to use Context Packs V2!**

✅ Dependencies installed
✅ V2 system deployed
✅ First query successful
✅ Know how to use basic features

**Next:**
1. Use V2 in your daily workflow
2. Collect session outcomes
3. Train after 50+ sessions
4. Watch it improve!

**Questions?** Read the [USER_GUIDE.md](USER_GUIDE.md) or [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md).

---

**Context Packs V2** - *5 minutes to world-class context management!* 🚀
