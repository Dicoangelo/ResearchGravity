# Context Packs System - Architecture Proposal

**Date:** 2026-01-18
**Innovation:** Transform 64MB session context → 8.5KB intelligent packs
**Goal:** Modular, composable, AI-routed context with cost tracking

---

## Problem Statement

Current prefetch: One monolithic 8.5KB context per session.

**Limitations:**
- Not adaptable to different session types
- Can't combine multiple knowledge domains
- No metrics on efficiency or savings
- Manual selection required

**Opportunity:** 64MB → 8.5KB proved 99% reduction possible. Can we make it intelligent, modular, and tracked?

---

## Solution: Context Packs System

### 1. Pack Types

#### Domain Packs (Topic-Specific)
```yaml
multi-agent-orchestration.pack:
  size: 3.2KB
  papers: [2511.15755, 2508.17536, 2505.19591]
  learnings: ACE, DQ Scoring, Bicameral voting
  projects: OS-App consensus engine

llm-optimization.pack:
  size: 2.8KB
  papers: [2512.14142, 2504.19413]
  learnings: KV cache, token savings, memory consolidation
  projects: Astraea scheduling, Mem0 integration
```

#### Project Packs (Codebase-Specific)
```yaml
os-app-architecture.pack:
  size: 5.2KB
  structure: App.tsx patterns, Zustand state, API modules
  conventions: Component naming, hook patterns
  gotchas: Large App.tsx, read sections not whole file

careercoach-agents.pack:
  size: 3.1KB
  structure: Next.js app router, AI integrations
  conventions: CareerResumeBuilder module
```

#### Pattern Packs (Workflow-Specific)
```yaml
debugging-patterns.pack:
  size: 1.8KB
  strategies: Root cause analysis, git bisect, logging
  tools: /debug skill, breakpoints

architecture-design.pack:
  size: 3.9KB
  strategies: Plan mode usage, system design, trade-offs
  tools: /arch skill, diagram generation
```

#### Paper Packs (Research Clusters)
```yaml
consensus-mechanisms.pack:
  size: 4.1KB
  papers: [2511.15755, 2508.17536, 2505.19591, 2505.13516]
  theme: Multi-agent voting, ACE, CIR3

agentic-memory.pack:
  size: 3.8KB
  papers: [2502.12110, 2504.19413, 2601.01885]
  theme: A-MEM, Mem0, AgeMem architectures
```

---

## 2. Intelligent Selection (DQ + ACE)

### DQ Scoring Framework

```python
def score_pack(pack, session_context):
    """
    DQ Scoring: Validity (40%) + Specificity (30%) + Correctness (30%)
    """

    # Validity: How relevant to current session?
    validity = semantic_similarity(pack.content, session_context)

    # Specificity: How targeted vs generic?
    specificity = pack.domain_focus / pack.generality

    # Correctness: How up-to-date and accurate?
    correctness = recency_score(pack.updated) * accuracy_rating(pack)

    return (validity * 0.4) + (specificity * 0.3) + (correctness * 0.3)
```

### ACE Consensus for Pack Selection

```python
def select_packs_via_ace(session_context, token_budget=50_000):
    """
    Adaptive Consensus Engine for pack selection
    Multiple scoring perspectives reach consensus
    """

    # Multiple agents score packs
    agent_scores = {
        'relevance_agent': score_by_relevance(all_packs),
        'efficiency_agent': score_by_token_efficiency(all_packs),
        'recency_agent': score_by_freshness(all_packs),
    }

    # ACE consensus
    consensus_scores = adaptive_consensus(agent_scores)

    # Greedy knapsack: maximize value within budget
    selected = []
    tokens_used = 0

    for pack in sorted(consensus_scores, key=lambda x: x.score, reverse=True):
        if tokens_used + pack.size <= token_budget:
            selected.append(pack)
            tokens_used += pack.size

    return merge_packs(selected), tokens_used
```

---

## 3. Pack Storage & Registry

### Directory Structure
```
~/.agent-core/context-packs/
├── domain/
│   ├── multi-agent-orchestration.pack.json
│   ├── llm-optimization.pack.json
│   └── ui-ux-patterns.pack.json
├── project/
│   ├── os-app-architecture.pack.json
│   └── careercoach-agents.pack.json
├── pattern/
│   ├── debugging-patterns.pack.json
│   └── architecture-design.pack.json
├── paper/
│   ├── consensus-mechanisms.pack.json
│   └── agentic-memory.pack.json
├── registry.json  # Pack metadata and DQ scores
└── metrics.json   # Usage stats, savings, costs
```

### Pack Format

```json
{
  "pack_id": "multi-agent-orchestration",
  "type": "domain",
  "version": "1.2.0",
  "created": "2026-01-16T12:00:00Z",
  "updated": "2026-01-17T20:43:00Z",
  "size_bytes": 3276,
  "size_tokens": 820,
  "content": {
    "papers": [
      {"arxiv_id": "2511.15755", "title": "MyAntFarm.ai DQ Scoring", "relevance": 5},
      {"arxiv_id": "2508.17536", "title": "Voting captures most gains", "relevance": 5}
    ],
    "learnings": [
      "ACE (Adaptive Consensus Engine) - validity 40% + specificity 30% + correctness 30%",
      "Bicameral architecture outperforms single-agent by 23%"
    ],
    "implementations": [
      "OS-App consensus engine (App.tsx:1250-1340)",
      "DQ scoring in agent evaluation"
    ],
    "keywords": ["multi-agent", "consensus", "voting", "orchestration"]
  },
  "dq_metadata": {
    "base_validity": 0.92,
    "base_specificity": 0.88,
    "base_correctness": 0.95,
    "base_score": 0.917
  },
  "usage_stats": {
    "times_selected": 32,
    "sessions": ["dr.-zero---self-evol-20260117-082802-6a0ec3", "..."],
    "avg_session_relevance": 0.89,
    "combined_with": ["agentic-memory", "os-app-architecture"]
  }
}
```

---

## 4. Metrics & Cost Tracking

### Tracked Metrics

```python
class PackMetrics:
    """Track context pack efficiency and cost savings"""

    # Session-level
    session_id: str
    packs_loaded: List[str]  # Pack IDs
    total_tokens: int  # Combined pack size
    baseline_tokens: int  # What would have been loaded without packs
    token_savings: int  # baseline - total
    cost_savings: float  # token_savings * model_rate

    # Pack-level
    pack_selection_time_ms: float
    dq_scores: Dict[str, float]
    ace_consensus: Dict[str, Any]

    # Aggregated (across all sessions)
    total_sessions: int
    total_token_savings: int
    total_cost_savings: float
    avg_cache_hit_rate: float
    most_used_packs: List[Tuple[str, int]]
    best_combinations: List[Dict]
```

### Cost Translation

```python
# Model pricing
PRICING = {
    'haiku': {'input': 0.25, 'output': 1.25},  # per MTok
    'sonnet': {'input': 3.00, 'output': 15.00},
    'opus': {'input': 15.00, 'output': 75.00},
}

def calculate_savings(baseline_tokens, pack_tokens, model='sonnet'):
    """Calculate $ saved by using packs vs full context"""
    saved_tokens = baseline_tokens - pack_tokens
    cost_per_token = PRICING[model]['input'] / 1_000_000
    return saved_tokens * cost_per_token

# Example: 64MB session
baseline = 16_000_000 tokens  # ~64MB transcript
pack_total = 12_000 tokens  # 3 packs combined
savings = calculate_savings(baseline, pack_total, 'sonnet')
# Result: $47.96 saved per session!
```

---

## 5. Dashboard Integration

### New Tab: "Context Efficiency"

```yaml
Overview Panel:
  Total Token Savings: 4,812,340 tokens
  Total Cost Savings: $14,437.02
  Sessions Optimized: 147
  Average Reduction: 99.2%

Live Metrics:
  Current Session: 3 packs loaded (9.2KB)
  Baseline Would Be: 28MB (7,000,000 tokens)
  This Session Saves: $20.97
  Model: Sonnet

Pack Performance (Top 5):
  1. multi-agent-orchestration
     - Uses: 32 sessions
     - Avg DQ: 0.917
     - Savings: $672.64

  2. os-app-architecture
     - Uses: 28 sessions
     - Avg DQ: 0.943
     - Savings: $589.20

Best Combinations:
  1. [multi-agent, agentic-memory, os-app]
     - Uses: 23 sessions
     - Combined DQ: 0.921
     - Avg tokens: 11,840
     - Savings: $458.70

Efficiency Timeline (Chart):
  - Jan 16: Baseline (no packs)
  - Jan 17 19:16: Packs deployed
  - Jan 18: Current
  [Line chart showing token usage dropping]

Cost Breakdown:
  Without Packs: $15,211.00 (estimated)
  With Packs: $773.98 (actual)
  Savings: $14,437.02 (94.9% reduction)
```

---

## 6. Usage Examples

### Auto-Select (Default)
```bash
# Session starts, auto-detects context
cc  # Starts Claude Code in current directory

# System automatically:
# 1. Detects project: OS-App
# 2. Analyzes recent git history
# 3. Uses ACE to select relevant packs
# 4. Loads: [os-app-architecture, multi-agent, debugging-patterns]
# 5. Injects 9.2KB context instead of 64MB
```

### Manual Selection
```bash
# Load specific packs
prefetch --packs multi-agent,agentic-memory,consensus-mechanisms

# Load by pattern
prefetch --pattern architecture  # Auto-selects architecture-related packs

# Load all packs for domain
prefetch --domain multi-agent  # All multi-agent related packs
```

### On-Demand Addition
```python
# Mid-session: need more context
/load-pack llm-optimization
# System fetches pack, scores against current context, injects if relevant
```

---

## 7. Pack Building Pipeline

### Automatic Pack Generation

```python
# From archived sessions
python3 build_packs.py --source sessions --since 2026-01-01

# From learnings.md
python3 build_packs.py --source learnings --cluster-by topic

# Manual pack creation
python3 build_packs.py --create \
  --type domain \
  --name "quantum-computing" \
  --papers "2601.12345,2601.23456" \
  --keywords "quantum,qubits,entanglement"
```

### Pack Versioning

```bash
# Update pack with new learnings
python3 update_pack.py multi-agent-orchestration \
  --add-paper 2601.09876 \
  --add-learning "New consensus mechanism from latest session"

# Result: multi-agent-orchestration v1.3.0
```

---

## 8. Implementation Phases

### Phase 1: Pack Builder (Week 1)
- [ ] `build_packs.py` - Generate packs from sessions/learnings
- [ ] Pack storage structure
- [ ] Registry system
- [ ] Basic metrics tracking

### Phase 2: Intelligent Selection (Week 1-2)
- [ ] `select_packs.py` - DQ scoring implementation
- [ ] ACE consensus for pack selection
- [ ] Session context analyzer
- [ ] Automatic pack loading in prefetch.py

### Phase 3: Metrics & Dashboard (Week 2)
- [ ] `pack_metrics.py` - Comprehensive tracking
- [ ] Cost calculation and translation
- [ ] Dashboard "Context Efficiency" tab
- [ ] Real-time savings display

### Phase 4: Advanced Features (Week 2-3)
- [ ] On-demand pack loading mid-session
- [ ] Pack versioning and updates
- [ ] Pack combination optimizer
- [ ] A/B testing for pack effectiveness

---

## 9. Success Metrics

**Target Outcomes:**

1. **Token Efficiency**
   - Maintain 99%+ token reduction
   - Average context load: <15KB
   - Baseline comparison: >10MB saved per session

2. **Cost Savings**
   - $500+ saved per week on Sonnet
   - 95%+ cost reduction vs no-pack baseline
   - ROI: System pays for itself in compute savings

3. **Relevance**
   - DQ scores avg >0.85 for loaded packs
   - <5% pack replacements mid-session
   - User satisfaction: packs are helpful 90%+ of time

4. **Performance**
   - Pack selection: <500ms
   - Zero session startup delay
   - Cache hit rate: >99%

---

## 10. Innovation Contribution

**This system represents:**

1. **First AI-routed context management** - No one else is using DQ+ACE for session context
2. **Proof of concept for Mem0** - 90% token savings, but with composability
3. **Cost transparency** - Developers see exactly what they save
4. **Modular memory** - Like package managers, but for AI context

**Research paper potential:** "Context Packs: Modular, AI-Routed Session Memory for Large Language Models"

---

## Files to Create

```
researchgravity/
├── build_packs.py          # Generate packs from sessions
├── select_packs.py         # DQ + ACE pack selection
├── pack_metrics.py         # Metrics tracking
├── update_pack.py          # Pack versioning
└── CONTEXT_PACKS_PROPOSAL.md  # This file

.agent-core/
└── context-packs/          # Pack storage
    ├── domain/
    ├── project/
    ├── pattern/
    ├── paper/
    ├── registry.json
    └── metrics.json

.claude/scripts/
├── dashboard.html          # Updated with Context Efficiency tab
└── command-center.html     # Updated with pack metrics
```

---

**Next Action:** Implement Phase 1 - Pack Builder and storage structure.
