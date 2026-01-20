# CPB Core (Python)

**Cognitive Precision Bridge** - Unified AI orchestration with precision-aware routing.

[![PyPI version](https://badge.fury.io/py/cpb-core.svg)](https://badge.fury.io/py/cpb-core)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install cpb-core
```

## Quick Start

```python
from cpb import cpb, analyze, score_response

# Analyze query complexity
result = analyze("Design a distributed cache system")
print(f"Complexity: {result['complexity_score']:.2f}")
print(f"Recommended path: {result['selected_path']}")

# Get routing decision
decision = cpb.route("Compare REST vs GraphQL for our API")
print(f"Path: {decision.selected_path}")
print(f"Reasoning: {decision.reasoning}")

# Build ACE consensus prompts
prompts = cpb.build_ace_prompts("What's the best auth strategy?")
for p in prompts:
    print(f"[{p['agent']}] {p['system_prompt'][:50]}...")

# Score a response
dq = score_response(
    query="Explain microservices",
    response="Microservices are an architectural style..."
)
print(f"DQ Score: {dq.overall:.2f}")
```

## Architecture

```
Query → [Complexity Analysis] → [Path Selection] → [Execution] → [DQ Scoring] → Result
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                 DIRECT             RLM/ACE            CASCADE
                 (Fast)            (Standard)          (Deep)
```

### Execution Paths

| Path | Complexity | Use Case | Model |
|------|------------|----------|-------|
| **Direct** | <0.2 | Simple queries, navigation | Sonnet |
| **RLM** | 0.2-0.5 | Long context, document analysis | Sonnet |
| **ACE** | 0.5-0.7 | Decisions, trade-offs, consensus | Opus |
| **Hybrid** | >0.7 | Complex + long context | Opus |
| **Cascade** | >0.7 | Critical decisions, research | Opus |

### 5-Agent ACE Ensemble

| Agent | Role | Focus |
|-------|------|-------|
| 🔬 Analyst | Evidence evaluator | Data, logic, consistency |
| 🤔 Skeptic | Challenge assumptions | Risks, edge cases |
| 🔄 Synthesizer | Pattern finder | Connections, frameworks |
| 🛠️ Pragmatist | Feasibility checker | Actionability, constraints |
| 🔭 Visionary | Strategic thinker | Long-term effects |

### DQ Scoring

```python
DQ = Validity (40%) + Specificity (30%) + Correctness (30%)
```

## ELITE Tier Configuration

```python
from cpb import DEFAULT_CPB_CONFIG

# ELITE defaults:
# - Default path: cascade
# - Context threshold: 100,000 chars
# - Complexity threshold: 0.35
# - DQ threshold: 0.75
# - ACE agents: 5
```

## CLI Usage

```bash
# Route a query
cpb route "Design a new API architecture"

# Analyze complexity
cpb analyze "What is TypeScript?"

# Score a response
cpb score --query "Explain REST" --response "REST is..."
```

## Related Packages

- **[@metaventionsai/cpb-core](https://www.npmjs.com/package/@metaventionsai/cpb-core)** - TypeScript version
- **[@metaventionsai/voice-nexus](https://www.npmjs.com/package/@metaventionsai/voice-nexus)** - Voice AI architecture

## License

MIT © Dicoangelo
