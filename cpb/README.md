# CPB Core (Python)

**Cognitive Precision Bridge** - Unified AI orchestration with precision-aware routing and research-grounded answers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.5.0-green.svg)]()

## Features

- **Precision Mode** - Research-grounded, evidence-verified answers (DQ ≥ 0.95)
- **7-Agent Ensemble** - Multi-perspective analysis with critic validation
- **Deep Research** - Gemini/Perplexity integration for real-time web research
- **Pioneer Mode** - Optimized weights for cutting-edge research queries
- **Trust Context** - User-provided context treated as Tier 1 sources
- **Ground Truth** - Claim extraction, cross-source validation, self-consistency

## Installation

```bash
# Core dependencies
pip install -r cpb/requirements.txt

# Or install individually
pip install aiohttp google-genai anthropic
```

## Quick Start

### Precision Mode (Recommended)

```bash
# Basic precision query
python -m cpb.precision_cli "What are best practices for multi-agent orchestration?"

# With deep research (Gemini/Perplexity)
python -m cpb.precision_cli "What are emerging TDP patterns in 2026?" --deep-research

# Pioneer mode for cutting-edge research
python -m cpb.precision_cli "Latest findings on agent consensus" --pioneer --deep-research

# With user context as trusted source
python -m cpb.precision_cli "Analyze my research" --context @research.md --trust-context

# Check system status
python -m cpb.precision_cli --status

# Dry run (show execution plan without API calls)
python -m cpb.precision_cli "test query" --dry-run
```

### Python API

```python
from cpb import execute_precision, check_dependencies, get_deep_research_status

# Check system status
deps = check_dependencies()
dr_status = get_deep_research_status()
print(f"Best provider: {dr_status['best_provider']}")

# Execute precision query
import asyncio

async def research():
    result = await execute_precision(
        "What are the architectural patterns for multi-agent AI?",
        pioneer=True,
        deep_research=True
    )
    print(f"DQ Score: {result.dq_score:.3f}")
    print(f"Sources: {len(result.sources)}")
    print(result.output)

asyncio.run(research())
```

## Architecture

```
Query → [Enhancement] → [Deep Research] → [Tiered Search] → [7-Agent Cascade] → [Verification] → Result
            │                 │                 │                  │                  │
         Haiku           Gemini/PPX        arXiv+GH+RG        Sonnet×7          Ground Truth
```

### Execution Phases

| Phase | Description | Model |
|-------|-------------|-------|
| Query Enhancement | Expand vague queries into research-grade prompts | Haiku |
| Deep Research | Real-time web research via Gemini or Perplexity | External API |
| Tiered Search | arXiv, GitHub, internal ResearchGravity sources | API + Qdrant |
| 7-Agent Cascade | Multi-perspective analysis | Sonnet × 7 |
| MAR Consensus | Multi-Agent Reasoning verification | Haiku × 3 + Sonnet |
| Verification | Ground truth validation, citation checking | Internal |

### 7-Agent Precision Ensemble

| Agent | Role | Focus |
|-------|------|-------|
| 🔬 Analyst | Evidence | Data, logic, citations |
| 🤔 Skeptic | Critique | Risks, edge cases, gaps |
| 🔄 Synthesizer | Integration | Patterns, frameworks |
| 🛠️ Pragmatist | Implementation | Feasibility, trade-offs |
| 🔭 Visionary | Strategy | Long-term implications |
| 📚 Historian | Context | Prior art, precedents |
| 💡 Innovator | Novelty | New approaches, breakthroughs |

### DQ Scoring

```
Default Mode:
  DQ = Validity (30%) + Specificity (20%) + Correctness (35%) + Ground Truth (15%)

Pioneer Mode (for cutting-edge research):
  DQ = Validity (25%) + Specificity (25%) + Correctness (30%) + Ground Truth (20%)

Trust Context Mode:
  DQ = Validity (28%) + Specificity (20%) + Correctness (40%) + Ground Truth (12%)
```

## v2.5 Enhancements

### Deep Research Hardening

- **Caching**: 15-minute TTL cache to avoid redundant API calls
- **Retry Logic**: Exponential backoff (1s, 2s, 4s) for transient failures
- **Provider Fallback**: Gemini → Perplexity automatic failover
- **Gemini Extraction**: 3 fallback methods for citation extraction

### Cost Tracking

```python
result = await execute_precision(query, deep_research=True)
print(f"Deep research cost: ${result.deep_research_cost_usd:.4f}")
print(f"Deep research tokens: {result.deep_research_tokens}")
print(f"Phase timings: {result.phase_timings}")
```

### CLI Enhancements

```bash
# System status (dependencies, providers, cache)
python -m cpb.precision_cli --status

# Execution plan preview
python -m cpb.precision_cli "query" --dry-run

# With all options
python -m cpb.precision_cli "query" \
    --pioneer \
    --deep-research \
    --deep-provider gemini \
    --trust-context \
    --context @file.md \
    --verbose \
    --output result.md
```

## Configuration

### Environment Variables

```bash
# Gemini (required for deep research)
export GOOGLE_API_KEY="your-key"

# Perplexity (optional fallback)
export PERPLEXITY_API_KEY="your-key"

# Anthropic (for LLM calls)
export ANTHROPIC_API_KEY="your-key"
```

### API Keys in Config

Store keys in `~/.agent-core/config.json`:

```json
{
  "gemini": {"api_key": "your-key"},
  "perplexity": {"api_key": "your-key"},
  "cohere": {"api_key": "your-key"}
}
```

## Testing

```bash
# Run all tests
python -m pytest cpb/tests/ -v

# Run specific test file
python -m pytest cpb/tests/test_precision_v24.py -v

# Run with coverage
python -m pytest cpb/tests/ --cov=cpb
```

## Directory Structure

```
cpb/
├── __init__.py              # Package exports, check_dependencies()
├── precision_orchestrator.py # 7-agent cascade, verification loop
├── precision_cli.py         # CLI interface (--status, --dry-run)
├── precision_config.py      # DQ weights, thresholds
├── deep_research.py         # Gemini/Perplexity integration (v2.5 hardened)
├── search_layer.py          # Tiered search (arXiv, GitHub, internal)
├── query_enhancer.py        # Query expansion with pioneer detection
├── critic_verifier.py       # Evidence, oracle, ground truth validation
├── ground_truth.py          # Claim extraction, cross-source validation
├── llm_client.py            # Multi-provider LLM client
├── run_logger.py            # Run documentation with cost tracking
├── requirements.txt         # Dependencies
└── tests/
    ├── conftest.py          # Pytest fixtures
    └── test_precision_v24.py # v2.4/2.5 feature tests
```

## Version History

| Version | Features |
|---------|----------|
| **2.5** | Caching, retry, fallback, cost tracking, --status, --dry-run |
| **2.4** | Pioneer mode, trust context, deep research integration |
| **2.3** | Query enhancement, run logging, follow-up queries |
| **2.2** | Ground truth corpus, diagnostic mode |
| **2.1** | Ground truth validation, claims extraction |
| **2.0** | Precision mode, 7-agent ensemble, tiered search |

## License

MIT © Dicoangelo
