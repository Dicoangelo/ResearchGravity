# Routing Research Workflow

Complete guide for research-driven optimization of the CLI routing system.

## Overview

This workflow integrates academic research (arXiv papers) with the autonomous routing system, allowing baselines to be updated with research-backed insights and full lineage tracking.

---

## Quick Start

```bash
# 1. Initialize research session
cd ~/researchgravity
python3 init_session.py "LLM routing optimization 2026" --impl-project cli-routing

# 2. Fetch recent papers
python3 routing-research-sync.py fetch-papers \
  --query "LLM routing OR model selection OR adaptive inference" \
  --days 90 \
  --output fetched-papers.json

# 3. Extract insights (LLM-assisted)
python3 routing-research-sync.py extract-insights \
  --papers fetched-papers.json \
  --focus "complexity thresholds,cost optimization,accuracy metrics" \
  --model sonnet \
  --output routing-insights.json

# 4. Preview baseline updates
python3 routing-research-sync.py update-baselines \
  --insights routing-insights.json \
  --dry-run

# 5. Apply updates (if satisfied)
python3 routing-research-sync.py update-baselines \
  --insights routing-insights.json \
  --apply

# 6. Archive session
python3 archive_session.py
```

---

## Detailed Workflow

### Step 1: Initialize Research Session

```bash
python3 ~/researchgravity/init_session.py \
  "LLM routing optimization 2026" \
  --impl-project cli-routing
```

**What this does:**
- Creates session ID: `llm-routing-opt-YYYYMMDD-HHMMSS-hash`
- Links to implementation project: `cli-routing`
- Sets up session directory in `~/.agent-core/sessions/`
- Generates multi-tier search queries

**Output:**
```
✓ Session initialized: llm-routing-opt-20260118-143022-a4f8
✓ Linked to project: cli-routing
✓ Search queries generated
```

### Step 2: Fetch Recent Papers

```bash
python3 ~/researchgravity/routing-research-sync.py fetch-papers \
  --query "LLM routing" \
  --days 90 \
  --max-results 50 \
  --output fetched-papers.json
```

**Search queries (auto-generated):**
- "LLM routing model selection"
- "LLM routing complexity estimation"
- "LLM routing adaptive inference"
- "cost optimization LLM 2026"
- "inference optimization language models"
- "query complexity classification"

**What this does:**
- Queries arXiv cs.AI category
- Filters to papers published in last 90 days
- Extracts: arXiv ID, title, abstract, authors, URLs
- Saves to JSON for processing

**Example output:**
```json
[
  {
    "arxiv_id": "2601.XXXXX",
    "title": "Optimal LLM Routing via Complexity Thresholds",
    "published": "2026-01-15T00:00:00",
    "summary": "We propose a novel approach...",
    "authors": ["Author Name"],
    "url": "https://arxiv.org/abs/2601.XXXXX",
    "pdf_url": "https://arxiv.org/pdf/2601.XXXXX",
    "categories": ["cs.AI", "cs.LG"]
  }
]
```

### Step 3: Log Papers (Manual Quality Control)

```bash
# Log Tier 1 papers for lineage tracking
for arxiv_id in $(jq -r '.[].arxiv_id' fetched-papers.json); do
  python3 ~/researchgravity/log_url.py \
    "https://arxiv.org/abs/$arxiv_id" \
    --tier 1 \
    --category research \
    --relevance 5
done
```

**What this does:**
- Registers papers in ResearchGravity tracking
- Tier 1 = Primary source (arXiv)
- Creates lineage for traceability

### Step 4: Extract Insights (LLM-Assisted)

```bash
python3 ~/researchgravity/routing-research-sync.py extract-insights \
  --papers fetched-papers.json \
  --focus "complexity thresholds,cost optimization,accuracy metrics" \
  --model sonnet \
  --output routing-insights.json
```

**What this does:**
- For each paper, calls Claude Sonnet with analysis prompt
- Extracts: thresholds, cost insights, accuracy metrics, strategies
- Assigns confidence score and applicability rating
- Saves structured insights as JSON

**Example insight:**
```json
{
  "thresholds": {
    "haiku": {"max": 0.32},
    "sonnet": {"max": 0.68},
    "opus": {"max": 1.0}
  },
  "cost_insights": [
    "Haiku maintains 92% accuracy at complexity 0.30-0.32",
    "Cost reduction of 25% vs always-largest-model"
  ],
  "accuracy": {
    "metric": "routing_accuracy",
    "value": 0.85
  },
  "strategies": [
    "Use token-based complexity estimation",
    "Prefer cheaper model if DQ scores within 0.05"
  ],
  "rationale": "Empirical study on 10K queries shows haiku underutilized",
  "confidence": 0.82,
  "applicability": "high",
  "source_paper": "2601.XXXXX",
  "title": "Optimal LLM Routing..."
}
```

### Step 5: Review Insights

```bash
# Human-readable view
cat routing-insights.json | jq '.[] | {
  paper: .source_paper,
  title: .title,
  confidence: .confidence,
  thresholds: .thresholds,
  rationale: .rationale
}'

# Check confidence distribution
cat routing-insights.json | jq '[.[] | .confidence] | add / length'
```

**Quality checks:**
- Confidence >0.6? (Filter low-confidence insights)
- Applicability = "high"? (Skip theoretical-only)
- Threshold changes >5%? (Ignore minor adjustments)

### Step 6: Preview Baseline Updates (Dry Run)

```bash
python3 ~/researchgravity/routing-research-sync.py update-baselines \
  --insights routing-insights.json \
  --dry-run
```

**What this does:**
- Loads current `baselines.json`
- Compares proposed thresholds with current values
- Calculates significance of changes (>5% threshold)
- Shows preview without modifying files

**Example output:**
```
📊 Analyzing 5 insights for baseline updates...
  📝 Proposed: complexity_thresholds.haiku.range[1]: 0.30 → 0.32
     Source: arXiv:2601.XXXXX
     Confidence: 0.82
     Rationale: 92% accuracy observed for haiku at 0.30-0.32
  📝 Proposed: complexity_thresholds.sonnet.range[1]: 0.70 → 0.68
     Source: arXiv:2601.YYYYY
     Confidence: 0.75
     Rationale: Sonnet over-provisioned in 15% of cases

✓ Found 2 potential updates

🔍 DRY RUN - No changes applied
Review modifications above and run with --apply to update baselines
```

### Step 7: Apply Updates (If Satisfied)

```bash
python3 ~/researchgravity/routing-research-sync.py update-baselines \
  --insights routing-insights.json \
  --apply
```

**What this does:**
- Applies threshold modifications to `baselines.json`
- Appends research lineage entries
- Updates `last_updated` timestamp
- Increments version (optional)

**Result:**
```json
{
  "version": "1.0.0",
  "last_updated": "2026-01-18T14:30:00Z",
  "research_lineage": [
    {
      "target": "complexity_thresholds.haiku.range[1]",
      "old_value": 0.30,
      "new_value": 0.32,
      "source_paper": "arXiv:2601.XXXXX",
      "paper_title": "Optimal LLM Routing via Complexity Thresholds",
      "rationale": "92% accuracy observed for haiku at complexity 0.30-0.32",
      "confidence": 0.82,
      "applied": "2026-01-18T14:30:00Z"
    }
  ],
  "complexity_thresholds": {
    "haiku": {
      "range": [0.0, 0.32],  // Updated from 0.30
      ...
    }
  }
}
```

### Step 8: Archive Research Session

```bash
python3 ~/researchgravity/archive_session.py
```

**What this does:**
- Archives session to `~/.agent-core/sessions/{session-id}/`
- Extracts findings and creates lineage records
- Auto-generates learnings for `learnings.md`
- Links research → implementation (cli-routing)

---

## Lineage Traceability

### Trace Parameter Origin

```bash
python3 ~/researchgravity/routing-research-sync.py trace \
  --parameter "complexity_thresholds.haiku.range[1]"
```

**Output:**
```json
{
  "parameter": "complexity_thresholds.haiku.range[1]",
  "current_value": 0.32,
  "previous_value": 0.30,
  "last_modified": "2026-01-18T14:30:00Z",
  "source_paper": "arXiv:2601.XXXXX",
  "paper_title": "Optimal LLM Routing via Complexity Thresholds",
  "rationale": "92% accuracy observed for haiku at complexity 0.30-0.32",
  "confidence": 0.82
}
```

### Full Lineage Chain

```
baselines.json:haiku.range[1] = 0.32
  └─ Modified: 2026-01-18T14:30:00Z
      └─ Research Session: llm-routing-opt-20260118-143022-a4f8
          └─ Papers:
              ├─ arXiv:2601.XXXXX - "Optimal LLM Routing"
              └─ arXiv:2512.14142 - "Astraea: State-Aware Scheduling"
          └─ Implementation Project: cli-routing
          └─ Applied By: routing-research-sync.py
          └─ Confidence: 0.82
```

---

## Integration with ResearchGravity

### Session Lifecycle

1. **init_session.py** - Initialize with `--impl-project cli-routing`
2. **log_url.py** - Log papers as Tier 1 sources
3. **routing-research-sync.py** - Fetch, analyze, update
4. **archive_session.py** - Archive with lineage

### Data Flow

```
arXiv API
  ↓
fetched-papers.json
  ↓
routing-research-sync.py extract-insights
  ↓
routing-insights.json
  ↓
routing-research-sync.py update-baselines
  ↓
baselines.json (updated with lineage)
  ↓
dq-scorer.js (loads baselines on startup)
  ↓
Improved routing decisions
```

---

## Scheduled Research Sync (Cron)

Add to crontab for automatic updates:

```bash
# Weekly: Fetch new routing papers (Mondays 9 AM)
0 9 * * 1 cd ~/researchgravity && python3 routing-research-sync.py fetch-papers --query "LLM routing" --days 7 --output /tmp/routing-papers-weekly.json

# Monthly: Full research cycle (1st of month, 10 AM)
0 10 1 * * cd ~/researchgravity && bash -c 'python3 init_session.py "Monthly routing research $(date +\%Y-\%m)" && python3 routing-research-sync.py fetch-papers --query "LLM routing" --days 30 --output /tmp/papers.json'
```

---

## Best Practices

### Research Quality

1. **Confidence threshold**: Only apply insights with confidence >0.6
2. **Applicability filter**: Skip "low" applicability papers
3. **Significance threshold**: Only update if change >5%
4. **Manual review**: Always preview with `--dry-run` first

### Baseline Updates

1. **Incremental changes**: Small adjustments (±0.02) preferred over large jumps
2. **A/B testing**: Test changes via metrics before permanent update
3. **Rollback plan**: Keep baseline backups for quick reversion
4. **Version tracking**: Increment version on major updates

### Lineage Tracking

1. **Full attribution**: Always link to source paper
2. **Rationale required**: Document why change was made
3. **Confidence scores**: Track quality of evidence
4. **Temporal ordering**: Maintain chronological lineage

---

## Troubleshooting

### No papers found

```bash
# Check arXiv is accessible
python3 -c "import arxiv; print('arxiv installed')"

# Try broader query
python3 routing-research-sync.py fetch-papers --query "LLM" --days 180
```

### LLM call failures

```bash
# Check claude CLI works
claude --model sonnet -p "test"

# Use haiku for faster (cheaper) analysis
python3 routing-research-sync.py extract-insights \
  --papers papers.json \
  --model haiku
```

### No insights extracted

- Check paper abstracts are substantive
- Adjust focus areas to match paper content
- Lower confidence threshold in manual review

### Baseline updates rejected

- Ensure changes are >5% (significance threshold)
- Check current value vs proposed value
- Verify confidence score >0.6

---

## Example: Complete Research Cycle

```bash
#!/bin/bash
# Complete research-driven optimization cycle

cd ~/researchgravity

# 1. Initialize
python3 init_session.py "LLM routing optimization $(date +%Y-%m)" \
  --impl-project cli-routing

# 2. Fetch papers
python3 routing-research-sync.py fetch-papers \
  --query "LLM routing OR model selection OR adaptive inference" \
  --days 90 \
  --output /tmp/routing-papers.json

# 3. Log papers
for arxiv_id in $(jq -r '.[].arxiv_id' /tmp/routing-papers.json); do
  python3 log_url.py "https://arxiv.org/abs/$arxiv_id" \
    --tier 1 --category research --relevance 5
done

# 4. Extract insights
python3 routing-research-sync.py extract-insights \
  --papers /tmp/routing-papers.json \
  --focus "complexity thresholds,cost optimization,accuracy" \
  --model sonnet \
  --output /tmp/routing-insights.json

# 5. Review insights
echo "=== EXTRACTED INSIGHTS ==="
cat /tmp/routing-insights.json | jq '.[] | {paper, confidence, thresholds}'

# 6. Preview updates
echo "=== PROPOSED BASELINE UPDATES ==="
python3 routing-research-sync.py update-baselines \
  --insights /tmp/routing-insights.json \
  --dry-run

# 7. Prompt for confirmation
read -p "Apply updates? (y/N): " confirm
if [[ "$confirm" == "y" ]]; then
  python3 routing-research-sync.py update-baselines \
    --insights /tmp/routing-insights.json \
    --apply
  echo "✓ Baselines updated"
fi

# 8. Archive session
python3 archive_session.py

echo "✓ Research cycle complete"
```

---

## Next Steps

After completing research integration:

1. **Monitor performance**: Use `routing-report 7` to track impact
2. **A/B testing**: Compare old vs new thresholds (Phase 4)
3. **Meta-analysis**: Let meta-analyzer propose further optimizations
4. **Continuous improvement**: Schedule weekly/monthly research syncs

---

**Status: Phase 3 Complete | Research-Driven Optimization Active**
