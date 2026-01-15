# Signal Verification Protocol (SVP)
## Invented: Jan 14, 2026 | Session: Frontier AI Breakthroughs

---

## Problem Statement

News aggregators (HN, Reddit, X) surface articles that APPEAR fresh but are actually:
- Late coverage of older breaks
- Rehashed stories from hours/days prior
- Discussion threads about stale news

**Result:** False positives contaminate research sessions with "old news dressed as new."

---

## The Method: Parallel Verification

### Step 1: Capture Raw Feed
```
Source: HN /newest, Reddit /new, X latest
Capture: Title + exact timestamp (minutes ago)
Filter: AI/ML related only
```

### Step 2: Classify Signal Type
| Type | Definition | Verification Needed |
|------|------------|---------------------|
| **Show HN / Launch** | Creator announcing their work | LOW - likely original |
| **Tell HN / Discussion** | Opinion or analysis | MEDIUM - check if rehash |
| **News Link** | Article about external event | HIGH - must verify origin |

### Step 3: Parallel Origin Search
For each News Link signal, run:
```
Search: "[key terms]" first reported earliest
Goal: Find ORIGINAL break date/source
```

### Step 4: Apply Freshness Criteria
| Verdict | Criteria |
|---------|----------|
| **FRESH** | Original source < target window (e.g., 60 min) |
| **SAME DAY** | Broke today but outside target window |
| **STALE** | Broke yesterday or earlier |
| **REHASH** | Secondary coverage of older story |

### Step 5: Log Only Verified Fresh
- Only signals passing verification get logged
- Note original source + break time
- Flag anything borderline

---

## Verification Query Templates

```
# Origin check
"[company]" "[event]" first reported earliest

# Date verification
"[topic]" announced January [day] 2026

# Source triangulation
site:theinformation.com OR site:reuters.com "[topic]"
```

---

## Signal Categories (Post-Verification)

| Category | Description |
|----------|-------------|
| **Product Launch** | New tool/feature announced by creator |
| **Breaking News** | Event happening NOW, first reports |
| **Funding/M&A** | Deal announced today |
| **Research Drop** | Paper/model released today |
| **Insider Leak** | Non-public info surfacing |

---

## Anti-Patterns (What to Reject)

1. **Late Coverage** - Article published today about yesterday's news
2. **Aggregator Lag** - HN post linking to 2-day-old article
3. **Analysis Pieces** - Commentary on older events
4. **Roundups** - "This week in AI" style compilations
5. **Prediction Posts** - Speculation, not news

---

## Implementation in ResearchGravity

```bash
# When logging verified fresh signals
python3 log_url.py <url> \
  --tier 1 \
  --category <category> \
  --relevance 5 \
  --used \
  --notes "VERIFIED [timestamp]: [original source] broke [time ago]"
```

---

## Session Example

**Target Window:** Last 60 minutes
**Raw Captures:** 10 AI signals from HN
**Post-Verification:** 1 truly fresh (Vercel AI Voice Elements)
**Rejection Rate:** 90%

**Lesson:** Most "fresh" signals are stale. Verification is essential.

---

## Metrics

| Metric | Definition |
|--------|------------|
| **Signal-to-Noise Ratio** | Fresh signals / Total captured |
| **Verification Time** | Seconds per signal to verify |
| **False Positive Rate** | Stale signals initially flagged as fresh |

---

## Evolution Notes

- v1.0: Manual parallel search verification
- Future: Automate with origin-date extraction
- Future: Build source freshness index (which outlets break first)

---

*Protocol developed during live research session. Refine with use.*
