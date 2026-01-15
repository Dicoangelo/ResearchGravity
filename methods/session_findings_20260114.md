# Session Findings: Jan 14, 2026
## Frontier AI Breakthroughs

---

## INVENTIONS (Process)

### Signal Verification Protocol (SVP)
**Problem:** News aggregators surface stale content disguised as fresh.

**Solution:** Parallel origin search before logging any signal.

**Process:**
1. Capture raw feed with timestamps
2. Classify signal type (Launch vs. News Link)
3. Run parallel search: `"[topic]" first reported earliest`
4. Apply freshness verdict: FRESH / SAME DAY / STALE / REHASH
5. Log only verified fresh signals

**Result:** 90% of "fresh" HN posts were actually stale.

**File:** `methods/signal_verification_protocol.md`

---

## VERIFIED FRESH SIGNALS (Last 60 min, strict)

| Signal | Verified |
|--------|----------|
| Vercel AI Voice Elements | ✅ Product launch today |

---

## VERIFIED SAME-DAY SIGNALS (Jan 14, 2026)

| Signal | First Broke | Source |
|--------|-------------|--------|
| Microsoft $500M/yr → Anthropic | ~14 hrs ago | The Information |
| China blocks H200 at customs | Jan 14 morning | Reuters |
| Airbnb hires Meta AI exec as CTO | ~8 hrs ago | Bloomberg |
| Thinking Machines Lab crisis | ~4 hrs ago | TechCrunch |
| Google Personal Intelligence | ~6 hrs ago | Google Blog |

---

## VERIFIED STALE (Rejected)

| Signal | Actually Broke |
|--------|----------------|
| OpenAI Torch acquisition | Jan 12 (2 days old) |
| GPT-5.2 Codex release | Earlier this week |
| Skild AI $1.4B | Jan 14 but widely covered |

---

## KEY LEARNINGS

1. **HN /newest ≠ fresh news** — Most posts discuss older stories
2. **Show HN posts ARE original** — Creator announcements are reliable
3. **Verification takes ~30 sec/signal** — Worth the accuracy
4. **90% rejection rate is normal** — When targeting <60 min window

---

## THESIS (Verified Signals Only)

```
JAN 14, 2026 — WHAT ACTUALLY BROKE TODAY

1. Microsoft making Claude DEFAULT for business
   → $500M/yr spend, parity with own products

2. China weaponizing chip access
   → H200 blocked hours after US approved
   → $54B frozen, pre-Xi-Trump leverage

3. Thinking Machines Lab imploding
   → 3 founders → OpenAI (Zoph fired)
   → $12B startup talent war

4. Voice AI infrastructure emerging
   → Vercel AI Voice Elements launched
   → ElevenLabs $330M ARR, Deepgram unicorn
```

---

## SESSION STATS

| Metric | Value |
|--------|-------|
| URLs logged | 52 |
| Verified fresh (<60 min) | 1 |
| Verified same-day | 5 |
| Rejected as stale | 90%+ |
| Method invented | 1 (SVP) |

---

*Session demonstrates that rigorous verification dramatically improves signal quality.*
