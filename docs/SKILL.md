---
name: agent-core
description: Unified agent orchestration for research, innovation scouting, and memory across CLI (Claude Code), Antigravity (VSCode OSS), and web. Use for multi-environment coordination, parallel sessions, persistent URL logging, and long-term memory. Triggers on "/innovation-scout", "/deep-research", "/remember", "/recall", "/sync", "/archive", "/parallel", or research requests.
---

# Agent Core v3.2 — Metaventions AI

**v3.2 Update:** Consolidated architecture, project registry, unified research index.
**v3.1 Update:** Automatic session capture and cross-project lineage tracking.

Frontier intelligence for meta-invention. Research that compounds.

## Quick Reference

```
Mode:        Auto-accept (always)
Model:       Opus 4.5 with thinking
Cadence:     Daily or per-session
Standard:    Metaventions-grade
```

---

## Cold Start Protocol

When user invokes "researchgravity" or similar, ALWAYS:

### 1. Check Session State
```bash
# Check for active session
cat .agent/research/session.json 2>/dev/null

# Check for recent archived sessions
ls -lt ~/.agent-core/sessions/ | head -5
```

### 2. Present Options
```
ResearchGravity — Metaventions AI

Current state:
☐ Active session: [topic] (last updated: [time])
  OR
☐ No active session

Recent sessions:
1. [session-id] — [topic] — [date]
2. [session-id] — [topic] — [date]

Options:
→ Continue active session
→ Resume archived session: [id]
→ Start fresh on new topic
```

### 3. Wait for User Choice
Do NOT assume. Ask:
- "Continue current session, resume an old one, or start fresh?"

### 4. Then Proceed
- Continue → Load session, show where we left off
- Resume → `python3 init_session.py --continue [session-id]`
- Fresh → Ask for topic, then `python3 init_session.py "[topic]"`

---

## Automatic Session Capture (v3.1)

**All research sessions are now automatically tracked.** No more lost research.

### What Gets Captured Automatically

| Artifact | Capture Method | Storage |
|----------|---------------|---------|
| **Full transcript** | Claude session linked | `~/.agent-core/sessions/[id]/full_transcript.txt` |
| **All URLs** | Extracted from transcript | `urls_captured.json` |
| **Key findings** | Pattern-matched extraction | `findings_captured.json` |
| **Lineage** | Cross-project links | `lineage.json` |

### How It Works

```
1. init_session.py → Auto-registers with tracker
2. During research → Claude transcript accumulates
3. On archive/capture → Full session extracted and stored
4. On project switch → Lineage link created automatically
```

### Pre-Link to Implementation

When starting research that will feed into a specific project:

```bash
python3 init_session.py "multi-agent orchestration" --impl-project OS-App
```

This creates a pending lineage link that gets completed when you move to implementation.

### Manual Capture (if needed)

```bash
# Capture current session state
python3 session_tracker.py capture

# Link research to implementation
python3 session_tracker.py link [research-session-id] [impl-project]

# Check status
python3 session_tracker.py status
```

### Backfill Historical Sessions

Recover research from old Claude sessions that weren't tracked:

```bash
# Scan last 24 hours
python3 auto_capture.py scan --hours 24

# Backfill specific session file
python3 auto_capture.py backfill ~/.claude/projects/-Users-foo/abc123.jsonl --topic "my research topic"
```

### Session Lineage

Every research session can be linked to its implementation:

```
ResearchGravity Session          →    Implementation Project
─────────────────────────────         ────────────────────────
multi-agent-orchestration-...    →    OS-App (ACE feature)
agentic-ai-framework-...         →    OS-App (agent system)
```

View lineage:
```bash
python3 session_tracker.py status
```

---

## Philosophy

> "Meta-invention: inventions that enable other inventions."

Research quality is measured by:
1. **Signal density** — Every item must move understanding forward
2. **Source diversity** — No single-source blindspots
3. **Editorial synthesis** — Raw links are not research
4. **Thesis framing** — What does this mean? What's the gap?
5. **Compounding potential** — Does this enable future innovation?

---

## Source Hierarchy

### Tier 1: Primary Sources (Check Daily)

| Category | Sources | Query Pattern |
|----------|---------|---------------|
| **Research** | arXiv (cs.AI, cs.SE, cs.LG), HuggingFace Papers | `site:arxiv.org [topic] 2026` |
| **Labs** | OpenAI Blog, Anthropic News, Google AI Blog, Meta AI | `site:openai.com OR site:anthropic.com [topic]` |
| **Industry** | TechCrunch, The Verge, Ars Technica, Wired | `site:techcrunch.com [topic] January 2026` |

### Tier 2: Signal Amplifiers

| Category | Sources | Query Pattern |
|----------|---------|---------------|
| **GitHub** | Trending, Topics, Releases | `[topic] stars:>500 pushed:>30d` |
| **Benchmarks** | METR, ARC Prize, LMSYS, PapersWithCode | `site:metr.org OR site:arcprize.org` |
| **Social** | X/Twitter key accounts, HN, Reddit ML | Key figures: @kaboris, @karpathy, @ylecun |

### Tier 3: Deep Context

| Category | Sources | Query Pattern |
|----------|---------|---------------|
| **Newsletters** | Import AI, The Batch, Latent Space | Benchmark calibration |
| **Podcasts** | Latent Space, Gradient Dissent, Lex Fridman | Long-form context |
| **Forums** | LessWrong, EA Forum, Alignment Forum | Frontier safety discourse |

---

## Source URLs (Copy-Paste Ready)

### Daily Scan
```
https://arxiv.org/list/cs.AI/new
https://arxiv.org/list/cs.LG/new
https://arxiv.org/list/cs.SE/new
https://huggingface.co/papers/trending
https://news.ycombinator.com/
https://github.com/trending
```

### Lab Blogs
```
https://openai.com/news/
https://www.anthropic.com/news
https://blog.google/technology/ai/
https://ai.meta.com/blog/
https://deepmind.google/discover/blog/
```

### Industry News
```
https://techcrunch.com/category/artificial-intelligence/
https://arstechnica.com/ai/
https://www.theverge.com/ai-artificial-intelligence
```

### Benchmarks & Trackers
```
https://metr.org/blog/
https://arcprize.org/blog
https://lmarena.ai/
https://paperswithcode.com/sota
```

---

## Research Workflow

### Phase 1: Signal Capture (30 min)

```
1. Scan Tier 1 sources for last 24-48 hours
2. Log ALL URLs (used or not) via log_url.py
3. Tag each with: source_tier, category, relevance (1-5)
```

### Phase 2: Synthesis (20 min)

```
1. Group findings by theme (not source)
2. Identify the GAP — what's missing from current landscape?
3. Draft thesis statement: "X is happening because Y, which means Z"
```

### Phase 3: Editorial Frame (10 min)

```
1. Write 1-paragraph summary with thesis
2. Each finding gets: [Name](URL) + quantitative signal + 1-line rationale
3. End with: "Innovation opportunity: ..."
```

---

## Filters (Updated)

### Viral Filter (High Adoption)
```
[topic] stars:>500 pushed:>[30 days ago]
```

### Groundbreaker Filter (Emerging)
```
[topic] stars:10..200 created:>[90 days ago]
```

### Frontier Filter (Bleeding Edge)
```
site:arxiv.org [topic] submitted:[last 48 hours]
```

### Protocol Filter (Infrastructure)
```
"protocol" OR "standard" OR "specification" [topic] 2026
```

---

## Output Standards

### Every Finding MUST Include:

1. **Link**: `[Name](URL)` — direct, not search results
2. **Signal**: Quantitative (stars, citations, date, benchmark score)
3. **Rationale**: One sentence — why does this matter?
4. **Category**: research | industry | benchmark | social | lab

### Synthesis MUST Include:

1. **Thesis**: What's the pattern across findings?
2. **Gap**: What's missing that represents opportunity?
3. **Innovation Direction**: Concrete next step

---

## Architecture (v3.2 Consolidated)

**Single source of truth:** `~/.agent-core/` for all persistent data.

```
~/.agent-core/                    # ALL RESEARCH DATA
├── config.json                   # System settings
├── projects.json                 # Project registry (v3.2)
├── session_tracker.json          # Auto-capture state
├── auto_capture_log.json         # Capture history
│
├── sessions/                     # Archived research sessions
│   ├── index.md
│   └── [session-id]/
│       ├── session.json
│       ├── full_transcript.txt
│       ├── urls_captured.json
│       ├── findings_captured.json
│       └── lineage.json
│
├── research/                     # Project research files (v3.2)
│   ├── INDEX.md                  # Unified research index
│   ├── careercoach/
│   │   ├── innovation-scout.md
│   │   └── sources.csv
│   ├── os-app/
│   │   ├── os_app_master_proposal.md
│   │   └── ...
│   └── metaventions/
│       └── metaventions_scout.md
│
├── memory/
│   ├── global.md                 # System-wide facts
│   └── projects/                 # Per-project identities (v3.2)
│       └── careercoach.md
│
└── workflows/                    # Workflow definitions
    ├── deep-research.md
    └── ...

~/.antigravity/                   # IDE CONFIG ONLY (not for research)
├── extensions/                   # VSCode extensions
└── argv.json                     # IDE settings

~/Desktop/Antigravity/ResearchGravity/   # SCRIPTS (git repo)
├── init_session.py               # Initialize + auto-register
├── log_url.py                    # Manual URL logging
├── archive_session.py            # Archive + capture
├── session_tracker.py            # Auto-capture engine
├── auto_capture.py               # Backfill + scan
├── project_context.py            # Project context loader (v3.2)
├── status.py                     # Cold start checker
├── SKILL.md                      # This file
└── .agent/research/              # LOCAL active session only
```

**Key Principle:** `~/.antigravity/` is for IDE config only. Never store research there.

---

## Commands

| Command | Script | Description |
|---------|--------|-------------|
| `/innovation-scout [topic]` | workflows/ | Multi-source frontier scan |
| `/deep-research [topic]` | workflows/ | Full Metaventions-grade investigation |
| `/remember [fact]` | workflows/ | Store to memory |
| `/recall [query]` | workflows/ | Query memory |
| `/parallel [task]` | workflows/ | Coordinate parallel sessions |
| `/sync` | sync_environments.py | Push/pull state |
| `/archive` | archive_session.py | Close session |
| `/status` | sync_environments.py status | Show state |
| `/capture` | session_tracker.py capture | Capture current session |
| `/link [session] [project]` | session_tracker.py link | Create lineage link |
| `/backfill` | auto_capture.py scan | Recover historical sessions |
| `/context` | project_context.py | Load project context from cwd |
| `/projects` | project_context.py --list | List all registered projects |
| `/index` | project_context.py --index | Show unified research index |

---

## Session Workflow

### 1. Initialize
```bash
python3 init_session.py "topic" --workflow deep-research
```

### 2. Research
```bash
# Scan sources (use generated queries from session_log.md)
# Log EVERY URL
python3 log_url.py <url> --tier 1 --category research --relevance 5 --used

# Checkpoint findings
# Update scratchpad.json with findings array
```

### 3. Synthesize
```bash
# Write thesis + gap + innovation direction
# Update session_log.md with editorial synthesis
```

### 4. Archive
```bash
python3 archive_session.py
```

---

## URL Logging (Critical)

**Log ALL URLs** — used or not, succeeded or failed:

```bash
# Tier 1 research paper you used
python3 log_url.py https://arxiv.org/abs/2601.05918 \
  --tier 1 --category research --relevance 5 --used

# Industry news you skipped
python3 log_url.py https://techcrunch.com/... \
  --tier 1 --category industry --relevance 2 --skipped --notes "Not relevant"

# Social signal from key figure
python3 log_url.py https://x.com/karpathy/status/... \
  --tier 2 --category social --relevance 4 --used
```

---

## Quality Checklist

Before archiving a session, verify:

- [ ] Scanned all Tier 1 sources for timeframe
- [ ] Logged 10+ URLs minimum
- [ ] Identified at least one GAP
- [ ] Wrote thesis statement
- [ ] Each finding has: link + signal + rationale
- [ ] Innovation direction is concrete, not vague

---

## Environments

| Environment | Type | Shortcuts | Best For |
|-------------|------|-----------|----------|
| **CLI** | `claude` command | — | Planning, parallel sessions, synthesis |
| **Antigravity** | VSCode OSS | ⌘E ⌘L ⌘I | Coding, preview, browser research |
| **Web** | claude.ai/code | — | Handoff, visual review |

### Antigravity Shortcuts
- **⌘E** — Switch to Agent Manager
- **⌘L** — Code with Agent
- **⌘I** — Edit code inline

---

## Parallel Sessions (Boris's Pattern)

```
Tab 1: Planning/Orchestration (this tab)
Tab 2: Tier 1 Research (arXiv, HuggingFace)
Tab 3: Tier 1 Industry (TechCrunch, labs)
Tab 4: Tier 2 GitHub + Benchmarks
Tab 5: Synthesis + Writing
```

---

## Setup

```bash
chmod +x setup.sh && ./setup.sh
```

---

## Contact

**Metaventions AI**
Dico Angelo
dicoangelo@metaventionsai.com
