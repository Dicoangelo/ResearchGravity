<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,100:00d9ff&height=200&section=header&text=ResearchGravity&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Metaventions%20AI%20Research%20Framework&descSize=20&descAlignY=55" />
</p>

<p align="center">
  <strong>Frontier intelligence for meta-invention. Research that compounds.</strong>
</p>

<p align="center">
  <em>"Let the invention be hidden in your vision"</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-3.5.0-00d9ff?style=for-the-badge" alt="Version" />
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Sessions-54+-4a0080?style=for-the-badge" alt="Sessions" />
  <img src="https://img.shields.io/badge/Papers-306+-00d9ff?style=for-the-badge" alt="Papers" />
  <img src="https://img.shields.io/badge/URLs-3,200+-1a1a2e?style=for-the-badge" alt="URLs" />
  <img src="https://img.shields.io/badge/Concepts-479+-success?style=for-the-badge" alt="Concepts" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Metaventions_AI-Architected_Intelligence-1a1a2e?style=for-the-badge" alt="Metaventions AI" />
</p>

---

## Why • What's New • Architecture • Quick Start • Auto-Capture • Sources • Contact

---

## What's New in v3.5 (January 2026)

| Feature | Description |
|---------|-------------|
| **Precision Bridge Research** | Tesla US20260017019A1 → RLM synthesis methodology |
| **Cognitive Wallet Tracking** | 54 sessions, 306 papers, 3,200+ URLs accumulated |
| **Deep Dive Workflow** | Multi-paper synthesis with implementation output |
| **Framework Extraction** | COMPRESS → EXPLORE → RECONSTRUCT pattern identified |

### Notable Research Sessions

| Session | Papers | Output |
|---------|--------|--------|
| Tesla Mixed-Precision RoPE | 15 arXiv | `recursiveLanguageModel.ts` implementation |
| Multi-Agent Orchestration | 12 arXiv | ACE/DQ Scoring in OS-App |
| 160+ Papers Meta-Synthesis | 160+ | Unified research index |

## What's New in v3.4

| Feature | Description |
|---------|-------------|
| **Context Prefetcher** | `prefetch.py` — Inject relevant learnings into Claude sessions |
| **Learnings Backfill** | `backfill_learnings.py` — Extract learnings from all archived sessions |
| **Memory Injection** | Auto-load project context, papers, and lineage at session start |
| **Shell Integration** | `prefetch`, `prefetch-clip`, `prefetch-inject` shell commands |

### v3.3 Changelog

| Feature | Description |
|---------|-------------|
| **YouTube Research** | `youtube_channel.py` — Channel analysis and transcript extraction |
| **Enhanced Backfill** | Improved session recovery with better transcript parsing |
| **Ecosystem Sync** | Deeper integration with Agent Core orchestration |

### v3.2 Changelog

| Feature | Description |
|---------|-------------|
| **Auto-Capture** | Sessions automatically tracked — URLs, findings, full transcripts extracted |
| **Lineage Tracking** | Link research sessions to implementation projects |
| **Project Registry** | 4 registered projects with cross-referenced research |
| **Context Loader** | Auto-load project context from any directory |
| **Unified Index** | Cross-reference by paper, topic, or session |
| **Backfill** | Recover research from historical Claude sessions |

---

## Why ResearchGravity?

Traditional research workflows fail at the frontier:

| Problem | Impact |
|---------|--------|
| Single-source blindspots | Missing critical signals |
| No synthesis | Raw links ≠ research |
| No session continuity | Context lost between sessions |
| No quality standard | Inconsistent output |

**ResearchGravity** solves this with:

- **Multi-tier source hierarchy** — Tier 1 (primary), Tier 2 (amplifiers), Tier 3 (context)
- **Cold Start Protocol** — Never lose session context
- **Synthesis workflow** — Thesis → Gap → Innovation Direction
- **Quality checklist** — Consistent Metaventions-grade output

---

## Architecture

```
ResearchGravity/                    # SCRIPTS (git repo)
├── prefetch.py                     # Context prefetcher for Claude sessions (v3.4)
├── backfill_learnings.py           # Extract learnings from archived sessions (v3.4)
├── init_session.py                 # Initialize + auto-register sessions
├── session_tracker.py              # Auto-capture engine (v3.1)
├── auto_capture.py                 # Backfill historical sessions (v3.1)
├── project_context.py              # Project context loader (v3.2)
├── youtube_channel.py              # YouTube research & transcripts (v3.3)
├── log_url.py                      # Manual URL logging
├── status.py                       # Cold start session checker
├── archive_session.py              # Archive completed sessions
├── sync_environments.py            # Cross-environment sync
└── SKILL.md                        # Agent Core v3.4 documentation

~/.agent-core/                      # DATA (single source of truth)
├── projects.json                   # Project registry (v3.2)
├── session_tracker.json            # Auto-capture state
├── research/                       # Project research files
│   ├── INDEX.md                    # Unified cross-reference index
│   ├── careercoach/
│   ├── os-app/
│   └── metaventions/
├── sessions/                       # Archived sessions
│   └── [session-id]/
│       ├── session.json
│       ├── full_transcript.txt
│       ├── urls_captured.json
│       ├── findings_captured.json
│       └── lineage.json
├── memory/
│   ├── learnings.md                # Extracted learnings archive (v3.4)
│   ├── global.md
│   └── projects/
└── workflows/
```

---

## Quick Start

### 1. Check Session State
```bash
python3 status.py
```

### 2. Initialize New Session
```bash
# Basic session
python3 init_session.py "your research topic"

# Pre-link to implementation project (v3.1)
python3 init_session.py "multi-agent consensus" --impl-project os-app
```

### 3. Research & Log URLs
```bash
# Log a Tier 1 research paper
python3 log_url.py https://arxiv.org/abs/2601.05918 \
  --tier 1 --category research --relevance 5 --used

# Log industry news
python3 log_url.py https://techcrunch.com/... \
  --tier 1 --category industry --relevance 4 --used
```

### 4. Archive When Complete
```bash
python3 archive_session.py
```

### 5. Check Tracker Status (v3.1)
```bash
python3 session_tracker.py status
```

### 6. Load Project Context (v3.2)
```bash
# Auto-detect from current directory
python3 project_context.py

# List all projects
python3 project_context.py --list

# View unified index
python3 project_context.py --index
```

---

## Research Workflow

### Phase 1: Signal Capture (30 min)
```
1. Scan Tier 1 sources for last 24-48 hours
2. Log ALL URLs (used or not) via log_url.py
3. Tag each with: tier, category, relevance (1-5)
```

### Phase 2: Synthesis (20 min)
```
1. Group findings by theme (not source)
2. Identify the GAP — what's missing?
3. Draft thesis: "X is happening because Y, which means Z"
```

### Phase 3: Editorial Frame (10 min)
```
1. Write 1-paragraph summary with thesis
2. Each finding: [Name](URL) + signal + rationale
3. End with: "Innovation opportunity: ..."
```

---

## Source Hierarchy

### Tier 1: Primary Sources (Check Daily)

| Category | Sources |
|----------|---------|
| **Research** | arXiv (cs.AI, cs.SE, cs.LG), HuggingFace Papers |
| **Labs** | OpenAI, Anthropic, Google AI, Meta AI, DeepMind |
| **Industry** | TechCrunch, The Verge, Ars Technica, Wired |

### Tier 2: Signal Amplifiers

| Category | Sources |
|----------|---------|
| **GitHub** | Trending, Topics, Releases |
| **Benchmarks** | METR, ARC Prize, LMSYS, PapersWithCode |
| **Social** | X/Twitter key accounts, HN, Reddit ML |

### Tier 3: Deep Context

| Category | Sources |
|----------|---------|
| **Newsletters** | Import AI, The Batch, Latent Space |
| **Forums** | LessWrong, Alignment Forum |

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

## Cold Start Protocol

When invoking ResearchGravity, always run `status.py` first:

```
==================================================
  ResearchGravity — Metaventions AI
==================================================

📍 ACTIVE SESSION
   Topic: [current topic]
   URLs logged: X | Findings: Y | Thesis: Yes/No

📚 RECENT SESSIONS
   1. [topic] — [date]
   2. [topic] — [date]

--------------------------------------------------
OPTIONS:
  → Continue active session
  → Resume archived session
  → Start fresh
--------------------------------------------------
```

---

## Auto-Capture & Lineage (v3.1)

**All research sessions are now automatically tracked.** No more lost research.

### What Gets Captured

| Artifact | Storage |
|----------|---------|
| Full transcript | `~/.agent-core/sessions/[id]/full_transcript.txt` |
| All URLs | `urls_captured.json` |
| Key findings | `findings_captured.json` |
| Cross-project links | `lineage.json` |

### Lineage Tracking

Link research sessions to implementation projects:

```bash
# Pre-link at session start
python3 init_session.py "multi-agent DQ" --impl-project os-app

# Manual link after research
python3 session_tracker.py link [session-id] [project]
```

### Backfill Historical Sessions

Recover research from old Claude sessions:

```bash
# Scan recent history
python3 auto_capture.py scan --hours 48

# Backfill specific session
python3 auto_capture.py backfill ~/.claude/projects/.../session.jsonl --topic "..."
```

---

## Context Prefetcher (v3.4)

**Memory injection for Claude sessions.** Automatically load relevant learnings, project memory, and research papers at session start.

### Basic Usage

```bash
# Auto-detect project from current directory
python3 prefetch.py

# Specific project with papers
python3 prefetch.py --project os-app --papers

# Filter by topic
python3 prefetch.py --topic multi-agent --days 30

# Copy to clipboard
python3 prefetch.py --project os-app --clipboard

# Inject into ~/CLAUDE.md
python3 prefetch.py --project os-app --inject
```

### Shell Commands

After sourcing `~/.claude/scripts/auto-context.sh`:

```bash
prefetch                    # Auto-detect project, last 14 days
prefetch os-app 7           # Specific project, last 7 days
prefetch-clip               # Copy context to clipboard
prefetch-inject             # Inject into ~/CLAUDE.md
prefetch-topic "consensus"  # Filter by topic across all sessions
backfill-learnings          # Regenerate learnings.md from all sessions
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--project, -p` | Project ID to load context for |
| `--topic, -t` | Filter by topic keywords |
| `--days, -d` | Limit to last N days (default: 14) |
| `--limit, -l` | Max learning entries (default: 5) |
| `--papers` | Include relevant arXiv papers |
| `--clipboard, -c` | Copy to clipboard (macOS) |
| `--inject, -i` | Inject into ~/CLAUDE.md |
| `--json` | Output as JSON |
| `--quiet, -q` | Suppress info output |

### Backfill Learnings

Extract learnings from all archived sessions:

```bash
# Process all sessions
python3 backfill_learnings.py

# Last 7 days only
python3 backfill_learnings.py --since 7

# Specific session
python3 backfill_learnings.py --session <session-id>

# Preview without writing
python3 backfill_learnings.py --dry-run
```

### What Gets Injected

| Component | Source |
|-----------|--------|
| Project info | `projects.json` — name, focus, tech stack, status |
| Project memory | `memory/projects/[project].md` |
| Recent learnings | `memory/learnings.md` — filtered by project/topic/days |
| Research papers | `paper_index` in projects.json |
| Lineage | Research sessions → features implemented |

---

## Integration

ResearchGravity integrates with the **Antigravity ecosystem**:

| Environment | Use Case |
|-------------|----------|
| **CLI** (Claude Code) | Planning, parallel sessions, synthesis |
| **Antigravity** (VSCode) | Coding, preview, browser research |
| **Web** (claude.ai) | Handoff, visual review |

---

## Roadmap

- [x] ~~Auto-capture sessions~~ (v3.1)
- [x] ~~Cross-project lineage tracking~~ (v3.1)
- [x] ~~Project registry & context loader~~ (v3.2)
- [x] ~~Unified research index~~ (v3.2)
- [x] ~~Context prefetcher & memory injection~~ (v3.4)
- [x] ~~Learnings backfill from archived sessions~~ (v3.4)
- [ ] MCP integration for tool context
- [ ] Auto-synthesis via LLM
- [ ] Browser extension for URL capture
- [ ] Team collaboration features

---

## License

MIT License — See [LICENSE](LICENSE)

---

## Contact

**Metaventions AI**
Dico Angelo
dicoangelo@metaventionsai.com

<p align="center">
  <a href="https://metaventions-ai-architected-intelligence-1061986917838.us-west1.run.app/">
    <img src="https://img.shields.io/badge/Metaventions_AI-Website-00d9ff?style=for-the-badge" alt="Website" />
  </a>
  <a href="https://github.com/Dicoangelo">
    <img src="https://img.shields.io/badge/GitHub-Dicoangelo-1a1a2e?style=for-the-badge&logo=github" alt="GitHub" />
  </a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,100:00d9ff&height=100&section=footer" />
</p>
