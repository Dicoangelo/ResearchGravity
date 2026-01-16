# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ResearchGravity is a Python research session tracking framework with auto-capture, lineage tracking, and multi-tier source management.

## Commands

```bash
# Session management
python3 status.py                              # Check session state (always run first)
python3 init_session.py "topic"                # Start new research session
python3 init_session.py "topic" --impl-project os-app  # Pre-link to implementation project

# URL logging
python3 log_url.py <url> --tier 1 --category research --relevance 5 --used
python3 log_url.py <url> --tier 2 --category industry --relevance 4

# Session lifecycle
python3 archive_session.py                     # Archive completed session
python3 session_tracker.py status              # Check auto-capture status
python3 session_tracker.py link <session-id> <project>  # Link session to project

# Context loading
python3 project_context.py                     # Auto-detect from current directory
python3 project_context.py --list              # List all projects
python3 project_context.py --index             # View unified index

# Backfill
python3 auto_capture.py scan --hours 48        # Scan recent history
python3 auto_capture.py backfill <path> --topic "..."  # Recover from old session

# Context Prefetcher (v3.4)
python3 prefetch.py                            # Auto-detect project, inject context
python3 prefetch.py --project os-app --papers  # Specific project with papers
python3 prefetch.py --topic multi-agent        # Filter by topic
python3 prefetch.py --clipboard                # Copy to clipboard
python3 prefetch.py --inject                   # Inject into ~/CLAUDE.md

# Learnings Backfill (v3.4)
python3 backfill_learnings.py                  # Regenerate learnings.md from all sessions
python3 backfill_learnings.py --since 7        # Last 7 days only
python3 backfill_learnings.py --dry-run        # Preview without writing
```

## Architecture

```
researchgravity/           # Scripts (this repo)
├── prefetch.py            # Context prefetcher for Claude sessions (v3.4)
├── backfill_learnings.py  # Extract learnings from archived sessions (v3.4)
├── init_session.py        # Session initialization
├── session_tracker.py     # Auto-capture engine
├── auto_capture.py        # Backfill historical sessions
├── project_context.py     # Project context loader
├── log_url.py             # URL logging
├── status.py              # Cold start checker
├── archive_session.py     # Session archival
├── sync_environments.py   # Cross-environment sync
└── SKILL.md               # Agent Core documentation

~/.agent-core/             # Data (single source of truth)
├── projects.json          # Project registry
├── session_tracker.json   # Auto-capture state
├── research/              # Per-project research files
│   ├── INDEX.md           # Unified cross-reference index
│   ├── careercoach/
│   ├── os-app/
│   └── metaventions/
├── sessions/              # Archived sessions
│   └── [session-id]/
│       ├── session.json
│       ├── full_transcript.txt
│       ├── urls_captured.json
│       ├── findings_captured.json
│       └── lineage.json
├── memory/
│   ├── learnings.md       # Extracted learnings archive (v3.4)
│   ├── global.md
│   └── projects/
└── workflows/
```

## Cold Start Protocol

Always run `status.py` first when starting a session. It shows:
- Active session state
- URLs logged, findings count, thesis status
- Recent archived sessions

## Source Hierarchy

**Tier 1 (Primary):** arXiv, HuggingFace Papers, OpenAI, Anthropic, Google AI, Meta AI, DeepMind, TechCrunch, The Verge

**Tier 2 (Amplifiers):** GitHub Trending, METR, ARC Prize, LMSYS, X/Twitter key accounts, HN, Reddit ML

**Tier 3 (Context):** Import AI, The Batch, Latent Space, LessWrong, Alignment Forum

## Research Workflow

1. **Signal Capture (30 min):** Scan Tier 1 sources, log all URLs via `log_url.py`
2. **Synthesis (20 min):** Group by theme, identify gaps, draft thesis
3. **Editorial Frame (10 min):** Write summary, link findings with rationale

## Lineage Tracking

Link research sessions to implementation projects for traceability:
```bash
python3 init_session.py "topic" --impl-project os-app
python3 session_tracker.py link <session-id> <project>
```

## Integration

Works across:
- **CLI (Claude Code):** Planning, parallel sessions, synthesis
- **Antigravity (VSCode):** Coding, preview, browser research
- **Web (claude.ai):** Handoff, visual review
