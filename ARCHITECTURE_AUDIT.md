# ResearchGravity Architecture Audit

**Generated:** 2026-01-14
**Purpose:** Map all research storage locations, identify overlaps, clarify architecture, plan consolidation

---

## Executive Summary

**3 storage locations with overlapping purposes:**

| Location | Intended Purpose | Actual State |
|----------|------------------|--------------|
| `~/.antigravity/` | IDE config only | IDE + orphaned project research |
| `~/.agent-core/` | Global research storage | Partial storage + outdated scripts |
| `ResearchGravity/` | Scripts repository | Scripts (v2.0) + local session state |

**Key Problems:**
1. Project-specific research (CareerCoach) dumped into IDE config folder
2. Scripts duplicated in two locations with different versions
3. Workflows duplicated with different levels of detail
4. No clear single source of truth

---

## 1. ~/.antigravity/ — Antigravity IDE Config

### Location
```
/Users/dicoangelo/.antigravity/
```

### What It Is
The **Antigravity IDE** (VSCode OSS fork) data directory — equivalent to `~/.vscode/`

### Structure
```
~/.antigravity/
├── argv.json              # VSCode crash reporter config
├── antigravity/           # IDE internal (empty)
├── extensions/            # VSCode extensions (1.1GB)
│   ├── anthropic.claude-code-2.1.7-darwin-arm64/
│   ├── ms-python.python-2026.0.0-universal/
│   ├── golang.go-0.52.1-universal/
│   └── ... (31 total extensions)
│
│   ⚠️ ORPHANED PROJECT DATA (shouldn't be here):
│
├── research/              # CareerCoachAntigravity research
│   ├── innovation-scout.md
│   ├── metaventions_scout.md
│   ├── os_app_10x_adaptive_ui.md
│   ├── os_app_master_proposal.md
│   ├── os_app_phase1_realtime_fix.md
│   ├── os_app_stabilization_master.md
│   ├── os_app_system_context.md
│   ├── focusproof_extension_migration.md
│   └── sources.csv
│
├── memory/
│   └── global_identity.md  # CareerCoach project identity
│
└── workflows/              # Simple workflow definitions
    ├── deep-research.md
    ├── innovation-scout.md
    ├── remember.md
    └── session-history.md
```

### Research Content Analysis

#### innovation-scout.md
**Topic:** AI Career Intelligence (CAPER, CareerScape, TraceTop)
**Project:** CareerCoachAntigravity
**Key Research:**
- CAPER: Ternary relationships (User-Position-Company)
- CareerScape: Graph-based resume validation
- TraceTop: Skill tech-trees with gamification

#### os_app_master_proposal.md
**Topic:** Agentic & Biometric OS Evolution
**Project:** OS-App
**Key Research:**
- MemOS: LLM-as-Kernel pattern (3.7k stars)
- AFS: Agentic File System (arXiv:2512.05470)
- Gaze-triggered context prefetching

#### sources.csv (Career Graph Research)
```
CAPER (2408.15620) - Temporal Knowledge Graphs
CareerScape (2509.19677) - Multi-layer career validation
TraceTop - Skill tech-trees
OpenResume - Graph benchmarking dataset
React Flow - Node-based visualization
Sigma.js - WebGL large-scale rendering
```

#### global_identity.md
**Content:** CareerCoachAntigravity project identity
- Tech stack: Next.js 14, TypeScript, Vanilla CSS
- Dual-brain: Antigravity + Claude Code CLI
- Active missions: Skill Graph Navigator, Resume Builder

### Workflows Analysis (Simple Versions)

| File | Lines | Focus |
|------|-------|-------|
| `deep-research.md` | 9 | browser_subagent launch |
| `innovation-scout.md` | 22 | Viral + Groundbreaker filters |
| `remember.md` | 6 | Memory storage |
| `session-history.md` | 8 | Tab archiving |

**Note:** These are simplified workflows that reference `.agent/` (project-local) paths and `browser_subagent` (Antigravity IDE feature).

### Verdict
- **Keep:** `extensions/`, `argv.json`, `antigravity/`
- **Migrate:** `research/`, `memory/`, `workflows/`
- **This folder is for IDE config, not research storage**

---

## 2. ~/.agent-core/ — Global Research Storage

### Location
```
/Users/dicoangelo/.agent-core/
```

### What It Is
The **global research storage** for ResearchGravity — meant to persist across all projects.

### Structure
```
~/.agent-core/
├── config.json                    # Multi-environment settings
├── session_tracker.json           # v3.1 auto-capture state (NEW)
├── auto_capture_log.json          # v3.1 capture history (NEW)
│
├── sessions/                      # Archived research sessions
│   ├── index.md
│   ├── agentic-ai-framework-20260109-160753-16a34f/
│   │   ├── session.json
│   │   ├── session_archive.md
│   │   └── innovation_scout_summary.md
│   ├── ai-testing-tooling-b-20260112-201347-427fbe/
│   ├── combinatorial-creati-20260109-180923-bfd4d0/
│   ├── combinatorial-creati-20260109-181002-bfd4d0/
│   ├── sssp-sorting-barrier-20260111-095808-7f516a/
│   ├── test-metaventions-20260112-203725-698f55/
│   ├── universal-verificati-20260112-203333-c338b3/
│   └── backfill-multi-agent-orchestr-20260113-225420-4627fa/
│       ├── session.json
│       ├── full_transcript.txt    (3.2MB)
│       ├── urls_captured.json     (154 URLs)
│       ├── findings_captured.json (88 findings)
│       └── lineage.json           (→ OS-App)
│
├── memory/
│   └── global.md                  # EMPTY template
│
├── scripts/                       # ⚠️ OUTDATED v1.0 scripts
│   ├── init_session.py            (7.9KB - old version)
│   ├── log_url.py                 (4.5KB)
│   ├── archive_session.py         (9.6KB)
│   └── sync_environments.py       (6.5KB)
│
├── specs/
│   └── pgcce-v3.0-spec.md
│
└── workflows/                     # Comprehensive workflow definitions
    ├── deep-research.md           (163 lines - full version)
    ├── innovation-scout.md
    ├── parallel-session.md
    └── research.md
```

### config.json
```json
{
  "version": "1.0",
  "environments": {
    "cli": {"enabled": true, "default_model": "claude", "web_search": true},
    "antigravity": {"enabled": true, "browser_subagent": true, "auto_sync": true}
  },
  "sync": {"enabled": true, "conflict_resolution": "latest_wins", "auto_push": true},
  "logging": {"log_all_urls": true, "checkpoint_interval_minutes": 5}
}
```

### Sessions Archive

| Session ID | Topic | URLs | Findings | Date |
|------------|-------|------|----------|------|
| `agentic-ai-framework-20260109` | Agentic AI Frameworks | 29 | — | Jan 9 |
| `ai-testing-tooling-b-20260112` | AI Testing Tooling | ? | — | Jan 12 |
| `combinatorial-creati-20260109` | Combinatorial Creativity | ? | — | Jan 9 |
| `sssp-sorting-barrier-20260111` | SSSP Algorithm | ? | — | Jan 11 |
| `backfill-multi-agent-orchestr` | **Multi-Agent DQ** | **154** | **88** | Jan 13 |

**Key Session:** `backfill-multi-agent-orchestr`
- Contains arXiv:2511.15755 (Decision Quality Scoring)
- Linked to OS-App (ACE feature)
- Full transcript archived (3.2MB)

### Scripts Version Comparison

| Script | ~/.agent-core/ | ResearchGravity/ | Winner |
|--------|----------------|------------------|--------|
| `init_session.py` | 7.9KB (v1.0) | 18KB (v2.0+tracker) | ResearchGravity |
| `log_url.py` | 4.5KB | 9.5KB | ResearchGravity |
| `archive_session.py` | 9.6KB | 9.6KB | Same |
| `sync_environments.py` | 6.5KB | 6.5KB | Same |
| `session_tracker.py` | ✗ | 17KB (NEW) | ResearchGravity |
| `auto_capture.py` | ✗ | 15KB (NEW) | ResearchGravity |
| `status.py` | ✗ | 3.8KB | ResearchGravity |

### Workflows Analysis (Full Versions)

**deep-research.md (163 lines)**
- Full execution protocol
- Browser subagent + CLI fallback
- URL logging requirements
- Synthesis template
- Environment handoff

**Significantly more detailed than ~/.antigravity/workflows/**

### Verdict
- **Keep:** `sessions/`, `config.json`, `session_tracker.json`
- **Delete:** `scripts/` (use ResearchGravity as canonical)
- **Merge:** `memory/` (needs CareerCoach identity migrated in)
- **Merge:** `workflows/` (already has full versions)

---

## 3. ~/Desktop/Antigravity/ResearchGravity/ — Scripts + Local State

### Location
```
/Users/dicoangelo/Desktop/Antigravity/ResearchGravity/
```

### What It Is
The **ResearchGravity git repository** — contains canonical scripts and local session state.

### Structure
```
ResearchGravity/
├── .git/                          # Git repository
├── .github/                       # GitHub workflows
├── .gitignore
├── LICENSE (MIT)
├── README.md                      # Project documentation
├── CONTRIBUTING.md
├── SKILL.md                       # Agent Core v3.1 skill definition
├── ARCHITECTURE_AUDIT.md          # This file
│
├── init_session.py                # v2.0 with auto-tracking
├── log_url.py                     # URL logging
├── archive_session.py             # Session archival
├── sync_environments.py           # Cross-env sync
├── status.py                      # Session status checker
├── session_tracker.py             # v3.1 auto-capture engine (NEW)
├── auto_capture.py                # v3.1 backfill/scan (NEW)
│
├── setup.sh                       # Setup script
├── agent-core-v2.skill            # Old skill file
├── CLAUDE.md.template             # Project template
│
├── .agent/                        # Local session state
│   └── research/
│       ├── session.json           # Active session
│       ├── session_log.md         # Research narrative
│       ├── scratchpad.json        # Machine-readable
│       ├── sources.csv            # URL log
│       ├── innovation_scout_summary.md
│       └── *_sources.csv
│
├── libs/                          # (empty)
└── temp_sssp_repo/                # Temp clone
```

### Script Versions (Canonical)

| Script | Version | Size | Features |
|--------|---------|------|----------|
| `init_session.py` | v2.0 | 18KB | Auto-tracking, `--impl-project`, lineage |
| `log_url.py` | v1.5 | 9.5KB | Auto-detect source/tier |
| `session_tracker.py` | v3.1 | 17KB | Auto-capture, lineage links |
| `auto_capture.py` | v3.1 | 15KB | Backfill, URL extraction |
| `status.py` | v1.0 | 3.8KB | Cold start protocol |

### Local .agent/research/

**Purpose:** Active session state for current project
**Relationship:** Gets archived to `~/.agent-core/sessions/` when complete

**Current Session:**
```json
{
  "session_id": "test-metaventions-20260112-203725-698f55",
  "topic": "Test Metaventions",
  "status": "active"
}
```

### Verdict
- **Keep:** All scripts (canonical source)
- **Keep:** `.agent/research/` (local active state)
- **Publish:** This is the git repo, should be the single source of truth for code

---

## 4. Overlap Matrix

### Folders

| Folder | ~/.antigravity | ~/.agent-core | ResearchGravity |
|--------|:--------------:|:-------------:|:---------------:|
| `research/` | ✓ (project) | ✗ | ✓ (local) |
| `memory/` | ✓ (project) | ✓ (empty) | ✗ |
| `workflows/` | ✓ (simple) | ✓ (full) | ✗ |
| `sessions/` | ✗ | ✓ | ✗ |
| `scripts/` | ✗ | ✓ (old) | ✓ (new) |
| `extensions/` | ✓ (IDE) | ✗ | ✗ |

### Key Files

| File | ~/.antigravity | ~/.agent-core | Notes |
|------|:--------------:|:-------------:|-------|
| `config.json` | ✗ | ✓ | Keep in agent-core |
| `argv.json` | ✓ (IDE) | ✗ | IDE config |
| `global.md` / `global_identity.md` | ✓ (content) | ✓ (empty) | Merge |
| `deep-research.md` | ✓ (9 lines) | ✓ (163 lines) | Keep agent-core |
| `innovation-scout.md` (workflow) | ✓ | ✓ | Keep agent-core |
| `innovation-scout.md` (research) | ✓ | ✗ | Migrate to sessions |

### Scripts

| Script | agent-core/scripts/ | ResearchGravity/ | Action |
|--------|:-------------------:|:----------------:|--------|
| `init_session.py` | v1.0 (7.9KB) | **v2.0 (18KB)** | Delete agent-core |
| `log_url.py` | v1.0 (4.5KB) | **v1.5 (9.5KB)** | Delete agent-core |
| `archive_session.py` | Same | Same | Delete agent-core |
| `sync_environments.py` | Same | Same | Delete agent-core |
| `session_tracker.py` | ✗ | **v3.1** | Keep ResearchGravity |
| `auto_capture.py` | ✗ | **v3.1** | Keep ResearchGravity |

---

## 5. Data at Risk (Would Be Lost Without Migration)

### In ~/.antigravity/research/ ONLY

| File | Content | Project |
|------|---------|---------|
| `innovation-scout.md` | CAPER, CareerScape, TraceTop research | CareerCoach |
| `metaventions_scout.md` | Agentic OS research | Metaventions |
| `os_app_*.md` (5 files) | OS-App proposals & plans | OS-App |
| `focusproof_extension_migration.md` | FocusProof extraction plan | OS-App |
| `sources.csv` | Career graph arXiv papers | CareerCoach |

### In ~/.antigravity/memory/ ONLY

| File | Content | Project |
|------|---------|---------|
| `global_identity.md` | CareerCoach identity, tech stack, missions | CareerCoach |

### Summary
**9 research files + 1 identity file** exist only in `~/.antigravity/` and would be lost if that folder is cleaned without migration.

---

## 6. Recommended Target Architecture

### Clean State
```
~/.antigravity/                    # IDE CONFIG ONLY
├── extensions/                    # VSCode extensions
├── argv.json                      # IDE settings
└── antigravity/                   # IDE internal

~/.agent-core/                     # ALL RESEARCH DATA
├── config.json                    # System settings
├── session_tracker.json           # Auto-capture state
├── auto_capture_log.json          # Capture history
│
├── sessions/                      # All archived sessions
│   └── [session-id]/
│       ├── session.json
│       ├── full_transcript.txt
│       ├── urls_captured.json
│       ├── findings_captured.json
│       └── lineage.json
│
├── memory/                        # Global memory
│   ├── global.md                  # System-wide facts
│   └── projects/                  # Per-project identities
│       ├── careercoach.md
│       └── os-app.md
│
├── research/                      # Migrated research files
│   ├── careercoach/
│   │   ├── innovation-scout.md
│   │   └── sources.csv
│   └── os-app/
│       ├── os_app_master_proposal.md
│       └── ...
│
└── workflows/                     # Workflow definitions
    ├── deep-research.md
    ├── innovation-scout.md
    └── ...

~/Desktop/Antigravity/ResearchGravity/   # SCRIPTS ONLY (git repo)
├── *.py                           # Canonical scripts
├── SKILL.md                       # Skill definition
├── README.md                      # Documentation
└── .agent/research/               # LOCAL active session state
```

### Key Principles
1. **One storage location:** `~/.agent-core/` for all persistent data
2. **One script source:** `ResearchGravity/` for all code
3. **Project separation:** Research files organized by project
4. **IDE isolation:** `~/.antigravity/` only for IDE config

---

## 7. Migration Actions

### Phase 1: Backup
```bash
# Backup everything before changes
cp -r ~/.antigravity ~/antigravity_backup_$(date +%Y%m%d)
cp -r ~/.agent-core ~/agent-core_backup_$(date +%Y%m%d)
```

### Phase 2: Migrate Research
```bash
# Create project folders in agent-core
mkdir -p ~/.agent-core/research/careercoach
mkdir -p ~/.agent-core/research/os-app

# Move CareerCoach research
mv ~/.antigravity/research/innovation-scout.md ~/.agent-core/research/careercoach/
mv ~/.antigravity/research/sources.csv ~/.agent-core/research/careercoach/

# Move OS-App research
mv ~/.antigravity/research/os_app_*.md ~/.agent-core/research/os-app/
mv ~/.antigravity/research/focusproof_extension_migration.md ~/.agent-core/research/os-app/
mv ~/.antigravity/research/metaventions_scout.md ~/.agent-core/research/os-app/
```

### Phase 3: Migrate Memory
```bash
# Create project memories
mkdir -p ~/.agent-core/memory/projects
mv ~/.antigravity/memory/global_identity.md ~/.agent-core/memory/projects/careercoach.md
```

### Phase 4: Clean Duplicates
```bash
# Remove outdated scripts from agent-core
rm -rf ~/.agent-core/scripts/

# Remove empty/migrated folders from antigravity
rm -rf ~/.antigravity/research/
rm -rf ~/.antigravity/memory/
rm -rf ~/.antigravity/workflows/
```

### Phase 5: Update References
- Update `SKILL.md` to reference only `~/.agent-core/`
- Update scripts to not write to `~/.antigravity/`

---

## 8. Verification Checklist

After migration, verify:

- [ ] `~/.antigravity/` contains ONLY: `extensions/`, `argv.json`, `antigravity/`
- [ ] `~/.agent-core/research/` contains all migrated research files
- [ ] `~/.agent-core/memory/projects/` contains project identities
- [ ] No scripts in `~/.agent-core/scripts/`
- [ ] All ResearchGravity scripts work with new paths
- [ ] Backups exist and are accessible

---

## 9. System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER'S RESEARCH SYSTEM                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  ~/.antigravity │    │    ~/.agent-core    │    │ ResearchGravity │
│    (IDE only)   │    │   (All research)    │    │  (Scripts/Git)  │
├─────────────────┤    ├─────────────────────┤    ├─────────────────┤
│ extensions/     │    │ sessions/           │    │ *.py scripts    │
│ argv.json       │    │ memory/             │    │ SKILL.md        │
│                 │    │ research/           │    │ .agent/research │
│                 │    │ workflows/          │    │   (local state) │
│                 │    │ config.json         │    │                 │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
         │                       │                        │
         │                       │                        │
         └───────────┬───────────┴────────────┬──────────┘
                     │                        │
                     ▼                        ▼
              ┌──────────────┐        ┌──────────────┐
              │  Antigravity │        │  Claude Code │
              │  IDE (VSCode)│        │     CLI      │
              └──────────────┘        └──────────────┘
```

---

*End of Architecture Audit*
