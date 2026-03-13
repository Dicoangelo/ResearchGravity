#!/bin/bash
# ============================================================
# Agent Core v2.0 - Global Setup Script
# ============================================================
# Unified agent system for CLI + Antigravity (VSCode OSS)
# Based on Boris's Claude Code patterns
#
# Usage:
#   ./setup.sh              # Full setup
#   ./setup.sh --minimal    # Just directories, no aliases
#   ./setup.sh --uninstall  # Remove agent-core
#   ./setup.sh --update     # Update existing installation
# ============================================================

# Don't exit on error - handle gracefully
set +e

AGENT_CORE="$HOME/.agent-core"
DESKTOP_EXPORT="$HOME/Desktop/AgentCore"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Colors (with fallback for non-color terminals)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }

echo ""
echo "============================================================"
echo "  Agent Core Setup v2.0"
echo "  CLI + Antigravity + Web Orchestration"
echo "============================================================"
echo ""

# Parse arguments
MINIMAL=false
UNINSTALL=false
UPDATE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --minimal) MINIMAL=true ;;
        --uninstall) UNINSTALL=true ;;
        --update) UPDATE=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --minimal    Install without shell aliases"
            echo "  --update     Update existing installation"
            echo "  --uninstall  Remove agent-core completely"
            echo "  --help       Show this help"
            exit 0
            ;;
        *) log_error "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Uninstall
if [ "$UNINSTALL" = true ]; then
    echo -e "${YELLOW}Uninstalling Agent Core...${NC}"
    rm -rf "$AGENT_CORE" 2>/dev/null
    rm -rf "$DESKTOP_EXPORT" 2>/dev/null
    log_success "Agent Core removed"
    echo ""
    echo "Note: Shell aliases in ~/.zshrc were not removed."
    echo "Remove them manually if needed."
    exit 0
fi

# ============================================================
# Step 1: Create directory structure
# ============================================================
echo "Step 1: Creating directory structure..."
mkdir -p "$AGENT_CORE/sessions"
mkdir -p "$AGENT_CORE/memory"
mkdir -p "$AGENT_CORE/workflows"
mkdir -p "$AGENT_CORE/scripts"
mkdir -p "$AGENT_CORE/assets"
mkdir -p "$AGENT_CORE/logs"
if [ $? -eq 0 ]; then
    log_success "Created $AGENT_CORE"
else
    log_error "Failed to create $AGENT_CORE"
    exit 1
fi

# ============================================================
# Step 2: Copy scripts (if source exists)
# ============================================================
echo "Step 2: Installing scripts..."
if [ -d "$SCRIPT_DIR/scripts" ]; then
    for f in "$SCRIPT_DIR/scripts/"*.py; do
        if [ -f "$f" ]; then
            cp "$f" "$AGENT_CORE/scripts/"
            chmod +x "$AGENT_CORE/scripts/$(basename "$f")"
        fi
    done
    log_success "Scripts installed from $SCRIPT_DIR/scripts/"
else
    log_warn "No scripts directory found - skipping"
fi

# ============================================================
# Step 3: Copy workflows (if source exists)
# ============================================================
echo "Step 3: Installing workflows..."
if [ -d "$SCRIPT_DIR/workflows" ]; then
    for f in "$SCRIPT_DIR/workflows/"*.md; do
        if [ -f "$f" ]; then
            cp "$f" "$AGENT_CORE/workflows/"
        fi
    done
    log_success "Workflows installed"
else
    log_warn "No workflows directory found - skipping"
fi

# ============================================================
# Step 4: Copy assets (if source exists)
# ============================================================
echo "Step 4: Installing assets..."
if [ -d "$SCRIPT_DIR/assets" ]; then
    for f in "$SCRIPT_DIR/assets/"*; do
        if [ -f "$f" ]; then
            cp "$f" "$AGENT_CORE/assets/"
        fi
    done
    log_success "Assets installed"
else
    log_warn "No assets directory found - skipping"
fi

# ============================================================
# Step 5: Create/update config.json
# ============================================================
echo "Step 5: Configuring..."
if [ ! -f "$AGENT_CORE/config.json" ] || [ "$UPDATE" = true ]; then
    cat > "$AGENT_CORE/config.json" << 'CONFIGEOF'
{
  "version": "2.0",
  "defaults": {
    "auto_accept": true,
    "model": "claude-opus-4-6",
    "thinking": true,
    "max_parallel_sessions": 5
  },
  "environments": {
    "cli": {
      "enabled": true,
      "auto_accept": true,
      "plan_mode_shortcut": "shift+tab twice"
    },
    "antigravity": {
      "enabled": true,
      "type": "vscode-oss",
      "version": "1.13.3",
      "shortcuts": {
        "agent_manager": "cmd+e",
        "code_with_agent": "cmd+l",
        "edit_inline": "cmd+i"
      },
      "browser_enabled": true
    },
    "web": {
      "enabled": true,
      "handoff_supported": true
    }
  },
  "sync": {
    "enabled": true,
    "conflict_resolution": "latest_wins",
    "auto_push": true,
    "sync_on_archive": true
  },
  "logging": {
    "log_all_urls": true,
    "log_unused_urls": true,
    "log_failed_urls": true,
    "checkpoint_interval_minutes": 5
  },
  "research": {
    "viral_filter": {
      "min_stars": 500,
      "recency_days": 30
    },
    "groundbreaker_filter": {
      "min_stars": 10,
      "max_stars": 200,
      "recency_days": 90
    }
  }
}
CONFIGEOF
    log_success "Config created (auto-accept: ON)"
else
    log_info "Config exists - skipping (use --update to overwrite)"
fi

# ============================================================
# Step 6: Initialize memory files
# ============================================================
echo "Step 6: Initializing memory..."
if [ ! -f "$AGENT_CORE/memory/global.md" ]; then
    cat > "$AGENT_CORE/memory/global.md" << MEMEOF
# Global Memory

Last updated: $TIMESTAMP

## Identity & Preferences

_User preferences stored here..._

## Technical Stack

_Preferred languages, frameworks, tools..._

## Architecture Patterns

_Preferred patterns and anti-patterns..._

## Key Resources

_Important URLs, APIs, references..._
MEMEOF
    log_success "Global memory initialized"
fi

if [ ! -f "$AGENT_CORE/memory/learnings.md" ]; then
    cat > "$AGENT_CORE/memory/learnings.md" << 'LEARNEOF'
# Research Learnings

Auto-extracted insights from archived sessions.

---

LEARNEOF
    log_success "Learnings memory initialized"
fi

# ============================================================
# Step 7: Create session index
# ============================================================
echo "Step 7: Creating session index..."
if [ ! -f "$AGENT_CORE/sessions/index.md" ]; then
    cat > "$AGENT_CORE/sessions/index.md" << 'INDEXEOF'
# Session Index

All research sessions logged here.

| Date | Session ID | Topic | Workflow | Duration | URLs | Key Finding |
|------|------------|-------|----------|----------|------|-------------|
INDEXEOF
    log_success "Session index created"
else
    log_info "Session index exists"
fi

# ============================================================
# Step 8: Create CLAUDE.md template (if not exists)
# ============================================================
echo "Step 8: Checking CLAUDE.md template..."
if [ ! -f "$AGENT_CORE/assets/CLAUDE.md.template" ]; then
    cat > "$AGENT_CORE/assets/CLAUDE.md.template" << 'CLAUDEEOF'
# CLAUDE.md

> Shared context for Claude sessions. **Commit this to git.**

## Project Context

**Project**: [PROJECT_NAME]
**Stack**: [LANGUAGES/FRAMEWORKS]

## Quick Reference

```
Mode:     Auto-accept (always)
Model:    Opus 4.5 with thinking
Plan:     Shift+Tab twice → iterate → execute
```

### Antigravity Shortcuts
| Key | Action |
|-----|--------|
| ⌘E | Switch to Agent Manager |
| ⌘L | Code with Agent |
| ⌘I | Edit code inline |

## Agent Commands

| Command | Description |
|---------|-------------|
| `/innovation-scout [topic]` | arXiv + GitHub research |
| `/deep-research [topic]` | Multi-source investigation |
| `/remember [fact]` | Store to memory |
| `/archive` | Close session |

## Architecture Decisions

_Key decisions for this project:_

## Do NOT

- Don't modify dependencies without asking
- Don't commit directly; create PRs

---

*This file is read by Claude at session start.*
CLAUDEEOF
    log_success "CLAUDE.md template created"
fi

# ============================================================
# Step 9: Desktop export for easy access
# ============================================================
echo "Step 9: Creating desktop export..."
mkdir -p "$DESKTOP_EXPORT/workflows"

# Copy workflows if they exist
if [ -d "$AGENT_CORE/workflows" ]; then
    for f in "$AGENT_CORE/workflows/"*.md; do
        if [ -f "$f" ]; then
            cp "$f" "$DESKTOP_EXPORT/workflows/"
        fi
    done
fi

# Create project installer
cat > "$DESKTOP_EXPORT/install-to-project.sh" << 'INSTALLEOF'
#!/bin/bash
# Install Agent Core into current project
set -e

echo "Installing Agent Core to $(pwd)..."

# Create .agent structure
mkdir -p .agent/workflows
mkdir -p .agent/research

# Copy workflows
if [ -d "$HOME/.agent-core/workflows" ]; then
    cp "$HOME/.agent-core/workflows/"*.md .agent/workflows/ 2>/dev/null || true
fi

# Create project memory
if [ ! -f ".agent/memory.md" ]; then
    cat > .agent/memory.md << 'EOF'
# Project Memory

## Context

_Project-specific context..._

## Decisions

_Key decisions and rationale..._
EOF
fi

# Create CLAUDE.md if not exists
if [ ! -f "CLAUDE.md" ] && [ -f "$HOME/.agent-core/assets/CLAUDE.md.template" ]; then
    cp "$HOME/.agent-core/assets/CLAUDE.md.template" ./CLAUDE.md
    echo "Created CLAUDE.md - edit with project details"
fi

echo "✅ Agent Core installed!"
echo ""
echo "Files created:"
echo "  .agent/workflows/     - Agent workflows"
echo "  .agent/memory.md      - Project memory"
echo "  CLAUDE.md             - Shared context (commit this)"
INSTALLEOF
chmod +x "$DESKTOP_EXPORT/install-to-project.sh"
log_success "Desktop export created at $DESKTOP_EXPORT"

# ============================================================
# Step 10: Configure Claude Code (if available)
# ============================================================
echo "Step 10: Configuring Claude Code..."
if command -v claude &> /dev/null; then
    claude config set autoAccept true 2>/dev/null && \
        log_success "Claude Code auto-accept enabled" || \
        log_warn "Could not configure Claude Code"
else
    log_info "Claude Code CLI not found - skipping"
fi

# ============================================================
# Step 11: Shell aliases
# ============================================================
if [ "$MINIMAL" = false ]; then
    echo ""
    echo "============================================================"
    echo "  Shell Configuration"
    echo "============================================================"
    echo ""
    echo "Add these to your ${CYAN}~/.zshrc${NC} or ${CYAN}~/.bashrc${NC}:"
    echo ""
    echo -e "${YELLOW}# Agent Core v2.0${NC}"
    echo "export AGENT_CORE=\"\$HOME/.agent-core\""
    echo ""
    echo "# Quick commands"
    echo "alias agent-init='python3 \$AGENT_CORE/scripts/init_session.py'"
    echo "alias agent-sync='python3 \$AGENT_CORE/scripts/sync_environments.py'"
    echo "alias agent-archive='python3 \$AGENT_CORE/scripts/archive_session.py'"
    echo "alias agent-log='python3 \$AGENT_CORE/scripts/log_url.py'"
    echo "alias agent-status='python3 \$AGENT_CORE/scripts/sync_environments.py status'"
    echo ""
    echo "# Project setup"
    echo "alias agent-install='$DESKTOP_EXPORT/install-to-project.sh'"
    echo "alias agent-claude-md='cp \$AGENT_CORE/assets/CLAUDE.md.template ./CLAUDE.md'"
    echo ""
fi

# ============================================================
# Complete!
# ============================================================
echo ""
echo "============================================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "============================================================"
echo ""
echo -e "${CYAN}Directory Structure:${NC}"
echo "  $AGENT_CORE/"
echo "  ├── config.json       (auto-accept: ON)"
echo "  ├── sessions/         (archived research)"
echo "  ├── memory/"
echo "  │   ├── global.md     (permanent facts)"
echo "  │   └── learnings.md  (research insights)"
echo "  ├── workflows/        (agent workflows)"
echo "  ├── scripts/          (Python tools)"
echo "  └── assets/           (templates)"
echo ""
echo -e "${CYAN}Antigravity Shortcuts:${NC}"
echo "  ⌘E  Switch to Agent Manager"
echo "  ⌘L  Code with Agent"
echo "  ⌘I  Edit code inline"
echo ""
echo -e "${CYAN}Boris's Workflow:${NC}"
echo "  1. ${YELLOW}Plan first${NC} (Shift+Tab twice)"
echo "  2. Iterate until plan is solid"
echo "  3. Auto-accept executes"
echo ""
echo -e "${CYAN}Quick Start:${NC}"
echo "  agent-init 'your topic'     # Start research"
echo "  agent-log <url> --used      # Log URL"
echo "  agent-status                # Check state"
echo "  agent-archive               # Close session"
echo ""
echo -e "${CYAN}Parallel Sessions:${NC}"
echo "  Tab 1: Planning    Tab 2-3: Features"
echo "  Tab 4: Testing     Tab 5: Documentation"
echo ""
