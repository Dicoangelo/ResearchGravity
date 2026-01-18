#!/bin/bash
# Context Packs V2 - Production Deployment Script
# Deploys V2 system to production with V1 backup

set -e  # Exit on error

echo "============================================================"
echo "Context Packs V2 - Production Deployment"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_CORE="$HOME/.agent-core/context-packs"
BACKUP_DIR="$HOME/.agent-core/context-packs-v1-backup-$(date +%Y%m%d-%H%M%S)"

echo -e "${BLUE}📁 Directories:${NC}"
echo "   Script: $SCRIPT_DIR"
echo "   Agent Core: $AGENT_CORE"
echo "   Backup: $BACKUP_DIR"
echo ""

# Step 1: Check dependencies
echo -e "${BLUE}[1/7] Checking dependencies...${NC}"

MISSING_DEPS=()

if ! python3 -c "import sentence_transformers" 2>/dev/null; then
    MISSING_DEPS+=("sentence-transformers")
fi

if ! python3 -c "import networkx" 2>/dev/null; then
    MISSING_DEPS+=("networkx")
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    MISSING_DEPS+=("numpy")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo "   Installing..."
    pip3 install "${MISSING_DEPS[@]}" --break-system-packages --quiet
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ All dependencies present${NC}"
fi

# Optional: Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  PyTorch not installed (optional for RL training)${NC}"
    echo "   To install: pip3 install torch --break-system-packages"
else
    echo -e "${GREEN}✓ PyTorch available (RL training enabled)${NC}"
fi

echo ""

# Step 2: Backup V1 system
echo -e "${BLUE}[2/7] Backing up V1 system...${NC}"

if [ -f "$SCRIPT_DIR/select_packs.py" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$SCRIPT_DIR/select_packs.py" "$BACKUP_DIR/"
    cp "$SCRIPT_DIR/build_packs.py" "$BACKUP_DIR/" 2>/dev/null || true
    cp "$SCRIPT_DIR/pack_metrics.py" "$BACKUP_DIR/" 2>/dev/null || true
    echo -e "${GREEN}✓ V1 system backed up to: $BACKUP_DIR${NC}"
else
    echo -e "${YELLOW}⚠️  No V1 system found to backup${NC}"
fi

echo ""

# Step 3: Deploy V2 files
echo -e "${BLUE}[3/7] Deploying V2 files...${NC}"

# Check V2 files exist
V2_FILES=(
    "context_packs_v2_prototype.py"
    "context_packs_v2_layer4_rl.py"
    "context_packs_v2_layer5_focus.py"
    "select_packs_v2_integrated.py"
)

ALL_PRESENT=true
for file in "${V2_FILES[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$file" ]; then
        echo -e "${RED}❌ Missing V2 file: $file${NC}"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo -e "${RED}Deployment aborted: Missing V2 files${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All V2 files present${NC}"

# Make scripts executable
chmod +x "$SCRIPT_DIR/select_packs_v2_integrated.py"
chmod +x "$SCRIPT_DIR/context_packs_v2_prototype.py"
chmod +x "$SCRIPT_DIR/context_packs_v2_layer4_rl.py"
chmod +x "$SCRIPT_DIR/context_packs_v2_layer5_focus.py"

echo -e "${GREEN}✓ V2 files deployed and made executable${NC}"
echo ""

# Step 4: Create symlinks
echo -e "${BLUE}[4/7] Creating convenience symlinks...${NC}"

# Create select-packs symlink (points to integrated V2)
if [ -L "$SCRIPT_DIR/select-packs" ] || [ -f "$SCRIPT_DIR/select-packs" ]; then
    rm "$SCRIPT_DIR/select-packs"
fi
ln -s "$SCRIPT_DIR/select_packs_v2_integrated.py" "$SCRIPT_DIR/select-packs"
echo -e "${GREEN}✓ Created: select-packs -> select_packs_v2_integrated.py${NC}"

# Create v2 symlink
if [ -L "$SCRIPT_DIR/v2" ] || [ -f "$SCRIPT_DIR/v2" ]; then
    rm "$SCRIPT_DIR/v2"
fi
ln -s "$SCRIPT_DIR/context_packs_v2_prototype.py" "$SCRIPT_DIR/v2"
echo -e "${GREEN}✓ Created: v2 -> context_packs_v2_prototype.py${NC}"

echo ""

# Step 5: Test V2 engine
echo -e "${BLUE}[5/7] Testing V2 engine...${NC}"

TEST_OUTPUT=$(python3 "$SCRIPT_DIR/select_packs_v2_integrated.py" \
    --context "test deployment" \
    --budget 1000 \
    --format json 2>&1)

if echo "$TEST_OUTPUT" | grep -q '"engine": "v2"'; then
    echo -e "${GREEN}✓ V2 engine operational${NC}"

    # Extract metrics
    LAYERS=$(echo "$TEST_OUTPUT" | grep -o '"layers": [0-9]*' | grep -o '[0-9]*' || echo "unknown")
    TIME=$(echo "$TEST_OUTPUT" | grep -o '"selection_time_ms": [0-9.]*' | grep -o '[0-9.]*' || echo "unknown")

    echo "   Layers: $LAYERS"
    echo "   Selection time: ${TIME}ms"
else
    echo -e "${YELLOW}⚠️  V2 engine not available, using V1 fallback${NC}"

    if echo "$TEST_OUTPUT" | grep -q '"engine": "v1"'; then
        echo -e "${GREEN}✓ V1 fallback operational${NC}"
    else
        echo -e "${RED}❌ Test failed${NC}"
        echo "$TEST_OUTPUT"
        exit 1
    fi
fi

echo ""

# Step 6: Initialize storage
echo -e "${BLUE}[6/7] Initializing V2 storage...${NC}"

# Create V2 storage files if they don't exist
mkdir -p "$AGENT_CORE"

if [ ! -f "$AGENT_CORE/rl_operations.jsonl" ]; then
    touch "$AGENT_CORE/rl_operations.jsonl"
    echo -e "${GREEN}✓ Created: rl_operations.jsonl${NC}"
fi

if [ ! -f "$AGENT_CORE/continuum_memory.json" ]; then
    echo "{}" > "$AGENT_CORE/continuum_memory.json"
    echo -e "${GREEN}✓ Created: continuum_memory.json${NC}"
fi

echo ""

# Step 7: Display usage
echo -e "${BLUE}[7/7] Deployment complete!${NC}"
echo ""
echo "============================================================"
echo -e "${GREEN}✅ Context Packs V2 - Production Deployment Complete${NC}"
echo "============================================================"
echo ""
echo -e "${BLUE}📖 Quick Start:${NC}"
echo ""
echo "  # Use V2 (default, 7 layers)"
echo "  python3 select-packs --context \"your query\" --budget 50000"
echo ""
echo "  # Auto-detect context"
echo "  python3 select-packs --auto"
echo ""
echo "  # Force V1 (2 layers)"
echo "  python3 select-packs --context \"your query\" --v1"
echo ""
echo "  # Direct V2 access"
echo "  python3 v2 --query \"your query\" --budget 50000"
echo ""
echo -e "${BLUE}🔧 Layer-Specific Tools:${NC}"
echo ""
echo "  # RL Pack Manager (Layer 4)"
echo "  python3 context_packs_v2_layer4_rl.py decide --pack-id ... --context \"...\" --session-id ..."
echo "  python3 context_packs_v2_layer4_rl.py reward --session-id ... --pack-id ... --reward 0.9"
echo "  python3 context_packs_v2_layer4_rl.py train --batch-size 32 --epochs 10"
echo ""
echo "  # Focus/Continuum/Trainable (Layers 5-7)"
echo "  python3 context_packs_v2_layer5_focus.py focus --pack-id ... --query \"...\""
echo "  python3 context_packs_v2_layer5_focus.py memory"
echo "  python3 context_packs_v2_layer5_focus.py weights --top 10"
echo ""
echo -e "${BLUE}📊 System Info:${NC}"
echo "  V2 Engine: $([ "$LAYERS" = "7" ] && echo "✓ Operational (7 layers)" || echo "Fallback to V1 (2 layers)")"
echo "  V1 Backup: $BACKUP_DIR"
echo "  Storage: $AGENT_CORE"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "  Research: $SCRIPT_DIR/CONTEXT_PACKS_V2_RESEARCH.md"
echo "  Design: $SCRIPT_DIR/CONTEXT_PACKS_V2_DESIGN.md"
echo "  Complete: $SCRIPT_DIR/CONTEXT_PACKS_V2_COMPLETE.md"
echo ""
echo -e "${GREEN}🚀 V2 is now your default context packs engine!${NC}"
echo ""
