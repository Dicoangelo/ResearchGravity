#!/bin/bash
# Quick backfill status check

COUNT=$(curl -s http://localhost:6333/collections/findings 2>/dev/null | jq -r '.result.points_count // 0')
TOTAL=2530
PERCENT=$((COUNT * 100 / TOTAL))
REMAINING=$((TOTAL - COUNT))
BATCHES=$((REMAINING / 50 + 1))
MINUTES=$((BATCHES * 70 / 60))

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}━━━ Backfill Status ━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓${NC} Vectors: $COUNT / $TOTAL ($PERCENT%)"
echo -e "${YELLOW}⏱${NC}  Remaining: ~$MINUTES minutes ($BATCHES batches)"

# Progress indicator
if [ "$PERCENT" -lt 25 ]; then
    echo -e "${YELLOW}▸${NC} Status: Getting started..."
elif [ "$PERCENT" -lt 50 ]; then
    echo -e "${YELLOW}▸▸${NC} Status: Making progress..."
elif [ "$PERCENT" -lt 75 ]; then
    echo -e "${GREEN}▸▸▸${NC} Status: More than halfway!"
elif [ "$PERCENT" -lt 100 ]; then
    echo -e "${GREEN}▸▸▸▸${NC} Status: Almost done!"
else
    echo -e "${GREEN}✓✓✓✓${NC} Status: COMPLETE!"
fi

# Check if process is running
if pgrep -f backfill_vectors.py > /dev/null; then
    PID=$(pgrep -f backfill_vectors.py)
    echo -e "${GREEN}✓${NC} Process: Running (PID $PID)"
else
    echo -e "${YELLOW}⚠${NC}  Process: Not running (may have completed or failed)"
fi

echo ""
