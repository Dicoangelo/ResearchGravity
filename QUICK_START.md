# ResearchGravity Quick Start Guide

**Status:** ✅ Fully Activated (100% - 2,530 vectors)

---

## Research Statistics

- **Sessions:** 114 archived research sessions
- **Findings:** 2,530 (812 thesis, 802 gaps, 772 findings, 144 innovations)
- **URLs:** 8,935 cataloged
- **Vectors:** 2,530 (100% embedded with Cohere)
- **Storage:** 10 MB SQLite + 40 MB Qdrant

---

## Quick Commands

### Search & Discovery

```bash
# Semantic search (finds conceptually similar research)
rg-semantic "multi-agent consensus mechanisms"
rg-semantic "agentic orchestration patterns"

# Keyword search (exact matches)
rg-search "consensus"

# Browse sessions
rg-sessions 20

# System stats
rg-stats
```

### API Server

```bash
# Start REST API
rg-api

# Test (in another terminal):
curl http://localhost:3847/api/sessions | jq
```

### Qdrant

```bash
rg-qdrant-start      # Start
rg-qdrant-stop       # Stop
rg-qdrant-dash       # Open dashboard
```

---

## See Also

- Full Guide: ~/researchgravity/SYSTEM_READY.md
- Storage Guide: ~/researchgravity/STORAGE_GUIDE.md
- Victory Report: /tmp/VICTORY_ACHIEVED.md
