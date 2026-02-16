# ResearchGravity Storage & Query Guide

## ✅ Current Status

**Storage Triad Active:**
- ✅ SQLite: 114 sessions, 2,530 findings, 8,935 URLs
- 🔄 Qdrant: 800/2,530 vectors uploaded (31% complete, ~30 min remaining)
- ✅ Supabase: Connected (voice sessions)

**All data is safe and queryable now via SQLite!**

---

## 🎯 Quick Commands (After Sourcing)

```bash
# Reload shell config
source ~/.zshrc

# Quick stats
rg-stats              # Show counts + vector progress
rg-backfill           # Check backfill progress
rg-projects           # Research by project
rg-sessions           # Recent sessions

# Search
rg-search "multi-agent"           # Search findings
rg-semantic "agentic consensus"   # Semantic search (after backfill)

# Qdrant management
rg-qdrant-start       # Start Qdrant container
rg-qdrant-stop        # Stop Qdrant
rg-qdrant-dash        # Open web dashboard
rg-qdrant-logs        # View logs

# API server
rg-api                # Start research API on :3847

# Direct database
rg-db                 # Open SQLite shell
```

---

## 📊 SQLite Queries (Available Now)

### 1. Find Sessions by Topic
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
SELECT id, topic, project, finding_count, url_count
FROM sessions
WHERE topic LIKE '%multi-agent%'
ORDER BY started_at DESC
LIMIT 10;
EOF
```

### 2. Search Findings
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
SELECT
  s.topic,
  f.type,
  f.content
FROM findings f
JOIN sessions s ON f.session_id = s.id
WHERE f.content LIKE '%consensus%'
LIMIT 10;
EOF
```

### 3. URL Distribution by Tier
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
SELECT
  tier,
  category,
  COUNT(*) as count,
  ROUND(AVG(relevance), 2) as avg_relevance
FROM urls
GROUP BY tier, category
ORDER BY tier, count DESC;
EOF
```

### 4. Top Research Topics
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
SELECT
  topic,
  finding_count,
  url_count,
  project
FROM sessions
ORDER BY finding_count DESC
LIMIT 20;
EOF
```

### 5. Export to CSV
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
.headers on
.mode csv
.output ~/research_export.csv
SELECT
  s.id as session_id,
  s.topic,
  s.project,
  f.type as finding_type,
  f.content as finding
FROM sessions s
LEFT JOIN findings f ON s.id = f.session_id;
.quit
EOF
```

---

## 🔍 Semantic Search (After Backfill Completes)

### Test Search
```bash
cd ~/researchgravity
source .venv/bin/activate
export COHERE_API_KEY=$(jq -r .cohere.api_key ~/.agent-core/config.json)

# Basic search
python3 test_semantic_search.py "multi-agent orchestration"

# Compare vector vs reranked
python3 test_semantic_search.py "agentic consensus" --compare
```

### Python API
```python
from storage.qdrant_db import get_qdrant

# Initialize
q = await get_qdrant()

# Search findings with reranking
results = await q.search_findings(
    "multi-agent consensus",
    limit=10,
    rerank=True,
    min_score=0.5
)

for r in results:
    print(f"[{r['relevance_score']:.3f}] {r['content']}")

# Search sessions
sessions = await q.search_sessions(
    "agentic orchestration",
    limit=5,
    filter_project="os-app"
)

# Unified search across all collections
all_results = await q.semantic_search(
    "research workflow",
    collections=["findings", "sessions"],
    limit=5
)

await q.close()
```

---

## 🌐 REST API

### Start Server
```bash
cd ~/researchgravity
source .venv/bin/activate
python3 -m api.server --port 3847

# Or use alias:
rg-api
```

### Endpoints

**Get Sessions:**
```bash
curl http://localhost:3847/api/sessions | jq
```

**Semantic Search:**
```bash
curl -X POST http://localhost:3847/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "multi-agent orchestration",
    "limit": 5,
    "rerank": true
  }' | jq
```

**Get Statistics:**
```bash
curl http://localhost:3847/api/v2/stats | jq
```

**Graph Endpoints:**
```bash
# Graph stats
curl http://localhost:3847/api/v2/graph/stats | jq

# Session subgraph
curl http://localhost:3847/api/v2/graph/session/<session-id> | jq

# Concept clusters
curl http://localhost:3847/api/v2/graph/clusters | jq
```

---

## 🐳 Docker Management

### Qdrant Container

**Status:**
```bash
docker ps | grep qdrant
```

**Start/Stop:**
```bash
docker start qdrant-researchgravity
docker stop qdrant-researchgravity
docker restart qdrant-researchgravity
```

**Logs:**
```bash
docker logs qdrant-researchgravity
docker logs -f qdrant-researchgravity  # Follow
```

**Storage Location:**
```bash
ls -lah ~/.agent-core/qdrant_storage/
```

**Web Dashboard:**
- http://localhost:6333/dashboard
- http://localhost:6333/collections

---

## 📦 Data Export

### Full Backup
```bash
# SQLite database
cp ~/.agent-core/storage/antigravity.db ~/backup_research_$(date +%Y%m%d).db

# All sessions
cp -r ~/.agent-core/sessions ~/backup_sessions_$(date +%Y%m%d)

# Qdrant storage
cp -r ~/.agent-core/qdrant_storage ~/backup_qdrant_$(date +%Y%m%d)
```

### Export to JSON
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
.mode json
.output ~/research_backup.json
SELECT
  json_object(
    'sessions', (SELECT json_group_array(json_object(
      'id', id,
      'topic', topic,
      'project', project,
      'finding_count', finding_count
    )) FROM sessions),
    'findings', (SELECT json_group_array(json_object(
      'content', content,
      'type', type,
      'session_id', session_id
    )) FROM findings LIMIT 100)
  ) as export;
EOF
```

---

## 🔧 Maintenance

### Check Vector Progress
```bash
# Quick check
rg-backfill

# Detailed
curl http://localhost:6333/collections/findings | jq '.result | {
  status,
  points_count,
  indexed_vectors_count,
  segments_count
}'
```

### Rebuild Vectors (if needed)
```bash
cd ~/researchgravity
source .venv/bin/activate
export COHERE_API_KEY=$(jq -r .cohere.api_key ~/.agent-core/config.json)

# Findings only
python3 backfill_vectors.py --findings-only

# Sessions only
python3 backfill_vectors.py --sessions-only

# Full rebuild
python3 backfill_vectors.py
```

### Database Maintenance
```bash
# Vacuum SQLite (optimize storage)
sqlite3 ~/.agent-core/storage/antigravity.db "VACUUM;"

# Analyze (update statistics)
sqlite3 ~/.agent-core/storage/antigravity.db "ANALYZE;"
```

---

## 📈 Monitoring Backfill

### Watch Progress
```bash
watch -n 5 ~/researchgravity/query_research.sh backfill

# Or tail the log
tail -f /tmp/qdrant_backfill.log
```

### Expected Timeline
- **Current:** 800/2,530 (31%)
- **Rate:** ~50 embeddings/minute
- **Remaining:** ~35 batches × 70s = ~40 minutes
- **Total:** ~50 minutes from start

---

## 🎓 Example Queries

### Find All Agentic Research
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
SELECT DISTINCT
  s.topic,
  COUNT(f.id) as findings,
  GROUP_CONCAT(DISTINCT f.type) as types
FROM sessions s
JOIN findings f ON s.id = f.session_id
WHERE s.topic LIKE '%agent%'
  OR f.content LIKE '%agent%'
GROUP BY s.id
ORDER BY findings DESC;
EOF
```

### Research Timeline
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
SELECT
  DATE(started_at) as date,
  COUNT(*) as sessions,
  SUM(finding_count) as findings,
  SUM(url_count) as urls
FROM sessions
WHERE started_at IS NOT NULL
GROUP BY DATE(started_at)
ORDER BY date DESC
LIMIT 30;
EOF
```

### Tier 1 Sources
```bash
sqlite3 ~/.agent-core/storage/antigravity.db <<EOF
SELECT
  url,
  category,
  relevance,
  title
FROM urls
WHERE tier = 1
  AND relevance >= 4
ORDER BY relevance DESC, used DESC
LIMIT 20;
EOF
```

---

## 🚀 Next Steps

1. **Wait for backfill** (~40 min) or upgrade Cohere for instant completion
2. **Test semantic search** with `rg-semantic "your query"`
3. **Start the API** with `rg-api` for cross-app integration
4. **Connect OS-App** to the research API

All your research data is safe and queryable now via SQLite!
