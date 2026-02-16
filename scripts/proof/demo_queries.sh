#!/bin/bash
# Demo queries to try while backfill completes

DB="$HOME/.agent-core/storage/antigravity.db"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ResearchGravity Demo Queries (Available NOW)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "1️⃣  Top Research Sessions by Findings:"
sqlite3 "$DB" -column -header << SQL
SELECT 
  substr(topic, 1, 50) as topic,
  finding_count,
  url_count
FROM sessions
WHERE finding_count > 0
ORDER BY finding_count DESC
LIMIT 10;
SQL

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "2️⃣  Multi-Agent Research Findings:"
sqlite3 "$DB" -column -header << SQL
SELECT 
  type,
  substr(content, 1, 70) as finding
FROM findings
WHERE content LIKE '%multi-agent%'
LIMIT 5;
SQL

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "3️⃣  URL Sources by Tier:"
sqlite3 "$DB" -column -header << SQL
SELECT 
  tier,
  category,
  COUNT(*) as urls,
  ROUND(AVG(relevance), 1) as avg_relevance
FROM urls
GROUP BY tier, category
ORDER BY tier, urls DESC
LIMIT 10;
SQL

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "4️⃣  Research Gaps Identified:"
sqlite3 "$DB" -column << SQL
SELECT 
  substr(content, 1, 80) as gap
FROM findings
WHERE type = 'gap'
ORDER BY RANDOM()
LIMIT 5;
SQL

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📊 Try more queries:"
echo "  rg-search \"your keyword\""
echo "  rg-sessions 20"
echo "  rg-projects"
echo ""
