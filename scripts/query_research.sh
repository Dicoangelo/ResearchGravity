#!/bin/bash
# Quick research queries for ResearchGravity

DB="$HOME/.agent-core/storage/antigravity.db"

case "$1" in
  stats)
    echo "━━━ Research Statistics ━━━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" <<EOF
SELECT
  'Sessions' as entity, COUNT(*) as count FROM sessions
UNION ALL
SELECT 'Findings', COUNT(*) FROM findings
UNION ALL
SELECT 'URLs', COUNT(*) FROM urls;
EOF
    echo ""
    echo "Qdrant Vectors:"
    curl -s http://localhost:6333/collections/findings 2>/dev/null | jq -r '"  findings: \(.result.points_count // 0) vectors"'
    curl -s http://localhost:6333/collections/sessions 2>/dev/null | jq -r '"  sessions: \(.result.points_count // 0) vectors"'
    ;;

  sessions)
    echo "━━━ Recent Sessions ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" -header -column <<EOF
SELECT
  substr(id, 1, 40) as session_id,
  substr(topic, 1, 60) as topic,
  project,
  finding_count as findings
FROM sessions
ORDER BY started_at DESC
LIMIT ${2:-10};
EOF
    ;;

  findings)
    QUERY="${2:-multi-agent}"
    echo "━━━ Findings matching '$QUERY' ━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" -header -column <<EOF
SELECT
  substr(session_id, 1, 30) as session,
  type,
  substr(content, 1, 80) as content
FROM findings
WHERE content LIKE '%$QUERY%'
LIMIT 10;
EOF
    ;;

  urls)
    echo "━━━ Top URLs by Tier ━━━━━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" -header -column <<EOF
SELECT
  tier,
  category,
  COUNT(*) as count
FROM urls
GROUP BY tier, category
ORDER BY tier, count DESC;
EOF
    ;;

  search)
    QUERY="${2:-agentic}"
    echo "━━━ Full-text search: '$QUERY' ━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" -header -column <<EOF
SELECT
  substr(s.topic, 1, 50) as session_topic,
  substr(f.content, 1, 70) as finding
FROM findings f
JOIN sessions s ON f.session_id = s.id
WHERE f.content LIKE '%$QUERY%'
LIMIT 10;
EOF
    ;;

  projects)
    echo "━━━ Research by Project ━━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" -header -column <<EOF
SELECT
  COALESCE(project, 'unassigned') as project,
  COUNT(DISTINCT id) as sessions,
  SUM(finding_count) as findings,
  SUM(url_count) as urls
FROM sessions
GROUP BY project
ORDER BY sessions DESC;
EOF
    ;;

  backfill)
    echo "━━━ Backfill Progress ━━━━━━━━━━━━━━━━━━━━━━━━━"
    FINDINGS_TOTAL=$(sqlite3 "$DB" "SELECT COUNT(*) FROM findings;")
    SESSIONS_TOTAL=$(sqlite3 "$DB" "SELECT COUNT(*) FROM sessions;")
    FINDINGS_VECTORS=$(curl -s http://localhost:6333/collections/findings 2>/dev/null | jq -r '.result.points_count // 0')
    SESSIONS_VECTORS=$(curl -s http://localhost:6333/collections/sessions 2>/dev/null | jq -r '.result.points_count // 0')

    echo "Findings: $FINDINGS_VECTORS / $FINDINGS_TOTAL"
    echo "Sessions: $SESSIONS_VECTORS / $SESSIONS_TOTAL"

    if [ "$FINDINGS_TOTAL" -gt 0 ]; then
      PERCENT=$((FINDINGS_VECTORS * 100 / FINDINGS_TOTAL))
      echo "Progress: $PERCENT%"
      echo ""
      echo "Log: tail -f /tmp/qdrant_backfill.log"
    fi
    ;;

  *)
    cat <<EOF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ResearchGravity Query Tool
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage: query_research.sh <command> [args]

Commands:
  stats               Show overall statistics
  sessions [N]        List recent N sessions (default: 10)
  findings <query>    Search findings by keyword
  urls                Show URL distribution by tier/category
  search <query>      Full-text search across findings
  projects            Show research grouped by project
  backfill            Check vector backfill progress

Examples:
  query_research.sh stats
  query_research.sh sessions 20
  query_research.sh findings "multi-agent"
  query_research.sh search "consensus"
  query_research.sh backfill
EOF
    ;;
esac
