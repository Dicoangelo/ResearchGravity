#!/bin/bash
# Interactive research explorer

DB="$HOME/.agent-core/storage/antigravity.db"

case "$1" in
  arxiv)
    echo "━━━ arXiv Papers in Research ━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" << SQL
SELECT 
  substr(url, 1, 60) as paper,
  tier,
  relevance
FROM urls
WHERE url LIKE '%arxiv.org%'
ORDER BY relevance DESC
LIMIT 15;
SQL
    ;;
    
  github)
    echo "━━━ GitHub Repos Referenced ━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" << SQL
SELECT 
  substr(url, 1, 70) as repo,
  tier
FROM urls
WHERE url LIKE '%github.com%'
LIMIT 15;
SQL
    ;;
    
  thesis)
    echo "━━━ Research Thesis Statements ━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" -column << SQL
SELECT 
  substr(content, 1, 100) as thesis
FROM findings
WHERE type = 'thesis'
ORDER BY RANDOM()
LIMIT 10;
SQL
    ;;
    
  gaps)
    echo "━━━ Innovation Gaps Identified ━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" -column << SQL
SELECT 
  substr(content, 1, 100) as gap
FROM findings
WHERE type = 'gap'
ORDER BY RANDOM()
LIMIT 10;
SQL
    ;;
    
  timeline)
    echo "━━━ Research Timeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" -column -header << SQL
SELECT 
  DATE(started_at) as date,
  COUNT(*) as sessions,
  SUM(finding_count) as findings
FROM sessions
WHERE started_at IS NOT NULL
GROUP BY DATE(started_at)
ORDER BY date DESC
LIMIT 15;
SQL
    ;;
    
  topics)
    echo "━━━ Common Research Topics ━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 "$DB" << SQL
.mode list
SELECT DISTINCT
  CASE
    WHEN topic LIKE '%multi-agent%' THEN 'Multi-Agent Systems'
    WHEN topic LIKE '%orchestration%' THEN 'Orchestration'
    WHEN topic LIKE '%agentic%' THEN 'Agentic AI'
    WHEN topic LIKE '%consensus%' THEN 'Consensus Mechanisms'
    WHEN topic LIKE '%routing%' THEN 'Query Routing'
    WHEN topic LIKE '%memory%' THEN 'AI Memory'
    WHEN topic LIKE '%voice%' THEN 'Voice Interfaces'
    ELSE 'Other'
  END as topic_area,
  COUNT(*) as sessions
FROM sessions
WHERE topic IS NOT NULL
GROUP BY topic_area
ORDER BY sessions DESC;
SQL
    ;;
    
  *)
    cat << HELP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Research Explorer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage: explore_research.sh <command>

Commands:
  arxiv       Show arXiv papers referenced
  github      Show GitHub repos referenced
  thesis      Show research thesis statements
  gaps        Show innovation gaps identified
  timeline    Show research activity timeline
  topics      Show common research topics

Examples:
  ~/researchgravity/explore_research.sh arxiv
  ~/researchgravity/explore_research.sh gaps
  ~/researchgravity/explore_research.sh timeline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HELP
    ;;
esac
