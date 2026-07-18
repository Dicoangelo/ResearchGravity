#!/bin/bash
# Frontier Scout daily run — METAVENTIONS AI ResearchGravity
# Scheduled by ~/Library/LaunchAgents/com.metaventions.frontier-scout.plist
# Appends each run to the append-only ledger; never overwrites (telemetry rule).

set -u
REPO="$HOME/projects/apps/researchgravity"
LEDGER="$HOME/.agent-core/storage/frontier_scout_runs.jsonl"
LOG="$HOME/.agent-core/logs/frontier-scout.log"
mkdir -p "$(dirname "$LEDGER")" "$(dirname "$LOG")"

# Key from the canonical config (keyless works, key lifts rate limits).
# Never clobber an inherited env key with an empty value.
CFG_KEY="$(python3 -c "
import json
try:
    print(json.load(open('$HOME/.agent-core/config.json')).get('firecrawl',{}).get('api_key',''))
except Exception:
    print('')
")"
if [ -n "$CFG_KEY" ]; then
  export FIRECRAWL_API_KEY="$CFG_KEY"
fi

cd "$REPO" || exit 1
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MODE="${1:-citers}"

OUT="$(python3 -m cpb.frontier_scout --mode "$MODE" --limit 10 --json 2>>"$LOG")"
STATUS=$?

if [ $STATUS -eq 0 ] && [ -n "$OUT" ]; then
  python3 -c "
import json, sys
out = json.loads(sys.argv[1])
rec = {'ts': '$TS', 'mode': '$MODE', 'result': out}
with open('$LEDGER', 'a') as f:
    f.write(json.dumps(rec) + '\n')
new = out.get('frontier', out.get('papers', []))
print(f'frontier-scout $TS mode=$MODE surfaced={len(new)}')
" "$OUT" >>"$LOG" 2>&1
else
  echo "frontier-scout $TS mode=$MODE FAILED status=$STATUS" >>"$LOG"
fi
exit $STATUS
