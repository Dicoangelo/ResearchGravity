#!/usr/bin/env python3
"""Persist FIRECRAWL_API_KEY from the current env into ~/.agent-core/config.json.

Run from a shell where the key is exported. The cron wrapper
(frontier_scout_cron.sh) reads config.json's firecrawl.api_key so launchd
runs get keyed Firecrawl access instead of keyless rate limits.
"""

import json
import os
import sys

path = os.path.expanduser("~/.agent-core/config.json")
key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
if not key:
    sys.exit("FIRECRAWL_API_KEY not set in this shell — nothing written.")

with open(path) as f:
    cfg = json.load(f)
cfg["firecrawl"] = {"api_key": key}
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"firecrawl.api_key persisted to {path} (len {len(key)})")
