#!/usr/bin/env python3
"""
Backfill Telemetry Data

Imports telemetry data into the Meta-Learning Engine:
- Session outcomes from ~/.claude/data/session-outcomes.jsonl
- Cognitive states from ~/.claude/kernel/cognitive-os/
- Error patterns from ~/.claude/data/recovery-outcomes.jsonl

This populates both SQLite and Qdrant for meta-learning predictions.

Usage:
    python3 backfill_telemetry.py                    # Backfill all
    python3 backfill_telemetry.py --outcomes         # Only outcomes
    python3 backfill_telemetry.py --cognitive        # Only cognitive states
    python3 backfill_telemetry.py --errors           # Only error patterns
    python3 backfill_telemetry.py --dry-run          # Preview without storing
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse

from storage.engine import get_engine


# Data paths
HOME = Path.home()
OUTCOMES_FILE = HOME / ".claude" / "data" / "session-outcomes.jsonl"
COGNITIVE_FILES = HOME / ".claude" / "kernel" / "cognitive-os"
RECOVERY_FILE = HOME / ".claude" / "data" / "recovery-outcomes.jsonl"
FATE_FILE = HOME / ".claude" / "kernel" / "cognitive-os" / "fate-predictions.jsonl"


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    if not file_path.exists():
        return []

    records = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return records


async def backfill_outcomes(dry_run: bool = False) -> int:
    """Backfill session outcomes."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Backfilling session outcomes...")

    if not OUTCOMES_FILE.exists():
        print(f"❌ File not found: {OUTCOMES_FILE}")
        return 0

    # Read outcomes
    all_outcomes = read_jsonl(OUTCOMES_FILE)

    # Filter out outcomes without intent or title
    outcomes = [
        o for o in all_outcomes
        if o.get("intent") or o.get("title")
    ]

    print(f"📊 Found {len(all_outcomes)} session outcomes ({len(outcomes)} valid)")

    if not outcomes:
        return 0

    # Preview
    print(f"\nSample outcome:")
    print(f"  Session: {outcomes[0].get('session_id', 'N/A')[:60]}...")
    print(f"  Intent: {outcomes[0].get('intent', 'N/A')[:60]}...")
    print(f"  Outcome: {outcomes[0].get('outcome')} | Quality: {outcomes[0].get('quality')}")

    if dry_run:
        return len(outcomes)

    # Store in smaller batches to avoid database locks
    engine = await get_engine()
    count = 0
    batch_size = 50

    print(f"Processing in batches of {batch_size}...")
    for i in range(0, len(outcomes), batch_size):
        batch = outcomes[i:i + batch_size]
        batch_count = await engine.store_outcomes_batch(batch)
        count += batch_count
        print(f"  Processed {count}/{len(outcomes)}...")
        await asyncio.sleep(0.1)  # Small delay to avoid lock contention

    await engine.close()

    print(f"✅ Stored {count} session outcomes")
    return count


async def backfill_cognitive_states(dry_run: bool = False) -> int:
    """Backfill cognitive states from fate predictions, routing decisions, and flow history."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Backfilling cognitive states...")

    states = []

    # 1. Load fate predictions
    fate_file = HOME / ".claude" / "kernel" / "cognitive-os" / "fate-predictions.jsonl"
    if fate_file.exists():
        fate_records = read_jsonl(fate_file)
        print(f"📊 Found {len(fate_records)} fate predictions")

        for record in fate_records:
            ts = record.get("timestamp", "")
            if ts:
                # Parse timestamp to extract hour
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hour = dt.hour
                    day = dt.strftime("%A")
                except:
                    hour = 0
                    day = ""

                # Map success probability to energy level
                success_prob = record.get("success_probability", 0.5)
                energy = 0.3 + (success_prob * 0.7)  # Map 0-1 to 0.3-1.0

                state = {
                    "id": f"fate-{ts}",
                    "mode": "unknown",  # Fate predictions don't include mode
                    "energy_level": energy,
                    "flow_score": 0.0,
                    "hour": hour,
                    "day": day,
                    "timestamp": ts,
                    "predictions": {
                        "predicted": record.get("predicted"),
                        "actual": record.get("actual"),
                        "success_probability": success_prob,
                        "partial_probability": record.get("partial_probability"),
                        "abandon_probability": record.get("abandon_probability")
                    }
                }
                states.append(state)

    # 2. Load routing decisions
    routing_file = HOME / ".claude" / "kernel" / "cognitive-os" / "routing-decisions.jsonl"
    if routing_file.exists():
        routing_records = read_jsonl(routing_file)
        print(f"📊 Found {len(routing_records)} routing decisions")

        for record in routing_records:
            ts = record.get("timestamp", "")
            if ts:
                # Map cognitive mode to energy level
                mode = record.get("cognitive_mode", "unknown")
                energy_map = {
                    "morning": 0.6,
                    "peak": 0.8,
                    "dip": 0.5,
                    "evening": 0.7,
                    "deep_night": 0.9
                }
                energy = energy_map.get(mode, 0.5)

                state = {
                    "id": f"routing-{ts}",
                    "mode": mode,
                    "energy_level": energy,
                    "flow_score": 0.0,
                    "hour": record.get("hour", 0),
                    "day": "",
                    "timestamp": ts,
                    "predictions": {
                        "recommended_model": record.get("recommended_model"),
                        "dq_score": record.get("dq_score"),
                        "task_complexity": record.get("task_complexity")
                    }
                }
                states.append(state)

    # 3. Load flow history
    flow_file = HOME / ".claude" / "kernel" / "cognitive-os" / "flow-history.jsonl"
    if flow_file.exists():
        flow_records = read_jsonl(flow_file)
        print(f"📊 Found {len(flow_records)} flow history records")

        for record in flow_records:
            ts = record.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hour = dt.hour
                    day = dt.strftime("%A")
                except:
                    hour = 0
                    day = ""

                flow_score = record.get("score", 0.0)

                # Map flow state to energy level
                state_name = record.get("state", "neutral")
                energy_map = {
                    "deep_flow": 1.0,
                    "flow": 0.8,
                    "focused": 0.7,
                    "neutral": 0.5,
                    "distracted": 0.3,
                    "struggling": 0.2
                }
                energy = energy_map.get(state_name, 0.5)

                state = {
                    "id": f"flow-{ts}",
                    "mode": state_name,
                    "energy_level": energy,
                    "flow_score": flow_score,
                    "hour": hour,
                    "day": day,
                    "timestamp": ts,
                    "predictions": {
                        "session_id": record.get("session_id"),
                        "messages": record.get("messages"),
                        "tools": record.get("tools")
                    }
                }
                states.append(state)

    print(f"📊 Total cognitive states: {len(states)}")

    if not states:
        return 0

    # Preview
    print(f"\nSample state:")
    print(f"  Mode: {states[0].get('mode')} | Energy: {states[0].get('energy_level'):.2f}")
    print(f"  Hour: {states[0].get('hour')} | Flow: {states[0].get('flow_score'):.2f}")
    print(f"  Timestamp: {states[0].get('timestamp')}")

    if dry_run:
        return len(states)

    # Store in smaller batches
    engine = await get_engine()
    count = 0
    batch_size = 50

    print(f"Processing in batches of {batch_size}...")
    for i in range(0, len(states), batch_size):
        batch = states[i:i + batch_size]
        batch_count = await engine.store_cognitive_states_batch(batch)
        count += batch_count
        print(f"  Processed {count}/{len(states)}...")
        await asyncio.sleep(0.1)

    await engine.close()

    print(f"✅ Stored {count} cognitive states")
    return count


async def backfill_error_patterns(dry_run: bool = False) -> int:
    """Backfill error patterns from recovery outcomes."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Backfilling error patterns...")

    if not RECOVERY_FILE.exists():
        print(f"⚠️  File not found: {RECOVERY_FILE}")
        return 0

    # Read recovery outcomes
    records = read_jsonl(RECOVERY_FILE)
    print(f"📊 Found {len(records)} recovery records")

    # Convert to error patterns
    errors = []
    seen_actions = set()

    for record in records:
        action = record.get("action", "unknown")
        category = record.get("category", "unknown")

        # Skip duplicates - group by action
        if action in seen_actions:
            continue

        error = {
            "id": f"error-{category}-{action}",
            "error_type": f"{category}:{action}",
            "context": record.get("details", "")[:500],
            "solution": action.replace("_", " ").title(),
            "success_rate": 1.0 if record.get("success") else 0.0,
            "occurrences": 1
        }
        errors.append(error)
        seen_actions.add(action)

    print(f"📊 Converted to {len(errors)} error patterns")

    if not errors:
        return 0

    # Preview
    print(f"\nSample error:")
    print(f"  Type: {errors[0].get('error_type')}")
    print(f"  Solution: {errors[0].get('solution')[:60]}...")
    print(f"  Success rate: {errors[0].get('success_rate'):.0%}")

    if dry_run:
        return len(errors)

    # Store
    engine = await get_engine()
    count = await engine.store_error_patterns_batch(errors)
    await engine.close()

    print(f"✅ Stored {count} error patterns")
    return count


async def main():
    parser = argparse.ArgumentParser(description="Backfill telemetry data for meta-learning")
    parser.add_argument("--outcomes", action="store_true", help="Only backfill session outcomes")
    parser.add_argument("--cognitive", action="store_true", help="Only backfill cognitive states")
    parser.add_argument("--errors", action="store_true", help="Only backfill error patterns")
    parser.add_argument("--dry-run", action="store_true", help="Preview without storing")
    args = parser.parse_args()

    # If no specific flag, do all
    do_all = not (args.outcomes or args.cognitive or args.errors)

    print("=" * 60)
    print("Meta-Learning Engine - Telemetry Backfill")
    print("=" * 60)

    total = 0

    if do_all or args.outcomes:
        total += await backfill_outcomes(args.dry_run)

    if do_all or args.cognitive:
        total += await backfill_cognitive_states(args.dry_run)

    if do_all or args.errors:
        total += await backfill_error_patterns(args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"[DRY RUN] Would backfill {total} records")
    else:
        print(f"✅ Backfill complete: {total} records stored")
        print("\nNext steps:")
        print("  1. Test search: python3 -c \"import asyncio; from storage.engine import get_engine; asyncio.run((lambda: asyncio.run(get_engine()))().search_outcomes('implement feature'))\"")
        print("  2. Run predictions: python3 predict_session.py 'implement authentication'")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
