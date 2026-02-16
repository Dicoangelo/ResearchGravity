#!/usr/bin/env python3
"""
Simple backfill - Direct SQLite and Qdrant writes

Avoids connection pool issues by using direct writes.
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # noqa: E402
from storage.qdrant_db import QdrantDB

HOME = Path.home()
OUTCOMES_FILE = HOME / ".claude" / "data" / "session-outcomes.jsonl"
FATE_FILE = HOME / ".claude" / "kernel" / "cognitive-os" / "fate-predictions.jsonl"
ROUTING_FILE = HOME / ".claude" / "kernel" / "cognitive-os" / "routing-decisions.jsonl"
FLOW_FILE = HOME / ".claude" / "kernel" / "cognitive-os" / "flow-history.jsonl"
DB_PATH = HOME / ".agent-core" / "storage" / "antigravity.db"


def read_jsonl(file_path: Path):
    records = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    return records


async def backfill_sqlite():
    """Backfill SQLite directly."""
    outcomes = read_jsonl(OUTCOMES_FILE)

    # Filter valid
    outcomes = [o for o in outcomes if o.get("intent") or o.get("title")]

    print(f"📊 Backfilling {len(outcomes)} outcomes to SQLite...")

    # Direct SQLite connection (not using aiosqlite)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    cursor = conn.cursor()

    count = 0
    for o in outcomes:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO session_outcomes
                (id, session_id, intent, outcome, quality, model_efficiency, models_used, date, messages, tools)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                o.get("session_id"),
                o.get("session_id"),
                o.get("intent", o.get("title", ""))[:500],
                o.get("outcome"),
                o.get("quality"),
                o.get("model_efficiency"),
                json.dumps(o.get("models_used", {})),
                o.get("date"),
                o.get("messages"),
                o.get("tools")
            ))
            count += 1
            if count % 50 == 0:
                print(f"  Progress: {count}/{len(outcomes)}")
                conn.commit()
        except Exception as e:
            print(f"  Error on record {count}: {e}")

    conn.commit()
    conn.close()

    print(f"✅ SQLite: Stored {count} outcomes")
    return count


async def backfill_qdrant():
    """Backfill Qdrant."""
    outcomes = read_jsonl(OUTCOMES_FILE)
    outcomes = [o for o in outcomes if o.get("intent") or o.get("title")]

    print(f"📊 Backfilling {len(outcomes)} outcomes to Qdrant...")

    qdrant = QdrantDB()
    await qdrant.initialize()

    # Process in batches
    batch_size = 50
    count = 0

    for i in range(0, len(outcomes), batch_size):
        batch = outcomes[i:i + batch_size]
        try:
            batch_count = await qdrant.upsert_outcomes_batch(batch)
            count += batch_count
            print(f"  Progress: {count}/{len(outcomes)}")
        except Exception as e:
            print(f"  Error on batch {i//batch_size}: {e}")

    await qdrant.close()

    print(f"✅ Qdrant: Stored {count} outcomes")
    return count


async def process_cognitive_states():
    """Process cognitive states from multiple sources."""
    states = []

    # 1. Fate predictions
    if FATE_FILE.exists():
        fate_records = read_jsonl(FATE_FILE)
        print(f"📊 Processing {len(fate_records)} fate predictions...")

        for record in fate_records:
            ts = record.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hour = dt.hour
                    day = dt.strftime("%A")
                except:
                    hour = 0
                    day = ""

                success_prob = record.get("success_probability", 0.5)
                energy = 0.3 + (success_prob * 0.7)

                state = {
                    "id": f"fate-{ts}",
                    "mode": "unknown",
                    "energy_level": energy,
                    "flow_score": 0.0,
                    "hour": hour,
                    "day": day,
                    "timestamp": ts,
                    "predictions": {
                        "predicted": record.get("predicted"),
                        "success_probability": success_prob
                    }
                }
                states.append(state)

    # 2. Routing decisions
    if ROUTING_FILE.exists():
        routing_records = read_jsonl(ROUTING_FILE)
        print(f"📊 Processing {len(routing_records)} routing decisions...")

        for record in routing_records:
            ts = record.get("timestamp", "")
            if ts:
                mode = record.get("cognitive_mode", "unknown")
                energy_map = {
                    "morning": 0.6, "peak": 0.8, "dip": 0.5,
                    "evening": 0.7, "deep_night": 0.9
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
                        "dq_score": record.get("dq_score")
                    }
                }
                states.append(state)

    # 3. Flow history
    if FLOW_FILE.exists():
        flow_records = read_jsonl(FLOW_FILE)
        print(f"📊 Processing {len(flow_records)} flow records...")

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
                state_name = record.get("state", "neutral")
                energy_map = {
                    "deep_flow": 1.0, "flow": 0.8, "focused": 0.7,
                    "neutral": 0.5, "distracted": 0.3, "struggling": 0.2
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
                        "session_id": record.get("session_id")
                    }
                }
                states.append(state)

    return states


async def backfill_cognitive_sqlite(states):
    """Backfill cognitive states to SQLite."""
    print(f"📊 Backfilling {len(states)} cognitive states to SQLite...")

    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    cursor = conn.cursor()

    count = 0
    for s in states:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO cognitive_states
                (id, mode, energy_level, flow_score, hour, day, predictions, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                s["id"],
                s["mode"],
                s["energy_level"],
                s["flow_score"],
                s["hour"],
                s["day"],
                json.dumps(s.get("predictions", {})),
                s["timestamp"]
            ))
            count += 1
            if count % 100 == 0:
                print(f"  Progress: {count}/{len(states)}")
                conn.commit()
        except Exception as e:
            print(f"  Error on record {count}: {e}")

    conn.commit()
    conn.close()

    print(f"✅ SQLite: Stored {count} cognitive states")
    return count


async def backfill_cognitive_qdrant(states):
    """Backfill cognitive states to Qdrant."""
    print(f"📊 Backfilling {len(states)} cognitive states to Qdrant...")

    qdrant = QdrantDB()
    await qdrant.initialize()

    batch_size = 50
    count = 0

    for i in range(0, len(states), batch_size):
        batch = states[i:i + batch_size]
        try:
            batch_count = await qdrant.upsert_cognitive_states_batch(batch)
            count += batch_count
            print(f"  Progress: {count}/{len(states)}")
        except Exception as e:
            print(f"  Error on batch {i//batch_size}: {e}")

    await qdrant.close()

    print(f"✅ Qdrant: Stored {count} cognitive states")
    return count


async def main():
    print("=" * 60)
    print("Simple Backfill - Session Outcomes & Cognitive States")
    print("=" * 60)

    # Backfill outcomes
    print("\n1. Session Outcomes")
    print("-" * 60)
    outcomes_sqlite = await backfill_sqlite()
    outcomes_qdrant = await backfill_qdrant()

    # Process and backfill cognitive states
    print("\n2. Cognitive States")
    print("-" * 60)
    states = await process_cognitive_states()

    cognitive_sqlite = 0
    cognitive_qdrant = 0

    if states:
        cognitive_sqlite = await backfill_cognitive_sqlite(states)
        cognitive_qdrant = await backfill_cognitive_qdrant(states)

    print("\n" + "=" * 60)
    print(f"✅ Total backfilled:")
    print(f"   Outcomes:  SQLite {outcomes_sqlite} | Qdrant {outcomes_qdrant}")
    print(f"   Cognitive: SQLite {cognitive_sqlite} | Qdrant {cognitive_qdrant}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
