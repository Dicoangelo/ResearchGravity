"""
UCW Value History — Persistent tracking of wallet value over time.

This module handles:
- Loading historical value snapshots from disk
- Recording new value snapshots
- Calculating appreciation trends
- Generating history visualizations (ASCII charts)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AGENT_CORE = Path.home() / ".agent-core"
HISTORY_FILE = AGENT_CORE / "wallet_history.json"


def load_history() -> List[Dict]:
    """Load value history from disk."""
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(HISTORY_FILE.read_text())
        return data.get("history", [])
    except Exception as e:
        print(f"Warning: Could not load history: {e}")
        return []


def save_history(history: List[Dict]) -> None:
    """Save value history to disk."""
    AGENT_CORE.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "1.0",
        "last_updated": datetime.now().isoformat(),
        "history": history,
    }

    HISTORY_FILE.write_text(json.dumps(data, indent=2))


def record_snapshot(
    value: float,
    concepts: int,
    sessions: int,
    papers: int = 0,
    urls: int = 0,
) -> Dict:
    """Record a new value snapshot."""
    history = load_history()

    # Check if we already have a snapshot today
    today = datetime.now().date().isoformat()
    for entry in history:
        ts = entry.get("timestamp", "")
        if ts.startswith(today):
            # Update existing entry for today
            entry["value"] = value
            entry["concepts"] = concepts
            entry["sessions"] = sessions
            entry["papers"] = papers
            entry["urls"] = urls
            save_history(history)
            return entry

    # Add new snapshot
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "date": today,
        "value": value,
        "concepts": concepts,
        "sessions": sessions,
        "papers": papers,
        "urls": urls,
    }

    history.append(snapshot)

    # Keep last 365 days of history
    if len(history) > 365:
        history = history[-365:]

    save_history(history)
    return snapshot


def get_recent_history(days: int = 30) -> List[Dict]:
    """Get history for the last N days."""
    history = load_history()

    cutoff = datetime.now() - timedelta(days=days)
    recent = []

    for entry in history:
        try:
            ts = datetime.fromisoformat(entry.get("timestamp", ""))
            if ts >= cutoff:
                recent.append(entry)
        except:
            pass

    return recent


def calculate_appreciation(days: int = 30) -> Optional[float]:
    """Calculate appreciation rate over the last N days."""
    history = get_recent_history(days)

    if len(history) < 2:
        return None

    first_value = history[0].get("value", 0)
    last_value = history[-1].get("value", 0)

    if first_value == 0:
        return None

    return (last_value - first_value) / first_value


def get_value_delta(days: int = 7) -> Tuple[float, float]:
    """Get value change over the last N days."""
    history = get_recent_history(days)

    if len(history) < 2:
        return (0.0, 0.0)

    first_value = history[0].get("value", 0)
    last_value = history[-1].get("value", 0)

    delta = last_value - first_value
    pct = (delta / first_value * 100) if first_value > 0 else 0

    return (delta, pct)


def format_history_chart(days: int = 30, width: int = 50) -> str:
    """Generate ASCII chart of value history."""
    history = get_recent_history(days)

    if not history:
        return "  No history available yet.\n"

    values = [e.get("value", 0) for e in history]
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val > min_val else 1

    lines = [
        "  VALUE HISTORY (Last {} days)".format(days),
        "  " + "═" * (width + 10),
        "",
    ]

    # Show min/max
    lines.append(f"  Max: ${max_val:,.2f}")

    # ASCII chart (vertical bars)
    chart_height = 8
    chart = []

    # Sample to fit width
    step = max(1, len(values) // width)
    sampled = values[::step][-width:]

    for row in range(chart_height):
        threshold = max_val - (row + 1) * (range_val / chart_height)
        line = "  "
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        chart.append(line)

    lines.extend(chart)
    lines.append("  " + "─" * len(sampled))
    lines.append(f"  Min: ${min_val:,.2f}")
    lines.append("")

    # Summary stats
    delta, pct = get_value_delta(days)
    arrow = "↑" if delta >= 0 else "↓"
    lines.append(f"  {days}-day change: {arrow} ${abs(delta):,.2f} ({pct:+.1f}%)")
    lines.append("")

    return "\n".join(lines)


def format_history_table(limit: int = 10) -> str:
    """Generate table of recent history entries."""
    history = load_history()

    if not history:
        return "  No history available yet.\n"

    lines = [
        "",
        "  RECENT VALUE SNAPSHOTS",
        "  " + "═" * 50,
        "",
        "  Date         Value        Concepts  Sessions",
        "  " + "─" * 50,
    ]

    for entry in history[-limit:]:
        date = entry.get("date", "")[:10]
        value = entry.get("value", 0)
        concepts = entry.get("concepts", 0)
        sessions = entry.get("sessions", 0)

        lines.append(f"  {date}   ${value:>10,.2f}   {concepts:>6}    {sessions:>5}")

    lines.append("  " + "─" * 50)
    lines.append("")

    # Show appreciation
    appreciation = calculate_appreciation(30)
    if appreciation is not None:
        pct = appreciation * 100
        arrow = "↑" if pct >= 0 else "↓"
        lines.append(f"  30-day appreciation: {arrow} {abs(pct):.1f}%")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test history display
    print(format_history_chart())
    print(format_history_table())
