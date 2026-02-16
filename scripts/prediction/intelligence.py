#!/usr/bin/env python3
"""
ResearchGravity Intelligence Layer
===================================

Exposes meta-learning predictions via CLI and API.

Available Predictions:
- Session quality (1-5 scale)
- Success probability
- Optimal hour for task type
- Likely errors with solutions
- Related research papers
- Session pattern detection

Usage:
  python3 intelligence.py predict "task description"
  python3 intelligence.py optimal-time
  python3 intelligence.py errors "git workflow"
  python3 intelligence.py research "multi-agent"
  python3 intelligence.py patterns
  python3 intelligence.py calibrate
  python3 intelligence.py status
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


# Paths
AGENT_CORE_DIR = Path.home() / ".agent-core"
CLAUDE_DATA_DIR = Path.home() / ".claude" / "data"
COGNITIVE_STATE_FILE = Path.home() / ".claude" / "kernel" / "cognitive-os" / "state.json"
SESSION_OUTCOMES_FILE = CLAUDE_DATA_DIR / "session-outcomes.jsonl"


@dataclass
class SessionPrediction:
    """Prediction for a session."""
    intent: str
    predicted_quality: float  # 1-5
    success_probability: float  # 0-1
    optimal_hour: int  # 0-23
    cognitive_mode: str  # morning, peak, dip, evening, deep_night
    energy_level: float  # 0-1
    likely_errors: List[Dict[str, str]]
    related_research: List[Dict[str, str]]
    confidence: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_current_hour() -> int:
    """Get current hour (0-23)."""
    return datetime.now().hour


def get_cognitive_mode(hour: int) -> tuple[str, float]:
    """Determine cognitive mode and energy level from hour."""
    # Based on typical patterns
    if 5 <= hour < 9:
        return "morning", 0.7
    elif 9 <= hour < 12:
        return "peak", 0.95
    elif 12 <= hour < 14:
        return "dip", 0.5
    elif 14 <= hour < 18:
        return "peak", 0.85
    elif 18 <= hour < 22:
        return "evening", 0.6
    else:
        return "deep_night", 0.4


def load_cognitive_state() -> Dict[str, Any]:
    """Load current cognitive state from file."""
    if COGNITIVE_STATE_FILE.exists():
        try:
            return json.loads(COGNITIVE_STATE_FILE.read_text())
        except Exception:
            pass

    # Default state based on current time
    hour = get_current_hour()
    mode, energy = get_cognitive_mode(hour)

    return {
        "mode": mode,
        "energy_level": energy,
        "hour": hour,
        "flow_score": 0.5,
    }


def load_session_outcomes(days: int = 30) -> List[Dict[str, Any]]:
    """Load recent session outcomes."""
    outcomes = []
    cutoff = datetime.now() - timedelta(days=days)

    if SESSION_OUTCOMES_FILE.exists():
        try:
            with open(SESSION_OUTCOMES_FILE, 'r') as f:
                for line in f:
                    try:
                        outcome = json.loads(line.strip())
                        date_str = outcome.get("date", "")
                        if date_str:
                            outcome_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                            if outcome_date.replace(tzinfo=None) >= cutoff:
                                outcomes.append(outcome)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception:
            pass

    return outcomes


async def predict_session_quality(
    intent: str,
    storage_engine=None
) -> SessionPrediction:
    """
    Predict session quality based on intent and context.

    Uses:
    - Historical session outcomes
    - Current cognitive state
    - Similar past intents
    - Error pattern history
    """
    # Get cognitive state
    cognitive_state = load_cognitive_state()
    hour = cognitive_state.get("hour", get_current_hour())
    mode = cognitive_state.get("mode", "unknown")
    energy = cognitive_state.get("energy_level", 0.5)
    flow = cognitive_state.get("flow_score", 0.5)

    # Load historical outcomes
    outcomes = load_session_outcomes(days=30)

    # Calculate base quality prediction
    # Start with average from similar hour outcomes
    hour_outcomes = [o for o in outcomes if o.get("hour") == hour]
    if hour_outcomes:
        avg_quality = sum(o.get("quality", 3) for o in hour_outcomes) / len(hour_outcomes)
    else:
        # Default based on cognitive mode
        mode_quality = {
            "peak": 4.2,
            "morning": 3.8,
            "evening": 3.5,
            "dip": 3.0,
            "deep_night": 2.8,
        }
        avg_quality = mode_quality.get(mode, 3.5)

    # Adjust for energy and flow
    quality_adjustment = (energy - 0.5) * 0.5 + (flow - 0.5) * 0.5
    predicted_quality = max(1.0, min(5.0, avg_quality + quality_adjustment))

    # Calculate success probability
    success_outcomes = [o for o in outcomes if o.get("outcome") == "success"]
    if outcomes:
        base_success = len(success_outcomes) / len(outcomes)
    else:
        base_success = 0.7  # Default

    # Adjust for current state
    success_probability = base_success * (0.5 + energy * 0.5)
    success_probability = max(0.1, min(0.95, success_probability))

    # Find optimal hour
    hour_success = {}
    for o in outcomes:
        h = o.get("hour", 12)
        if h not in hour_success:
            hour_success[h] = {"success": 0, "total": 0}
        hour_success[h]["total"] += 1
        if o.get("outcome") == "success":
            hour_success[h]["success"] += 1

    if hour_success:
        optimal_hour = max(
            hour_success.keys(),
            key=lambda h: hour_success[h]["success"] / max(hour_success[h]["total"], 1)
        )
    else:
        optimal_hour = 10  # Default peak hour

    # Find likely errors
    likely_errors = await get_likely_errors(intent, storage_engine)

    # Find related research
    related_research = await get_related_research(intent, storage_engine)

    # Calculate confidence
    confidence = min(0.9, len(outcomes) / 100 + 0.3)

    return SessionPrediction(
        intent=intent,
        predicted_quality=round(predicted_quality, 2),
        success_probability=round(success_probability, 2),
        optimal_hour=optimal_hour,
        cognitive_mode=mode,
        energy_level=round(energy, 2),
        likely_errors=likely_errors[:3],
        related_research=related_research[:3],
        confidence=round(confidence, 2),
        timestamp=datetime.now().isoformat(),
    )


async def get_likely_errors(
    context: str,
    storage_engine=None
) -> List[Dict[str, str]]:
    """Get likely errors based on context."""
    errors = []

    # Try storage engine first
    if storage_engine:
        try:
            results = await storage_engine.search_error_patterns(
                context, limit=5, min_success_rate=0.5
            )
            for r in results:
                errors.append({
                    "error_type": r.get("error_type", "unknown"),
                    "context": r.get("context", "")[:100],
                    "solution": r.get("solution", "")[:200],
                    "success_rate": r.get("success_rate", 0),
                })
        except Exception:
            pass

    # Fallback to common patterns based on context keywords
    if not errors:
        patterns = [
            ("git", "merge_conflict", "git conflicts during merge", "Use git status, resolve conflicts manually"),
            ("git", "push_rejected", "push rejected by remote", "Pull first with rebase: git pull --rebase"),
            ("npm", "node_modules_issue", "node_modules corruption", "Delete node_modules and package-lock.json, then npm install"),
            ("python", "import_error", "module not found", "Check virtual environment activation"),
            ("docker", "container_stopped", "container exits immediately", "Check logs with docker logs <container>"),
            ("test", "flaky_test", "inconsistent test results", "Add retry logic or fix race conditions"),
            ("build", "cache_stale", "stale build cache", "Clear build cache and rebuild"),
        ]

        context_lower = context.lower()
        for keyword, error_type, ctx, solution in patterns:
            if keyword in context_lower:
                errors.append({
                    "error_type": error_type,
                    "context": ctx,
                    "solution": solution,
                    "success_rate": 0.8,
                })

    return errors


async def get_related_research(
    query: str,
    storage_engine=None
) -> List[Dict[str, str]]:
    """Get related research papers/findings."""
    research = []

    if storage_engine:
        try:
            results = await storage_engine.search_findings(
                query, limit=5, filter_type="research"
            )
            for r in results:
                research.append({
                    "content": r.get("content", "")[:200],
                    "score": r.get("score", 0),
                    "session_id": r.get("session_id", ""),
                })
        except Exception:
            pass

    return research


def get_optimal_time(task_type: str = "general") -> Dict[str, Any]:
    """Get optimal time for a task type."""
    outcomes = load_session_outcomes(days=30)

    # Group by hour and task type
    hour_stats = {}
    for o in outcomes:
        hour = o.get("hour", 12)
        outcome = o.get("outcome", "")
        quality = o.get("quality", 3)

        if hour not in hour_stats:
            hour_stats[hour] = {"success": 0, "total": 0, "quality_sum": 0}

        hour_stats[hour]["total"] += 1
        hour_stats[hour]["quality_sum"] += quality
        if outcome == "success":
            hour_stats[hour]["success"] += 1

    # Find best hours
    scored_hours = []
    for hour, stats in hour_stats.items():
        if stats["total"] < 3:
            continue
        success_rate = stats["success"] / stats["total"]
        avg_quality = stats["quality_sum"] / stats["total"]
        score = success_rate * 0.5 + (avg_quality / 5) * 0.5
        scored_hours.append({
            "hour": hour,
            "success_rate": round(success_rate, 2),
            "avg_quality": round(avg_quality, 2),
            "score": round(score, 2),
            "sample_size": stats["total"],
        })

    scored_hours.sort(key=lambda x: x["score"], reverse=True)

    return {
        "task_type": task_type,
        "current_hour": get_current_hour(),
        "optimal_hours": scored_hours[:5],
        "recommendation": scored_hours[0] if scored_hours else {"hour": 10, "note": "insufficient data"},
    }


def get_session_patterns() -> Dict[str, Any]:
    """Analyze session patterns."""
    outcomes = load_session_outcomes(days=30)

    if not outcomes:
        return {"message": "No session data available"}

    # Analyze patterns
    patterns = {
        "total_sessions": len(outcomes),
        "by_outcome": {},
        "by_mode": {},
        "by_hour": {},
        "avg_quality": 0,
        "success_rate": 0,
    }

    quality_sum = 0
    success_count = 0

    for o in outcomes:
        # By outcome
        outcome = o.get("outcome", "unknown")
        patterns["by_outcome"][outcome] = patterns["by_outcome"].get(outcome, 0) + 1
        if outcome == "success":
            success_count += 1

        # By mode
        mode = o.get("cognitive_mode", "unknown")
        patterns["by_mode"][mode] = patterns["by_mode"].get(mode, 0) + 1

        # By hour
        hour = str(o.get("hour", "unknown"))
        patterns["by_hour"][hour] = patterns["by_hour"].get(hour, 0) + 1

        # Quality
        quality_sum += o.get("quality", 0)

    patterns["avg_quality"] = round(quality_sum / len(outcomes), 2)
    patterns["success_rate"] = round(success_count / len(outcomes), 2)

    # Find dominant pattern
    if patterns["by_mode"]:
        dominant_mode = max(patterns["by_mode"].items(), key=lambda x: x[1])
        patterns["dominant_mode"] = {"mode": dominant_mode[0], "count": dominant_mode[1]}

    return patterns


async def run_calibration(storage_engine=None) -> Dict[str, Any]:
    """Run calibration loop to measure prediction accuracy."""
    if not storage_engine:
        return {"error": "Storage engine required for calibration"}

    try:
        accuracy = await storage_engine.get_prediction_accuracy(days=30)
        return {
            "calibration_metrics": accuracy,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}


async def get_status() -> Dict[str, Any]:
    """Get intelligence layer status."""
    cognitive_state = load_cognitive_state()
    outcomes = load_session_outcomes(days=7)

    return {
        "status": "active",
        "cognitive_state": cognitive_state,
        "recent_sessions": len(outcomes),
        "predictions_available": [
            "session_quality",
            "success_probability",
            "optimal_time",
            "likely_errors",
            "related_research",
            "session_patterns",
        ],
        "timestamp": datetime.now().isoformat(),
    }


async def main():
    parser = argparse.ArgumentParser(
        description="ResearchGravity Intelligence Layer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict session quality")
    predict_parser.add_argument("intent", help="Task/intent description")

    # Optimal time
    time_parser = subparsers.add_parser("optimal-time", help="Get optimal time for tasks")
    time_parser.add_argument("--task-type", default="general", help="Task type")

    # Errors
    errors_parser = subparsers.add_parser("errors", help="Get likely errors")
    errors_parser.add_argument("context", help="Context to search for errors")

    # Research
    research_parser = subparsers.add_parser("research", help="Find related research")
    research_parser.add_argument("query", help="Research query")

    # Patterns
    subparsers.add_parser("patterns", help="Show session patterns")

    # Calibrate
    subparsers.add_parser("calibrate", help="Run calibration loop")

    # Status
    subparsers.add_parser("status", help="Show intelligence status")

    args = parser.parse_args()

    # Try to get storage engine
    storage_engine = None
    try:
        from storage.engine import get_engine
        storage_engine = await get_engine()
    except ImportError:
        pass

    try:
        if args.command == "predict":
            result = await predict_session_quality(args.intent, storage_engine)
            print(json.dumps(result.to_dict(), indent=2))

        elif args.command == "optimal-time":
            result = get_optimal_time(args.task_type)
            print(json.dumps(result, indent=2))

        elif args.command == "errors":
            errors = await get_likely_errors(args.context, storage_engine)
            print(json.dumps({"likely_errors": errors}, indent=2))

        elif args.command == "research":
            research = await get_related_research(args.query, storage_engine)
            print(json.dumps({"related_research": research}, indent=2))

        elif args.command == "patterns":
            patterns = get_session_patterns()
            print(json.dumps(patterns, indent=2))

        elif args.command == "calibrate":
            result = await run_calibration(storage_engine)
            print(json.dumps(result, indent=2))

        elif args.command == "status":
            status = await get_status()
            print(json.dumps(status, indent=2))

        else:
            parser.print_help()

    finally:
        if storage_engine:
            await storage_engine.close()


if __name__ == "__main__":
    asyncio.run(main())
