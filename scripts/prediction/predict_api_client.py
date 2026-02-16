#!/usr/bin/env python3
"""
Prediction API Client

CLI tool to interact with Meta-Learning prediction endpoints.

Usage:
    python3 predict_api_client.py predict "implement authentication"
    python3 predict_api_client.py errors "git clone repository"
    python3 predict_api_client.py optimal-time "architecture design"
    python3 predict_api_client.py accuracy
    python3 predict_api_client.py multi-search "multi-agent system"
"""

import argparse
import asyncio
import json
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("httpx not installed. Run: pip install httpx")


API_BASE_URL = "http://localhost:3847"


async def predict_session(
    intent: str,
    cognitive_state: Optional[Dict[str, Any]] = None,
    track: bool = False
):
    """Call session prediction endpoint."""
    if not HTTPX_AVAILABLE:
        return

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/api/v2/predict/session",
            json={
                "intent": intent,
                "cognitive_state": cognitive_state,
                "track_prediction": track
            },
            timeout=30.0
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None


async def predict_errors(intent: str, preventable_only: bool = True):
    """Call error prediction endpoint."""
    if not HTTPX_AVAILABLE:
        return

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/api/v2/predict/errors",
            json={
                "intent": intent,
                "include_preventable_only": preventable_only
            },
            timeout=30.0
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None


async def predict_optimal_time(intent: str, current_hour: Optional[int] = None):
    """Call optimal time prediction endpoint."""
    if not HTTPX_AVAILABLE:
        return

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/api/v2/predict/optimal-time",
            json={
                "intent": intent,
                "current_hour": current_hour
            },
            timeout=30.0
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None


async def get_accuracy(days: int = 30):
    """Call accuracy metrics endpoint."""
    if not HTTPX_AVAILABLE:
        return

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/api/v2/predict/accuracy",
            params={"days": days},
            timeout=30.0
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None


async def multi_search(query: str, limit: int = 5):
    """Call multi-vector search endpoint."""
    if not HTTPX_AVAILABLE:
        return

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/api/v2/predict/multi-search",
            params={"query": query, "limit": limit},
            timeout=30.0
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None


def format_prediction(prediction: Dict[str, Any]):
    """Format prediction output."""
    print("\n" + "=" * 70)
    print("🔮 Session Outcome Prediction")
    print("=" * 70)

    quality = prediction.get("predicted_quality", 0)
    success_prob = prediction.get("success_probability", 0)
    confidence = prediction.get("confidence", 0)
    optimal_hour = prediction.get("optimal_time", 0)

    # Quality stars
    stars = "⭐" * int(quality)
    if quality >= 4.0:
        emoji = "🟢"
    elif quality >= 3.0:
        emoji = "🟡"
    else:
        emoji = "🔴"

    print(f"\n{emoji} Predicted Quality: {quality}/5 {stars}")
    print(f"   Success Probability: {success_prob:.0%}")
    print(f"   Confidence: {confidence:.0%}")
    print(f"   Optimal Time: {optimal_hour}:00")

    # Similar sessions
    similar = prediction.get("similar_sessions", [])
    if similar:
        print(f"\n📚 Similar Sessions: {len(similar)}")
        for i, session in enumerate(similar[:3], 1):
            print(f"   {i}. {session.get('intent', 'Unknown')} → {session.get('outcome', 'unknown')} ({session.get('quality', 0)}/5)")

    # Recommended research
    research = prediction.get("recommended_research", [])
    if research:
        print(f"\n🔬 Recommended Research: {len(research)}")
        for i, paper in enumerate(research[:3], 1):
            print(f"   {i}. {paper.get('content', '')[:60]}...")

    # Potential errors
    errors = prediction.get("potential_errors", [])
    if errors:
        print(f"\n⚠️  Potential Errors: {len(errors)}")
        for i, error in enumerate(errors[:3], 1):
            error_type = error.get("error_type", "unknown")
            success_rate = error.get("success_rate", 0)
            print(f"   {i}. {error_type.upper()} - {success_rate:.0%} preventable")

    # Signals
    signals = prediction.get("signals", {})
    if signals:
        print(f"\n📊 Signal Breakdown:")
        print(f"   Outcome Score: {signals.get('outcome_score', 0):.2f}")
        print(f"   Cognitive Alignment: {signals.get('cognitive_alignment', 0):.2f}")
        print(f"   Research Availability: {signals.get('research_availability', 0):.2f}")
        print(f"   Error Probability: {signals.get('error_probability', 0):.2f}")

    # Prediction ID (if tracked)
    prediction_id = prediction.get("prediction_id")
    if prediction_id:
        print(f"\n🔗 Prediction ID: {prediction_id}")
        print("   (Stored for calibration tracking)")

    print("\n" + "=" * 70)


def format_errors(errors_response: Dict[str, Any]):
    """Format error predictions output."""
    print("\n" + "=" * 70)
    print("⚠️  Error Prediction & Prevention")
    print("=" * 70)

    errors = errors_response.get("errors", [])

    if not errors:
        print("\n✅ No significant error patterns detected")
        print("   You're in the clear!")
        print("=" * 70)
        return

    print(f"\n🔍 Found {len(errors)} potential error patterns")

    for i, error in enumerate(errors, 1):
        error_type = error.get("error_type", "unknown")
        success_rate = error.get("success_rate", 0)
        severity = error.get("severity", "medium")
        score = error.get("score", 0)

        severity_emoji = "🔴" if severity == "high" else "🟡"

        print(f"\n{i}. {severity_emoji} {error_type.upper()}")
        print(f"   Relevance: {score:.2f} | Prevention: {success_rate:.0%}")

        solution = error.get("solution", "")
        if solution:
            brief = solution.split('\n')[0][:80]
            print(f"   💡 {brief}...")

    print("\n" + "=" * 70)


def format_optimal_time(result: Dict[str, Any]):
    """Format optimal time output."""
    print("\n" + "=" * 70)
    print("⏰ Optimal Timing Prediction")
    print("=" * 70)

    optimal_hour = result.get("optimal_hour", 0)
    is_optimal = result.get("is_optimal_now", False)
    wait_hours = result.get("wait_hours", 0)
    reasoning = result.get("reasoning", "")

    print(f"\n📍 Optimal Hour: {optimal_hour}:00")
    print(f"   Current Status: {'✅ Optimal now!' if is_optimal else f'⏳ Wait {wait_hours}h'}")
    print(f"   Reasoning: {reasoning}")

    print("\n" + "=" * 70)


def format_accuracy(accuracy: Dict[str, Any]):
    """Format accuracy metrics output."""
    print("\n" + "=" * 70)
    print("📊 Prediction Accuracy Metrics")
    print("=" * 70)

    total = accuracy.get("total_predictions", 0)
    accurate = accuracy.get("accurate_predictions", 0)
    acc_rate = accuracy.get("accuracy", 0)
    quality_error = accuracy.get("avg_quality_error", 0)
    success_rate = accuracy.get("success_prediction_rate", 0)
    days = accuracy.get("period_days", 30)

    print(f"\n📈 Period: Last {days} days")
    print(f"   Total Predictions: {total}")
    print(f"   Accurate: {accurate}")
    print(f"   Accuracy: {acc_rate:.0%}")
    print(f"   Avg Quality Error: {quality_error:.2f}")
    print(f"   Success Prediction Rate: {success_rate:.0%}")

    if total == 0:
        print("\n💡 No predictions tracked yet")
        print("   Use --track flag when making predictions to enable calibration")

    print("\n" + "=" * 70)


def format_multi_search(results: Dict[str, Any]):
    """Format multi-search results."""
    print("\n" + "=" * 70)
    print("🔍 Multi-Vector Search Results")
    print("=" * 70)

    total = results.get("total_results", 0)
    print(f"\n📊 Total Results: {total}")

    # Outcomes
    outcomes = results.get("outcomes", [])
    if outcomes:
        print(f"\n📝 Session Outcomes ({len(outcomes)}):")
        for i, outcome in enumerate(outcomes[:3], 1):
            print(f"   {i}. {outcome.get('intent', 'Unknown')} → {outcome.get('outcome', 'unknown')} ({outcome.get('quality', 0)}/5)")

    # Cognitive
    cognitive = results.get("cognitive", [])
    if cognitive:
        print(f"\n🧠 Cognitive States ({len(cognitive)}):")
        for i, state in enumerate(cognitive[:3], 1):
            mode = state.get("mode", "unknown")
            hour = state.get("hour", 0)
            energy = state.get("energy_level", 0)
            print(f"   {i}. {mode} @ hour {hour} (energy: {energy:.2f})")

    # Research
    research = results.get("research", [])
    if research:
        print(f"\n🔬 Research Findings ({len(research)}):")
        for i, finding in enumerate(research[:3], 1):
            content = finding.get("content", "")[:60]
            print(f"   {i}. {content}...")

    # Errors
    errors = results.get("errors", [])
    if errors:
        print(f"\n⚠️  Error Patterns ({len(errors)}):")
        for i, error in enumerate(errors[:3], 1):
            error_type = error.get("error_type", "unknown")
            success_rate = error.get("success_rate", 0)
            print(f"   {i}. {error_type.upper()} - {success_rate:.0%} preventable")

    print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Prediction API Client")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Predict session
    predict_parser = subparsers.add_parser("predict", help="Predict session outcome")
    predict_parser.add_argument("intent", help="Task intent/description")
    predict_parser.add_argument("--hour", type=int, help="Current hour (0-23)")
    predict_parser.add_argument("--mode", help="Cognitive mode (peak, dip, etc.)")
    predict_parser.add_argument("--track", action="store_true", help="Store prediction for tracking")

    # Predict errors
    errors_parser = subparsers.add_parser("errors", help="Predict potential errors")
    errors_parser.add_argument("intent", help="Task intent/description")

    # Optimal time
    time_parser = subparsers.add_parser("optimal-time", help="Find optimal time for task")
    time_parser.add_argument("intent", help="Task intent/description")
    time_parser.add_argument("--hour", type=int, help="Current hour (0-23)")

    # Accuracy
    accuracy_parser = subparsers.add_parser("accuracy", help="Get prediction accuracy")
    accuracy_parser.add_argument("--days", type=int, default=30, help="Days to analyze")

    # Multi-search
    search_parser = subparsers.add_parser("multi-search", help="Multi-vector search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Results per dimension")

    args = parser.parse_args()

    if not HTTPX_AVAILABLE:
        print("❌ httpx not installed. Run: pip install httpx")
        return

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "predict":
            cognitive_state = None
            if args.hour is not None or args.mode:
                cognitive_state = {}
                if args.hour is not None:
                    cognitive_state["hour"] = args.hour
                if args.mode:
                    cognitive_state["mode"] = args.mode

            result = await predict_session(
                args.intent,
                cognitive_state=cognitive_state,
                track=args.track
            )
            if result:
                format_prediction(result)

        elif args.command == "errors":
            result = await predict_errors(args.intent)
            if result:
                format_errors(result)

        elif args.command == "optimal-time":
            result = await predict_optimal_time(args.intent, args.hour)
            if result:
                format_optimal_time(result)

        elif args.command == "accuracy":
            result = await get_accuracy(args.days)
            if result:
                format_accuracy(result)

        elif args.command == "multi-search":
            result = await multi_search(args.query, args.limit)
            if result:
                format_multi_search(result)

    except httpx.ConnectError:
        print("❌ Cannot connect to API server")
        print("   Make sure the server is running: python3 -m api.server")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
