#!/usr/bin/env python3
"""
Session Outcome Predictor

Uses the Meta-Learning Engine to predict session outcomes before you start working.

Usage:
    python3 predict_session.py "implement authentication system"
    python3 predict_session.py "fix bug in API" --hour 20
    python3 predict_session.py "add multi-agent feature" --verbose
"""

import asyncio
import argparse
from datetime import datetime
from typing import Optional

from storage.meta_learning import get_meta_engine


def format_prediction(prediction: dict, verbose: bool = False, simulated_hour: Optional[int] = None) -> str:
    """Format prediction for display."""
    output = []

    # Header
    output.append("\n" + "=" * 70)
    output.append("🔮 Session Outcome Prediction")
    output.append("=" * 70)

    # Core prediction
    quality = prediction["predicted_quality"]
    prob = prediction["success_probability"]
    confidence = prediction["confidence"]

    # Visual quality indicator
    stars = "⭐" * int(quality)
    quality_color = "🟢" if quality >= 4 else "🟡" if quality >= 3 else "🔴"

    output.append(f"\n{quality_color} Predicted Quality: {quality:.1f}/5 {stars}")
    output.append(f"   Success Probability: {prob:.0%}")
    output.append(f"   Confidence: {confidence:.0%}")

    # Timing recommendation
    optimal_hour = prediction["optimal_time"]
    current_hour = simulated_hour if simulated_hour is not None else datetime.now().hour

    if abs(current_hour - optimal_hour) <= 1:
        output.append(f"\n✅ Good timing! Current hour ({current_hour}:00) is near optimal ({optimal_hour}:00)")
    else:
        hours_diff = (optimal_hour - current_hour) % 24
        output.append(f"\n⏰ Suboptimal timing")
        output.append(f"   Current: {current_hour}:00")
        output.append(f"   Optimal: {optimal_hour}:00 (wait {hours_diff}h)")

    # Recommended research
    if prediction["recommended_research"]:
        output.append(f"\n📚 Recommended Research:")
        for i, r in enumerate(prediction["recommended_research"], 1):
            content = r.get("content", "")[:60]
            score = r.get("relevance_score", r.get("score", 0))
            output.append(f"   {i}. [{score:.2f}] {content}...")
    else:
        output.append(f"\n📚 No specific research recommendations")

    # Potential errors
    if prediction["potential_errors"]:
        output.append(f"\n⚠️  Potential Issues to Avoid:")
        for e in prediction["potential_errors"]:
            error_type = e.get("error_type", "Unknown")
            success_rate = e.get("success_rate", 0)
            solution = e.get("solution", "")[:50]
            output.append(f"   - {error_type} (prevention success: {success_rate:.0%})")
            if solution:
                output.append(f"     Solution: {solution}...")
    else:
        output.append(f"\n✅ No known error patterns detected")

    # Similar sessions
    if prediction["similar_sessions"]:
        output.append(f"\n🔍 Similar Past Sessions:")
        for s in prediction["similar_sessions"]:
            intent = s.get("intent", "")[:50]
            outcome = s.get("outcome", "unknown")
            quality = s.get("quality", 0)
            score = s.get("relevance_score", s.get("score", 0))
            outcome_emoji = "✅" if outcome == "success" else "⚠️" if outcome == "partial" else "❌"
            output.append(f"   {outcome_emoji} [{score:.2f}] {intent}... (Q: {quality}/5)")

    # Verbose mode - show signals
    if verbose:
        output.append(f"\n📊 Signal Breakdown:")
        signals = prediction.get("signals", {})
        output.append(f"   Outcome Score: {signals.get('outcome_score', 0):.2f}")
        output.append(f"   Cognitive Alignment: {signals.get('cognitive_alignment', 0):.2f}")
        output.append(f"   Research Availability: {signals.get('research_availability', 0):.2f}")
        output.append(f"   Error Probability: {signals.get('error_probability', 0):.2f}")

    # Recommendation
    output.append("\n" + "-" * 70)
    if prob >= 0.7:
        output.append("💡 Recommendation: Strong indicators for success. Proceed!")
    elif prob >= 0.5:
        output.append("💡 Recommendation: Moderate success probability. Consider waiting for optimal time.")
    else:
        output.append("💡 Recommendation: Low success probability. Review research and wait for optimal conditions.")
    output.append("=" * 70)

    return "\n".join(output)


async def predict(intent: str, hour: Optional[int] = None, verbose: bool = False):
    """Run prediction for a session intent."""
    engine = await get_meta_engine()

    # Build cognitive state
    current_hour = hour if hour is not None else datetime.now().hour
    current_day = datetime.now().strftime("%A")

    # Simple cognitive mode mapping (would use actual cognitive-os in production)
    if 5 <= current_hour < 9:
        mode = "morning"
        energy = 0.6
    elif 9 <= current_hour < 12 or 14 <= current_hour < 18:
        mode = "peak"
        energy = 0.8
    elif 12 <= current_hour < 14:
        mode = "dip"
        energy = 0.5
    elif 18 <= current_hour < 22:
        mode = "evening"
        energy = 0.7
    else:
        mode = "deep_night"
        energy = 0.9 if current_hour in [2, 20] else 0.6

    cognitive_state = {
        "mode": mode,
        "hour": current_hour,
        "day": current_day,
        "energy_level": energy,
        "flow_score": 0.5  # Neutral
    }

    # Get prediction
    prediction = await engine.predict_session_outcome(
        intent=intent,
        cognitive_state=cognitive_state
    )

    # Display
    print(format_prediction(prediction, verbose, current_hour))

    await engine.close()


async def predict_optimal_time(intent: str):
    """Predict optimal time for a task."""
    engine = await get_meta_engine()

    result = await engine.predict_optimal_time(intent)

    print("\n" + "=" * 70)
    print("⏰ Optimal Timing Analysis")
    print("=" * 70)
    print(f"\nTask: {intent}")
    print(f"\nOptimal Hour: {result['optimal_hour']}:00")
    print(f"Is Optimal Now: {'✅ Yes' if result['is_optimal_now'] else '❌ No'}")
    if result['wait_hours'] > 0:
        print(f"Wait Time: {result['wait_hours']} hours")
    print(f"\nReasoning: {result['reasoning']}")
    print("=" * 70)

    await engine.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Predict session outcomes using Meta-Learning Engine"
    )
    parser.add_argument("intent", nargs="?", help="Session intent (what you want to accomplish)")
    parser.add_argument("--hour", type=int, help="Hour to simulate (0-23)")
    parser.add_argument("--optimal-time", action="store_true", help="Show optimal time for task")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed signal breakdown")
    args = parser.parse_args()

    if not args.intent:
        parser.print_help()
        return

    if args.optimal_time:
        await predict_optimal_time(args.intent)
    else:
        await predict(args.intent, args.hour, args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
