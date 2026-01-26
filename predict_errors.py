#!/usr/bin/env python3
"""
Error Prediction Tool

Predicts potential errors before you start a task and provides prevention strategies.

Usage:
    python3 predict_errors.py "git clone repository"
    python3 predict_errors.py "implement authentication" --verbose
    python3 predict_errors.py --strategies git
"""

import asyncio
import argparse
from storage.meta_learning import get_meta_engine


def format_errors(errors: list, verbose: bool = False) -> str:
    """Format error predictions for display."""
    output = []

    output.append("\n" + "=" * 70)
    output.append("⚠️  Error Prediction & Prevention")
    output.append("=" * 70)

    if not errors:
        output.append("\n✅ No significant error patterns detected for this task")
        output.append("\nYou're in the clear! Proceed with confidence.")
        output.append("=" * 70)
        return "\n".join(output)

    output.append(f"\n🔍 Found {len(errors)} potential error patterns")
    output.append("\nTop Preventable Errors:")

    for i, error in enumerate(errors, 1):
        error_type = error.get("error_type", "unknown")
        success_rate = error.get("success_rate", 0.0)
        context = error.get("context", "")
        solution = error.get("solution", "")
        score = error.get("score", 0.0)

        # Severity indicator
        severity = error.get("severity", "medium")
        severity_emoji = "🔴" if severity == "high" else "🟡"

        output.append(f"\n{i}. {severity_emoji} {error_type.upper()}")
        output.append(f"   Relevance: {score:.2f} | Prevention success: {success_rate:.0%}")

        if verbose:
            if context:
                output.append(f"\n   Context: {context[:150]}...")
            if solution:
                output.append(f"\n   ✅ Prevention:")
                # Split solution into lines
                for line in solution.split('\n')[:3]:
                    if line.strip():
                        output.append(f"      {line.strip()[:60]}...")
        else:
            # Brief solution
            if solution:
                brief = solution.split('\n')[0][:80]
                output.append(f"   💡 {brief}...")

    output.append("\n" + "-" * 70)
    output.append("💡 Recommendation:")

    if errors:
        high_risk = [e for e in errors if e.get("severity") == "high"]
        if high_risk:
            output.append(f"   ⚠️  {len(high_risk)} high-risk patterns detected")
            output.append("   Review prevention strategies before proceeding")
        else:
            output.append("   ✅ Moderate risk - be aware of potential issues")

    output.append("=" * 70)
    return "\n".join(output)


def format_strategies(strategies: dict) -> str:
    """Format prevention strategies for display."""
    output = []

    output.append("\n" + "=" * 70)
    output.append(f"🛡️  Prevention Strategies: {strategies['error_type'].upper()}")
    output.append("=" * 70)

    output.append(f"\nSuccess Rate: {strategies['success_rate']:.0%}")
    output.append(f"Patterns Analyzed: {strategies['pattern_count']}")

    if strategies['strategies']:
        output.append("\n📋 Prevention Strategies:")
        for i, strategy in enumerate(strategies['strategies'], 1):
            output.append(f"\n{i}. {strategy[:200]}")
            if len(strategy) > 200:
                output.append("   ...")

    if strategies['examples']:
        output.append("\n📝 Common Examples:")
        for i, example in enumerate(strategies['examples'], 1):
            output.append(f"\n{i}. {example}")

    output.append("\n" + "=" * 70)
    return "\n".join(output)


async def predict_errors(task: str, verbose: bool = False):
    """Predict errors for a task."""
    engine = await get_meta_engine()

    errors = await engine.predict_errors(
        intent=task,
        include_preventable_only=True
    )

    print(format_errors(errors, verbose))

    await engine.close()


async def get_strategies(error_type: str):
    """Get prevention strategies for an error type."""
    engine = await get_meta_engine()

    strategies = await engine.get_prevention_strategies(error_type)

    print(format_strategies(strategies))

    await engine.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Predict and prevent errors before they happen"
    )
    parser.add_argument("task", nargs="?", help="Task description")
    parser.add_argument("--strategies", metavar="TYPE", help="Get prevention strategies for error type (git, concurrency, permissions, etc.)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed context and solutions")
    args = parser.parse_args()

    if args.strategies:
        await get_strategies(args.strategies)
    elif args.task:
        await predict_errors(args.task, args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
