#!/usr/bin/env python3
"""
Cognitive Precision Bridge (CPB) - CLI Interface

Command-line tools for CPB analysis, scoring, and management.

Usage:
    python3 -m cpb.cli analyze "Your query here"
    python3 -m cpb.cli score --query "Q" --response "R"
    python3 -m cpb.cli stats
    python3 -m cpb.cli ace-prompts "What's the best approach?"
    python3 -m cpb.cli status
"""

import argparse
import json

from .orchestrator import cpb, analyze
from .dq_scorer import dq_scorer


def cmd_analyze(args):
    """Analyze query complexity"""
    result = analyze(args.query, args.context)

    print("\n" + "=" * 60)
    print("  CPB QUERY ANALYSIS")
    print("=" * 60)

    print(f"\nQuery: {result['query']}")
    if args.context:
        print(f"Context: {len(args.context):,} chars")

    print(f"\n📊 Complexity Score: {result['complexity_score']:.3f}")
    print(f"🔀 Selected Path: {result['selected_path'].upper()}")
    print(f"📝 Reasoning: {result['reasoning']}")
    print(f"🎯 Confidence: {result['confidence']}%")

    print("\n📡 Signals:")
    signals = result['signals']
    for key, value in signals.items():
        indicator = "✓" if value else "✗" if isinstance(value, bool) else str(value)
        print(f"   {key}: {indicator}")

    if result['alternatives']:
        print("\n🔄 Alternatives:")
        for alt in result['alternatives']:
            print(f"   • {alt['path']} (score: {alt['score']:.2f})")
            print(f"     Tradeoff: {alt['tradeoff']}")

    print("\n" + "=" * 60)

    if args.json:
        print("\nJSON Output:")
        print(json.dumps(result, indent=2, default=str))


def cmd_score(args):
    """Score a query-response pair"""
    dq = dq_scorer.score(args.query, args.response, args.context)

    print("\n" + "=" * 60)
    print("  DQ SCORE ANALYSIS")
    print("=" * 60)

    print(f"\nQuery: {args.query[:80]}{'...' if len(args.query) > 80 else ''}")
    print(f"Response: {args.response[:80]}{'...' if len(args.response) > 80 else ''}")

    print(f"\n📊 Overall DQ Score: {dq.overall:.3f}")

    # Score bar
    bar_length = 30
    filled = int(dq.overall * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"   [{bar}] {dq.overall*100:.1f}%")

    print("\n📈 Component Breakdown:")
    print(f"   Validity (40%):    {dq.validity:.3f}")
    print(f"   Specificity (30%): {dq.specificity:.3f}")
    print(f"   Correctness (30%): {dq.correctness:.3f}")

    tier = dq_scorer.get_quality_tier(dq)
    tier_emoji = {"excellent": "🌟", "good": "✅", "acceptable": "⚠️", "below_threshold": "❌"}
    print(f"\n🏆 Quality Tier: {tier_emoji.get(tier, '')} {tier.upper()}")

    suggestions = dq_scorer.suggest_improvements(dq)
    if suggestions:
        print("\n💡 Suggestions:")
        for s in suggestions:
            print(f"   • {s}")

    if args.log:
        dq_scorer.log_score(args.query, args.response, dq, model=args.model or 'cli')
        print("\n✅ Score logged to metrics")

    print("\n" + "=" * 60)


def cmd_stats(args):
    """Show DQ scoring statistics"""
    stats = dq_scorer.get_stats(args.days)

    print("\n" + "=" * 60)
    print(f"  DQ STATISTICS (Last {args.days} days)")
    print("=" * 60)

    if 'message' in stats:
        print(f"\n{stats['message']}")
        return

    print(f"\n📊 Total Scored: {stats['total_scored']}")
    print(f"📈 Average DQ:   {stats['avg_dq']:.3f}")
    print(f"📉 Min/Max:      {stats['min_dq']:.3f} / {stats['max_dq']:.3f}")
    print(f"✅ Above 0.75:   {stats['above_threshold']}")
    print(f"❌ Below 0.60:   {stats['below_min']}")

    if stats.get('by_model'):
        print("\n📊 By Model:")
        for model, data in stats['by_model'].items():
            if model:
                print(f"   {model}: {data['count']} scored, avg {data['avg_dq']:.3f}")

    if stats.get('by_path'):
        print("\n🔀 By Path:")
        for path, data in stats['by_path'].items():
            if path:
                print(f"   {path}: {data['count']} scored, avg {data['avg_dq']:.3f}")

    print("\n" + "=" * 60)


def cmd_ace_prompts(args):
    """Generate ACE consensus prompts"""
    prompts = cpb.build_ace_prompts(args.query, args.context, args.agents)

    print("\n" + "=" * 60)
    print("  ACE CONSENSUS PROMPTS")
    print("=" * 60)

    print(f"\nQuery: {args.query}")
    print(f"Agent Count: {len(prompts)}")

    for i, p in enumerate(prompts, 1):
        print(f"\n{'─' * 50}")
        print(f"🤖 Agent {i}: {p['agent']}")
        print(f"{'─' * 50}")

        if args.full:
            print(f"\n[System Prompt]\n{p['system_prompt']}")
            print(f"\n[User Prompt]\n{p['user_prompt']}")
        else:
            print(f"\nSystem: {p['system_prompt'][:100]}...")
            print(f"User: {p['user_prompt'][:100]}...")

    print("\n" + "=" * 60)

    if args.json:
        print("\nJSON Output:")
        print(json.dumps(prompts, indent=2))


def cmd_status(args):
    """Show CPB status and configuration"""
    status = cpb.get_status()
    warnings = cpb.validate_config()

    print("\n" + "=" * 60)
    print("  CPB STATUS")
    print("=" * 60)

    print(f"\n🏷️  Tier: {status['tier'].upper()}")
    print(f"📚 Learning: {'Enabled' if status['learning_enabled'] else 'Disabled'}")
    print(f"✅ Verification: {'Enabled' if status['verification_enabled'] else 'Disabled'}")

    print("\n⚙️  Configuration:")
    for key, value in status['config'].items():
        print(f"   {key}: {value}")

    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"   • {w}")

    # Get learned preferences
    prefs = cpb.get_learned_preferences()
    if 'by_path' in prefs:
        print(f"\n📊 Learned Patterns ({prefs['total_patterns']} recorded):")
        for path, data in prefs['by_path'].items():
            print(f"   {path}: {data['count']} uses, avg DQ {data['avg_dq']:.3f}")
        print(f"   Recommended default: {prefs['recommended_default']}")

    print("\n" + "=" * 60)


def cmd_learn(args):
    """View and manage learned patterns"""
    prefs = cpb.get_learned_preferences()

    print("\n" + "=" * 60)
    print("  CPB LEARNED PATTERNS")
    print("=" * 60)

    if 'message' in prefs:
        print(f"\n{prefs['message']}")
        return

    print(f"\n📊 Total Patterns: {prefs['total_patterns']}")

    if prefs.get('by_path'):
        print("\n📈 Performance by Path:")
        for path, data in prefs['by_path'].items():
            success_pct = data.get('success_rate', 0) * 100
            print(f"\n   🔀 {path.upper()}")
            print(f"      Uses: {data['count']}")
            print(f"      Avg DQ: {data['avg_dq']:.3f}")
            print(f"      Avg Time: {data['avg_time_ms']:.0f}ms")
            print(f"      Success: {success_pct:.1f}%")

    print(f"\n🎯 Recommended Default Path: {prefs.get('recommended_default', 'cascade')}")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Precision Bridge (CPB) CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m cpb.cli analyze "Design a microservices architecture"
  python3 -m cpb.cli score --query "What is X?" --response "X is..."
  python3 -m cpb.cli stats --days 30
  python3 -m cpb.cli ace-prompts "Compare REST vs GraphQL"
  python3 -m cpb.cli status
  python3 -m cpb.cli learn
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_p = subparsers.add_parser('analyze', help='Analyze query complexity')
    analyze_p.add_argument('query', help='Query to analyze')
    analyze_p.add_argument('--context', '-c', help='Optional context')
    analyze_p.add_argument('--json', '-j', action='store_true', help='Output JSON')

    # Score command
    score_p = subparsers.add_parser('score', help='Score a response')
    score_p.add_argument('--query', '-q', required=True, help='Original query')
    score_p.add_argument('--response', '-r', required=True, help='Response to score')
    score_p.add_argument('--context', '-c', help='Optional context')
    score_p.add_argument('--log', '-l', action='store_true', help='Log score to metrics')
    score_p.add_argument('--model', '-m', help='Model name for logging')

    # Stats command
    stats_p = subparsers.add_parser('stats', help='Show DQ statistics')
    stats_p.add_argument('--days', '-d', type=int, default=7, help='Days to analyze')

    # ACE prompts command
    ace_p = subparsers.add_parser('ace-prompts', help='Generate ACE consensus prompts')
    ace_p.add_argument('query', help='Query for ACE analysis')
    ace_p.add_argument('--context', '-c', help='Optional context')
    ace_p.add_argument('--agents', '-a', type=int, default=5, help='Number of agents')
    ace_p.add_argument('--full', '-f', action='store_true', help='Show full prompts')
    ace_p.add_argument('--json', '-j', action='store_true', help='Output JSON')

    # Status command
    status_p = subparsers.add_parser('status', help='Show CPB status')

    # Learn command
    learn_p = subparsers.add_parser('learn', help='View learned patterns')

    args = parser.parse_args()

    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'score':
        cmd_score(args)
    elif args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'ace-prompts':
        cmd_ace_prompts(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'learn':
        cmd_learn(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
