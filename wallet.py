#!/usr/bin/env python3
"""
Universal Cognitive Wallet CLI

Commands:
    wallet.py status     Show wallet status and value
    wallet.py value      Show detailed value breakdown
    wallet.py export     Export wallet to UCW format
    wallet.py history    Show value history over time
    wallet.py inject     Inject context into CLAUDE.md

Examples:
    python3 wallet.py status
    python3 wallet.py value
    python3 wallet.py export --output my-wallet.ucw.json
    python3 wallet.py inject
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ucw.schema import CognitiveWallet
from ucw.export import build_wallet_from_agent_core, export_wallet, export_wallet_summary
from ucw.value import (
    CognitiveAppreciationEngine,
    format_value_display,
    get_value_breakdown,
)
from ucw.adapters.claude import ClaudeAdapter
from ucw.history import format_history_chart, format_history_table, get_value_delta


def cmd_status(args):
    """Show wallet status and value."""
    print("\n  Loading wallet from agent-core...")
    wallet = build_wallet_from_agent_core()

    stats = wallet.get_stats()

    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║           UNIVERSAL COGNITIVE WALLET                      ║
  ╠═══════════════════════════════════════════════════════════╣
  ║                                                           ║
  ║   Status: ACTIVE                                          ║
  ║   Version: {wallet.version}                                          ║
  ║   Integrity: {wallet.integrity_hash[:12] if wallet.integrity_hash else 'Not set'}...                        ║
  ║                                                           ║
  ╠═══════════════════════════════════════════════════════════╣
  ║                                                           ║
  ║   Sessions:    {stats['sessions']:>5}                                     ║
  ║   Concepts:    {stats['concepts']:>5}                                     ║
  ║   Papers:      {stats['papers']:>5}                                     ║
  ║   URLs:        {stats['urls']:>5}                                     ║
  ║   Connections: {stats['connections']:>5}                                     ║
  ║                                                           ║
  ╠═══════════════════════════════════════════════════════════╣
  ║                                                           ║
  ║   💰 WALLET VALUE: ${stats['value']:>10,.2f}                         ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝
""")

    # Show domains
    if stats['domains']:
        print("  Domains:")
        for domain, weight in sorted(stats['domains'].items(), key=lambda x: -x[1]):
            bar_len = int(weight * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    {domain:20} {bar} {weight*100:>4.0f}%")
        print()


def cmd_value(args):
    """Show detailed value breakdown."""
    print("\n  Calculating wallet value...")
    wallet = build_wallet_from_agent_core()

    print(format_value_display(wallet))

    # Show top concepts if verbose
    if args.verbose:
        breakdown = get_value_breakdown(wallet)
        print("\n  Top Concepts by Value:")
        for content, value in breakdown.top_concepts[:10]:
            print(f"    ${value:>6.2f}  {content}")
        print()


def cmd_export(args):
    """Export wallet to UCW format."""
    print("\n  Building wallet from agent-core...")
    wallet = build_wallet_from_agent_core()

    output_path = args.output or f"wallet-{datetime.now().strftime('%Y%m%d-%H%M%S')}.ucw.json"

    print(f"  Exporting to: {output_path}")
    export_wallet(wallet, output_path, pretty=not args.compact)

    print(f"""
  ✓ Wallet exported successfully!

  Stats:
    Sessions:  {len(wallet.sessions)}
    Concepts:  {len(wallet.concepts)}
    Papers:    {len(wallet.papers)}
    Value:     ${wallet.value_metrics.total_value:,.2f}

  File: {output_path}
""")


def cmd_history(args):
    """Show value history over time."""
    # Build wallet to ensure history is recorded
    wallet = build_wallet_from_agent_core()

    # Show chart if requested
    if args.chart:
        days = args.days or 30
        print(format_history_chart(days=days))
    else:
        # Show table by default
        print(format_history_table(limit=args.limit or 10))

    # Show 7-day delta
    delta, pct = get_value_delta(7)
    if delta != 0:
        arrow = "↑" if delta >= 0 else "↓"
        print(f"  7-day change: {arrow} ${abs(delta):,.2f} ({pct:+.1f}%)")
        print()


def cmd_inject(args):
    """Inject wallet context into CLAUDE.md."""
    print("\n  Building wallet from agent-core...")
    wallet = build_wallet_from_agent_core()

    adapter = ClaudeAdapter()
    claude_md = Path(args.target) if args.target else Path.home() / "CLAUDE.md"

    if adapter.inject_into_claude_md(wallet, claude_md):
        print(f"""
  ✓ Context injected successfully!

  Target: {claude_md}
  Value: ${wallet.value_metrics.total_value:,.2f}
  Concepts: {len(wallet.concepts)}
  Papers: {len(wallet.papers)}
""")
    else:
        print(f"\n  ✗ Failed to inject context into {claude_md}")


def cmd_summary(args):
    """Show markdown summary."""
    wallet = build_wallet_from_agent_core()
    print(export_wallet_summary(wallet))


def main():
    parser = argparse.ArgumentParser(
        description="Universal Cognitive Wallet CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 wallet.py status           # Show wallet status
  python3 wallet.py value -v         # Detailed value breakdown
  python3 wallet.py export -o w.json # Export to file
  python3 wallet.py inject           # Inject into CLAUDE.md
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status
    status_parser = subparsers.add_parser("status", help="Show wallet status and value")

    # value
    value_parser = subparsers.add_parser("value", help="Show detailed value breakdown")
    value_parser.add_argument("-v", "--verbose", action="store_true", help="Show top concepts")

    # export
    export_parser = subparsers.add_parser("export", help="Export wallet to UCW format")
    export_parser.add_argument("-o", "--output", help="Output file path")
    export_parser.add_argument("--compact", action="store_true", help="Compact JSON output")

    # history
    history_parser = subparsers.add_parser("history", help="Show value history over time")
    history_parser.add_argument("-c", "--chart", action="store_true", help="Show ASCII chart")
    history_parser.add_argument("-d", "--days", type=int, default=30, help="Days to show (default: 30)")
    history_parser.add_argument("-l", "--limit", type=int, default=10, help="Number of entries to show (default: 10)")

    # inject
    inject_parser = subparsers.add_parser("inject", help="Inject context into CLAUDE.md")
    inject_parser.add_argument("-t", "--target", help="Target CLAUDE.md path")

    # summary
    summary_parser = subparsers.add_parser("summary", help="Show markdown summary")

    args = parser.parse_args()

    if args.command == "status" or args.command is None:
        cmd_status(args)
    elif args.command == "value":
        cmd_value(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "inject":
        cmd_inject(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
