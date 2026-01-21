#!/usr/bin/env python3
"""
CPB Precision Mode - CLI Interface

Command-line interface for research-grounded, evidence-verified answers.

Usage:
    cpb-precision "What are best practices for multi-agent orchestration?"
    cpb-precision "Compare RLM vs ACE" --context @design.md
    cpb-precision "Analyze v2 improvements" --verbose --output result.md
    cpb-precision --interactive

Or via module:
    python3 -m cpb.precision_cli "Your query here"
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from .precision_config import (
    PRECISION_CONFIG, PRECISION_AGENT_PERSONAS,
    validate_precision_config
)
from .precision_orchestrator import (
    PrecisionResult, PrecisionStatus,
    execute_precision, get_precision_status
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_header():
    """Print CLI header."""
    print()
    print("=" * 70)
    print("  CPB PRECISION MODE")
    print("  Research-grounded, evidence-verified answers (DQ ≥ 0.95)")
    print("=" * 70)


def print_status(status: PrecisionStatus):
    """Print real-time status update."""
    bar_width = 30
    filled = int(status.progress / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    dq_display = f"DQ: {status.current_dq:.2f}/{status.target_dq:.2f}" if status.current_dq > 0 else ""
    retry_display = f"[Retry {status.retry_attempt}]" if status.retry_attempt > 0 else ""

    print(f"\r[{bar}] {status.progress:3d}% {status.current_step[:40]:40s} {dq_display} {retry_display}", end="", flush=True)


def print_result(result: PrecisionResult, verbose: bool = False):
    """Print formatted result."""
    print("\n")
    print("=" * 70)
    print(f"  ANSWER (DQ: {result.dq_score:.3f})")
    print("=" * 70)
    print()
    print(result.output)

    print()
    print("=" * 70)
    print("  EVIDENCE SUMMARY")
    print("=" * 70)

    print(f"\nSources ({result.citations_found} cited, {result.citations_verified} verified):")
    for i, source in enumerate(result.sources[:15], 1):
        source_type = source.get('type', 'unknown')
        if source_type == 'arxiv':
            arxiv_id = source.get('arxiv_id', 'unknown')
            print(f"  [{i}] arXiv:{arxiv_id}")
        elif source_type == 'session':
            session_id = source.get('session_id', 'unknown')
            print(f"  [{i}] Session: {session_id}")
        else:
            print(f"  [{i}] {source}")

    print("\nCritic Validation:")
    if result.verification:
        v = result.verification
        evidence_icon = "✅" if v.evidence_score >= 0.85 else "⚠️" if v.evidence_score >= 0.7 else "❌"
        oracle_icon = "✅" if v.oracle_score >= 0.85 else "⚠️" if v.oracle_score >= 0.7 else "❌"
        conf_icon = "✅" if v.confidence_score >= 0.80 else "⚠️" if v.confidence_score >= 0.7 else "❌"

        print(f"  {evidence_icon} EvidenceCritic: {v.evidence_score:.2f} ({v.citations_verified}/{v.citations_found} citations verified)")
        print(f"  {oracle_icon} OracleConsensus: {v.oracle_score:.2f}")
        print(f"  {conf_icon} ConfidenceScorer: {v.confidence_score:.2f}")

        # Ground Truth Diagnostic (v2.2)
        gt_icon = "✅" if v.ground_truth_score >= 0.7 else "⚠️" if v.ground_truth_score >= 0.5 else "🔬"
        print(f"\n  Ground Truth (diagnostic):")
        print(f"    {gt_icon} Score: {v.ground_truth_score:.2f} ({v.claims_verified}/{v.claims_checked} claims matched)")
        if v.factual_accuracy > 0:
            print(f"       Factual accuracy: {v.factual_accuracy:.2f}")
        if v.cross_source_score > 0:
            print(f"       Cross-source: {v.cross_source_score:.2f}")
        if v.self_consistency > 0:
            print(f"       Self-consistency: {v.self_consistency:.2f}")

        # Needs review flag
        if v.needs_review:
            print(f"\n  🔍 NEEDS REVIEW: Ground truth issues detected - recommend human verification")

        if v.issues:
            print(f"\n  Issues ({len(v.issues)}):")
            for issue in v.issues[:5]:
                severity_icon = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}
                icon = severity_icon.get(issue.get('severity', 'info'), "•")
                print(f"    {icon} [{issue.get('code')}] {issue.get('message')}")

    print("\nExecution:")
    print(f"  Path: {result.path.value}")
    print(f"  Agents: {result.agent_count}")
    print(f"  Retries: {result.retry_count}/{PRECISION_CONFIG.max_retries}")
    print(f"  Time: {result.execution_time_ms}ms")
    print(f"  RG Mode: {result.rg_connection_mode}")

    if result.warnings:
        print("\n⚠️ Warnings:")
        for w in result.warnings:
            print(f"  • {w}")

    if verbose:
        print("\n--- Verbose Details ---")
        print(f"Validity: {result.validity:.3f}")
        print(f"Specificity: {result.specificity:.3f}")
        print(f"Correctness: {result.correctness:.3f}")

        if result.feedback_history:
            print("\nFeedback History:")
            for i, fb in enumerate(result.feedback_history, 1):
                print(f"  Retry {i}:")
                for line in fb.split('\n'):
                    print(f"    {line}")

    print()
    print("=" * 70)


def print_agents():
    """Print agent information."""
    print("\n7-Agent Precision Ensemble:")
    print("-" * 50)

    for agent in PRECISION_AGENT_PERSONAS:
        print(f"\n{agent['name']} ({agent['role']})")
        # Print first sentence of prompt
        prompt_intro = agent['prompt'].split('\n')[0]
        print(f"  {prompt_intro[:70]}...")

    print()


def print_config_status():
    """Print configuration status."""
    status = get_precision_status()
    warnings = validate_precision_config(PRECISION_CONFIG)

    print("\nPrecision Mode Configuration:")
    print("-" * 50)

    config = status.get('config', {})
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nResearchGravity Status:")
    rg = status.get('rg_status', {})
    for key, value in rg.items():
        print(f"  {key}: {value}")

    # v2.2: Ground Truth Corpus stats
    try:
        from .ground_truth import get_corpus
        corpus = get_corpus()
        stats = corpus.get_corpus_stats()
        print("\nGround Truth Corpus (v2.2):")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Total claims: {stats['total_claims']}")
        print(f"  High confidence: {stats['high_confidence']}")
        print(f"  Medium confidence: {stats['medium_confidence']}")
        print(f"  Baseline: {stats['baseline']}")
        if stats['avg_dq_score'] > 0:
            print(f"  Avg DQ score: {stats['avg_dq_score']:.3f}")
    except Exception:
        pass  # Corpus stats optional

    if warnings:
        print("\n⚠️ Configuration Warnings:")
        for w in warnings:
            print(f"  • {w}")

    print()


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

async def cmd_query(args):
    """Execute a precision query."""
    print_header()

    # Load context if provided
    context = None
    if args.context:
        if args.context.startswith('@'):
            # Load from file
            context_path = Path(args.context[1:])
            if context_path.exists():
                context = context_path.read_text()
                print(f"\n📄 Loaded context from {context_path} ({len(context):,} chars)")
            else:
                print(f"\n⚠️ Context file not found: {context_path}")
        else:
            context = args.context

    print(f"\n📝 Query: {args.query}")
    print(f"🎯 Target DQ: {PRECISION_CONFIG.dq_threshold}")
    print(f"🤖 Agents: {PRECISION_CONFIG.ace_config.agent_count}")
    print()

    # Status callback
    def on_status(status: PrecisionStatus):
        if not args.quiet:
            print_status(status)

    # Execute
    start = time.time()
    result = await execute_precision(
        args.query,
        context=context,
        on_status=on_status if not args.quiet else None
    )
    elapsed = time.time() - start

    # Clear status line
    if not args.quiet:
        print()

    # Display result
    print_result(result, verbose=args.verbose)

    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_content = format_output(result, args.query, context)

        if args.json:
            output_content = json.dumps(result.to_dict(), indent=2)

        output_path.write_text(output_content)
        print(f"\n💾 Output saved to {output_path}")

    return 0 if result.verified else 1


def format_output(result: PrecisionResult, query: str, context: Optional[str]) -> str:
    """Format result for file output."""
    lines = [
        "# CPB Precision Mode Result",
        "",
        f"**Query:** {query}",
        f"**DQ Score:** {result.dq_score:.3f}",
        f"**Verified:** {'✅ Yes' if result.verified else '❌ No'}",
        f"**Retries:** {result.retry_count}",
        "",
        "---",
        "",
        result.output,
        "",
        "---",
        "",
        "## Evidence",
        ""
    ]

    for i, source in enumerate(result.sources[:15], 1):
        source_type = source.get('type', 'unknown')
        if source_type == 'arxiv':
            arxiv_id = source.get('arxiv_id', 'unknown')
            lines.append(f"- [{i}] https://arxiv.org/abs/{arxiv_id}")
        elif source_type == 'session':
            session_id = source.get('session_id', 'unknown')
            lines.append(f"- [{i}] Session: {session_id}")

    lines.extend([
        "",
        "## Validation",
        "",
        f"- Validity: {result.validity:.3f}",
        f"- Specificity: {result.specificity:.3f}",
        f"- Correctness: {result.correctness:.3f}",
        f"- Citations: {result.citations_verified}/{result.citations_found} verified",
    ])

    return '\n'.join(lines)


async def cmd_interactive(args):
    """Run interactive REPL mode."""
    print_header()
    print("\n🔄 Interactive Mode - Type 'quit' to exit")
    print("-" * 50)

    while True:
        try:
            query = input("\n📝 Query: ").strip()

            if query.lower() in ('quit', 'exit', 'q'):
                print("👋 Goodbye!")
                break

            if not query:
                continue

            if query.lower() == 'status':
                print_config_status()
                continue

            if query.lower() == 'agents':
                print_agents()
                continue

            # Execute query
            def on_status(status: PrecisionStatus):
                print_status(status)

            result = await execute_precision(query, on_status=on_status)
            print()
            print_result(result, verbose=False)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

    return 0


async def cmd_status(args):
    """Show precision mode status."""
    print_header()
    print_config_status()
    return 0


async def cmd_agents(args):
    """Show agent information."""
    print_header()
    print_agents()
    return 0


# =============================================================================
# MAIN
# =============================================================================

def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="CPB Precision Mode - Research-grounded, evidence-verified answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cpb-precision "What are best practices for multi-agent orchestration?"
  cpb-precision "Compare RLM vs ACE" --context @design.md
  cpb-precision "Analyze architecture" --verbose --output result.md
  cpb-precision --interactive
  cpb-precision --status
  cpb-precision --agents
        """
    )

    # Positional argument for query
    parser.add_argument(
        'query',
        nargs='?',
        help='Query to analyze (omit for interactive mode)'
    )

    # Context
    parser.add_argument(
        '--context', '-c',
        help='Additional context (use @filename to load from file)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Save output to file'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with detailed metrics'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    # Mode flags
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive REPL mode'
    )
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show configuration status'
    )
    parser.add_argument(
        '--agents', '-a',
        action='store_true',
        help='Show agent information'
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle mode flags
    if args.status:
        return asyncio.run(cmd_status(args))

    if args.agents:
        return asyncio.run(cmd_agents(args))

    if args.interactive or not args.query:
        return asyncio.run(cmd_interactive(args))

    # Execute query
    return asyncio.run(cmd_query(args))


if __name__ == '__main__':
    sys.exit(main())
