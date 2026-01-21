#!/usr/bin/env python3
"""
CPB Precision Mode - Feedback CLI

Human feedback collection for ground truth learning.

Usage:
    python3 -m cpb feedback --query "query" --output "output" --rating 4
    python3 -m cpb feedback --query "query" --output "output" --rating 5 --verified-claims "claim1" "claim2"
    python3 -m cpb feedback --query "query" --output "output" --rating 2 --false-claims "bad claim"
    python3 -m cpb feedback --list                  # List recent feedback
    python3 -m cpb feedback --stats                 # Show feedback statistics
    python3 -m cpb feedback --export feedback.json  # Export feedback data

Research Foundation:
- arXiv:2512.00047 (Emergent Convergence) - Self-consistency
- arXiv:2508.17536 (Voting vs Debate) - Agreement as quality signal
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Optional, List

from .ground_truth import FeedbackCollector


def print_banner():
    """Print feedback CLI banner."""
    print("=" * 70)
    print("CPB PRECISION MODE - FEEDBACK COLLECTOR")
    print("=" * 70)
    print("Contribute to ground truth learning by providing feedback.")
    print()


def record_feedback(
    query: str,
    output: str,
    rating: int,
    corrections: Optional[str] = None,
    verified_claims: Optional[List[str]] = None,
    false_claims: Optional[List[str]] = None
):
    """Record human feedback on a response."""
    collector = FeedbackCollector()
    collector.record_feedback(
        query=query,
        output=output,
        rating=rating,
        corrections=corrections,
        verified_claims=verified_claims,
        false_claims=false_claims
    )

    print("✅ Feedback recorded successfully!")
    print()
    print(f"  Query: {query[:50]}...")
    print(f"  Rating: {'★' * rating}{'☆' * (5 - rating)} ({rating}/5)")
    if verified_claims:
        print(f"  Verified claims: {len(verified_claims)}")
    if false_claims:
        print(f"  False claims: {len(false_claims)}")
    if corrections:
        print(f"  Corrections: {corrections[:100]}...")


def list_feedback(limit: int = 10):
    """List recent feedback entries."""
    collector = FeedbackCollector()
    feedback = collector._load_feedback()

    if not feedback:
        print("No feedback recorded yet.")
        return

    print(f"Recent feedback ({min(limit, len(feedback))} of {len(feedback)} entries):")
    print("-" * 70)

    for entry in feedback[-limit:][::-1]:  # Most recent first
        timestamp = entry.get('timestamp', 'Unknown')
        query = entry.get('query', '')[:40]
        rating = entry.get('rating', 0)
        verified = len(entry.get('verified_claims', []))
        false = len(entry.get('false_claims', []))

        stars = '★' * rating + '☆' * (5 - rating)
        print(f"[{timestamp[:16]}] {stars} | V:{verified} F:{false} | {query}...")


def show_stats():
    """Show feedback statistics."""
    collector = FeedbackCollector()
    feedback = collector._load_feedback()

    if not feedback:
        print("No feedback recorded yet.")
        return

    total = len(feedback)
    ratings = [e.get('rating', 0) for e in feedback]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0

    verified_count = sum(len(e.get('verified_claims', [])) for e in feedback)
    false_count = sum(len(e.get('false_claims', [])) for e in feedback)

    print("═" * 50)
    print("FEEDBACK STATISTICS")
    print("═" * 50)
    print(f"Total entries:     {total}")
    print(f"Average rating:    {avg_rating:.2f}/5.0 {'★' * round(avg_rating)}{'☆' * (5 - round(avg_rating))}")
    print(f"Verified claims:   {verified_count}")
    print(f"False claims:      {false_count}")
    print()

    # Rating distribution
    print("Rating distribution:")
    for r in range(5, 0, -1):
        count = ratings.count(r)
        bar = '█' * count
        print(f"  {r}★: {bar} ({count})")

    # Ground truth claims
    gt_claims = collector.get_ground_truth_claims()
    positive = [c for c in gt_claims if c.confidence > 0.5]
    negative = [c for c in gt_claims if c.confidence < 0.5]

    print()
    print("Ground truth claims:")
    print(f"  Positive (verified):   {len(positive)}")
    print(f"  Negative (false):      {len(negative)}")


def export_feedback(output_path: str):
    """Export feedback to JSON file."""
    collector = FeedbackCollector()
    feedback = collector._load_feedback()

    with open(output_path, 'w') as f:
        json.dump({
            'exported_at': datetime.now().isoformat(),
            'total_entries': len(feedback),
            'feedback': feedback
        }, f, indent=2)

    print(f"✅ Exported {len(feedback)} feedback entries to {output_path}")


def interactive_feedback():
    """Interactive feedback collection mode."""
    print_banner()
    print("Enter feedback interactively. Type 'quit' to exit.")
    print()

    collector = FeedbackCollector()

    while True:
        print("-" * 50)
        query = input("Query (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break

        output = input("Response snippet: ").strip()
        if not output:
            print("Skipping (no response provided)")
            continue

        try:
            rating = int(input("Rating (1-5): ").strip())
            if rating < 1 or rating > 5:
                print("Invalid rating, using 3")
                rating = 3
        except ValueError:
            print("Invalid rating, using 3")
            rating = 3

        # Optional verified claims
        verified_input = input("Verified claims (comma-separated, or blank): ").strip()
        verified_claims = [c.strip() for c in verified_input.split(',')] if verified_input else None

        # Optional false claims
        false_input = input("False claims (comma-separated, or blank): ").strip()
        false_claims = [c.strip() for c in false_input.split(',')] if false_input else None

        # Optional corrections
        corrections = input("Corrections (or blank): ").strip() or None

        collector.record_feedback(
            query=query,
            output=output,
            rating=rating,
            corrections=corrections,
            verified_claims=verified_claims,
            false_claims=false_claims
        )

        print("✅ Feedback recorded!")

    print()
    print("Thank you for contributing to ground truth learning!")


def main():
    """Main entry point for feedback CLI."""
    parser = argparse.ArgumentParser(
        description="CPB Precision Mode - Feedback Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record feedback on a response
    python3 -m cpb feedback --query "What is CPB?" --output "CPB is..." --rating 4

    # Record with verified/false claims
    python3 -m cpb feedback --query "Q" --output "A" --rating 5 \\
        --verified-claims "claim 1" "claim 2" \\
        --false-claims "bad claim"

    # List recent feedback
    python3 -m cpb feedback --list

    # Show statistics
    python3 -m cpb feedback --stats

    # Interactive mode
    python3 -m cpb feedback --interactive

    # Export feedback
    python3 -m cpb feedback --export feedback.json
        """
    )

    parser.add_argument('--query', '-q', type=str, help='Query that was asked')
    parser.add_argument('--output', '-o', type=str, help='Response that was generated')
    parser.add_argument('--rating', '-r', type=int, choices=[1, 2, 3, 4, 5],
                        help='Rating (1-5, where 5 is best)')
    parser.add_argument('--corrections', '-c', type=str, help='Text corrections')
    parser.add_argument('--verified-claims', nargs='*', default=None,
                        help='Claims verified as true')
    parser.add_argument('--false-claims', nargs='*', default=None,
                        help='Claims identified as false')

    parser.add_argument('--list', '-l', action='store_true',
                        help='List recent feedback entries')
    parser.add_argument('--limit', type=int, default=10,
                        help='Number of entries to show in list')
    parser.add_argument('--stats', '-s', action='store_true',
                        help='Show feedback statistics')
    parser.add_argument('--export', '-e', type=str, metavar='FILE',
                        help='Export feedback to JSON file')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive feedback mode')

    args = parser.parse_args()

    # Handle subcommands
    if args.list:
        list_feedback(args.limit)
    elif args.stats:
        show_stats()
    elif args.export:
        export_feedback(args.export)
    elif args.interactive:
        interactive_feedback()
    elif args.query and args.output and args.rating:
        record_feedback(
            query=args.query,
            output=args.output,
            rating=args.rating,
            corrections=args.corrections,
            verified_claims=args.verified_claims,
            false_claims=args.false_claims
        )
    else:
        parser.print_help()
        print()
        print("Tip: Use --interactive for guided feedback collection")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
