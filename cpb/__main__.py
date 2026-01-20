#!/usr/bin/env python3
"""
Allow running CPB as a module: python3 -m cpb

Subcommands:
    python3 -m cpb analyze "query"       # Standard CPB analysis
    python3 -m cpb precision "query"     # Precision mode (95% DQ target)
    python3 -m cpb feedback --stats      # Feedback collection for ground truth
    python3 -m cpb score --query Q --response R
    python3 -m cpb status
"""

import sys


def main():
    """Main entry point with precision mode and feedback support."""
    # Check for precision subcommand
    if len(sys.argv) > 1 and sys.argv[1] == 'precision':
        # Remove 'precision' from args and delegate to precision CLI
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .precision_cli import main as precision_main
        return precision_main()

    # Check for feedback subcommand (v2)
    if len(sys.argv) > 1 and sys.argv[1] == 'feedback':
        # Remove 'feedback' from args and delegate to feedback CLI
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .feedback_cli import main as feedback_main
        return feedback_main()

    # Default to standard CLI
    from .cli import main as cli_main
    return cli_main()


if __name__ == '__main__':
    sys.exit(main() or 0)
