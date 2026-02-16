#!/usr/bin/env python3
"""
Pack Selector - V2 Integrated (Production)

Intelligent pack selection with automatic V1/V2 routing:
- V2 (default): 7-layer world-class system
- V1 (fallback): Original DQ + ACE system

Usage:
  python3 select_packs_v2_integrated.py --context "debugging React performance"
  python3 select_packs_v2_integrated.py --auto  # Auto-detect
  python3 select_packs_v2_integrated.py --context "..." --v1  # Force V1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Try to import V2 engine
V2_AVAILABLE = False
try:
    from context_packs_v2_prototype import ContextPacksV2Engine
    V2_AVAILABLE = True
except ImportError:
    print("⚠️  V2 engine not available, using V1 only")

# Import V1 components
from select_packs import PackSelector as V1PackSelector

AGENT_CORE = Path.home() / ".agent-core"
PACK_DIR = AGENT_CORE / "context-packs"


class PackSelectorV2Integrated:
    """
    Integrated pack selector with V1/V2 routing

    V2 Features (when available):
    - 7 layers: Multi-graph, Multi-agent, Attention, RL, Focus, Continuum, Trainable
    - Real semantic embeddings
    - 4-graph memory architecture
    - RL-based operations
    - Active focus compression
    - Continuum memory evolution
    - Trainable pack weights

    V1 Fallback:
    - DQ Scoring + ACE Consensus
    - Keyword-based matching
    - 2-layer system
    """

    def __init__(self, force_v1: bool = False):
        self.force_v1 = force_v1
        self.v2_engine = None
        self.v1_selector = None

        # Initialize engines
        if not force_v1 and V2_AVAILABLE:
            try:
                print("🚀 Initializing V2 Engine (7 layers)...")
                self.v2_engine = ContextPacksV2Engine()
                print("✓ V2 Engine ready")
            except Exception as e:
                print(f"⚠️  V2 initialization failed: {e}")
                print("   Falling back to V1")
                self.v1_selector = V1PackSelector()
        else:
            print("📦 Using V1 Engine (2 layers)")
            self.v1_selector = V1PackSelector()

    def select_packs(
        self,
        context: str = None,
        token_budget: int = 50000,
        min_packs: int = 1,
        max_packs: int = 5,
        enable_pruning: bool = True
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Select packs using V2 (if available) or V1 (fallback)

        Returns: (selected_packs, metadata)
        """

        # Auto-detect context if not provided
        if not context:
            context = self._auto_detect_context()

        # Route to appropriate engine
        if self.v2_engine:
            return self._select_v2(context, token_budget, enable_pruning)
        else:
            return self._select_v1(context, token_budget, min_packs, max_packs)

    def _select_v2(
        self,
        context: str,
        token_budget: int,
        enable_pruning: bool
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Select using V2 engine"""

        packs, metrics = self.v2_engine.select_and_compress(
            query=context,
            context={},
            token_budget=token_budget,
            enable_pruning=enable_pruning
        )

        # Format metadata for compatibility
        metadata = {
            'engine': 'v2',
            'layers': metrics.get('total_layers', 7),
            'selection_time_ms': metrics.get('selection_time_ms', 0),
            'packs_selected': metrics.get('packs_selected', 0),
            'budget_used': sum(
                pack.get('size_tokens', 0)
                for pack in packs
            ),
            'v2_metrics': metrics
        }

        return packs, metadata

    def _select_v1(
        self,
        context: str,
        token_budget: int,
        min_packs: int,
        max_packs: int
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Select using V1 engine"""

        selected_pack_ids, v1_metadata = self.v1_selector.select_packs(
            context=context,
            token_budget=token_budget,
            min_packs=min_packs,
            max_packs=max_packs
        )

        # Load full pack data
        packs = []
        for pack_id in selected_pack_ids:
            pack_file = self._find_pack_file(pack_id)
            if pack_file:
                with open(pack_file, 'r') as f:
                    packs.append(json.load(f))

        # Format metadata
        metadata = {
            'engine': 'v1',
            'layers': 2,
            'selection_time_ms': v1_metadata.get('selection_time_ms', 0),
            'packs_selected': len(packs),
            'budget_used': sum(
                pack.get('size_tokens', 0)
                for pack in packs
            ),
            'v1_metadata': v1_metadata
        }

        return packs, metadata

    def _find_pack_file(self, pack_id: str) -> Optional[Path]:
        """Find pack file by ID"""
        for pack_type in ['domain', 'project', 'pattern', 'paper']:
            pack_file = PACK_DIR / pack_type / f'{pack_id}.pack.json'
            if pack_file.exists():
                return pack_file
        return None

    def _auto_detect_context(self) -> str:
        """Auto-detect context from current directory"""
        cwd = Path.cwd()

        # Get directory name
        dir_name = cwd.name

        # Get recent git log (if git repo)
        context_parts = [dir_name]

        try:
            import subprocess
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=%s'],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                context_parts.append(result.stdout.strip())
        except:
            pass

        # Get package.json name (if exists)
        package_json = cwd / 'package.json'
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    if 'name' in data:
                        context_parts.append(data['name'])
            except:
                pass

        return ' '.join(context_parts)


def format_output(packs: List[Dict], metadata: Dict[str, Any], output_format: str = 'text'):
    """Format selection results for output"""

    if output_format == 'json':
        return json.dumps({
            'packs': packs,
            'metadata': metadata
        }, indent=2)

    # Text format
    output = []
    output.append("=" * 60)
    output.append(f"PACK SELECTION RESULTS ({metadata['engine'].upper()})")
    output.append("=" * 60)
    output.append(f"\nEngine: {metadata['engine'].upper()}")
    output.append(f"Layers: {metadata['layers']}")
    output.append(f"Selected: {metadata['packs_selected']} packs")
    output.append(f"Budget Used: {metadata['budget_used']} tokens")
    output.append(f"Time: {metadata['selection_time_ms']:.1f}ms")

    if metadata['engine'] == 'v2':
        v2_metrics = metadata.get('v2_metrics', {})
        layers_used = v2_metrics.get('layers_used', [])
        output.append(f"Layers Active: {', '.join(layers_used)}")

    output.append("\n" + "-" * 60)
    output.append("SELECTED PACKS:")
    output.append("-" * 60)

    for i, pack in enumerate(packs, 1):
        pack_id = pack.get('pack_id', pack.get('id', 'unknown'))
        pack_type = pack.get('type', 'unknown')

        # Get content (may be nested)
        content = pack.get('content', pack)

        output.append(f"\n{i}. {pack_id} (type: {pack_type})")
        output.append(f"   Size: {pack.get('size_tokens', 0)} tokens")

        # Papers
        papers = content.get('papers', [])
        if papers:
            output.append(f"   Papers: {', '.join(p.get('arxiv_id', '') for p in papers[:3])}")
            if len(papers) > 3:
                output.append(f"           (+{len(papers) - 3} more)")

        # Keywords
        keywords = content.get('keywords', [])
        if keywords:
            output.append(f"   Keywords: {', '.join(keywords[:5])}")
            if len(keywords) > 5:
                output.append(f"             (+{len(keywords) - 5} more)")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Pack Selector V2 Integrated (Production)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use V2 (default)
  python3 select_packs_v2_integrated.py --context "debugging React performance"

  # Auto-detect context
  python3 select_packs_v2_integrated.py --auto

  # Force V1 engine
  python3 select_packs_v2_integrated.py --context "..." --v1

  # JSON output
  python3 select_packs_v2_integrated.py --context "..." --format json

  # Disable pruning (V2 only)
  python3 select_packs_v2_integrated.py --context "..." --no-pruning
        """
    )

    parser.add_argument(
        '--context',
        type=str,
        help='Context for pack selection'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-detect context from current directory'
    )
    parser.add_argument(
        '--budget',
        type=int,
        default=50000,
        help='Token budget (default: 50000)'
    )
    parser.add_argument(
        '--min-packs',
        type=int,
        default=1,
        help='Minimum packs to select (V1 only, default: 1)'
    )
    parser.add_argument(
        '--max-packs',
        type=int,
        default=5,
        help='Maximum packs to select (V1 only, default: 5)'
    )
    parser.add_argument(
        '--v1',
        action='store_true',
        help='Force V1 engine (2 layers)'
    )
    parser.add_argument(
        '--no-pruning',
        action='store_true',
        help='Disable pruning (V2 only)'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.context and not args.auto:
        parser.error("Either --context or --auto must be specified")

    # Initialize selector
    selector = PackSelectorV2Integrated(force_v1=args.v1)

    # Select packs
    try:
        packs, metadata = selector.select_packs(
            context=args.context,
            token_budget=args.budget,
            min_packs=args.min_packs,
            max_packs=args.max_packs,
            enable_pruning=not args.no_pruning
        )

        # Format and print output
        output = format_output(packs, metadata, args.format)
        print(output)

        # Exit with success
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
