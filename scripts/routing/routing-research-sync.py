#!/usr/bin/env python3
"""
Routing Research Sync - Integrate academic research into routing baselines

Fetches recent papers on LLM routing, extracts insights, updates baselines
with full research lineage tracking.

Usage:
  python3 routing-research-sync.py fetch-papers --query "LLM routing" --days 90
  python3 routing-research-sync.py extract-insights --papers papers.json
  python3 routing-research-sync.py update-baselines --insights insights.json --apply
"""

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import arxiv
except ImportError:
    print("⚠️  arxiv package not found. Install: pip install arxiv")
    arxiv = None

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

HOME = Path.home()
BASELINES_FILE = HOME / ".claude/kernel/baselines.json"
AGENT_CORE = HOME / ".agent-core"
SESSIONS_DIR = AGENT_CORE / "sessions"

# ═══════════════════════════════════════════════════════════════════════════
# PAPER FETCHING
# ═══════════════════════════════════════════════════════════════════════════

class ResearchSync:
    def __init__(self):
        self.baselines_file = BASELINES_FILE
        self.sessions_dir = SESSIONS_DIR

    def generate_search_queries(self, topic: str = "LLM routing") -> Dict[str, List[str]]:
        """Generate multi-tier search queries"""
        year = datetime.now().year

        return {
            "tier1_research": [
                f"{topic} model selection",
                f"{topic} complexity estimation",
                f"{topic} adaptive inference",
                f"query complexity LLM {year}"
            ],
            "tier1_optimization": [
                f"cost optimization LLM {year}",
                "inference optimization language models",
                "query complexity classification"
            ],
            "tier2_empirical": [
                f"LLM routing evaluation {year}",
                "model selection benchmarks",
                f"adaptive model routing {year}"
            ]
        }

    def fetch_papers(
        self,
        query: str,
        days: int = 90,
        max_results: int = 50
    ) -> List[Dict]:
        """Fetch recent papers from arXiv"""
        if arxiv is None:
            print("❌ arxiv package not installed")
            print("Install with: pip install arxiv")
            return []

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        print(f"🔍 Searching arXiv for: {query}")
        print(f"📅 Date range: Last {days} days")

        client = arxiv.Client()
        search = arxiv.Search(
            query=f"cat:cs.AI AND ({query})",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        try:
            for result in client.results(search):
                # Filter by date
                if result.published < cutoff_date:
                    continue

                papers.append({
                    "arxiv_id": result.get_short_id(),
                    "title": result.title,
                    "published": result.published.isoformat(),
                    "summary": result.summary,
                    "authors": [a.name for a in result.authors],
                    "url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "categories": result.categories
                })

            print(f"✓ Found {len(papers)} recent papers")
            return papers

        except Exception as e:
            print(f"❌ Error fetching papers: {e}")
            return []

    def extract_insights(
        self,
        papers: List[Dict],
        focus_areas: Optional[List[str]] = None,
        model: str = "sonnet"
    ) -> List[Dict]:
        """Use LLM to extract actionable insights from papers"""
        if focus_areas is None:
            focus_areas = [
                "complexity thresholds",
                "cost optimization",
                "accuracy metrics",
                "routing strategies"
            ]

        print(f"\n🤖 Extracting insights from {len(papers)} papers...")
        print(f"Focus areas: {', '.join(focus_areas)}")

        insights = []

        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Analyzing: {paper['title'][:60]}...")

            # Construct prompt for LLM analysis
            prompt = f"""Analyze this research paper on LLM routing and extract actionable insights.

Title: {paper['title']}
Published: {paper['published'][:10]}
arXiv ID: {paper['arxiv_id']}

Abstract (first 600 chars):
{paper['summary'][:600]}

Extract insights for: {', '.join(focus_areas)}

Provide your analysis in this JSON format:
{{
  "thresholds": {{"haiku": {{"max": 0.XX}}, "sonnet": {{"max": 0.XX}}, "opus": {{"max": 1.0}}}},
  "cost_insights": ["insight 1", "insight 2"],
  "accuracy": {{"metric": "...", "value": 0.XX}},
  "strategies": ["strategy 1", "strategy 2"],
  "rationale": "Brief explanation of why these insights matter",
  "confidence": 0.X,
  "applicability": "high|medium|low"
}}

Only include insights that are directly applicable to routing decisions. Use null for unavailable data.
"""

            try:
                # Call Claude via CLI for analysis
                result = self._call_llm(prompt, model=model)

                # Try to parse JSON from response
                insight = self._extract_json(result)

                if insight:
                    insight['source_paper'] = paper['arxix_id']
                    insight['title'] = paper['title']
                    insight['published'] = paper['published']
                    insights.append(insight)
                    print(f"  ✓ Extracted insights (confidence: {insight.get('confidence', 'N/A')})")
                else:
                    print("  ⚠️  Could not parse insights")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        print(f"\n✓ Extracted insights from {len(insights)} papers")
        return insights

    def _call_llm(self, prompt: str, model: str = "sonnet") -> str:
        """Call Claude via CLI for analysis"""
        try:
            result = subprocess.run(
                ['claude', '--model', model, '-p', prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            raise Exception("LLM call timed out")
        except Exception as e:
            raise Exception(f"LLM call failed: {e}")

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        import re

        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return None

        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

    def update_baselines(
        self,
        insights: List[Dict],
        dry_run: bool = True
    ) -> List[Dict]:
        """Update baselines.json with research-backed modifications"""
        if not self.baselines_file.exists():
            print("❌ baselines.json not found")
            return []

        with open(self.baselines_file) as f:
            baselines = json.load(f)

        modifications = []

        print(f"\n📊 Analyzing {len(insights)} insights for baseline updates...")

        for insight in insights:
            # Filter by confidence and applicability
            confidence = insight.get('confidence', 0)
            applicability = insight.get('applicability', 'low')

            if confidence < 0.6 or applicability == 'low':
                print(f"  ⊘ Skipping low-confidence insight (conf: {confidence})")
                continue

            # Parse threshold recommendations
            if 'thresholds' in insight and insight['thresholds']:
                for model, threshold in insight['thresholds'].items():
                    if model not in baselines['complexity_thresholds']:
                        continue

                    if threshold is None or 'max' not in threshold:
                        continue

                    current_path = f"complexity_thresholds.{model}.range[1]"
                    old_value = baselines['complexity_thresholds'][model]['range'][1]
                    new_value = threshold['max']

                    # Only update if change is significant (>5%)
                    if abs(new_value - old_value) / old_value > 0.05:
                        modifications.append({
                            "target": current_path,
                            "old_value": old_value,
                            "new_value": new_value,
                            "source_paper": insight.get('source_paper', 'unknown'),
                            "paper_title": insight.get('title', 'Unknown'),
                            "rationale": insight.get('rationale', 'Research-backed threshold'),
                            "confidence": confidence,
                            "applied": datetime.now().isoformat()
                        })

                        print(f"  📝 Proposed: {current_path}: {old_value} → {new_value}")
                        print(f"     Source: {insight.get('source_paper', 'unknown')}")
                        print(f"     Confidence: {confidence:.2f}")

        if not modifications:
            print("\n  No significant updates found")
            return []

        print(f"\n✓ Found {len(modifications)} potential updates")

        if dry_run:
            print("\n🔍 DRY RUN - No changes applied")
            print("Review modifications above and run with --apply to update baselines")
            return modifications

        # Apply modifications
        for mod in modifications:
            # Parse target path
            target = mod['target']
            if 'complexity_thresholds' in target:
                model = target.split('.')[1]
                baselines['complexity_thresholds'][model]['range'][1] = mod['new_value']

        # Update research lineage
        if 'research_lineage' not in baselines:
            baselines['research_lineage'] = []

        baselines['research_lineage'].extend(modifications)
        baselines['last_updated'] = datetime.now().isoformat()

        # Save updated baselines
        with open(self.baselines_file, 'w') as f:
            json.dump(baselines, indent=2, fp=f)

        print(f"\n✅ Applied {len(modifications)} updates to baselines.json")
        print(f"   Version: {baselines['version']}")
        print(f"   Updated: {baselines['last_updated']}")

        return modifications

    def trace_parameter(self, parameter: str) -> Optional[Dict]:
        """Trace origin of a baseline parameter"""
        if not self.baselines_file.exists():
            return None

        with open(self.baselines_file) as f:
            baselines = json.load(f)

        lineage = baselines.get('research_lineage', [])

        # Find most recent modification for this parameter
        for mod in reversed(lineage):
            if mod['target'] == parameter:
                return {
                    "parameter": parameter,
                    "current_value": mod['new_value'],
                    "previous_value": mod['old_value'],
                    "last_modified": mod['applied'],
                    "source_paper": mod['source_paper'],
                    "paper_title": mod.get('paper_title', 'Unknown'),
                    "rationale": mod['rationale'],
                    "confidence": mod.get('confidence', 'N/A')
                }

        return None

# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Research-driven baseline optimization for routing system"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Fetch papers
    fetch_parser = subparsers.add_parser('fetch-papers', help='Fetch papers from arXiv')
    fetch_parser.add_argument('--query', default="LLM routing", help='Search query')
    fetch_parser.add_argument('--days', type=int, default=90, help='Days back to search')
    fetch_parser.add_argument('--max-results', type=int, default=50)
    fetch_parser.add_argument('--output', help='Output JSON file')

    # Extract insights
    insights_parser = subparsers.add_parser('extract-insights', help='Extract insights from papers')
    insights_parser.add_argument('--papers', required=True, help='Papers JSON file')
    insights_parser.add_argument('--focus', help='Comma-separated focus areas')
    insights_parser.add_argument('--model', default='sonnet', choices=['haiku', 'sonnet', 'opus'])
    insights_parser.add_argument('--output', help='Output JSON file')

    # Update baselines
    update_parser = subparsers.add_parser('update-baselines', help='Update baselines from insights')
    update_parser.add_argument('--insights', required=True, help='Insights JSON file')
    update_parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    update_parser.add_argument('--apply', action='store_true', help='Apply changes to baselines')

    # Trace parameter
    trace_parser = subparsers.add_parser('trace', help='Trace origin of a parameter')
    trace_parser.add_argument('--parameter', required=True, help='Parameter path (e.g., complexity_thresholds.haiku.range[1])')

    # Generate queries
    queries_parser = subparsers.add_parser('generate-queries', help='Generate search queries')
    queries_parser.add_argument('--topic', default='LLM routing')

    args = parser.parse_args()

    sync = ResearchSync()

    if args.command == 'fetch-papers':
        papers = sync.fetch_papers(
            query=args.query,
            days=args.days,
            max_results=args.max_results
        )

        if args.output:
            Path(args.output).write_text(json.dumps(papers, indent=2))
            print(f"\n💾 Saved to {args.output}")
        else:
            print(json.dumps(papers, indent=2))

    elif args.command == 'extract-insights':
        papers_data = json.loads(Path(args.papers).read_text())
        focus_areas = args.focus.split(',') if args.focus else None

        insights = sync.extract_insights(
            papers=papers_data,
            focus_areas=focus_areas,
            model=args.model
        )

        if args.output:
            Path(args.output).write_text(json.dumps(insights, indent=2))
            print(f"\n💾 Saved to {args.output}")
        else:
            print(json.dumps(insights, indent=2))

    elif args.command == 'update-baselines':
        insights_data = json.loads(Path(args.insights).read_text())

        # Determine mode
        dry_run = not args.apply

        modifications = sync.update_baselines(
            insights=insights_data,
            dry_run=dry_run
        )

        if modifications:
            print("\n📋 Modifications:")
            for mod in modifications:
                print(f"  {mod['target']}: {mod['old_value']} → {mod['new_value']}")

    elif args.command == 'trace':
        result = sync.trace_parameter(args.parameter)

        if result:
            print(json.dumps(result, indent=2))
        else:
            print(f"No lineage found for parameter: {args.parameter}")

    elif args.command == 'generate-queries':
        queries = sync.generate_search_queries(args.topic)
        print(json.dumps(queries, indent=2))

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
