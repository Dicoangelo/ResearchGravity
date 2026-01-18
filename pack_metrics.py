#!/usr/bin/env python3
"""
Pack Metrics Tracker - Track efficiency, cost savings, and performance.

Monitors:
- Token savings per session
- Cost translation ($ saved)
- Pack performance (usage, relevance)
- Selection quality (DQ scores, ACE consensus)

Usage:
  python3 pack_metrics.py --record SESSION_ID --packs pack1,pack2
  python3 pack_metrics.py --stats
  python3 pack_metrics.py --dashboard-data
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional


AGENT_CORE = Path.home() / ".agent-core"
PACK_DIR = AGENT_CORE / "context-packs"
METRICS_FILE = PACK_DIR / "metrics.json"


# Model pricing (per million tokens)
PRICING = {
    'haiku': {'input': 0.25, 'output': 1.25},
    'sonnet': {'input': 3.00, 'output': 15.00},
    'opus': {'input': 15.00, 'output': 75.00},
}


class PackMetrics:
    """Track and analyze pack performance metrics"""

    def __init__(self):
        self.metrics_file = METRICS_FILE
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict:
        """Load metrics data"""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                return json.load(f)

        return {
            "version": "1.0.0",
            "tracking_started": datetime.utcnow().isoformat() + "Z",
            "global_stats": {
                "total_sessions": 0,
                "total_token_savings": 0,
                "total_cost_savings": 0.0,
                "avg_reduction_rate": 0.0,
                "cache_hit_rate": 0.0
            },
            "pack_stats": {},
            "session_history": [],
            "daily_stats": {}
        }

    def _save_metrics(self):
        """Save metrics data"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def record_session(
        self,
        session_id: str,
        packs_loaded: List[str],
        context: str,
        baseline_tokens: int,
        pack_tokens: int,
        dq_scores: Dict[str, float],
        consensus_scores: Dict[str, float],
        model: str = 'sonnet',
        user_feedback: Optional[Dict] = None
    ):
        """Record a session's pack usage and metrics"""

        token_savings = baseline_tokens - pack_tokens
        reduction_rate = (token_savings / baseline_tokens * 100) if baseline_tokens > 0 else 0
        cost_savings = self._calculate_cost_savings(token_savings, model)

        session_record = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "packs_loaded": packs_loaded,
            "context": context[:200],  # Truncate for storage
            "baseline_tokens": baseline_tokens,
            "pack_tokens": pack_tokens,
            "token_savings": token_savings,
            "reduction_rate": reduction_rate,
            "cost_savings": cost_savings,
            "model": model,
            "dq_scores": dq_scores,
            "consensus_scores": consensus_scores,
            "user_feedback": user_feedback or {}
        }

        # Add to session history
        self.metrics['session_history'].append(session_record)

        # Update global stats
        global_stats = self.metrics['global_stats']
        global_stats['total_sessions'] += 1
        global_stats['total_token_savings'] += token_savings
        global_stats['total_cost_savings'] += cost_savings

        # Update average reduction rate
        all_rates = [s['reduction_rate'] for s in self.metrics['session_history']]
        global_stats['avg_reduction_rate'] = sum(all_rates) / len(all_rates) if all_rates else 0

        # Update pack-level stats
        for pack_id in packs_loaded:
            if pack_id not in self.metrics['pack_stats']:
                self.metrics['pack_stats'][pack_id] = {
                    "times_selected": 0,
                    "sessions": [],
                    "total_tokens_loaded": 0,
                    "avg_dq_score": 0.0,
                    "avg_consensus_score": 0.0,
                    "combined_with": {}
                }

            pack_stat = self.metrics['pack_stats'][pack_id]
            pack_stat['times_selected'] += 1
            pack_stat['sessions'].append(session_id)
            pack_stat['total_tokens_loaded'] += pack_tokens

            # Update average scores
            if pack_id in dq_scores:
                current_avg = pack_stat['avg_dq_score']
                count = pack_stat['times_selected']
                new_score = dq_scores[pack_id]
                pack_stat['avg_dq_score'] = ((current_avg * (count - 1)) + new_score) / count

            if pack_id in consensus_scores:
                current_avg = pack_stat['avg_consensus_score']
                count = pack_stat['times_selected']
                new_score = consensus_scores[pack_id]
                pack_stat['avg_consensus_score'] = ((current_avg * (count - 1)) + new_score) / count

            # Track co-occurrence
            for other_pack in packs_loaded:
                if other_pack != pack_id:
                    pack_stat['combined_with'][other_pack] = pack_stat['combined_with'].get(other_pack, 0) + 1

        # Update daily stats
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if today not in self.metrics['daily_stats']:
            self.metrics['daily_stats'][today] = {
                "sessions": 0,
                "token_savings": 0,
                "cost_savings": 0.0
            }

        daily = self.metrics['daily_stats'][today]
        daily['sessions'] += 1
        daily['token_savings'] += token_savings
        daily['cost_savings'] += cost_savings

        self._save_metrics()

        print(f"✓ Recorded session: {session_id}")
        print(f"  Token savings: {token_savings:,} ({reduction_rate:.1f}%)")
        print(f"  Cost savings: ${cost_savings:.2f}")

    def _calculate_cost_savings(self, tokens_saved: int, model: str = 'sonnet') -> float:
        """Calculate dollar savings from token reduction"""
        cost_per_token = PRICING[model]['input'] / 1_000_000
        return tokens_saved * cost_per_token

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "global": self.metrics['global_stats'],
            "recent_sessions": self.metrics['session_history'][-10:],
            "top_packs": self._get_top_packs(limit=10),
            "best_combinations": self._get_best_combinations(limit=5),
            "daily_trend": self._get_daily_trend(days=7)
        }

    def _get_top_packs(self, limit: int = 10) -> List[Dict]:
        """Get top performing packs"""
        packs = []
        for pack_id, stats in self.metrics['pack_stats'].items():
            packs.append({
                "pack_id": pack_id,
                "times_selected": stats['times_selected'],
                "avg_dq_score": stats['avg_dq_score'],
                "avg_consensus_score": stats['avg_consensus_score']
            })

        # Sort by times_selected
        packs.sort(key=lambda x: x['times_selected'], reverse=True)
        return packs[:limit]

    def _get_best_combinations(self, limit: int = 5) -> List[Dict]:
        """Get most common pack combinations"""
        # Find combinations that appear together
        combinations = {}

        for session in self.metrics['session_history']:
            packs = tuple(sorted(session['packs_loaded']))
            if len(packs) > 1:
                if packs not in combinations:
                    combinations[packs] = {
                        'count': 0,
                        'avg_savings': 0,
                        'sessions': []
                    }

                combinations[packs]['count'] += 1
                combinations[packs]['avg_savings'] += session['token_savings']
                combinations[packs]['sessions'].append(session['session_id'])

        # Calculate averages
        for combo_data in combinations.values():
            combo_data['avg_savings'] = combo_data['avg_savings'] / combo_data['count']

        # Sort and format
        sorted_combos = sorted(
            combinations.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        result = []
        for packs, data in sorted_combos[:limit]:
            result.append({
                'packs': list(packs),
                'uses': data['count'],
                'avg_savings': data['avg_savings']
            })

        return result

    def _get_daily_trend(self, days: int = 7) -> List[Dict]:
        """Get daily stats for trend analysis"""
        trend = []
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
            if date in self.metrics['daily_stats']:
                trend.append({
                    'date': date,
                    **self.metrics['daily_stats'][date]
                })
            else:
                trend.append({
                    'date': date,
                    'sessions': 0,
                    'token_savings': 0,
                    'cost_savings': 0.0
                })

        return list(reversed(trend))

    def generate_dashboard_data(self) -> str:
        """Generate JSON data for dashboard"""
        stats = self.get_stats()
        return json.dumps(stats, indent=2)

    def print_summary(self):
        """Print human-readable summary"""
        global_stats = self.metrics['global_stats']

        print("\n" + "="*60)
        print("CONTEXT PACKS EFFICIENCY REPORT")
        print("="*60 + "\n")

        print("📊 GLOBAL STATISTICS")
        print(f"  Total Sessions: {global_stats['total_sessions']}")
        print(f"  Total Token Savings: {global_stats['total_token_savings']:,}")
        print(f"  Total Cost Savings: ${global_stats['total_cost_savings']:.2f}")
        print(f"  Avg Reduction Rate: {global_stats['avg_reduction_rate']:.1f}%\n")

        # Top packs
        print("🏆 TOP PERFORMING PACKS")
        top_packs = self._get_top_packs(limit=5)
        for i, pack in enumerate(top_packs, 1):
            print(f"  {i}. {pack['pack_id']}")
            print(f"     Uses: {pack['times_selected']} | DQ: {pack['avg_dq_score']:.3f} | Consensus: {pack['avg_consensus_score']:.3f}")

        # Best combinations
        print("\n🔗 BEST PACK COMBINATIONS")
        combos = self._get_best_combinations(limit=3)
        for i, combo in enumerate(combos, 1):
            print(f"  {i}. [{', '.join(combo['packs'])}]")
            print(f"     Uses: {combo['uses']} | Avg Savings: {combo['avg_savings']:,.0f} tokens")

        # Daily trend
        print("\n📈 LAST 7 DAYS")
        trend = self._get_daily_trend(days=7)
        for day in trend:
            if day['sessions'] > 0:
                print(f"  {day['date']}: {day['sessions']} sessions | {day['token_savings']:,} tokens | ${day['cost_savings']:.2f} saved")

        print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Track and analyze context pack metrics"
    )

    parser.add_argument(
        '--record',
        help="Record session metrics (provide session ID)"
    )

    parser.add_argument(
        '--packs',
        help="Comma-separated pack IDs loaded in session"
    )

    parser.add_argument(
        '--context',
        help="Session context"
    )

    parser.add_argument(
        '--baseline',
        type=int,
        help="Baseline tokens (what would have been loaded without packs)"
    )

    parser.add_argument(
        '--pack-tokens',
        type=int,
        help="Actual tokens from packs"
    )

    parser.add_argument(
        '--model',
        default='sonnet',
        choices=['haiku', 'sonnet', 'opus'],
        help="Model used (for cost calculation)"
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help="Show statistics summary"
    )

    parser.add_argument(
        '--dashboard-data',
        action='store_true',
        help="Output dashboard JSON data"
    )

    args = parser.parse_args()

    tracker = PackMetrics()

    if args.record:
        if not all([args.packs, args.context, args.baseline, args.pack_tokens]):
            print("Error: --record requires --packs, --context, --baseline, --pack-tokens")
            return

        packs = [p.strip() for p in args.packs.split(',')]

        # Mock DQ and consensus scores
        dq_scores = {p: 0.85 for p in packs}
        consensus_scores = {p: 0.88 for p in packs}

        tracker.record_session(
            session_id=args.record,
            packs_loaded=packs,
            context=args.context,
            baseline_tokens=args.baseline,
            pack_tokens=args.pack_tokens,
            dq_scores=dq_scores,
            consensus_scores=consensus_scores,
            model=args.model
        )

    elif args.stats:
        tracker.print_summary()

    elif args.dashboard_data:
        print(tracker.generate_dashboard_data())

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
