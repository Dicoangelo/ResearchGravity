#!/usr/bin/env python3
"""
Routing Metrics - Performance tracking and analysis for CLI routing system

Tracks: accuracy, cost efficiency, latency, DQ scores
Supports: reporting, A/B testing, regression testing
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

HOME = Path.home()
METRICS_FILE = HOME / ".claude/data/routing-metrics.jsonl"
HISTORY_FILE = HOME / ".claude/kernel/dq-scores.jsonl"
BASELINES_FILE = HOME / ".claude/kernel/baselines.json"

# Cost per million tokens
COST_PER_MTOK = {
    "haiku": {"input": 0.25, "output": 1.25},
    "sonnet": {"input": 3.0, "output": 15.0},
    "opus": {"input": 15.0, "output": 75.0}
}

# ═══════════════════════════════════════════════════════════════════════════
# METRICS COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

class RoutingMetrics:
    def __init__(self):
        self.metrics_file = METRICS_FILE
        self.history_file = HISTORY_FILE
        self.baselines_file = BASELINES_FILE

        self.baselines = self._load_baselines()

    def _load_baselines(self) -> Dict:
        """Load baseline configuration"""
        if not self.baselines_file.exists():
            return {
                "performance_targets": {
                    "routing_accuracy": 0.75,
                    "cost_reduction_vs_random": 0.20,
                    "routing_latency_ms": 50,
                    "avg_dq_score": 0.70
                }
            }

        with open(self.baselines_file) as f:
            return json.load(f)

    def load_metrics(self, days: int = 7) -> List[Dict]:
        """Load metrics from last N days"""
        if not self.metrics_file.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        metrics = []

        with open(self.metrics_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    entry_time = datetime.fromtimestamp(entry['ts'])
                    if entry_time > cutoff:
                        metrics.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue

        return metrics

    def _load_history(self) -> List[Dict]:
        """Load DQ scoring history"""
        if not self.history_file.exists():
            return []

        history = []
        with open(self.history_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return history

    def calculate_accuracy(self, metrics: List[Dict]) -> Optional[float]:
        """Calculate routing accuracy via feedback"""
        history = self._load_history()

        # Filter to entries with feedback
        with_feedback = [h for h in history if 'success' in h]

        if not with_feedback:
            return None

        successful = [h for h in with_feedback if h['success']]
        return len(successful) / len(with_feedback)

    def _estimate_cost(self, entry: Dict) -> float:
        """Estimate cost for a routing decision"""
        model = entry.get('model', 'sonnet')

        # Estimate tokens (rough heuristic: 4 chars = 1 token)
        # Average query ~100 tokens, average response ~500 tokens
        est_input_tokens = 100
        est_output_tokens = 500

        cost_config = COST_PER_MTOK.get(model, COST_PER_MTOK['sonnet'])
        cost = (est_input_tokens * cost_config['input'] / 1_000_000 +
                est_output_tokens * cost_config['output'] / 1_000_000)

        return cost

    def calculate_cost_efficiency(self, metrics: List[Dict]) -> Optional[float]:
        """Calculate cost reduction vs random baseline"""
        if not metrics:
            return None

        # Actual cost
        actual_cost = sum(self._estimate_cost(m) for m in metrics)

        # Baseline: random selection (33% each model)
        avg_cost_haiku = (100 * COST_PER_MTOK['haiku']['input'] / 1_000_000 +
                          500 * COST_PER_MTOK['haiku']['output'] / 1_000_000)
        avg_cost_sonnet = (100 * COST_PER_MTOK['sonnet']['input'] / 1_000_000 +
                           500 * COST_PER_MTOK['sonnet']['output'] / 1_000_000)
        avg_cost_opus = (100 * COST_PER_MTOK['opus']['input'] / 1_000_000 +
                         500 * COST_PER_MTOK['opus']['output'] / 1_000_000)

        baseline_cost = len(metrics) * (
            0.33 * avg_cost_haiku +
            0.33 * avg_cost_sonnet +
            0.33 * avg_cost_opus
        )

        if baseline_cost == 0:
            return None

        reduction = (baseline_cost - actual_cost) / baseline_cost
        return reduction

    def _model_distribution(self, metrics: List[Dict]) -> Dict[str, float]:
        """Calculate model usage distribution"""
        if not metrics:
            return {"haiku": 0, "sonnet": 0, "opus": 0}

        counts = defaultdict(int)
        for m in metrics:
            counts[m.get('model', 'sonnet')] += 1

        total = len(metrics)
        return {
            model: counts[model] / total
            for model in ['haiku', 'sonnet', 'opus']
        }

    def _avg_dq(self, metrics: List[Dict]) -> Optional[float]:
        """Calculate average DQ score"""
        scores = [m.get('dq', 0) for m in metrics if 'dq' in m]
        return statistics.mean(scores) if scores else None

    def _avg_complexity(self, metrics: List[Dict]) -> Optional[float]:
        """Calculate average complexity"""
        complexities = [m.get('complexity', 0) for m in metrics if 'complexity' in m]
        return statistics.mean(complexities) if complexities else None

    def _avg_latency(self, metrics: List[Dict]) -> Optional[float]:
        """Calculate average routing latency"""
        latencies = [m.get('latency_ms', 0) for m in metrics if 'latency_ms' in m]
        return statistics.mean(latencies) if latencies else None

    def _percentile(self, metrics: List[Dict], p: int) -> Optional[float]:
        """Calculate percentile for latency"""
        latencies = sorted([m.get('latency_ms', 0) for m in metrics if 'latency_ms' in m])

        if not latencies:
            return None

        k = (len(latencies) - 1) * p / 100
        f = int(k)
        c = f + 1

        if c >= len(latencies):
            return latencies[-1]

        return latencies[f] + (k - f) * (latencies[c] - latencies[f])

    def _check_targets(self, metrics: List[Dict]) -> Dict[str, bool]:
        """Check if performance targets are met"""
        targets = self.baselines.get('performance_targets', {})

        accuracy = self.calculate_accuracy(metrics)
        cost_reduction = self.calculate_cost_efficiency(metrics)
        avg_dq = self._avg_dq(metrics)
        p95_latency = self._percentile(metrics, 95)

        results = {}

        if 'routing_accuracy' in targets and accuracy is not None:
            results['accuracy'] = accuracy >= targets['routing_accuracy']

        if 'cost_reduction_vs_random' in targets and cost_reduction is not None:
            results['cost_reduction'] = cost_reduction >= targets['cost_reduction_vs_random']

        if 'avg_dq_score' in targets and avg_dq is not None:
            results['avg_dq'] = avg_dq >= targets['avg_dq_score']

        if 'routing_latency_ms' in targets and p95_latency is not None:
            results['p95_latency'] = p95_latency <= targets['routing_latency_ms']

        return results

    def generate_report(self, days: int = 7, format: str = 'text') -> str:
        """Generate performance report"""
        metrics = self.load_metrics(days)

        if not metrics:
            return f"No metrics found for last {days} days"

        report_data = {
            "period": f"Last {days} days",
            "total_queries": len(metrics),
            "model_distribution": self._model_distribution(metrics),
            "avg_dq_score": self._avg_dq(metrics),
            "avg_complexity": self._avg_complexity(metrics),
            "routing_latency": {
                "avg": self._avg_latency(metrics),
                "p50": self._percentile(metrics, 50),
                "p95": self._percentile(metrics, 95)
            },
            "accuracy": self.calculate_accuracy(metrics),
            "cost_reduction": self.calculate_cost_efficiency(metrics),
            "targets_met": self._check_targets(metrics)
        }

        if format == 'json':
            return json.dumps(report_data, indent=2)

        # Text format
        lines = []
        lines.append("═" * 60)
        lines.append("  ROUTING PERFORMANCE REPORT")
        lines.append("═" * 60)
        lines.append(f"\nPeriod: {report_data['period']}")
        lines.append(f"Total Queries: {report_data['total_queries']}")

        lines.append("\n📊 Model Distribution:")
        dist = report_data['model_distribution']
        for model in ['haiku', 'sonnet', 'opus']:
            pct = dist.get(model, 0) * 100
            lines.append(f"  {model:8s}: {pct:5.1f}%")

        lines.append("\n📈 Performance Metrics:")
        if report_data['avg_dq_score'] is not None:
            lines.append(f"  Avg DQ Score:   {report_data['avg_dq_score']:.3f}")
        if report_data['avg_complexity'] is not None:
            lines.append(f"  Avg Complexity: {report_data['avg_complexity']:.3f}")

        lines.append("\n⏱️  Routing Latency:")
        lat = report_data['routing_latency']
        if lat['avg'] is not None:
            lines.append(f"  Average: {lat['avg']:.1f}ms")
        if lat['p50'] is not None:
            lines.append(f"  p50:     {lat['p50']:.1f}ms")
        if lat['p95'] is not None:
            lines.append(f"  p95:     {lat['p95']:.1f}ms")

        lines.append("\n🎯 Quality & Cost:")
        if report_data['accuracy'] is not None:
            lines.append(f"  Accuracy:       {report_data['accuracy']*100:.1f}%")
        else:
            lines.append(f"  Accuracy:       N/A (no feedback yet)")

        if report_data['cost_reduction'] is not None:
            lines.append(f"  Cost Reduction: {report_data['cost_reduction']*100:.1f}% vs random")

        lines.append("\n✅ Targets Met:")
        targets = report_data['targets_met']
        if targets:
            met_count = sum(1 for v in targets.values() if v)
            total_count = len(targets)
            lines.append(f"  {met_count}/{total_count} targets")
            for key, met in targets.items():
                symbol = "✓" if met else "✗"
                lines.append(f"    {symbol} {key}")
        else:
            lines.append("  Insufficient data for validation")

        lines.append("\n" + "═" * 60)

        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Routing performance metrics and analysis"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('--days', type=int, default=7, help='Days to analyze')
    report_parser.add_argument('--format', choices=['text', 'json'], default='text')

    # Check targets command
    targets_parser = subparsers.add_parser('check-targets', help='Check if targets are met')
    targets_parser.add_argument('--days', type=int, default=7)

    # Export command
    export_parser = subparsers.add_parser('export', help='Export metrics')
    export_parser.add_argument('--days', type=int, default=30)
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json')
    export_parser.add_argument('--output', help='Output file')

    args = parser.parse_args()

    metrics = RoutingMetrics()

    if args.command == 'report':
        report = metrics.generate_report(days=args.days, format=args.format)
        print(report)

    elif args.command == 'check-targets':
        data = metrics.load_metrics(days=args.days)
        targets_met = metrics._check_targets(data)

        if not targets_met:
            print("Insufficient data for target validation")
            return

        met_count = sum(1 for v in targets_met.values() if v)
        total_count = len(targets_met)

        print(f"Targets Met: {met_count}/{total_count}")
        for key, met in targets_met.items():
            symbol = "✓" if met else "✗"
            print(f"  {symbol} {key}")

        sys.exit(0 if met_count == total_count else 1)

    elif args.command == 'export':
        data = metrics.load_metrics(days=args.days)

        if args.format == 'json':
            output = json.dumps(data, indent=2)
        else:  # CSV
            import csv
            import io

            output_buffer = io.StringIO()
            if data:
                writer = csv.DictWriter(output_buffer, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            output = output_buffer.getvalue()

        if args.output:
            Path(args.output).write_text(output)
            print(f"Exported to {args.output}")
        else:
            print(output)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
