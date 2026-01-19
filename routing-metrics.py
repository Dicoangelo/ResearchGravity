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
import random
import hashlib

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
        """Load metrics from last N days (from either metrics or history file)"""
        # Try metrics file first, fall back to history file
        source_file = self.metrics_file if self.metrics_file.exists() else self.history_file

        if not source_file.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        metrics = []

        with open(source_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    # Handle both ms and seconds timestamps
                    ts = entry.get('ts', 0)
                    if ts > 1e12:  # Milliseconds
                        ts = ts / 1000
                    entry_time = datetime.fromtimestamp(ts)
                    if entry_time > cutoff:
                        metrics.append(entry)
                except (json.JSONDecodeError, KeyError, ValueError):
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

    def load_last_n_queries(self, n: int) -> List[Dict]:
        """Load last N queries (usage-based)"""
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

        # Return last N entries
        return history[-n:] if len(history) >= n else history

    def calculate_data_quality(self, data: List[Dict]) -> float:
        """
        Calculate data quality score (0.0-1.0)

        Quality factors:
        - Variance in DQ scores (consistency)
        - Feedback rate (% with feedback)
        - Model distribution (not all one model)
        - Sample size adequacy
        """
        if len(data) < 10:
            return 0.0

        # Factor 1: DQ score variance (lower is better, 0.0-0.3)
        dq_scores = [d.get('dq', d.get('dqScore', 0.5)) for d in data]
        try:
            dq_variance = statistics.variance(dq_scores) if len(dq_scores) > 1 else 0.5
            variance_score = max(0, 1.0 - (dq_variance / 0.3))  # Normalize to 0-1
        except:
            variance_score = 0.5

        # Factor 2: Feedback rate (higher is better)
        history = self._load_history()
        feedback_count = sum(1 for h in history if 'success' in h or 'failure' in h)
        feedback_rate = feedback_count / max(len(data), 1)
        feedback_score = min(1.0, feedback_rate * 2)  # 50% feedback = 1.0 score

        # Factor 3: Model distribution (not all one model)
        models = [d.get('model', 'sonnet') for d in data]
        unique_models = len(set(models))
        distribution_score = min(1.0, unique_models / 3)  # All 3 models = 1.0

        # Factor 4: Sample size (more is better)
        sample_score = min(1.0, len(data) / 200)  # 200+ samples = 1.0

        # Weighted average
        quality = (
            variance_score * 0.25 +
            feedback_score * 0.35 +
            distribution_score * 0.20 +
            sample_score * 0.20
        )

        return quality

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
        # Support both 'dq' and 'dqScore' field names
        scores = [m.get('dq', m.get('dqScore', 0)) for m in metrics if 'dq' in m or 'dqScore' in m]
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
# A/B TESTING FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════

class ABTest:
    """A/B testing framework for routing optimization"""

    def __init__(self):
        self.experiments_dir = HOME / ".claude/data/ab-experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_file = BASELINES_FILE

    def create_experiment(self, config: Dict) -> str:
        """Create new A/B test experiment"""
        experiment_id = config.get('experiment_id', self._generate_id())

        experiment = {
            "id": experiment_id,
            "name": config['name'],
            "created": datetime.now().isoformat(),
            "active": True,
            "control": config['control'],
            "variant": config['variant'],
            "split": config.get('split', 0.5),
            "primary_metric": config['primary_metric'],
            "secondary_metrics": config.get('secondary_metrics', []),
            "min_samples": config.get('min_samples', 100),
            "success_criteria": config.get('success_criteria', {}),
            "results": {
                "control": [],
                "variant": []
            }
        }

        exp_file = self.experiments_dir / f"{experiment_id}.json"
        exp_file.write_text(json.dumps(experiment, indent=2))

        return experiment_id

    def _generate_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:4]
        return f"exp-{timestamp}-{random_hash}"

    def should_use_variant(self, experiment_id: str, query_hash: str) -> bool:
        """Deterministically assign query to control or variant"""
        experiment = self.load_experiment(experiment_id)
        if not experiment or not experiment['active']:
            return False

        # Use query hash for deterministic assignment
        hash_value = int(hashlib.md5(query_hash.encode()).hexdigest(), 16)
        return (hash_value % 100) < (experiment['split'] * 100)

    def record_result(self, experiment_id: str, is_variant: bool, metrics: Dict):
        """Record result for experiment"""
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            return

        target = 'variant' if is_variant else 'control'
        experiment['results'][target].append({
            "ts": datetime.now().isoformat(),
            "metrics": metrics
        })

        self._save_experiment(experiment)

    def load_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Load experiment by ID"""
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        if not exp_file.exists():
            return None

        return json.loads(exp_file.read_text())

    def list_experiments(self, active_only: bool = False) -> List[Dict]:
        """List all experiments"""
        experiments = []

        for exp_file in self.experiments_dir.glob("*.json"):
            try:
                exp = json.loads(exp_file.read_text())
                if not active_only or exp.get('active', False):
                    experiments.append(exp)
            except (json.JSONDecodeError, KeyError):
                continue

        return sorted(experiments, key=lambda x: x.get('created', ''), reverse=True)

    def analyze_experiment(self, experiment_id: str) -> Dict:
        """Analyze experiment results with statistical significance"""
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        control_results = experiment['results']['control']
        variant_results = experiment['results']['variant']

        if len(control_results) < 2 or len(variant_results) < 2:
            return {
                "experiment_id": experiment_id,
                "status": "insufficient_data",
                "control_samples": len(control_results),
                "variant_samples": len(variant_results),
                "min_required": experiment['min_samples']
            }

        primary_metric = experiment['primary_metric']

        # Extract primary metric values
        control_values = [r['metrics'].get(primary_metric, 0) for r in control_results]
        variant_values = [r['metrics'].get(primary_metric, 0) for r in variant_results]

        # Calculate statistics
        control_mean = statistics.mean(control_values)
        variant_mean = statistics.mean(variant_values)
        improvement = ((variant_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0

        # Simple t-test approximation
        pooled_std = statistics.stdev(control_values + variant_values) if len(control_values + variant_values) > 1 else 0
        n_control = len(control_values)
        n_variant = len(variant_values)

        if pooled_std > 0:
            t_stat = (variant_mean - control_mean) / (pooled_std * ((1/n_control + 1/n_variant) ** 0.5))
            # Rough p-value approximation (for t_stat > 2, p < 0.05)
            p_value = 0.05 if abs(t_stat) > 2 else 0.1
        else:
            t_stat = 0
            p_value = 1.0

        significant = p_value < 0.05

        # Check success criteria
        success_criteria = experiment.get('success_criteria', {})
        meets_criteria = True

        if 'min_improvement' in success_criteria:
            meets_criteria = meets_criteria and (improvement >= success_criteria['min_improvement'])

        if 'max_p_value' in success_criteria:
            meets_criteria = meets_criteria and (p_value <= success_criteria['max_p_value'])

        return {
            "experiment_id": experiment_id,
            "name": experiment['name'],
            "status": "complete" if len(control_results) + len(variant_results) >= experiment['min_samples'] else "in_progress",
            "samples": {
                "control": len(control_results),
                "variant": len(variant_results),
                "total": len(control_results) + len(variant_results),
                "min_required": experiment['min_samples']
            },
            "primary_metric": primary_metric,
            "results": {
                "control_mean": control_mean,
                "variant_mean": variant_mean,
                "improvement_pct": improvement,
                "statistically_significant": significant,
                "p_value": p_value,
                "t_statistic": t_stat
            },
            "recommendation": "apply_variant" if (meets_criteria and significant and improvement > 0) else "keep_control",
            "meets_success_criteria": meets_criteria
        }

    def apply_variant(self, experiment_id: str, dry_run: bool = True) -> Dict:
        """Apply winning variant to baselines"""
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        analysis = self.analyze_experiment(experiment_id)

        if analysis.get('recommendation') != 'apply_variant':
            return {
                "error": "Variant not recommended",
                "reason": f"Recommendation: {analysis.get('recommendation')}",
                "analysis": analysis
            }

        # Load baselines
        if not self.baselines_file.exists():
            return {"error": "Baselines file not found"}

        baselines = json.loads(self.baselines_file.read_text())

        # Apply variant configuration
        variant_config = experiment['variant']
        modifications = []

        for target, new_value in variant_config.items():
            # Parse target path (e.g., "complexity_thresholds.haiku.range[1]")
            old_value = self._get_nested_value(baselines, target)

            modifications.append({
                "target": target,
                "old_value": old_value,
                "new_value": new_value,
                "experiment_id": experiment_id,
                "improvement": analysis['results']['improvement_pct']
            })

            if not dry_run:
                self._set_nested_value(baselines, target, new_value)

        if not dry_run:
            # Update research lineage
            if 'ab_test_lineage' not in baselines:
                baselines['ab_test_lineage'] = []

            baselines['ab_test_lineage'].extend(modifications)
            baselines['last_updated'] = datetime.now().isoformat()

            # Save baselines
            self.baselines_file.write_text(json.dumps(baselines, indent=2))

            # Mark experiment as inactive
            experiment['active'] = False
            experiment['applied'] = datetime.now().isoformat()
            self._save_experiment(experiment)

        return {
            "status": "dry_run" if dry_run else "applied",
            "modifications": modifications,
            "experiment": experiment_id
        }

    def _get_nested_value(self, data: Dict, path: str):
        """Get value from nested dict using dot notation"""
        parts = path.replace('[', '.').replace(']', '').split('.')
        current = data

        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = current.get(part, None)
                if current is None:
                    return None

        return current

    def _set_nested_value(self, data: Dict, path: str, value):
        """Set value in nested dict using dot notation"""
        parts = path.replace('[', '.').replace(']', '').split('.')
        current = data

        for i, part in enumerate(parts[:-1]):
            if part.isdigit():
                current = current[int(part)]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]

        last_part = parts[-1]
        if last_part.isdigit():
            current[int(last_part)] = value
        else:
            current[last_part] = value

    def _save_experiment(self, experiment: Dict):
        """Save experiment to file"""
        exp_file = self.experiments_dir / f"{experiment['id']}.json"
        exp_file.write_text(json.dumps(experiment, indent=2))

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

    # Check targets command (usage-based options)
    targets_parser = subparsers.add_parser('check-targets', help='Check if targets are met')
    targets_parser.add_argument('--days', type=int, default=None, help='Days to analyze (time-based)')
    targets_parser.add_argument('--last-n-queries', type=int, default=None, help='Check last N queries (usage-based)')
    targets_parser.add_argument('--all-time', action='store_true', help='Check all-time performance')

    # Data quality check (usage-based)
    quality_parser = subparsers.add_parser('check-data-quality', help='Check data quality metrics')
    quality_parser.add_argument('--all-time', action='store_true', help='Check all-time data quality')
    quality_parser.add_argument('--days', type=int, default=None, help='Days to analyze')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export metrics')
    export_parser.add_argument('--days', type=int, default=30)
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json')
    export_parser.add_argument('--output', help='Output file')

    # A/B test commands
    ab_parser = subparsers.add_parser('ab-test', help='A/B testing framework')
    ab_subparsers = ab_parser.add_subparsers(dest='ab_command', help='A/B test commands')

    # Create experiment
    ab_create = ab_subparsers.add_parser('create', help='Create new experiment')
    ab_create.add_argument('--config', required=True, help='Experiment config JSON file')

    # List experiments
    ab_list = ab_subparsers.add_parser('list', help='List experiments')
    ab_list.add_argument('--active-only', action='store_true', help='Show only active experiments')

    # Analyze experiment
    ab_analyze = ab_subparsers.add_parser('analyze', help='Analyze experiment results')
    ab_analyze.add_argument('--experiment', required=True, help='Experiment ID')

    # Apply variant
    ab_apply = ab_subparsers.add_parser('apply', help='Apply winning variant')
    ab_apply.add_argument('--experiment', required=True, help='Experiment ID')
    ab_apply.add_argument('--dry-run', action='store_true', help='Preview without applying')

    # Experiment status
    ab_status = ab_subparsers.add_parser('status', help='Show all active experiments status')

    # CPB commands (Cognitive Precision Bridge)
    cpb_parser = subparsers.add_parser('cpb', help='CPB precision routing')
    cpb_subparsers = cpb_parser.add_subparsers(dest='cpb_command', help='CPB commands')

    # CPB analyze
    cpb_analyze = cpb_subparsers.add_parser('analyze', help='Analyze query complexity')
    cpb_analyze.add_argument('query', help='Query to analyze')
    cpb_analyze.add_argument('--context', '-c', help='Optional context')

    # CPB score
    cpb_score = cpb_subparsers.add_parser('score', help='Score a response')
    cpb_score.add_argument('--query', '-q', required=True, help='Original query')
    cpb_score.add_argument('--response', '-r', required=True, help='Response to score')

    # CPB stats
    cpb_stats = cpb_subparsers.add_parser('stats', help='Show CPB statistics')
    cpb_stats.add_argument('--days', '-d', type=int, default=7, help='Days to analyze')

    # CPB status
    cpb_status = cpb_subparsers.add_parser('status', help='Show CPB status')

    args = parser.parse_args()

    metrics = RoutingMetrics()
    ab_test = ABTest()

    if args.command == 'report':
        report = metrics.generate_report(days=args.days, format=args.format)
        print(report)

    elif args.command == 'check-targets':
        # Usage-based: last-n-queries or all-time
        if args.last_n_queries:
            data = metrics.load_last_n_queries(args.last_n_queries)
        elif args.all_time:
            data = metrics.load_metrics(days=999)
        else:
            # Default to 7 days if no option specified
            data = metrics.load_metrics(days=args.days or 7)

        targets_met = metrics._check_targets(data)

        if not targets_met:
            print("Insufficient data for target validation")
            sys.exit(1)

        met_count = sum(1 for v in targets_met.values() if v)
        total_count = len(targets_met)

        print(f"Targets Met: {met_count}/{total_count}")
        for key, met in targets_met.items():
            symbol = "✓" if met else "✗"
            print(f"  {symbol} {key}")

        sys.exit(0 if met_count == total_count else 1)

    elif args.command == 'check-data-quality':
        # Calculate data quality score
        if args.all_time:
            data = metrics.load_metrics(days=999)
        else:
            data = metrics.load_metrics(days=args.days or 30)

        quality_score = metrics.calculate_data_quality(data)
        print(f"{quality_score:.2f}")
        sys.exit(0)

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

    elif args.command == 'ab-test':
        if args.ab_command == 'create':
            config = json.loads(Path(args.config).read_text())
            exp_id = ab_test.create_experiment(config)
            print(f"✓ Created experiment: {exp_id}")
            print(f"  Name: {config['name']}")
            print(f"  Control: {config['control']}")
            print(f"  Variant: {config['variant']}")

        elif args.ab_command == 'list':
            experiments = ab_test.list_experiments(active_only=args.active_only)

            if not experiments:
                print("No experiments found")
                return

            print(f"\n{'ID':<30} {'Name':<30} {'Status':<10} {'Samples':<10}")
            print("=" * 80)

            for exp in experiments:
                exp_id = exp['id']
                name = exp['name'][:28]
                status = "active" if exp.get('active') else "inactive"
                control_samples = len(exp['results']['control'])
                variant_samples = len(exp['results']['variant'])
                samples = f"{control_samples + variant_samples}"

                print(f"{exp_id:<30} {name:<30} {status:<10} {samples:<10}")

        elif args.ab_command == 'analyze':
            analysis = ab_test.analyze_experiment(args.experiment)

            if 'error' in analysis:
                print(f"❌ {analysis['error']}")
                return

            print("\n" + "=" * 60)
            print(f"  EXPERIMENT ANALYSIS: {args.experiment}")
            print("=" * 60)
            print(f"\nName: {analysis['name']}")
            print(f"Status: {analysis['status']}")

            print(f"\n📊 Samples:")
            print(f"  Control: {analysis['samples']['control']}")
            print(f"  Variant: {analysis['samples']['variant']}")
            print(f"  Total:   {analysis['samples']['total']} / {analysis['samples']['min_required']} required")

            print(f"\n📈 Results ({analysis['primary_metric']}):")
            results = analysis['results']
            print(f"  Control Mean: {results['control_mean']:.4f}")
            print(f"  Variant Mean: {results['variant_mean']:.4f}")
            print(f"  Improvement:  {results['improvement_pct']:+.2f}%")

            print(f"\n📉 Statistical Significance:")
            print(f"  t-statistic: {results['t_statistic']:.3f}")
            print(f"  p-value:     {results['p_value']:.3f}")
            print(f"  Significant: {'Yes' if results['statistically_significant'] else 'No'}")

            print(f"\n🎯 Recommendation:")
            rec = analysis['recommendation']
            if rec == 'apply_variant':
                print(f"  ✅ APPLY VARIANT (statistically significant improvement)")
            else:
                print(f"  ⊘ KEEP CONTROL (no significant improvement)")

            print(f"\nSuccess Criteria Met: {'Yes' if analysis['meets_success_criteria'] else 'No'}")
            print("\n" + "=" * 60)

        elif args.ab_command == 'apply':
            result = ab_test.apply_variant(args.experiment, dry_run=args.dry_run)

            if 'error' in result:
                print(f"❌ {result['error']}")
                if 'reason' in result:
                    print(f"   {result['reason']}")
                return

            print(f"\n{'🔍 DRY RUN' if result['status'] == 'dry_run' else '✅ APPLIED'}")
            print(f"\nExperiment: {result['experiment']}")
            print("\nModifications:")

            for mod in result['modifications']:
                print(f"  {mod['target']}")
                print(f"    Old: {mod['old_value']}")
                print(f"    New: {mod['new_value']}")
                print(f"    Improvement: {mod['improvement']:+.2f}%")

            if result['status'] == 'dry_run':
                print("\n💡 Run without --dry-run to apply changes")

        elif args.ab_command == 'status':
            experiments = ab_test.list_experiments(active_only=True)

            if not experiments:
                print("No active experiments")
                return

            print("\n" + "=" * 70)
            print("  ACTIVE A/B EXPERIMENTS")
            print("=" * 70)

            for exp in experiments:
                analysis = ab_test.analyze_experiment(exp['id'])

                print(f"\n📊 {exp['name']} ({exp['id']})")
                print(f"   Status: {analysis['status']}")
                print(f"   Samples: {analysis['samples']['total']} / {analysis['samples']['min_required']}")

                if analysis['status'] == 'complete':
                    results = analysis['results']
                    print(f"   Improvement: {results['improvement_pct']:+.2f}%")
                    print(f"   Significant: {'Yes' if results['statistically_significant'] else 'No'}")
                    print(f"   Recommendation: {analysis['recommendation']}")

            print("\n" + "=" * 70)

    elif args.command == 'cpb':
        # Import CPB module
        try:
            from cpb import cpb, dq_scorer
        except ImportError:
            print("❌ CPB module not found. Ensure cpb/ directory exists.")
            sys.exit(1)

        if args.cpb_command == 'analyze':
            result = cpb.analyze(args.query, args.context)

            print("\n" + "=" * 60)
            print("  CPB QUERY ANALYSIS")
            print("=" * 60)
            print(f"\nQuery: {result['query']}")
            print(f"\n📊 Complexity: {result['complexity_score']:.3f}")
            print(f"🔀 Path: {result['selected_path'].upper()}")
            print(f"📝 {result['reasoning']}")
            print("=" * 60)

        elif args.cpb_command == 'score':
            dq = dq_scorer.score(args.query, args.response)

            print("\n" + "=" * 60)
            print("  DQ SCORE")
            print("=" * 60)
            print(f"\n📊 Overall: {dq.overall:.3f}")
            print(f"   Validity:    {dq.validity:.3f}")
            print(f"   Specificity: {dq.specificity:.3f}")
            print(f"   Correctness: {dq.correctness:.3f}")

            tier = dq_scorer.get_quality_tier(dq)
            print(f"\n🏆 Tier: {tier.upper()}")
            print("=" * 60)

        elif args.cpb_command == 'stats':
            stats = dq_scorer.get_stats(args.days)

            print("\n" + "=" * 60)
            print(f"  CPB DQ STATISTICS ({args.days} days)")
            print("=" * 60)

            if 'message' in stats:
                print(f"\n{stats['message']}")
            else:
                print(f"\n📊 Total: {stats['total_scored']}")
                print(f"📈 Avg DQ: {stats['avg_dq']:.3f}")
                print(f"✅ Above 0.75: {stats['above_threshold']}")
                print(f"❌ Below 0.60: {stats['below_min']}")

            print("=" * 60)

        elif args.cpb_command == 'status':
            status = cpb.get_status()

            print("\n" + "=" * 60)
            print("  CPB STATUS")
            print("=" * 60)
            print(f"\n🏷️  Tier: {status['tier'].upper()}")
            print(f"📚 Learning: {'✓' if status['learning_enabled'] else '✗'}")
            print(f"✅ Verification: {'✓' if status['verification_enabled'] else '✗'}")
            print(f"\n⚙️  Config:")
            for k, v in status['config'].items():
                print(f"   {k}: {v}")
            print("=" * 60)

        else:
            print("Usage: routing-metrics.py cpb <analyze|score|stats|status>")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
