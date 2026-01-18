#!/usr/bin/env python3
"""
Routing Test Suite - Comprehensive testing for autonomous routing system

Tests:
- Regression: Known queries → expected models
- Integration: End-to-end routing decisions
- Performance: Latency, accuracy, cost efficiency
- Validation: Baseline consistency, target compliance
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

HOME = Path.home()
KERNEL_DIR = HOME / ".claude/kernel"
DQ_SCORER = KERNEL_DIR / "dq-scorer.js"
BASELINES_FILE = KERNEL_DIR / "baselines.json"

# Test query dataset with expected optimal models
REGRESSION_QUERIES = [
    # Haiku-optimal (simple, quick)
    {"query": "What is 2+2?", "expected": "haiku", "complexity_range": [0.0, 0.3]},
    {"query": "Define recursion", "expected": "haiku", "complexity_range": [0.0, 0.3]},
    {"query": "Capital of France", "expected": "haiku", "complexity_range": [0.0, 0.3]},
    {"query": "Format this JSON: {a:1}", "expected": "haiku", "complexity_range": [0.0, 0.3]},
    {"query": "Convert 100 USD to EUR", "expected": "haiku", "complexity_range": [0.0, 0.3]},

    # Sonnet-optimal (moderate complexity)
    {"query": "Implement binary search in Python", "expected": "sonnet", "complexity_range": [0.3, 0.7]},
    {"query": "Explain async/await in JavaScript with examples", "expected": "sonnet", "complexity_range": [0.3, 0.7]},
    {"query": "Write a React hook for API calls with error handling", "expected": "sonnet", "complexity_range": [0.3, 0.7]},
    {"query": "Refactor this code to use dependency injection", "expected": "sonnet", "complexity_range": [0.3, 0.7]},
    {"query": "Create SQL query to join 3 tables and aggregate results", "expected": "sonnet", "complexity_range": [0.3, 0.7]},

    # Opus-optimal (complex, architectural)
    {"query": "Design a distributed caching system with consistency guarantees", "expected": "opus", "complexity_range": [0.7, 1.0]},
    {"query": "Architect a microservices system for real-time data processing at scale", "expected": "opus", "complexity_range": [0.7, 1.0]},
    {"query": "Explain trade-offs between CAP theorem choices for financial transactions", "expected": "opus", "complexity_range": [0.7, 1.0]},
    {"query": "Design fault-tolerant distributed queue with exactly-once delivery", "expected": "opus", "complexity_range": [0.7, 1.0]},
    {"query": "Analyze security vulnerabilities in OAuth 2.0 implementations", "expected": "opus", "complexity_range": [0.7, 1.0]},
]

# ═══════════════════════════════════════════════════════════════════════════
# TEST FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════

class RoutingTestSuite:
    def __init__(self):
        self.results = {
            "regression": [],
            "integration": [],
            "performance": {},
            "validation": {}
        }

    def run_regression_tests(self) -> Dict:
        """Test known queries against expected routing decisions"""
        print("\n" + "="*70)
        print("REGRESSION TEST SUITE")
        print("="*70 + "\n")

        results = []
        correct = 0
        total = len(REGRESSION_QUERIES)

        for i, test_case in enumerate(REGRESSION_QUERIES, 1):
            query = test_case['query']
            expected = test_case['expected']
            complexity_range = test_case.get('complexity_range', [0, 1])

            print(f"[{i}/{total}] Testing: {query[:50]}...")

            # Route query via DQ scorer
            try:
                start = time.time()
                result = subprocess.run(
                    ['node', str(DQ_SCORER), 'route', query],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                latency_ms = (time.time() - start) * 1000

                if result.returncode != 0:
                    print(f"  ✗ FAIL: DQ scorer error")
                    results.append({
                        "query": query,
                        "expected": expected,
                        "actual": None,
                        "passed": False,
                        "error": "dq_scorer_error"
                    })
                    continue

                # Parse result
                decision = json.loads(result.stdout)
                actual_model = decision.get('model', 'unknown')
                complexity = decision.get('complexity', 0)
                dq_score = decision.get('dq', {}).get('score', 0)

                # Check if correct
                passed = (actual_model == expected)

                # Complexity should be in expected range
                complexity_valid = (complexity_range[0] <= complexity <= complexity_range[1])

                if passed and complexity_valid:
                    print(f"  ✓ PASS: {actual_model} (complexity: {complexity:.2f}, DQ: {dq_score:.2f}, {latency_ms:.0f}ms)")
                    correct += 1
                else:
                    if not passed:
                        print(f"  ✗ FAIL: Expected {expected}, got {actual_model}")
                    if not complexity_valid:
                        print(f"  ⚠ Complexity {complexity:.2f} outside range {complexity_range}")

                results.append({
                    "query": query,
                    "expected": expected,
                    "actual": actual_model,
                    "complexity": complexity,
                    "complexity_valid": complexity_valid,
                    "dq_score": dq_score,
                    "latency_ms": latency_ms,
                    "passed": passed and complexity_valid
                })

            except subprocess.TimeoutExpired:
                print(f"  ✗ FAIL: Timeout")
                results.append({
                    "query": query,
                    "expected": expected,
                    "actual": None,
                    "passed": False,
                    "error": "timeout"
                })
            except Exception as e:
                print(f"  ✗ FAIL: {e}")
                results.append({
                    "query": query,
                    "expected": expected,
                    "actual": None,
                    "passed": False,
                    "error": str(e)
                })

        accuracy = correct / total if total > 0 else 0

        print(f"\n{'='*70}")
        print(f"REGRESSION TEST RESULTS: {correct}/{total} passed ({accuracy:.1%})")
        print(f"{'='*70}\n")

        self.results['regression'] = {
            "total": total,
            "passed": correct,
            "accuracy": accuracy,
            "tests": results
        }

        return self.results['regression']

    def run_performance_tests(self) -> Dict:
        """Test routing performance (latency, overhead)"""
        print("\n" + "="*70)
        print("PERFORMANCE TEST SUITE")
        print("="*70 + "\n")

        # Test routing latency
        print("Testing routing latency...")

        latencies = []
        test_query = "test routing latency"

        for i in range(10):
            try:
                start = time.time()
                subprocess.run(
                    ['node', str(DQ_SCORER), 'route', test_query],
                    capture_output=True,
                    timeout=1
                )
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)
            except:
                pass

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p50_latency = sorted(latencies)[len(latencies) // 2]
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            max_latency = max(latencies)

            print(f"  Avg: {avg_latency:.1f}ms")
            print(f"  p50: {p50_latency:.1f}ms")
            print(f"  p95: {p95_latency:.1f}ms")
            print(f"  Max: {max_latency:.1f}ms")

            # Check against target (<50ms p95)
            latency_pass = p95_latency < 50

            print(f"\n  Latency Target (p95 < 50ms): {'✓ PASS' if latency_pass else '✗ FAIL'}")

            self.results['performance'] = {
                "latency": {
                    "avg_ms": avg_latency,
                    "p50_ms": p50_latency,
                    "p95_ms": p95_latency,
                    "max_ms": max_latency,
                    "target_met": latency_pass
                }
            }
        else:
            print("  ✗ FAIL: Could not measure latency")
            self.results['performance'] = {"error": "latency_measurement_failed"}

        print(f"\n{'='*70}\n")

        return self.results['performance']

    def run_validation_tests(self) -> Dict:
        """Validate baseline configuration and consistency"""
        print("\n" + "="*70)
        print("VALIDATION TEST SUITE")
        print("="*70 + "\n")

        validations = {}

        # Check 1: Baselines file exists and is valid JSON
        print("Checking baselines.json...")
        if BASELINES_FILE.exists():
            try:
                baselines = json.loads(BASELINES_FILE.read_text())
                print("  ✓ File exists and is valid JSON")

                # Check required fields
                required_fields = [
                    'version',
                    'complexity_thresholds',
                    'dq_weights',
                    'performance_targets'
                ]

                missing = [f for f in required_fields if f not in baselines]

                if not missing:
                    print("  ✓ All required fields present")
                    validations['baselines_structure'] = True
                else:
                    print(f"  ✗ Missing fields: {missing}")
                    validations['baselines_structure'] = False

                # Check threshold consistency
                thresholds = baselines.get('complexity_thresholds', {})

                haiku_max = thresholds.get('haiku', {}).get('range', [0, 0.3])[1]
                sonnet_min = thresholds.get('sonnet', {}).get('range', [0.3, 0.7])[0]
                sonnet_max = thresholds.get('sonnet', {}).get('range', [0.3, 0.7])[1]
                opus_min = thresholds.get('opus', {}).get('range', [0.7, 1.0])[0]

                if haiku_max == sonnet_min and sonnet_max == opus_min:
                    print("  ✓ Threshold boundaries are consistent")
                    validations['threshold_consistency'] = True
                else:
                    print(f"  ⚠ Threshold gaps detected: haiku({haiku_max}) → sonnet({sonnet_min}-{sonnet_max}) → opus({opus_min})")
                    validations['threshold_consistency'] = False

                # Check DQ weights sum to 1.0
                dq_weights = baselines.get('dq_weights', {})
                weight_sum = sum(dq_weights.values())

                if abs(weight_sum - 1.0) < 0.01:
                    print("  ✓ DQ weights sum to 1.0")
                    validations['dq_weights_valid'] = True
                else:
                    print(f"  ✗ DQ weights sum to {weight_sum} (should be 1.0)")
                    validations['dq_weights_valid'] = False

            except json.JSONDecodeError:
                print("  ✗ Invalid JSON")
                validations['baselines_structure'] = False
        else:
            print("  ✗ File not found")
            validations['baselines_structure'] = False

        # Check 2: DQ scorer kernel availability
        print("\nChecking DQ scorer kernel...")
        if DQ_SCORER.exists():
            print("  ✓ dq-scorer.js exists")

            try:
                result = subprocess.run(
                    ['node', str(DQ_SCORER), 'route', 'test'],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0:
                    print("  ✓ Kernel operational")
                    validations['kernel_operational'] = True
                else:
                    print("  ✗ Kernel returned error")
                    validations['kernel_operational'] = False
            except:
                print("  ✗ Kernel execution failed")
                validations['kernel_operational'] = False
        else:
            print("  ✗ Kernel not found")
            validations['kernel_operational'] = False

        print(f"\n{'='*70}\n")

        self.results['validation'] = validations

        return validations

    def generate_report(self, output_format: str = 'text') -> str:
        """Generate test report"""
        if output_format == 'json':
            return json.dumps(self.results, indent=2)

        # Text report
        lines = []
        lines.append("\n" + "="*70)
        lines.append("ROUTING SYSTEM - TEST SUITE REPORT")
        lines.append("="*70 + "\n")

        lines.append(f"Generated: {datetime.now().isoformat()}\n")

        # Regression results
        if 'regression' in self.results and self.results['regression']:
            reg = self.results['regression']
            lines.append(f"📊 Regression Tests: {reg['passed']}/{reg['total']} passed ({reg['accuracy']:.1%})")

        # Performance results
        if 'performance' in self.results and 'latency' in self.results['performance']:
            perf = self.results['performance']['latency']
            lines.append(f"⏱️  Routing Latency: p95={perf['p95_ms']:.1f}ms (target: <50ms) {'✓' if perf['target_met'] else '✗'}")

        # Validation results
        if 'validation' in self.results:
            val = self.results['validation']
            passed_val = sum(1 for v in val.values() if v)
            total_val = len(val)
            lines.append(f"✅ Validation Checks: {passed_val}/{total_val} passed")

        lines.append("\n" + "="*70)

        # Overall status
        all_pass = (
            self.results.get('regression', {}).get('accuracy', 0) >= 0.80 and
            self.results.get('performance', {}).get('latency', {}).get('target_met', False) and
            sum(1 for v in self.results.get('validation', {}).values() if v) == len(self.results.get('validation', {}))
        )

        if all_pass:
            lines.append("\n✅ OVERALL: PASS - System ready for production")
        else:
            lines.append("\n⚠️  OVERALL: REVIEW REQUIRED - Some tests failed")

        lines.append("\n" + "="*70 + "\n")

        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Routing system test suite"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run all tests
    all_parser = subparsers.add_parser('all', help='Run all tests')
    all_parser.add_argument('--format', choices=['text', 'json'], default='text')
    all_parser.add_argument('--output', help='Output file')

    # Individual test suites
    reg_parser = subparsers.add_parser('regression', help='Run regression tests only')
    reg_parser.add_argument('--format', choices=['text', 'json'], default='text')

    perf_parser = subparsers.add_parser('performance', help='Run performance tests only')
    perf_parser.add_argument('--format', choices=['text', 'json'], default='text')

    val_parser = subparsers.add_parser('validation', help='Run validation tests only')
    val_parser.add_argument('--format', choices=['text', 'json'], default='text')

    args = parser.parse_args()

    suite = RoutingTestSuite()

    if args.command == 'all':
        suite.run_regression_tests()
        suite.run_performance_tests()
        suite.run_validation_tests()

        report = suite.generate_report(output_format=args.format)

        if args.output:
            Path(args.output).write_text(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)

        # Exit with status code
        all_pass = (
            suite.results.get('regression', {}).get('accuracy', 0) >= 0.80 and
            suite.results.get('performance', {}).get('latency', {}).get('target_met', False) and
            sum(1 for v in suite.results.get('validation', {}).values() if v) == len(suite.results.get('validation', {}))
        )

        sys.exit(0 if all_pass else 1)

    elif args.command == 'regression':
        results = suite.run_regression_tests()

        if args.format == 'json':
            print(json.dumps(results, indent=2))

        sys.exit(0 if results.get('accuracy', 0) >= 0.80 else 1)

    elif args.command == 'performance':
        results = suite.run_performance_tests()

        if args.format == 'json':
            print(json.dumps(results, indent=2))

        sys.exit(0 if results.get('latency', {}).get('target_met', False) else 1)

    elif args.command == 'validation':
        results = suite.run_validation_tests()

        if args.format == 'json':
            print(json.dumps(results, indent=2))

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        sys.exit(0 if passed == total else 1)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
