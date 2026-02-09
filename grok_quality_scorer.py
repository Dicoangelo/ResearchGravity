#!/usr/bin/env python3
"""
Grok Conversation Quality Scorer
==================================

Scores Grok (xAI) export conversations for quality to prevent garbage pollution
of the Universal Cognitive Wallet database.

Grok export format (prod-grok-backend.json):
  conversations[].conversation  — metadata (id, title, create_time, modify_time)
  conversations[].responses[]   — messages (sender, message, create_time, model)

Metrics:
- Depth (40%): Message length, technical keywords, question complexity, exchanges
- Focus (30%): Topic consistency across messages
- Signal (30%): Substance vs noise ratio

Cognitive Modes:
- deep_work:   quality ≥ 0.75 AND depth ≥ 0.70
- exploration: quality ≥ 0.50
- casual:      quality ≥ 0.30
- garbage:     quality < 0.30

Grok Strategic Boost:
  Grok excels at strategic/market/geopolitical analysis.
  Conversations with 2+ strategic keywords get a depth boost.

Usage:
    python3 grok_quality_scorer.py ~/Desktop/ttl/30d/export_data/<user_id>
    python3 grok_quality_scorer.py ~/Desktop/ttl/30d/export_data/<user_id> --dry-run
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from collections import Counter


@dataclass
class QualityMetrics:
    """Quality scoring metrics for a conversation."""

    depth: float
    focus: float
    signal: float
    quality_score: float
    cognitive_mode: str
    purpose: str
    signal_strength: float
    import_recommended: bool
    message_count: int
    total_chars: int
    avg_message_length: float
    topic_consistency: float
    model: str


class GrokQualityScorer:
    """Score Grok conversations for quality."""

    DEEP_WORK_THRESHOLD = 0.75
    EXPLORATION_THRESHOLD = 0.50
    CASUAL_THRESHOLD = 0.30
    IMPORT_THRESHOLD = 0.40

    # Depth signals
    DEPTH_KEYWORDS = {
        'code', 'function', 'algorithm', 'database', 'schema', 'implementation',
        'architecture', 'design', 'analysis', 'research', 'theory', 'model',
        'system', 'protocol', 'framework', 'optimization', 'performance',
        'explain', 'understand', 'concept', 'principle', 'mechanism',
        'cognitive', 'semantic', 'temporal', 'coherence', 'emergence',
        'infrastructure', 'sovereign', 'autonomous', 'intelligence',
    }

    # Grok-specific strategic keywords
    STRATEGIC_KEYWORDS = {
        'strategy', 'market', 'trend', 'prediction', 'future', 'forecast',
        'geopolitics', 'economics', 'policy', 'innovation', 'disruption',
        'investment', 'growth', 'sector', 'competitive', 'landscape',
        'trade', 'regulation', 'valuation', 'macro', 'thesis',
    }

    GARBAGE_KEYWORDS = {
        'hi', 'hello', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no',
        'lol', 'haha', 'cool', 'nice', 'great', 'awesome',
        '?', '!', 'test', 'testing',
    }

    FOCUS_TOPICS = {
        'programming': ['code', 'function', 'class', 'variable', 'debug', 'error'],
        'database': ['sql', 'query', 'schema', 'table', 'database', 'postgresql'],
        'ai': ['ai', 'ml', 'model', 'neural', 'training', 'inference', 'llm'],
        'architecture': ['architecture', 'design', 'system', 'infrastructure', 'pattern'],
        'research': ['research', 'paper', 'study', 'analysis', 'methodology'],
        'ucw': ['ucw', 'cognitive', 'wallet', 'coherence', 'sovereignty', 'emergence'],
        'strategy': ['strategy', 'market', 'trend', 'geopolitics', 'economics', 'growth'],
        'career': ['career', 'job', 'resume', 'interview', 'skills', 'role'],
    }

    def score_conversation(self, conv_entry: Dict) -> QualityMetrics:
        """Score a single Grok conversation.

        Args:
            conv_entry: Dict with 'conversation' (metadata) and 'responses' (messages)
        """
        messages = self._extract_messages(conv_entry)

        if not messages:
            return self._create_garbage_metrics(conv_entry)

        depth = self._calculate_depth(messages)
        focus = self._calculate_focus(messages)
        signal = self._calculate_signal(messages)

        # Grok strategic boost
        all_text = ' '.join(m['content'] for m in messages).lower()
        strategic_count = sum(1 for kw in self.STRATEGIC_KEYWORDS if kw in all_text)
        if strategic_count >= 2:
            depth = min(depth + 0.10, 1.0)

        quality_score = (depth * 0.4) + (focus * 0.3) + (signal * 0.3)
        cognitive_mode = self._classify_cognitive_mode(quality_score, depth, focus)
        purpose = self._infer_purpose(messages, depth, focus)
        signal_strength = (depth + signal) / 2
        import_recommended = quality_score >= self.IMPORT_THRESHOLD

        message_count = len(messages)
        total_chars = sum(len(m['content']) for m in messages)
        avg_message_length = total_chars / message_count if message_count > 0 else 0
        topic_consistency = self._calculate_topic_consistency(messages)

        # Dominant model used
        model_counts = Counter(m.get('model', '') for m in messages if m.get('model'))
        dominant_model = model_counts.most_common(1)[0][0] if model_counts else ""

        return QualityMetrics(
            depth=round(depth, 4),
            focus=round(focus, 4),
            signal=round(signal, 4),
            quality_score=round(quality_score, 4),
            cognitive_mode=cognitive_mode,
            purpose=purpose,
            signal_strength=round(signal_strength, 4),
            import_recommended=import_recommended,
            message_count=message_count,
            total_chars=total_chars,
            avg_message_length=round(avg_message_length, 1),
            topic_consistency=round(topic_consistency, 4),
            model=dominant_model,
        )

    def _extract_messages(self, conv_entry: Dict) -> List[Dict]:
        """Extract messages from Grok conversation structure."""
        messages = []
        responses = conv_entry.get('responses', [])

        for resp_wrapper in responses:
            resp = resp_wrapper.get('response', {})
            sender = resp.get('sender', '')
            text = resp.get('message', '')
            model = resp.get('model', '')

            if not text or not text.strip():
                continue

            # Parse MongoDB timestamp
            create_time = resp.get('create_time', {})
            ts_ms = 0
            if isinstance(create_time, dict):
                date_obj = create_time.get('$date', {})
                if isinstance(date_obj, dict):
                    ts_ms = int(date_obj.get('$numberLong', '0'))
                elif isinstance(date_obj, (int, float)):
                    ts_ms = int(date_obj)
            elif isinstance(create_time, (int, float)):
                ts_ms = int(create_time)

            messages.append({
                'role': 'user' if sender == 'human' else 'assistant',
                'content': text.strip(),
                'create_time': ts_ms / 1000.0,  # Convert ms to seconds
                'model': model,
            })

        messages.sort(key=lambda m: m['create_time'])
        return messages

    def _calculate_depth(self, messages: List[Dict]) -> float:
        if not messages:
            return 0.0

        score = 0.0

        # 1. Message length (25%)
        avg_length = sum(len(m['content']) for m in messages) / len(messages)
        score += min(avg_length / 500, 1.0) * 0.25

        # 2. Technical keyword density (35%)
        all_text = ' '.join(m['content'] for m in messages).lower()
        depth_count = sum(1 for kw in self.DEPTH_KEYWORDS if kw in all_text)
        score += min(depth_count / 10, 1.0) * 0.35

        # 3. Question complexity (20%)
        score += self._analyze_question_complexity(messages) * 0.20

        # 4. Back-and-forth exchanges (20%)
        score += min(len(messages) / 20, 1.0) * 0.20

        return min(score, 1.0)

    def _calculate_focus(self, messages: List[Dict]) -> float:
        if not messages:
            return 0.0

        topics_per_message = []
        for msg in messages:
            content_lower = msg['content'].lower()
            msg_topics = [
                topic for topic, keywords in self.FOCUS_TOPICS.items()
                if any(kw in content_lower for kw in keywords)
            ]
            topics_per_message.append(msg_topics)

        all_topics = [t for topics in topics_per_message for t in topics]
        if not all_topics:
            return 0.3

        topic_counts = Counter(all_topics)
        _, count = topic_counts.most_common(1)[0]
        consistency = count / len(messages)
        return min(consistency, 1.0)

    def _calculate_signal(self, messages: List[Dict]) -> float:
        if not messages:
            return 0.0

        signal_score = 0.0
        noise_score = 0.0

        for msg in messages:
            content = msg['content'].lower().strip()

            if content in self.GARBAGE_KEYWORDS or len(content) < 10:
                noise_score += 1
                continue

            # Count depth keywords as signal
            kw_count = sum(1 for kw in self.DEPTH_KEYWORDS if kw in content)
            if kw_count >= 2:
                signal_score += 2
            elif kw_count >= 1:
                signal_score += 1
            elif len(content) > 100:
                signal_score += 0.5
            else:
                noise_score += 0.3

        total = signal_score + noise_score
        if total == 0:
            return 0.5
        return min(signal_score / total, 1.0)

    def _analyze_question_complexity(self, messages: List[Dict]) -> float:
        complex_patterns = [
            'how does', 'why does', 'what is the', 'explain',
            'difference between', 'compare', 'analyze', 'implement',
            'optimize', 'design', 'build', 'create a', 'develop',
            'what are the', 'could you', 'walk me through',
        ]

        user_msgs = [m for m in messages if m['role'] == 'user']
        if not user_msgs:
            return 0.0

        complex_count = 0
        for msg in user_msgs:
            content_lower = msg['content'].lower()
            if any(p in content_lower for p in complex_patterns):
                complex_count += 1

        return min(complex_count / max(len(user_msgs), 1), 1.0)

    def _classify_cognitive_mode(self, quality: float, depth: float, focus: float) -> str:
        if quality >= self.DEEP_WORK_THRESHOLD and depth >= 0.70:
            return "deep_work"
        elif quality >= self.EXPLORATION_THRESHOLD:
            return "exploration"
        elif quality >= self.CASUAL_THRESHOLD:
            return "casual"
        else:
            return "garbage"

    def _infer_purpose(self, messages: List[Dict], depth: float, focus: float) -> str:
        all_text = ' '.join(m['content'] for m in messages).lower()

        purposes = {
            'coding': ['code', 'function', 'debug', 'error', 'script', 'import', 'class'],
            'research': ['research', 'paper', 'study', 'methodology', 'literature', 'arxiv'],
            'thinking': ['think', 'philosophy', 'meaning', 'consciousness', 'emergence', 'cognitive'],
            'learning': ['learn', 'explain', 'tutorial', 'concept', 'understand', 'how does'],
            'strategy': ['strategy', 'market', 'trend', 'investment', 'growth', 'competitive'],
        }

        scores = {}
        for purpose, keywords in purposes.items():
            scores[purpose] = sum(1 for kw in keywords if kw in all_text)

        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return "random"

    def _calculate_topic_consistency(self, messages: List[Dict]) -> float:
        topics_per_message = []
        for msg in messages:
            content_lower = msg['content'].lower()
            msg_topics = [
                topic for topic, keywords in self.FOCUS_TOPICS.items()
                if any(kw in content_lower for kw in keywords)
            ]
            topics_per_message.append(msg_topics)

        messages_with_topics = sum(1 for t in topics_per_message if t)
        if messages_with_topics == 0:
            return 0.0
        return messages_with_topics / len(messages)

    def _create_garbage_metrics(self, conv_entry: Dict = None) -> QualityMetrics:
        return QualityMetrics(
            depth=0.0, focus=0.0, signal=0.0,
            quality_score=0.0, cognitive_mode="garbage",
            purpose="random", signal_strength=0.0,
            import_recommended=False,
            message_count=0, total_chars=0,
            avg_message_length=0.0, topic_consistency=0.0,
            model="",
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 grok_quality_scorer.py <export_dir> [--dry-run]")
        sys.exit(1)

    export_path = Path(sys.argv[1]).expanduser()
    dry_run = "--dry-run" in sys.argv

    # Find the Grok backend JSON
    grok_file = export_path / "prod-grok-backend.json"
    if not grok_file.exists():
        print(f"prod-grok-backend.json not found in {export_path}")
        sys.exit(1)

    print(f"Loading Grok export from {grok_file}...")
    with open(grok_file, "r") as f:
        data = json.load(f)

    conversations = data.get("conversations", [])
    print(f"  Loaded {len(conversations)} conversations\n")

    scorer = GrokQualityScorer()
    results = []
    mode_counts = Counter()
    purpose_counts = Counter()
    model_counts = Counter()

    for i, conv_entry in enumerate(conversations):
        conv = conv_entry.get("conversation", {})
        conv_id = conv.get("id", "")
        title = conv.get("title", "Untitled")

        metrics = scorer.score_conversation(conv_entry)
        mode_counts[metrics.cognitive_mode] += 1
        purpose_counts[metrics.purpose] += 1
        if metrics.model:
            model_counts[metrics.model] += 1

        results.append({
            "conversation_id": conv_id,
            "title": title,
            "create_time": conv.get("create_time", ""),
            "modify_time": conv.get("modify_time", ""),
            "metrics": asdict(metrics),
        })

        if (i + 1) % 500 == 0:
            print(f"  Scored {i + 1}/{len(conversations)}...")

    # Summary
    importable = sum(1 for r in results if r["metrics"]["import_recommended"])
    total_msgs = sum(r["metrics"]["message_count"] for r in results)
    quality_scores = [r["metrics"]["quality_score"] for r in results if r["metrics"]["quality_score"] > 0]

    print(f"\n{'='*60}")
    print(f"GROK QUALITY SCORING COMPLETE")
    print(f"{'='*60}")
    print(f"  Conversations scored: {len(results)}")
    print(f"  Total messages:       {total_msgs}")
    print(f"  Import recommended:   {importable} ({importable*100/len(results):.1f}%)")
    if quality_scores:
        print(f"  Avg quality score:    {sum(quality_scores)/len(quality_scores):.3f}")
    print()

    print("Cognitive mode breakdown:")
    for mode in ["deep_work", "exploration", "casual", "garbage"]:
        count = mode_counts.get(mode, 0)
        pct = count * 100 / len(results) if results else 0
        print(f"  {mode:15s}: {count:6d} ({pct:5.1f}%)")
    print()

    print("Purpose breakdown:")
    for purpose, count in purpose_counts.most_common():
        pct = count * 100 / len(results) if results else 0
        print(f"  {purpose:15s}: {count:6d} ({pct:5.1f}%)")
    print()

    print("Model breakdown:")
    for model, count in model_counts.most_common():
        print(f"  {model or '(empty)':40s}: {count:6d}")
    print()

    # Top deep_work conversations
    deep_work = [r for r in results if r["metrics"]["cognitive_mode"] == "deep_work"]
    deep_work.sort(key=lambda r: -r["metrics"]["quality_score"])
    if deep_work:
        print(f"Top deep_work conversations ({len(deep_work)} total):")
        for r in deep_work[:15]:
            m = r["metrics"]
            print(f"  [{m['quality_score']:.3f}] [{m['purpose']:10s}] [{m['model']:30s}] {r['title'][:50]}")
    print()

    if not dry_run:
        out_path = export_path / "grok_quality_scores.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Scores saved to: {out_path}")
    else:
        print("DRY RUN — no files written")


if __name__ == "__main__":
    main()
