#!/usr/bin/env python3
"""
ChatGPT Conversation Quality Scorer
====================================

Scores ChatGPT export conversations for quality to prevent garbage pollution
of the Universal Cognitive Wallet database.

Metrics:
- Depth: How deep is the thinking? (message length, back-and-forth, complexity)
- Focus: How focused is the conversation? (topic consistency, drift)
- Signal: How much signal vs noise? (substance vs fluff)
- Purpose: Is there a clear cognitive goal?

Cognitive Modes:
- deep_work: High-quality focused research/coding (>0.75)
- exploration: Quality thinking and learning (0.5-0.75)
- casual: Light conversations (0.3-0.5)
- garbage: Low-quality noise (<0.3)

Usage:
    python3 chatgpt_quality_scorer.py ~/Downloads/chatgpt-export-2026-02-06
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import re


@dataclass
class QualityMetrics:
    """Quality scoring metrics for a conversation"""

    # Core metrics (0.0-1.0)
    depth: float
    focus: float
    signal: float

    # Overall quality (weighted average)
    quality_score: float

    # Classification
    cognitive_mode: str  # deep_work, exploration, casual, garbage
    purpose: str  # research, coding, thinking, learning, random

    # Signal strength
    signal_strength: float

    # Import decision
    import_recommended: bool

    # Raw data
    message_count: int
    total_chars: int
    avg_message_length: float
    topic_consistency: float


class ConversationQualityScorer:
    """Score ChatGPT conversations for quality"""

    # Quality thresholds
    DEEP_WORK_THRESHOLD = 0.75
    EXPLORATION_THRESHOLD = 0.50
    CASUAL_THRESHOLD = 0.30
    IMPORT_THRESHOLD = 0.40

    # Depth signals (technical/research terms)
    DEPTH_KEYWORDS = {
        'code', 'function', 'algorithm', 'database', 'schema', 'implementation',
        'architecture', 'design', 'analysis', 'research', 'theory', 'model',
        'system', 'protocol', 'framework', 'optimization', 'performance',
        'explain', 'understand', 'concept', 'principle', 'mechanism',
        'why', 'how does', 'what is the', 'difference between',
        'cognitive', 'semantic', 'temporal', 'coherence', 'emergence',
        'infrastructure', 'sovereign', 'autonomous', 'intelligence',
    }

    # Garbage signals (low-value conversation)
    GARBAGE_KEYWORDS = {
        'hi', 'hello', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no',
        'lol', 'haha', 'cool', 'nice', 'great', 'awesome',
        '?', '!', 'test', 'testing', 'can you', 'help me',
    }

    # Focus topics (technical domains)
    FOCUS_TOPICS = {
        'programming': ['code', 'function', 'class', 'variable', 'debug', 'error'],
        'database': ['sql', 'query', 'schema', 'table', 'database', 'postgresql'],
        'ai': ['ai', 'ml', 'model', 'neural', 'training', 'inference', 'llm'],
        'architecture': ['architecture', 'design', 'system', 'infrastructure', 'pattern'],
        'research': ['research', 'paper', 'study', 'analysis', 'methodology'],
        'ucw': ['ucw', 'cognitive', 'wallet', 'coherence', 'sovereignty', 'emergence'],
    }

    def score_conversation(self, conversation: Dict) -> QualityMetrics:
        """
        Score a single ChatGPT conversation

        Args:
            conversation: ChatGPT export conversation object

        Returns:
            QualityMetrics with scoring results
        """

        # Extract messages
        messages = self._extract_messages(conversation)

        if not messages:
            return self._create_garbage_metrics()

        # Calculate metrics
        depth = self._calculate_depth(messages)
        focus = self._calculate_focus(messages)
        signal = self._calculate_signal(messages)

        # Overall quality (weighted average)
        quality_score = (depth * 0.4) + (focus * 0.3) + (signal * 0.3)

        # Classify cognitive mode
        cognitive_mode = self._classify_cognitive_mode(quality_score, depth, focus)

        # Infer purpose
        purpose = self._infer_purpose(messages, depth, focus)

        # Signal strength (how valuable is this data?)
        signal_strength = (depth + signal) / 2

        # Should we import?
        import_recommended = quality_score >= self.IMPORT_THRESHOLD

        # Calculate stats
        message_count = len(messages)
        total_chars = sum(len(m['content']) for m in messages)
        avg_message_length = total_chars / message_count if message_count > 0 else 0
        topic_consistency = self._calculate_topic_consistency(messages)

        return QualityMetrics(
            depth=depth,
            focus=focus,
            signal=signal,
            quality_score=quality_score,
            cognitive_mode=cognitive_mode,
            purpose=purpose,
            signal_strength=signal_strength,
            import_recommended=import_recommended,
            message_count=message_count,
            total_chars=total_chars,
            avg_message_length=avg_message_length,
            topic_consistency=topic_consistency,
        )

    def _extract_messages(self, conversation: Dict) -> List[Dict]:
        """Extract messages from ChatGPT conversation structure"""
        messages = []

        mapping = conversation.get('mapping', {})

        for msg_id, msg_data in mapping.items():
            message = msg_data.get('message')

            if not message:
                continue

            author = message.get('author', {})
            role = author.get('role', '')
            content = message.get('content', {})
            parts = content.get('parts', [])

            # Skip system messages
            if role == 'system':
                continue

            # Extract text (filter out non-string parts like image/file dicts)
            text = '\n'.join(p if isinstance(p, str) else p.get('text', '') if isinstance(p, dict) else str(p) for p in parts) if parts else ''

            if not text or not text.strip():
                continue

            messages.append({
                'role': role,
                'content': text,
                'create_time': message.get('create_time', 0),
            })

        # Sort by create_time
        messages.sort(key=lambda m: m['create_time'])

        return messages

    def _calculate_depth(self, messages: List[Dict]) -> float:
        """
        Calculate conversation depth

        Indicators:
        - Average message length (longer = deeper)
        - Technical keyword density
        - Question complexity
        - Back-and-forth exchanges
        """

        if not messages:
            return 0.0

        score = 0.0

        # 1. Message length (25% weight)
        avg_length = sum(len(m['content']) for m in messages) / len(messages)
        length_score = min(avg_length / 500, 1.0)  # Cap at 500 chars
        score += length_score * 0.25

        # 2. Technical keyword density (35% weight)
        all_text = ' '.join(m['content'] for m in messages).lower()
        depth_keyword_count = sum(1 for kw in self.DEPTH_KEYWORDS if kw in all_text)
        keyword_density = min(depth_keyword_count / 10, 1.0)  # Cap at 10 keywords
        score += keyword_density * 0.35

        # 3. Question complexity (20% weight)
        question_complexity = self._analyze_question_complexity(messages)
        score += question_complexity * 0.20

        # 4. Back-and-forth exchanges (20% weight)
        exchange_score = min(len(messages) / 20, 1.0)  # Cap at 20 messages
        score += exchange_score * 0.20

        return min(score, 1.0)

    def _calculate_focus(self, messages: List[Dict]) -> float:
        """
        Calculate conversation focus

        Indicators:
        - Topic consistency across messages
        - Minimal drift
        - Clear domain (programming, research, etc.)
        """

        if not messages:
            return 0.0

        # Detect topics across messages
        topics_per_message = []

        for msg in messages:
            content_lower = msg['content'].lower()
            msg_topics = []

            for topic, keywords in self.FOCUS_TOPICS.items():
                if any(kw in content_lower for kw in keywords):
                    msg_topics.append(topic)

            topics_per_message.append(msg_topics)

        # Calculate consistency
        if not any(topics_per_message):
            return 0.3  # No clear topic = low but not zero focus

        # Find most common topic
        all_topics = [t for topics in topics_per_message for t in topics]
        if not all_topics:
            return 0.3

        topic_counts = Counter(all_topics)
        most_common_topic, count = topic_counts.most_common(1)[0]

        # Consistency = how many messages mention the main topic
        consistency = count / len(messages)

        return min(consistency, 1.0)

    def _calculate_signal(self, messages: List[Dict]) -> float:
        """
        Calculate signal-to-noise ratio

        Indicators:
        - Substance vs fluff
        - Meaningful content vs greetings/acknowledgments
        - Technical density
        """

        if not messages:
            return 0.0

        signal_score = 0.0
        noise_score = 0.0

        for msg in messages:
            content = msg['content'].lower()

            # Count garbage signals (noise)
            garbage_count = sum(1 for kw in self.GARBAGE_KEYWORDS if kw == content.strip())
            if garbage_count > 0 or len(content.strip()) < 10:
                noise_score += 1
                continue

            # Count depth signals (signal)
            depth_count = sum(1 for kw in self.DEPTH_KEYWORDS if kw in content)
            if depth_count > 0:
                signal_score += 1
            elif len(content) > 50:  # Substantial message even without keywords
                signal_score += 0.5

        total = signal_score + noise_score
        if total == 0:
            return 0.5  # Neutral

        return signal_score / total

    def _analyze_question_complexity(self, messages: List[Dict]) -> float:
        """Analyze complexity of questions asked"""

        user_messages = [m for m in messages if m['role'] == 'user']

        if not user_messages:
            return 0.5

        complexity_score = 0.0

        for msg in user_messages:
            content = msg['content'].lower()

            # Complex question indicators
            if any(phrase in content for phrase in ['how does', 'why is', 'explain', 'what is the difference']):
                complexity_score += 1.0
            elif any(phrase in content for phrase in ['how to', 'what is', 'can you']):
                complexity_score += 0.5
            elif '?' in content:
                complexity_score += 0.3

        return min(complexity_score / len(user_messages), 1.0)

    def _calculate_topic_consistency(self, messages: List[Dict]) -> float:
        """Calculate how consistent the topic is across messages"""
        return self._calculate_focus(messages)  # Reuse focus calculation

    def _classify_cognitive_mode(self, quality: float, depth: float, focus: float) -> str:
        """Classify conversation into cognitive mode"""

        if quality >= self.DEEP_WORK_THRESHOLD and depth >= 0.7:
            return 'deep_work'
        elif quality >= self.EXPLORATION_THRESHOLD:
            return 'exploration'
        elif quality >= self.CASUAL_THRESHOLD:
            return 'casual'
        else:
            return 'garbage'

    def _infer_purpose(self, messages: List[Dict], depth: float, focus: float) -> str:
        """Infer the purpose of the conversation"""

        all_text = ' '.join(m['content'] for m in messages).lower()

        # Check for specific purposes
        if any(kw in all_text for kw in ['code', 'function', 'debug', 'implement', 'write a']):
            return 'coding'
        elif any(kw in all_text for kw in ['research', 'paper', 'study', 'analyze', 'explain']):
            return 'research'
        elif any(kw in all_text for kw in ['design', 'architecture', 'system', 'plan']):
            return 'thinking'
        elif any(kw in all_text for kw in ['learn', 'understand', 'teach', 'how to']):
            return 'learning'
        else:
            return 'random'

    def _create_garbage_metrics(self) -> QualityMetrics:
        """Create metrics for garbage/empty conversation"""
        return QualityMetrics(
            depth=0.0,
            focus=0.0,
            signal=0.0,
            quality_score=0.0,
            cognitive_mode='garbage',
            purpose='random',
            signal_strength=0.0,
            import_recommended=False,
            message_count=0,
            total_chars=0,
            avg_message_length=0,
            topic_consistency=0.0,
        )


class ChatGPTExportAnalyzer:
    """Analyze entire ChatGPT export"""

    def __init__(self, export_path: Path):
        self.export_path = export_path
        self.conversations_file = export_path / "conversations.json"
        self.scorer = ConversationQualityScorer()

    def load_conversations(self) -> List[Dict]:
        """Load all conversations from export"""
        with open(self.conversations_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_export(self) -> Tuple[List[Tuple[Dict, QualityMetrics]], Dict]:
        """
        Analyze entire export and return scored conversations + summary

        Returns:
            (scored_conversations, summary)
        """

        print("📦 Loading ChatGPT export...")
        conversations = self.load_conversations()
        print(f"   Found {len(conversations)} conversations\n")

        print("🎯 Scoring conversations...")
        scored_conversations = []

        for i, conv in enumerate(conversations):
            if (i + 1) % 100 == 0:
                print(f"   Scored {i + 1}/{len(conversations)}...")

            metrics = self.scorer.score_conversation(conv)
            scored_conversations.append((conv, metrics))

        print(f"   ✅ Scored all {len(conversations)} conversations\n")

        # Generate summary
        summary = self._generate_summary(scored_conversations)

        return scored_conversations, summary

    def _generate_summary(self, scored_conversations: List[Tuple[Dict, QualityMetrics]]) -> Dict:
        """Generate summary statistics"""

        total = len(scored_conversations)

        # Quality distribution
        deep_work = sum(1 for _, m in scored_conversations if m.cognitive_mode == 'deep_work')
        exploration = sum(1 for _, m in scored_conversations if m.cognitive_mode == 'exploration')
        casual = sum(1 for _, m in scored_conversations if m.cognitive_mode == 'casual')
        garbage = sum(1 for _, m in scored_conversations if m.cognitive_mode == 'garbage')

        # Import recommendation
        recommended = sum(1 for _, m in scored_conversations if m.import_recommended)
        not_recommended = total - recommended

        # Purpose distribution
        purposes = Counter(m.purpose for _, m in scored_conversations)

        # Average scores
        avg_quality = sum(m.quality_score for _, m in scored_conversations) / total if total > 0 else 0
        avg_depth = sum(m.depth for _, m in scored_conversations) / total if total > 0 else 0
        avg_signal = sum(m.signal_strength for _, m in scored_conversations) / total if total > 0 else 0

        return {
            'total_conversations': total,
            'quality_distribution': {
                'deep_work': deep_work,
                'exploration': exploration,
                'casual': casual,
                'garbage': garbage,
            },
            'import_recommendation': {
                'recommended': recommended,
                'not_recommended': not_recommended,
                'percentage_recommended': (recommended / total * 100) if total > 0 else 0,
            },
            'purpose_distribution': dict(purposes),
            'average_scores': {
                'quality': avg_quality,
                'depth': avg_depth,
                'signal_strength': avg_signal,
            },
        }

    def print_summary(self, summary: Dict):
        """Print summary report"""

        print("=" * 70)
        print("CHATGPT EXPORT QUALITY ANALYSIS")
        print("=" * 70)
        print()

        print(f"📊 Total Conversations: {summary['total_conversations']}")
        print()

        print("🎯 Quality Distribution:")
        qd = summary['quality_distribution']
        total = summary['total_conversations']
        print(f"   ⭐⭐⭐⭐⭐ Deep Work:    {qd['deep_work']:4d} ({qd['deep_work']/total*100:5.1f}%)")
        print(f"   ⭐⭐⭐   Exploration:  {qd['exploration']:4d} ({qd['exploration']/total*100:5.1f}%)")
        print(f"   ⭐     Casual:       {qd['casual']:4d} ({qd['casual']/total*100:5.1f}%)")
        print(f"   ❌     Garbage:      {qd['garbage']:4d} ({qd['garbage']/total*100:5.1f}%)")
        print()

        print("📥 Import Recommendation:")
        ir = summary['import_recommendation']
        print(f"   ✅ Recommended:     {ir['recommended']:4d} ({ir['percentage_recommended']:5.1f}%)")
        print(f"   ❌ Not Recommended: {ir['not_recommended']:4d}")
        print()

        print("🎯 Purpose Distribution:")
        for purpose, count in sorted(summary['purpose_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {purpose:15s}: {count:4d}")
        print()

        print("📈 Average Scores:")
        avg = summary['average_scores']
        print(f"   Quality:        {avg['quality']:.3f}")
        print(f"   Depth:          {avg['depth']:.3f}")
        print(f"   Signal Strength: {avg['signal_strength']:.3f}")
        print()

        print("=" * 70)
        print()

        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print()

        if ir['percentage_recommended'] > 50:
            print(f"   ✅ {ir['percentage_recommended']:.0f}% of conversations are quality.")
            print(f"   → Import {ir['recommended']} conversations")
            print(f"   → Skip {ir['not_recommended']} low-quality conversations")
        else:
            print(f"   ⚠️  Only {ir['percentage_recommended']:.0f}% of conversations are quality.")
            print(f"   → Consider raising import threshold")
            print(f"   → Or manually review borderline conversations")

        print()

        if qd['deep_work'] > 0:
            print(f"   ⭐ You have {qd['deep_work']} deep work conversations!")
            print(f"   → These are your highest-value cognitive assets")
            print(f"   → Import these FIRST")

        print()
        print("=" * 70)

    def export_scores(self, scored_conversations: List[Tuple[Dict, QualityMetrics]], output_path: Path):
        """Export scored conversations to JSON"""

        output_data = []

        for conv, metrics in scored_conversations:
            output_data.append({
                'conversation_id': conv.get('id'),
                'title': conv.get('title'),
                'create_time': conv.get('create_time'),
                'update_time': conv.get('update_time'),
                'metrics': asdict(metrics),
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"📝 Exported scores to: {output_path}")


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python3 chatgpt_quality_scorer.py <export_path>")
        print()
        print("Example:")
        print("  python3 chatgpt_quality_scorer.py ~/Downloads/chatgpt-export-2026-02-06")
        sys.exit(1)

    export_path = Path(sys.argv[1]).expanduser()

    if not export_path.exists():
        print(f"❌ Export path not found: {export_path}")
        sys.exit(1)

    # Analyze export
    analyzer = ChatGPTExportAnalyzer(export_path)
    scored_conversations, summary = analyzer.analyze_export()

    # Print summary
    analyzer.print_summary(summary)

    # Export scores
    output_path = export_path / "quality_scores.json"
    analyzer.export_scores(scored_conversations, output_path)

    print()
    print("✅ Quality analysis complete!")
    print()
    print("Next steps:")
    print("  1. Review quality_scores.json")
    print("  2. Decide import threshold")
    print("  3. Run import with quality filtering")


if __name__ == '__main__':
    main()
