#!/usr/bin/env python3
"""
Pack Builder - Generate context packs from sessions and learnings.

Creates modular, composable context packs optimized for token efficiency.

Usage:
  python3 build_packs.py --source sessions --since 2026-01-01
  python3 build_packs.py --source learnings --cluster-by topic
  python3 build_packs.py --create --type domain --name "quantum-computing"
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


AGENT_CORE = Path.home() / ".agent-core"
PACK_DIR = AGENT_CORE / "context-packs"
SESSIONS_DIR = AGENT_CORE / "sessions"


class PackBuilder:
    """Build context packs from various sources"""

    def __init__(self):
        self.pack_dir = PACK_DIR
        self.registry_path = self.pack_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load pack registry"""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {
            "version": "1.0.0",
            "created": datetime.utcnow().isoformat() + "Z",
            "packs": {},
            "metadata": {
                "total_packs": 0,
                "total_size_bytes": 0,
                "total_size_tokens": 0,
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }
        }

    def _save_registry(self):
        """Save pack registry"""
        self.registry["metadata"]["last_updated"] = datetime.utcnow().isoformat() + "Z"
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token"""
        return len(text) // 4

    def _extract_papers_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract arXiv paper references"""
        papers = []
        # Pattern: arXiv:XXXX.XXXXX or [XXXX.XXXXX]
        patterns = [
            r'arXiv:(\d{4}\.\d{5})',
            r'\[(\d{4}\.\d{5})\]',
            r'arxiv\.org/abs/(\d{4}\.\d{5})'
        ]

        seen = set()
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    papers.append({
                        "arxiv_id": match,
                        "relevance": 5  # Default, can be refined
                    })

        return papers

    def _extract_keywords(self, content: str, top_n: int = 10) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction - get frequent significant words
        words = re.findall(r'\b[a-z]{4,}\b', content.lower())

        # Filter out common words
        stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were',
                    'will', 'would', 'could', 'should', 'their', 'there', 'where'}
        words = [w for w in words if w not in stopwords]

        # Count frequencies
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1

        # Return top N
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]

    def create_pack_from_sessions(
        self,
        topic: str,
        since_days: int = 14,
        pack_type: str = "domain"
    ) -> Optional[str]:
        """Create pack by analyzing recent sessions on a topic"""

        cutoff = datetime.now() - timedelta(days=since_days)

        # Find relevant sessions
        relevant_sessions = []
        for session_dir in SESSIONS_DIR.glob("*"):
            if not session_dir.is_dir():
                continue

            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file) as f:
                    session_data = json.load(f)

                # Check if session matches topic
                session_topic = session_data.get('topic', '').lower()
                if topic.lower() in session_topic:
                    session_date = datetime.fromisoformat(
                        session_data['started'].replace('Z', '+00:00')
                    )

                    if session_date > cutoff:
                        relevant_sessions.append({
                            'id': session_data['session_id'],
                            'topic': session_data['topic'],
                            'date': session_date,
                            'path': session_dir
                        })
            except Exception as e:
                continue

        if not relevant_sessions:
            print(f"No sessions found for topic: {topic}")
            return None

        print(f"Found {len(relevant_sessions)} sessions for '{topic}'")

        # Aggregate content from sessions
        learnings = []
        papers = []
        implementations = []
        keywords_all = []

        for session in relevant_sessions:
            # Try to read session log
            log_file = session['path'] / "session_log.md"
            if log_file.exists():
                with open(log_file) as f:
                    log_content = f.read()

                # Extract papers
                session_papers = self._extract_papers_from_content(log_content)
                papers.extend(session_papers)

                # Extract learnings (look for bullet points)
                learning_lines = re.findall(r'^[-*]\s+(.+)$', log_content, re.MULTILINE)
                learnings.extend(learning_lines[:5])  # Top 5 per session

                # Extract keywords
                keywords = self._extract_keywords(log_content)
                keywords_all.extend(keywords)

        # Deduplicate papers
        unique_papers = {}
        for paper in papers:
            unique_papers[paper['arxiv_id']] = paper

        # Deduplicate and rank keywords
        keyword_freq = {}
        for kw in keywords_all:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        # Create pack content
        pack_content = {
            "papers": list(unique_papers.values()),
            "learnings": learnings[:10],  # Top 10 learnings
            "implementations": implementations,
            "keywords": [kw for kw, _ in top_keywords]
        }

        # Generate pack ID
        pack_id = topic.lower().replace(' ', '-').replace('_', '-')

        # Create pack
        pack = self._create_pack(
            pack_id=pack_id,
            pack_type=pack_type,
            content=pack_content,
            source_sessions=[s['id'] for s in relevant_sessions]
        )

        # Save pack
        pack_file = self.pack_dir / pack_type / f"{pack_id}.pack.json"
        with open(pack_file, 'w') as f:
            json.dump(pack, f, indent=2)

        # Update registry
        self.registry['packs'][pack_id] = {
            'type': pack_type,
            'file': str(pack_file.relative_to(self.pack_dir)),
            'version': pack['version'],
            'size_tokens': pack['size_tokens'],
            'created': pack['created']
        }
        self.registry['metadata']['total_packs'] += 1
        self.registry['metadata']['total_size_tokens'] += pack['size_tokens']
        self._save_registry()

        print(f"✓ Created pack: {pack_id} ({pack['size_tokens']} tokens)")
        return pack_id

    def create_pack_from_learnings(
        self,
        cluster_by: str = "topic",
        learnings_file: Optional[Path] = None
    ) -> List[str]:
        """Create packs from learnings.md by clustering"""

        if learnings_file is None:
            learnings_file = AGENT_CORE / "learnings.md"

        if not learnings_file.exists():
            print(f"Learnings file not found: {learnings_file}")
            return []

        with open(learnings_file) as f:
            content = f.read()

        # Parse learnings by date sections
        sections = re.split(r'^## (\d{4}-\d{2}-\d{2})', content, flags=re.MULTILINE)

        # Group by topic/keyword
        topic_content = {}

        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                date = sections[i]
                section_content = sections[i + 1]

                # Extract topic keywords
                keywords = self._extract_keywords(section_content, top_n=3)

                for keyword in keywords:
                    if keyword not in topic_content:
                        topic_content[keyword] = []

                    topic_content[keyword].append({
                        'date': date,
                        'content': section_content[:500]  # First 500 chars
                    })

        # Create pack for each significant topic
        created_packs = []
        for topic, entries in topic_content.items():
            if len(entries) >= 3:  # At least 3 entries to create pack
                pack_id = self.create_pack_from_topic_cluster(topic, entries)
                if pack_id:
                    created_packs.append(pack_id)

        return created_packs

    def create_pack_from_topic_cluster(
        self,
        topic: str,
        entries: List[Dict]
    ) -> Optional[str]:
        """Create pack from clustered topic entries"""

        # Aggregate content
        all_content = "\n".join([e['content'] for e in entries])

        # Extract papers
        papers = self._extract_papers_from_content(all_content)

        # Extract learnings
        learnings = []
        for entry in entries:
            lines = re.findall(r'^[-*]\s+(.+)$', entry['content'], re.MULTILINE)
            learnings.extend(lines)

        # Create pack content
        pack_content = {
            "papers": papers[:10],
            "learnings": learnings[:10],
            "implementations": [],
            "keywords": self._extract_keywords(all_content)
        }

        pack_id = f"{topic}-cluster"

        # Create and save pack
        pack = self._create_pack(
            pack_id=pack_id,
            pack_type="domain",
            content=pack_content,
            source_sessions=[]
        )

        pack_file = self.pack_dir / "domain" / f"{pack_id}.pack.json"
        with open(pack_file, 'w') as f:
            json.dump(pack, f, indent=2)

        # Update registry
        self.registry['packs'][pack_id] = {
            'type': 'domain',
            'file': str(pack_file.relative_to(self.pack_dir)),
            'version': pack['version'],
            'size_tokens': pack['size_tokens'],
            'created': pack['created']
        }
        self.registry['metadata']['total_packs'] += 1
        self.registry['metadata']['total_size_tokens'] += pack['size_tokens']
        self._save_registry()

        print(f"✓ Created cluster pack: {pack_id} ({pack['size_tokens']} tokens)")
        return pack_id

    def _create_pack(
        self,
        pack_id: str,
        pack_type: str,
        content: Dict,
        source_sessions: List[str]
    ) -> Dict:
        """Create pack structure"""

        # Serialize content for size calculation
        content_str = json.dumps(content, indent=2)
        size_bytes = len(content_str.encode('utf-8'))
        size_tokens = self._estimate_tokens(content_str)

        pack = {
            "pack_id": pack_id,
            "type": pack_type,
            "version": "1.0.0",
            "created": datetime.utcnow().isoformat() + "Z",
            "updated": datetime.utcnow().isoformat() + "Z",
            "size_bytes": size_bytes,
            "size_tokens": size_tokens,
            "content": content,
            "dq_metadata": {
                "base_validity": 0.85,  # Default, will be refined
                "base_specificity": 0.80,
                "base_correctness": 0.90,
                "base_score": 0.85
            },
            "usage_stats": {
                "times_selected": 0,
                "sessions": [],
                "avg_session_relevance": 0.0,
                "combined_with": []
            },
            "source_sessions": source_sessions
        }

        return pack

    def create_manual_pack(
        self,
        pack_id: str,
        pack_type: str,
        papers: List[str] = None,
        learnings: List[str] = None,
        keywords: List[str] = None
    ) -> str:
        """Manually create a pack"""

        content = {
            "papers": [{"arxiv_id": p, "relevance": 5} for p in (papers or [])],
            "learnings": learnings or [],
            "implementations": [],
            "keywords": keywords or []
        }

        pack = self._create_pack(
            pack_id=pack_id,
            pack_type=pack_type,
            content=content,
            source_sessions=[]
        )

        # Save pack
        pack_file = self.pack_dir / pack_type / f"{pack_id}.pack.json"
        with open(pack_file, 'w') as f:
            json.dump(pack, f, indent=2)

        # Update registry
        self.registry['packs'][pack_id] = {
            'type': pack_type,
            'file': str(pack_file.relative_to(self.pack_dir)),
            'version': pack['version'],
            'size_tokens': pack['size_tokens'],
            'created': pack['created']
        }
        self.registry['metadata']['total_packs'] += 1
        self.registry['metadata']['total_size_tokens'] += pack['size_tokens']
        self._save_registry()

        print(f"✓ Created manual pack: {pack_id} ({pack['size_tokens']} tokens)")
        return pack_id

    def list_packs(self):
        """List all packs"""
        print(f"\nTotal packs: {self.registry['metadata']['total_packs']}")
        print(f"Total tokens: {self.registry['metadata']['total_size_tokens']:,}")
        print("\nPacks:")

        for pack_id, pack_info in self.registry['packs'].items():
            print(f"  {pack_id:40s} | {pack_info['type']:10s} | {pack_info['size_tokens']:6d} tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Build context packs from sessions and learnings"
    )

    parser.add_argument(
        '--source',
        choices=['sessions', 'learnings', 'manual'],
        help="Source to build packs from"
    )

    parser.add_argument(
        '--topic',
        help="Topic to create pack for (when source=sessions)"
    )

    parser.add_argument(
        '--since',
        type=int,
        default=14,
        help="Days to look back (default: 14)"
    )

    parser.add_argument(
        '--cluster-by',
        default='topic',
        choices=['topic', 'date'],
        help="How to cluster learnings"
    )

    parser.add_argument(
        '--create',
        action='store_true',
        help="Create manual pack"
    )

    parser.add_argument(
        '--type',
        choices=['domain', 'project', 'pattern', 'paper'],
        default='domain',
        help="Pack type"
    )

    parser.add_argument(
        '--name',
        help="Pack name/ID"
    )

    parser.add_argument(
        '--papers',
        help="Comma-separated arXiv IDs"
    )

    parser.add_argument(
        '--keywords',
        help="Comma-separated keywords"
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help="List all packs"
    )

    args = parser.parse_args()

    builder = PackBuilder()

    if args.list:
        builder.list_packs()
        return

    if args.create:
        if not args.name:
            print("Error: --name required for manual pack creation")
            return

        papers = args.papers.split(',') if args.papers else []
        keywords = args.keywords.split(',') if args.keywords else []

        builder.create_manual_pack(
            pack_id=args.name,
            pack_type=args.type,
            papers=papers,
            keywords=keywords
        )

    elif args.source == 'sessions':
        if not args.topic:
            print("Error: --topic required when source=sessions")
            return

        builder.create_pack_from_sessions(
            topic=args.topic,
            since_days=args.since,
            pack_type=args.type
        )

    elif args.source == 'learnings':
        builder.create_pack_from_learnings(cluster_by=args.cluster_by)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
