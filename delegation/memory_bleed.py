"""
Memory Bleed — Cross-Conversation Knowledge Transfer via Supermemory

Implements memory bleed from arXiv:2602.11865 Section 6.

Memory bleed enables agents to learn from past delegations across conversations:
- Success patterns (what worked before)
- Failure patterns (what didn't work)
- Agent performance trends
- Task-specific learnings
- Context carryover

Knowledge transfer mechanisms:
- Semantic similarity matching via SBERT embeddings (reuses coherence_engine pipeline)
- Error pattern extraction from supermemory.db (read-only)
- Domain expertise scoring based on memory coverage
- Write-back: delegation outcomes written to supermemory via SM-2 spaced repetition

Supermemory Integration:
- Reads from ~/.claude/memory/supermemory.db (read-only for safety)
- Uses SBERT embeddings from mcp_raw.embeddings (same as coherence engine)
- Writes delegation outcomes back to supermemory for spaced repetition learning
- Graceful degradation: returns empty results if DB unavailable (never blocks delegation)

Usage:
    from delegation.memory_bleed import get_relevant_context, get_error_patterns, get_domain_expertise

    # Retrieve relevant context
    context = get_relevant_context("Implement user auth", limit=5)

    # Find past error patterns
    errors = get_error_patterns("authentication")

    # Score domain expertise
    expertise = get_domain_expertise("authentication")
"""

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import SBERT embeddings from mcp_raw (same pipeline as coherence engine)
try:
    from mcp_raw.embeddings import embed_single, cosine_similarity
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

log = logging.getLogger("delegation.memory_bleed")

# Default supermemory path
DEFAULT_SUPERMEMORY_PATH = os.path.expanduser("~/.claude/memory/supermemory.db")


@dataclass
class MemoryContext:
    """Relevant context item from supermemory."""
    content: str
    source: str
    quality: float
    date: Optional[str] = None
    project: Optional[str] = None
    similarity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorPattern:
    """Error pattern from past failures."""
    category: str
    pattern: str
    count: int
    solution: Optional[str] = None
    last_seen: Optional[str] = None


def _get_db_path() -> Optional[str]:
    """Get supermemory.db path, return None if unavailable."""
    path = Path(DEFAULT_SUPERMEMORY_PATH)
    if path.exists():
        return str(path)
    log.warning(f"Supermemory DB not found at {path}, memory bleed disabled")
    return None


def _connect_readonly(db_path: str) -> Optional[sqlite3.Connection]:
    """Connect to supermemory.db in read-only mode for safety."""
    try:
        # SQLite URI syntax for read-only mode
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=1.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        log.error(f"Failed to connect to supermemory.db: {e}")
        return None


def get_relevant_context(task: str, limit: int = 5) -> List[MemoryContext]:
    """
    Retrieve relevant context from supermemory.db via embedding similarity.

    Uses SBERT embeddings (same as coherence engine) to find similar past learnings,
    memory items, and session data. Scores by cosine similarity and filters by quality.

    Args:
        task: Task description to search for
        limit: Maximum number of context items to return (default: 5)

    Returns:
        List of MemoryContext objects sorted by similarity (descending)
        Returns empty list if DB unavailable or embeddings disabled

    Performance:
        Target: <500ms using precomputed embeddings
        Graceful degradation: returns [] if DB unavailable (never blocks delegation)
    """
    db_path = _get_db_path()
    if not db_path:
        return []

    if not HAS_EMBEDDINGS:
        log.warning("SBERT embeddings unavailable, memory bleed disabled")
        return []

    # Embed the task query
    try:
        task_embedding = embed_single(task, prefix="search_query")
    except Exception as e:
        log.error(f"Failed to embed task query: {e}")
        return []

    # Query supermemory for relevant items
    conn = _connect_readonly(db_path)
    if not conn:
        return []

    try:
        # Query memory_items with quality >= 0.5 (filter out low-quality noise)
        rows = conn.execute("""
            SELECT id, content, source, quality, date, project, metadata
            FROM memory_items
            WHERE quality >= 0.5
            ORDER BY date DESC
            LIMIT 200
        """).fetchall()

        # Score by similarity
        results = []
        for row in rows:
            content = row["content"]
            if not content or len(content) < 10:
                continue

            # Embed and score similarity
            try:
                content_embedding = embed_single(content, prefix="search_document")
                similarity = cosine_similarity(task_embedding, content_embedding)
            except Exception as e:
                log.debug(f"Embedding error for item {row['id']}: {e}")
                continue

            # Filter by similarity threshold (0.6 = moderate relevance)
            if similarity < 0.6:
                continue

            # Parse metadata JSON
            metadata = {}
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                except Exception:
                    pass

            results.append(MemoryContext(
                content=content,
                source=row["source"],
                quality=row["quality"] or 0.0,
                date=row["date"],
                project=row["project"],
                similarity=similarity,
                metadata=metadata,
            ))

        # Sort by similarity (descending) and return top N
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    except Exception as e:
        log.error(f"Memory context query failed: {e}")
        return []
    finally:
        conn.close()


def get_error_patterns(task_type: str) -> List[ErrorPattern]:
    """
    Find past error patterns for similar task types.

    Searches supermemory.error_patterns table for matching categories/patterns.
    Used to avoid repeating past mistakes in delegation.

    Args:
        task_type: Task type/category to search for (e.g., "authentication", "database")

    Returns:
        List of ErrorPattern objects sorted by count (descending)
        Returns empty list if DB unavailable

    Performance:
        Target: <100ms (simple SQLite query with LIKE matching)
    """
    db_path = _get_db_path()
    if not db_path:
        return []

    conn = _connect_readonly(db_path)
    if not conn:
        return []

    try:
        # Search by category and pattern (case-insensitive LIKE)
        search_term = f"%{task_type.lower()}%"
        rows = conn.execute("""
            SELECT category, pattern, count, solution, last_seen
            FROM error_patterns
            WHERE LOWER(category) LIKE ? OR LOWER(pattern) LIKE ?
            ORDER BY count DESC
            LIMIT 10
        """, (search_term, search_term)).fetchall()

        results = []
        for row in rows:
            results.append(ErrorPattern(
                category=row["category"],
                pattern=row["pattern"],
                count=row["count"],
                solution=row["solution"],
                last_seen=row["last_seen"],
            ))

        return results

    except Exception as e:
        log.error(f"Error pattern query failed: {e}")
        return []
    finally:
        conn.close()


def get_domain_expertise(domain: str) -> float:
    """
    Score domain expertise based on memory coverage for this domain.

    Measures how much supermemory knows about a domain by counting relevant
    memory items, learnings, and error patterns. Higher scores = more context available.

    Args:
        domain: Domain to score (e.g., "authentication", "database", "frontend")

    Returns:
        Expertise score [0.0, 1.0]:
        - 0.0 = no memory coverage (first time seeing this domain)
        - 0.5 = moderate coverage (10-50 relevant items)
        - 1.0 = high coverage (100+ relevant items with quality learnings)

    Performance:
        Target: <100ms (COUNT queries with indexes)
    """
    db_path = _get_db_path()
    if not db_path:
        return 0.0

    conn = _connect_readonly(db_path)
    if not conn:
        return 0.0

    try:
        search_term = f"%{domain.lower()}%"

        # Count memory items
        memory_count = conn.execute("""
            SELECT COUNT(*) as cnt
            FROM memory_items
            WHERE LOWER(content) LIKE ? AND quality >= 0.5
        """, (search_term,)).fetchone()["cnt"]

        # Count learnings
        learning_count = conn.execute("""
            SELECT COUNT(*) as cnt
            FROM learnings
            WHERE LOWER(content) LIKE ? AND quality >= 0.5
        """, (search_term,)).fetchone()["cnt"]

        # Count error patterns (indicates experience with failures)
        error_count = conn.execute("""
            SELECT COUNT(*) as cnt
            FROM error_patterns
            WHERE LOWER(category) LIKE ? OR LOWER(pattern) LIKE ?
        """, (search_term, search_term)).fetchone()["cnt"]

        # Score formula (logarithmic scale to handle high counts gracefully)
        total = memory_count + learning_count * 2 + error_count * 3
        if total == 0:
            return 0.0

        # Logarithmic scaling: 1 item = 0.1, 10 items = 0.5, 100 items = 1.0
        import math
        score = min(1.0, math.log10(total + 1) / 2.0)
        return max(0.0, score)

    except Exception as e:
        log.error(f"Domain expertise query failed: {e}")
        return 0.0
    finally:
        conn.close()


def write_delegation_outcome(
    task: str,
    outcome: str,
    quality: float,
    category: str = "delegation",
    project: Optional[str] = None,
) -> bool:
    """
    Write delegation outcome to supermemory for spaced repetition learning.

    Creates a review item using SM-2 algorithm (SuperMemo 2) for spaced repetition.
    This enables delegation system to learn from outcomes over time.

    Args:
        task: Task description
        outcome: Delegation outcome (success/failure details)
        quality: Quality score [0.0, 1.0]
        category: Category for organization (default: "delegation")
        project: Optional project name

    Returns:
        True if written successfully, False otherwise

    SM-2 Parameters:
        - ease_factor: 2.5 (default for new items)
        - interval_days: 1 (first review tomorrow)
        - repetitions: 0 (first time)
        - next_review: tomorrow's date
    """
    db_path = _get_db_path()
    if not db_path:
        return False

    # Open in read-write mode for writing
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        log.error(f"Failed to connect to supermemory.db for writing: {e}")
        return False

    try:
        import uuid
        from datetime import datetime, timedelta

        review_id = uuid.uuid4().hex[:16]
        content = f"[Delegation] {task}\nOutcome: {outcome}"
        next_review = (datetime.now() + timedelta(days=1)).date().isoformat()
        last_review = datetime.now().date().isoformat()

        conn.execute("""
            INSERT INTO reviews (
                id, content, category, ease_factor, interval_days,
                repetitions, next_review, last_review, source_id
            ) VALUES (?, ?, ?, 2.5, 1, 0, ?, ?, ?)
        """, (review_id, content, category, next_review, last_review, project or ""))

        conn.commit()
        log.info(f"Wrote delegation outcome to supermemory: {review_id}")
        return True

    except Exception as e:
        log.error(f"Failed to write delegation outcome: {e}")
        return False
    finally:
        conn.close()


# Convenience function for batch context injection
def inject_context(subtasks: List[Any], context_limit: int = 3) -> None:
    """
    Inject relevant context into SubTask objects before delegation.

    Modifies subtasks in-place by adding memory_context to metadata field.
    This is called by the coordinator before routing subtasks to agents.

    Args:
        subtasks: List of SubTask objects to inject context into
        context_limit: Max context items per subtask (default: 3)

    Side Effects:
        Modifies subtask.metadata["memory_context"] for each subtask
    """
    for subtask in subtasks:
        # Get relevant context for this subtask
        context = get_relevant_context(subtask.description, limit=context_limit)

        # Inject into metadata
        if not hasattr(subtask, "metadata"):
            subtask.metadata = {}

        subtask.metadata["memory_context"] = [
            {
                "content": c.content[:200],  # Truncate for brevity
                "source": c.source,
                "similarity": round(c.similarity, 3),
                "quality": round(c.quality, 2),
            }
            for c in context
        ]

        # Also inject error patterns if available
        if hasattr(subtask, "profile") and subtask.profile:
            # Infer task type from profile complexity/criticality
            if subtask.profile.criticality >= 0.7:
                task_type = "critical"
            elif subtask.profile.complexity >= 0.7:
                task_type = "complex"
            else:
                task_type = "general"

            errors = get_error_patterns(task_type)
            if errors:
                subtask.metadata["error_patterns"] = [
                    {
                        "category": e.category,
                        "pattern": e.pattern[:100],
                        "solution": e.solution[:100] if e.solution else None,
                    }
                    for e in errors[:3]  # Top 3 patterns only
                ]


# Backward-compatible class interface (deprecated, use functions instead)
class MemoryBleedEngine:
    """
    Legacy class interface for memory bleed (deprecated).

    Use functions instead: get_relevant_context(), get_error_patterns(),
    get_domain_expertise(), write_delegation_outcome()
    """

    def __init__(self, db_path: str = ""):
        """Initialize with optional custom DB path."""
        self.db_path = db_path or DEFAULT_SUPERMEMORY_PATH
        log.warning("MemoryBleedEngine class is deprecated, use functions instead")

    def retrieve_context(
        self,
        task_description: str,
        max_items: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context (legacy method)."""
        contexts = get_relevant_context(task_description, limit=max_items)
        # Filter by threshold and convert to dict
        return [
            {
                "content": c.content,
                "source": c.source,
                "quality": c.quality,
                "similarity": c.similarity,
            }
            for c in contexts if c.similarity >= similarity_threshold
        ]
