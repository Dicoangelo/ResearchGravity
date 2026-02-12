"""
Coherence Engine — Session-Level Coherence

Instead of comparing individual events, compare entire conversation THEMES
across platforms. Detect "conversation coherence" — same intellectual thread
explored across sessions on different platforms.

Example: "On Claude you spent 2 hours debugging state management.
On ChatGPT you explored agent architecture. The underlying pattern:
you're solving distributed state."

Phase 1 (US-010): Generate session summary embeddings
Phase 2 (US-011): Cross-platform session comparison
Phase 3 (US-012): Integrate into daemon loop
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import numpy as np

from . import config as cfg
from mcp_raw.embeddings import embed_single, embed_texts

log = logging.getLogger("coherence.session_coherence")


@dataclass
class SessionCoherence:
    """A detected session-level coherence across platforms."""
    session_a_id: str
    session_b_id: str
    platform_a: str
    platform_b: str
    similarity: float
    summary_a: str
    summary_b: str
    shared_themes: List[str]


# ── Session Summary Embeddings (US-010) ──────────────────────────────────────


async def generate_session_embedding(
    pool: asyncpg.Pool,
    session_id: str,
) -> Optional[List[float]]:
    """
    Generate a summary embedding for a session by mean-pooling its event embeddings.

    Steps:
    1. Fetch all event embeddings for this session from embedding_cache
    2. Mean-pool them into a single 768d vector
    3. Generate a text summary from event topics/intents
    4. Store both on cognitive_sessions
    """
    async with pool.acquire() as conn:
        # Get session info
        session = await conn.fetchrow(
            "SELECT session_id, platform, event_count FROM cognitive_sessions WHERE session_id = $1",
            session_id,
        )
        if not session:
            return None

        # Get embeddings for all events in this session
        rows = await conn.fetch(
            f"""SELECT ec.{cfg.EMBED_COLUMN} AS embedding
                FROM embedding_cache ec
                JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                WHERE ce.session_id = $1
                  AND ec.{cfg.EMBED_COLUMN} IS NOT NULL""",
            session_id,
        )

        if not rows:
            log.debug(f"No embeddings for session {session_id}")
            return None

        # Mean pool
        embeddings = []
        for r in rows:
            emb = r["embedding"]
            if emb is not None:
                if isinstance(emb, (bytes, memoryview)):
                    # pgvector returns binary — parse it
                    arr = np.frombuffer(bytes(emb), dtype=np.float32)
                elif isinstance(emb, str):
                    arr = np.array(json.loads(emb), dtype=np.float32)
                elif isinstance(emb, list):
                    arr = np.array(emb, dtype=np.float32)
                else:
                    continue
                if len(arr) == cfg.EMBED_DIMENSIONS:
                    embeddings.append(arr)

        if not embeddings:
            return None

        mean_embedding = np.mean(embeddings, axis=0)
        # L2 normalize
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        # Generate text summary from topics
        topic_rows = await conn.fetch(
            """SELECT light_layer->>'topic' AS topic,
                      light_layer->>'intent' AS intent,
                      light_layer->>'summary' AS summary
               FROM cognitive_events
               WHERE session_id = $1
                 AND light_layer IS NOT NULL
               ORDER BY timestamp_ns ASC""",
            session_id,
        )

        topics = {}
        intents = {}
        summaries = []
        for r in topic_rows:
            t = r.get("topic")
            i = r.get("intent")
            s = r.get("summary")
            if t and t != "general":
                topics[t] = topics.get(t, 0) + 1
            if i:
                intents[i] = intents.get(i, 0) + 1
            if s and len(s) > 20:
                summaries.append(s[:200])

        # Build session summary text
        top_topics = sorted(topics.items(), key=lambda x: -x[1])[:5]
        top_intents = sorted(intents.items(), key=lambda x: -x[1])[:3]
        summary_text = (
            f"Session on {session['platform']} with {session['event_count']} events. "
            f"Topics: {', '.join(t for t, _ in top_topics)}. "
            f"Intents: {', '.join(i for i, _ in top_intents)}."
        )
        if summaries:
            summary_text += f" Key content: {summaries[0]}"

        # Store on cognitive_sessions
        emb_list = mean_embedding.tolist()
        await conn.execute(
            """UPDATE cognitive_sessions
               SET session_embedding = $2::vector,
                   session_summary = $3
               WHERE session_id = $1""",
            session_id,
            str(emb_list),
            summary_text,
        )

        log.debug(
            f"Session {session_id}: {len(embeddings)} event embeddings → "
            f"1 session embedding. Topics: {[t for t, _ in top_topics[:3]]}"
        )
        return emb_list


async def backfill_session_embeddings(
    pool: asyncpg.Pool,
    limit: int = 5000,
    skip_existing: bool = True,
) -> int:
    """Generate embeddings for all sessions that don't have them yet."""
    async with pool.acquire() as conn:
        where = "AND session_embedding IS NULL" if skip_existing else ""
        sessions = await conn.fetch(
            f"""SELECT session_id FROM cognitive_sessions
                WHERE event_count > 0 {where}
                ORDER BY started_ns DESC
                LIMIT $1""",
            limit,
        )

    processed = 0
    for i, row in enumerate(sessions):
        result = await generate_session_embedding(pool, row["session_id"])
        if result:
            processed += 1
        if (i + 1) % 100 == 0:
            log.info(f"Session embeddings: {i+1}/{len(sessions)} processed, {processed} generated")

    log.info(f"Session embedding backfill: {processed}/{len(sessions)} generated")
    return processed


# ── Session-Level Cross-Platform Comparison (US-011) ─────────────────────────


async def find_session_coherence(
    pool: asyncpg.Pool,
    since_hours: int = 168,
    min_similarity: float = 0.70,
    limit: int = 50,
) -> List[SessionCoherence]:
    """
    Find sessions on different platforms exploring the same intellectual thread.

    Uses cosine similarity on session_embedding vectors.
    Only compares across different platform families.
    """
    async with pool.acquire() as conn:
        # Find cross-platform similar sessions
        rows = await conn.fetch(
            f"""SELECT
                    a.session_id AS session_a,
                    b.session_id AS session_b,
                    a.platform AS platform_a,
                    b.platform AS platform_b,
                    a.session_summary AS summary_a,
                    b.session_summary AS summary_b,
                    1 - (a.session_embedding <=> b.session_embedding) AS similarity
                FROM cognitive_sessions a
                CROSS JOIN cognitive_sessions b
                WHERE a.session_id < b.session_id
                  AND a.platform != b.platform
                  AND a.session_embedding IS NOT NULL
                  AND b.session_embedding IS NOT NULL
                  AND a.started_ns > extract(epoch from NOW() - make_interval(hours => $1))::BIGINT * 1000000000
                  AND 1 - (a.session_embedding <=> b.session_embedding) >= $2
                ORDER BY similarity DESC
                LIMIT $3""",
            since_hours,
            min_similarity,
            limit,
        )

    results = []
    families = cfg.PLATFORM_FAMILIES
    for r in rows:
        # Skip same-family pairs
        family_a = families.get(r["platform_a"], r["platform_a"])
        family_b = families.get(r["platform_b"], r["platform_b"])
        if family_a == family_b:
            continue

        # Extract shared themes from summaries
        shared = _extract_shared_themes(r["summary_a"] or "", r["summary_b"] or "")

        results.append(SessionCoherence(
            session_a_id=r["session_a"],
            session_b_id=r["session_b"],
            platform_a=r["platform_a"],
            platform_b=r["platform_b"],
            similarity=float(r["similarity"]),
            summary_a=r["summary_a"] or "",
            summary_b=r["summary_b"] or "",
            shared_themes=shared,
        ))

    return results


def _extract_shared_themes(summary_a: str, summary_b: str) -> List[str]:
    """Extract shared topic words between two session summaries."""
    words_a = set(summary_a.lower().split())
    words_b = set(summary_b.lower().split())
    # Filter to meaningful words (>4 chars, not stopwords)
    stopwords = {"with", "from", "that", "this", "have", "been", "about", "their",
                 "would", "could", "should", "events", "session", "topics", "intents",
                 "content"}
    shared = words_a & words_b - stopwords
    return [w for w in shared if len(w) > 4][:10]
