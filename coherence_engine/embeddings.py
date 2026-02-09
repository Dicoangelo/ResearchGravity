"""
Coherence Engine — Embedding Pipeline

Wraps mcp_raw.embeddings for the coherence engine's needs.
Adds:
  - Async batch embedding with DB storage
  - Event-to-text conversion for coherence context
  - Cohere upgrade path (when API key set)
"""

import json
import time
from typing import Any, Dict, List, Optional

from . import config as cfg

# Import core embedding functions from mcp_raw (same repo)
from mcp_raw.embeddings import (
    build_embed_text,
    build_embed_text_contextual,
    embed_single,
    embed_texts,
    content_hash,
    cosine_similarity,
    _model_name,
    _dimensions,
    _embedding_column,
)

import logging

log = logging.getLogger("coherence.embeddings")


def event_to_text(event_row: Dict[str, Any]) -> str:
    """
    Convert a database row (dict) to embeddable text.

    Uses contextual embedding text (includes platform, cognitive mode, topic)
    for better cross-platform coherence detection.
    Falls back to base text if contextual produces nothing.
    """
    text = build_embed_text_contextual(event_row)
    if not text:
        text = build_embed_text(event_row)
    return text


async def embed_event_row(pool, event_row: Dict[str, Any]) -> Optional[List[float]]:
    """
    Embed a single event row and store in embedding_cache.

    Returns the embedding vector, or None if text too short.
    """
    text = event_to_text(event_row)
    if not text or len(text) < 10:
        return None

    embedding = embed_single(text)
    ch = content_hash(text)

    if pool:
        await _store_embedding(
            pool, ch, text[:200], embedding,
            source_event_id=event_row.get("event_id"),
        )

    return embedding


async def batch_embed_events(
    pool,
    limit: int = 0,
    skip_existing: bool = True,
) -> int:
    """
    Batch-embed events that don't have embeddings yet.
    Returns count of newly embedded events.
    """
    if not pool:
        log.error("No database pool for batch embedding")
        return 0

    async with pool.acquire() as conn:
        if skip_existing:
            rows = await conn.fetch(
                """SELECT event_id, data_layer, light_layer
                   FROM cognitive_events ce
                   WHERE NOT EXISTS (
                       SELECT 1 FROM embedding_cache ec
                       WHERE ec.source_event_id = ce.event_id
                   )
                   ORDER BY timestamp_ns DESC
                   LIMIT $1""",
                limit if limit > 0 else 100000,
            )
        else:
            rows = await conn.fetch(
                """SELECT event_id, data_layer, light_layer
                   FROM cognitive_events
                   ORDER BY timestamp_ns DESC
                   LIMIT $1""",
                limit if limit > 0 else 100000,
            )

    if not rows:
        log.info("No events to embed")
        return 0

    log.info(f"Embedding {len(rows)} events...")
    t0 = time.time()

    # Build texts
    texts = []
    event_ids = []
    for row in rows:
        text = event_to_text(dict(row))
        if text and len(text) >= 10:
            texts.append(text)
            event_ids.append(row["event_id"])

    if not texts:
        return 0

    # Batch embed
    batch_size = cfg.EMBED_BATCH_SIZE
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = embed_texts(batch, batch_size=min(batch_size, 64))
        all_embeddings.extend(embs)
        done = min(i + batch_size, len(texts))
        if done % 1000 < batch_size:
            log.info(f"  Embedded {done}/{len(texts)}")

    # Store in database
    stored = 0
    async with pool.acquire() as conn:
        for eid, text, emb in zip(event_ids, texts, all_embeddings):
            try:
                ch = content_hash(text)
                vec_str = "[" + ",".join(str(x) for x in emb) + "]"
                await conn.execute(
                    f"""INSERT INTO embedding_cache
                       (content_hash, content_preview, {_embedding_column}, model, dimensions, source_event_id)
                       VALUES ($1, $2, $3::vector, $4, $5, $6)
                       ON CONFLICT (content_hash) DO UPDATE SET
                           {_embedding_column} = $3::vector,
                           model = $4,
                           dimensions = $5""",
                    ch, text[:200], vec_str, _model_name, _dimensions, eid,
                )
                stored += 1
            except Exception as e:
                log.error(f"Store error for {eid}: {e}")

    elapsed = time.time() - t0
    rate = len(texts) / elapsed if elapsed > 0 else 0
    log.info(f"Embedded {stored} events in {elapsed:.1f}s ({rate:.0f}/sec)")
    return stored


async def _store_embedding(
    pool,
    ch: str,
    preview: str,
    embedding: List[float],
    source_event_id: Optional[str] = None,
):
    """Store a single embedding in the cache."""
    try:
        async with pool.acquire() as conn:
            vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
            await conn.execute(
                f"""INSERT INTO embedding_cache
                   (content_hash, content_preview, {_embedding_column}, model, dimensions, source_event_id)
                   VALUES ($1, $2, $3::vector, $4, $5, $6)
                   ON CONFLICT (content_hash) DO UPDATE SET
                       {_embedding_column} = $3::vector,
                       model = $4,
                       dimensions = $5""",
                ch, preview, vec_str, _model_name, _dimensions, source_event_id,
            )
    except Exception as e:
        log.error(f"Store embedding error: {e}")
