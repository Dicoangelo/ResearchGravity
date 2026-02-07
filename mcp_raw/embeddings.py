"""
Embedding Pipeline — Semantic vectors for cognitive events

Embeds event content using SBERT (local) with optional Cohere upgrade.
Stores vectors in embedding_cache table for similarity search.

Supports:
  - Real-time: embed single events as they're captured
  - Batch: embed all existing events in the database
  - Search: find similar events by cosine similarity
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .logger import get_logger

log = get_logger("embeddings")

# Lazy-load model to avoid import-time overhead
_model = None
_model_name = "all-MiniLM-L6-v2"
_dimensions = 384


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
        log.info(f"Loaded SBERT model: {_model_name} ({_dimensions}d)")
    return _model


def build_embed_text(event_or_dict) -> str:
    """
    Build the text to embed from a cognitive event.

    Format: "{intent}: {topic} | {summary} | {concepts}"
    Works with CaptureEvent objects or dicts from the database.
    """
    if hasattr(event_or_dict, "light_layer"):
        # CaptureEvent object
        light = event_or_dict.light_layer or {}
        data = event_or_dict.data_layer or {}
    elif isinstance(event_or_dict, dict):
        # Database row dict
        light_raw = event_or_dict.get("light_layer", "{}")
        data_raw = event_or_dict.get("data_layer", "{}")
        light = json.loads(light_raw) if isinstance(light_raw, str) else (light_raw or {})
        data = json.loads(data_raw) if isinstance(data_raw, str) else (data_raw or {})
    else:
        return ""

    intent = light.get("intent", "explore")
    topic = light.get("topic", "general")
    summary = light.get("summary", "")
    concepts = light.get("concepts", [])
    content = data.get("content", "")

    # Build text: intent + topic + summary + concepts
    parts = [f"{intent}: {topic}"]
    if summary:
        parts.append(summary[:300])
    elif content:
        parts.append(content[:300])
    if concepts:
        parts.append(" ".join(concepts))

    return " | ".join(parts)


def content_hash(text: str) -> str:
    """SHA-256 hash for dedup in embedding_cache."""
    return hashlib.sha256(text.encode()).hexdigest()


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    model = _get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_single(text: str) -> List[float]:
    """Embed a single text. Returns float vector."""
    model = _get_model()
    return model.encode(text).tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class EmbeddingPipeline:
    """
    Full embedding pipeline with database integration.

    Usage:
        pipeline = EmbeddingPipeline(db_pool)
        await pipeline.embed_event(event)           # Real-time
        await pipeline.batch_embed(limit=1000)       # Batch
        results = await pipeline.find_similar(text)  # Search
    """

    def __init__(self, pool=None):
        self._pool = pool

    async def embed_event(self, event) -> Optional[List[float]]:
        """Embed a single capture event and store in cache."""
        text = build_embed_text(event)
        if not text or len(text) < 10:
            return None

        embedding = embed_single(text)
        ch = content_hash(text)

        if self._pool:
            await self._store_embedding(
                ch, text[:200], embedding,
                source_event_id=getattr(event, "event_id", None),
            )

        return embedding

    async def batch_embed(
        self,
        limit: int = 0,
        skip_existing: bool = True,
    ) -> int:
        """
        Batch-embed events from the database that don't have embeddings yet.
        Returns count of newly embedded events.
        """
        if not self._pool:
            log.error("No database pool for batch embedding")
            return 0

        # Find events without embeddings
        async with self._pool.acquire() as conn:
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
            text = build_embed_text(dict(row))
            if text and len(text) >= 10:
                texts.append(text)
                event_ids.append(row["event_id"])

        if not texts:
            return 0

        # Batch embed
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = embed_texts(batch, batch_size=batch_size)
            all_embeddings.extend(embs)
            if (i + batch_size) % 1000 == 0:
                log.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

        # Store in database
        stored = 0
        skipped_dupes = 0
        async with self._pool.acquire() as conn:
            for eid, text, emb in zip(event_ids, texts, all_embeddings):
                try:
                    ch = content_hash(text)
                    vec_str = '[' + ','.join(str(x) for x in emb) + ']'
                    result = await conn.execute(
                        """INSERT INTO embedding_cache
                           (content_hash, content_preview, embedding, model, dimensions, source_event_id)
                           VALUES ($1, $2, $3::vector, $4, $5, $6)
                           ON CONFLICT (content_hash) DO NOTHING""",
                        ch, text[:200], vec_str, _model_name, _dimensions, eid,
                    )
                    # Check if row was actually inserted (not a dupe)
                    if result and result.endswith("1"):
                        stored += 1
                    else:
                        skipped_dupes += 1
                except Exception as e:
                    log.error(f"Store error for {eid}: {e}")

        elapsed = time.time() - t0
        rate = len(texts) / elapsed if elapsed > 0 else 0
        log.info(f"Embedded {stored} events in {elapsed:.1f}s ({rate:.0f}/sec)"
                 + (f", {skipped_dupes} content dupes skipped" if skipped_dupes else ""))
        return stored

    async def find_similar(
        self,
        text: str,
        threshold: float = 0.75,
        limit: int = 20,
        exclude_platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find events similar to the given text.

        Uses brute-force cosine similarity (fast enough for <100K events).
        For pgvector HNSW, use find_similar_pgvector().
        """
        query_emb = embed_single(text)

        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            # Get all embeddings with event info
            if exclude_platform:
                rows = await conn.fetch(
                    """SELECT ec.content_hash, ec.content_preview, ec.embedding,
                              ec.source_event_id, ce.platform, ce.session_id,
                              ce.light_layer, ce.coherence_sig
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ce.platform != $1""",
                    exclude_platform,
                )
            else:
                rows = await conn.fetch(
                    """SELECT ec.content_hash, ec.content_preview, ec.embedding,
                              ec.source_event_id, ce.platform, ce.session_id,
                              ce.light_layer, ce.coherence_sig
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id""",
                )

        # Compute similarities
        results = []
        for row in rows:
            try:
                emb_str = row["embedding"]
                emb = json.loads(emb_str) if isinstance(emb_str, str) else list(emb_str)
                sim = cosine_similarity(query_emb, emb)
                if sim >= threshold:
                    results.append({
                        "event_id": row["source_event_id"],
                        "platform": row["platform"],
                        "session_id": row["session_id"],
                        "similarity": round(sim, 4),
                        "preview": row["content_preview"],
                        "coherence_sig": row["coherence_sig"],
                    })
            except Exception:
                continue

        # Sort by similarity
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:limit]

    async def find_similar_pgvector(
        self,
        text: str,
        limit: int = 20,
        exclude_platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find similar events using pgvector's HNSW index (cosine distance).
        Much faster than brute-force for large datasets.
        """
        if not self._pool:
            return []

        query_emb = embed_single(text)
        vec_str = '[' + ','.join(str(x) for x in query_emb) + ']'

        async with self._pool.acquire() as conn:
            if exclude_platform:
                rows = await conn.fetch(
                    """SELECT ec.content_preview, ec.source_event_id,
                              ce.platform, ce.session_id, ce.coherence_sig,
                              ce.cognitive_mode,
                              1 - (ec.embedding <=> $1::vector) AS similarity
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ce.platform != $2
                       ORDER BY ec.embedding <=> $1::vector
                       LIMIT $3""",
                    vec_str, exclude_platform, limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT ec.content_preview, ec.source_event_id,
                              ce.platform, ce.session_id, ce.coherence_sig,
                              ce.cognitive_mode,
                              1 - (ec.embedding <=> $1::vector) AS similarity
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       ORDER BY ec.embedding <=> $1::vector
                       LIMIT $2""",
                    vec_str, limit,
                )

        return [
            {
                "event_id": row["source_event_id"],
                "platform": row["platform"],
                "session_id": row["session_id"],
                "similarity": round(float(row["similarity"]), 4),
                "preview": row["content_preview"],
                "coherence_sig": row["coherence_sig"],
                "cognitive_mode": row["cognitive_mode"],
            }
            for row in rows
        ]

    async def find_cross_platform_matches(
        self,
        platform: str,
        threshold: float = 0.70,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find the best semantic matches between one platform and all others.
        Returns pairs of events with high cross-platform similarity.
        """
        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """WITH source_events AS (
                       SELECT ec.embedding, ec.content_preview AS src_preview,
                              ec.source_event_id AS src_id, ce.cognitive_mode AS src_mode
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ce.platform = $1 AND ce.cognitive_mode IN ('deep_work', 'exploration')
                       LIMIT 1000
                   )
                   SELECT src.src_preview, src.src_id, src.src_mode,
                          tgt.content_preview AS tgt_preview,
                          tgt.source_event_id AS tgt_id,
                          tgt_ce.platform AS tgt_platform,
                          tgt_ce.cognitive_mode AS tgt_mode,
                          1 - (src.embedding <=> tgt.embedding) AS similarity
                   FROM source_events src
                   CROSS JOIN LATERAL (
                       SELECT ec2.content_preview, ec2.source_event_id, ec2.embedding
                       FROM embedding_cache ec2
                       JOIN cognitive_events ce2 ON ec2.source_event_id = ce2.event_id
                       WHERE ce2.platform != $1
                       ORDER BY ec2.embedding <=> src.embedding
                       LIMIT 1
                   ) tgt
                   JOIN cognitive_events tgt_ce ON tgt.source_event_id = tgt_ce.event_id
                   WHERE 1 - (src.embedding <=> tgt.embedding) >= $2
                   ORDER BY similarity DESC
                   LIMIT $3""",
                platform, threshold, limit,
            )

        return [
            {
                "source_preview": row["src_preview"],
                "source_id": row["src_id"],
                "source_mode": row["src_mode"],
                "target_preview": row["tgt_preview"],
                "target_id": row["tgt_id"],
                "target_platform": row["tgt_platform"],
                "target_mode": row["tgt_mode"],
                "similarity": round(float(row["similarity"]), 4),
            }
            for row in rows
        ]

    async def _store_embedding(
        self,
        ch: str,
        preview: str,
        embedding: List[float],
        source_event_id: Optional[str] = None,
    ):
        if not self._pool:
            return
        try:
            async with self._pool.acquire() as conn:
                vec_str = '[' + ','.join(str(x) for x in embedding) + ']'
                await conn.execute(
                    """INSERT INTO embedding_cache
                       (content_hash, content_preview, embedding, model, dimensions, source_event_id)
                       VALUES ($1, $2, $3::vector, $4, $5, $6)
                       ON CONFLICT (content_hash) DO NOTHING""",
                    ch, preview, vec_str, _model_name, _dimensions, source_event_id,
                )
        except Exception as e:
            log.error(f"Store embedding error: {e}")
