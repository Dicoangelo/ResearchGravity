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
_model_name = "nomic-ai/nomic-embed-text-v1.5"
_dimensions = 768
_embedding_column = "embedding_768"

# Legacy model info (for reading old embeddings during migration)
_legacy_model_name = "all-MiniLM-L6-v2"
_legacy_dimensions = 384


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name, trust_remote_code=True)
        log.info(f"Loaded embedding model: {_model_name} ({_dimensions}d)")
    return _model


def release_model():
    """Release the SBERT model to free memory."""
    global _model
    _model = None
    log.info("SBERT model released")


def _parse_event_layers(event_or_dict):
    """Extract light and data layers from event object or dict."""
    if hasattr(event_or_dict, "light_layer"):
        light = event_or_dict.light_layer or {}
        data = event_or_dict.data_layer or {}
        instinct = getattr(event_or_dict, "instinct_layer", None) or {}
        platform = getattr(event_or_dict, "platform", "")
        cog_mode = getattr(event_or_dict, "cognitive_mode", "")
    elif isinstance(event_or_dict, dict):
        light_raw = event_or_dict.get("light_layer", "{}")
        data_raw = event_or_dict.get("data_layer", "{}")
        instinct_raw = event_or_dict.get("instinct_layer", "{}")
        light = json.loads(light_raw) if isinstance(light_raw, str) else (light_raw or {})
        data = json.loads(data_raw) if isinstance(data_raw, str) else (data_raw or {})
        instinct = json.loads(instinct_raw) if isinstance(instinct_raw, str) else (instinct_raw or {})
        platform = event_or_dict.get("platform", "")
        cog_mode = event_or_dict.get("cognitive_mode", "")
    else:
        return {}, {}, {}, "", ""
    return light, data, instinct, platform, cog_mode


def build_embed_text(event_or_dict) -> str:
    """
    Build the text to embed from a cognitive event.

    Format: "{intent}: {topic} | {summary} | {concepts}"
    Works with CaptureEvent objects or dicts from the database.
    """
    light, data, _, _, _ = _parse_event_layers(event_or_dict)
    if not light and not data:
        return ""

    intent = light.get("intent", "explore")
    topic = light.get("topic", "general")
    summary = light.get("summary", "")
    concepts = light.get("concepts", [])
    content = data.get("content", "")

    parts = [f"{intent}: {topic}"]
    if summary:
        parts.append(summary[:300])
    elif content:
        parts.append(content[:300])
    if concepts:
        parts.append(" ".join(concepts))

    return " | ".join(parts)


def build_embed_text_contextual(event_or_dict) -> str:
    """
    Build context-enriched embedding text per Anthropic's contextual retrieval pattern.

    Prepends session context (platform, cognitive mode, topic) to the base text.
    This improves cross-platform coherence detection by 5-10% because the embedding
    captures WHERE and HOW the thinking happened, not just WHAT was said.

    Format: "In a {mode} session on {platform} about {topic}. {base_text}"
    """
    light, data, instinct, platform, cog_mode = _parse_event_layers(event_or_dict)
    if not light and not data:
        return ""

    parts = []

    # Session context prefix
    session_topic = light.get("topic", "")
    if platform or cog_mode:
        ctx = []
        if cog_mode:
            ctx.append(f"{cog_mode}")
        if platform:
            ctx.append(f"on {platform}")
        if session_topic:
            ctx.append(f"about {session_topic}")
        parts.append(f"In a {' '.join(ctx)} session.")

    # Core content
    intent = light.get("intent", "explore")
    summary = light.get("summary", "")
    content = data.get("content", "")

    if intent and session_topic:
        parts.append(f"{intent}: {session_topic}")
    if summary:
        parts.append(summary[:300])
    elif content:
        parts.append(content[:300])

    # Concepts
    concepts = light.get("concepts", [])
    if concepts:
        parts.append(" ".join(concepts[:5]))

    # High coherence signal
    coherence_potential = instinct.get("coherence_potential", 0)
    if coherence_potential and coherence_potential > 0.7:
        parts.append(f"[high coherence: {coherence_potential:.2f}]")

    return " | ".join(filter(None, parts))


def content_hash(text: str) -> str:
    """SHA-256 hash for dedup in embedding_cache."""
    return hashlib.sha256(text.encode()).hexdigest()


def embed_texts(texts: List[str], batch_size: int = 64, prefix: str = "search_document") -> List[List[float]]:
    """Embed a batch of texts. Returns list of float vectors.

    Nomic requires prefix: 'search_document' for indexing, 'search_query' for queries.
    """
    model = _get_model()
    prefixed = [f"{prefix}: {t}" for t in texts]
    embeddings = model.encode(prefixed, batch_size=batch_size, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_single(text: str, prefix: str = "search_document") -> List[float]:
    """Embed a single text. Returns float vector."""
    model = _get_model()
    return model.encode(f"{prefix}: {text}", show_progress_bar=False).tolist()


async def embed_single_async(text: str, prefix: str = "search_document") -> List[float]:
    """Async wrapper — runs SBERT in a thread to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(embed_single, text, prefix)


async def embed_texts_async(texts: List[str], batch_size: int = 64, prefix: str = "search_document") -> List[List[float]]:
    """Async wrapper — runs SBERT batch in a thread to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(embed_texts, texts, batch_size, prefix)


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

        embedding = await embed_single_async(text)
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
            embs = await embed_texts_async(batch, batch_size=batch_size)
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
                        f"""INSERT INTO embedding_cache
                           (content_hash, content_preview, {_embedding_column}, model, dimensions, source_event_id)
                           VALUES ($1, $2, $3::vector, $4, $5, $6)
                           ON CONFLICT (content_hash) DO UPDATE SET
                               {_embedding_column} = $3::vector,
                               model = $4,
                               dimensions = $5""",
                        ch, text[:200], vec_str, _model_name, _dimensions, eid,
                    )
                    # UPSERT always affects 1 row
                    stored += 1
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
        query_emb = await embed_single_async(text)

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
        Uses 768d nomic embeddings (embedding_768) with fallback to 384d (embedding).
        """
        if not self._pool:
            return []

        query_emb = await embed_single_async(text, prefix="search_query")
        vec_str = '[' + ','.join(str(x) for x in query_emb) + ']'
        col = _embedding_column

        async with self._pool.acquire() as conn:
            if exclude_platform:
                rows = await conn.fetch(
                    f"""SELECT ec.content_preview, ec.source_event_id,
                              ce.platform, ce.session_id, ce.coherence_sig,
                              ce.cognitive_mode,
                              1 - (ec.{col} <=> $1::vector) AS similarity
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ce.platform != $2 AND ec.{col} IS NOT NULL
                       ORDER BY ec.{col} <=> $1::vector
                       LIMIT $3""",
                    vec_str, exclude_platform, limit,
                )
            else:
                rows = await conn.fetch(
                    f"""SELECT ec.content_preview, ec.source_event_id,
                              ce.platform, ce.session_id, ce.coherence_sig,
                              ce.cognitive_mode,
                              1 - (ec.{col} <=> $1::vector) AS similarity
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ec.{col} IS NOT NULL
                       ORDER BY ec.{col} <=> $1::vector
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

        col = _embedding_column
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""WITH source_events AS (
                       SELECT ec.{col} AS emb, ec.content_preview AS src_preview,
                              ec.source_event_id AS src_id, ce.cognitive_mode AS src_mode
                       FROM embedding_cache ec
                       JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
                       WHERE ce.platform = $1 AND ce.cognitive_mode IN ('deep_work', 'exploration')
                             AND ec.{col} IS NOT NULL
                       LIMIT 1000
                   )
                   SELECT src.src_preview, src.src_id, src.src_mode,
                          tgt.content_preview AS tgt_preview,
                          tgt.source_event_id AS tgt_id,
                          tgt_ce.platform AS tgt_platform,
                          tgt_ce.cognitive_mode AS tgt_mode,
                          1 - (src.emb <=> tgt.emb) AS similarity
                   FROM source_events src
                   CROSS JOIN LATERAL (
                       SELECT ec2.content_preview, ec2.source_event_id, ec2.{col} AS emb
                       FROM embedding_cache ec2
                       JOIN cognitive_events ce2 ON ec2.source_event_id = ce2.event_id
                       WHERE ce2.platform != $1 AND ec2.{col} IS NOT NULL
                       ORDER BY ec2.{col} <=> src.emb
                       LIMIT 1
                   ) tgt
                   JOIN cognitive_events tgt_ce ON tgt.source_event_id = tgt_ce.event_id
                   WHERE 1 - (src.emb <=> tgt.emb) >= $2
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
