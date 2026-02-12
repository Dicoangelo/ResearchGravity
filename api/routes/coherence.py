"""
Coherence Dashboard API Routes

Provides REST endpoints for the UCW coherence web dashboard.
All data sourced from PostgreSQL (ucw_cognitive database).

Endpoints:
    GET /api/v2/coherence/overview   — Aggregate stats
    GET /api/v2/coherence/moments    — List coherence moments
    GET /api/v2/coherence/network    — Platform-pair graph data
    GET /api/v2/coherence/pulse      — Real-time activity + arcs
    GET /api/v2/coherence/moment/{id}/signals — Signal breakdown
"""

import json
from datetime import datetime, timezone
from typing import Optional

import asyncpg
from fastapi import APIRouter, Query

from coherence_engine import config as cfg

router = APIRouter(prefix="/api/v2/coherence", tags=["coherence"])

_pool: Optional[asyncpg.Pool] = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            cfg.PG_DSN, min_size=1, max_size=5, command_timeout=60
        )
    return _pool


def _ns_to_iso(ns: int) -> str:
    """Convert nanosecond timestamp to ISO string."""
    if not ns:
        return ""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def _dt_to_iso(dt) -> Optional[str]:
    """Convert datetime to ISO string, handling None."""
    if dt is None:
        return None
    return dt.isoformat()


# ── Overview ─────────────────────────────────────────────


@router.get("/overview")
async def coherence_overview():
    """Aggregate dashboard stats in a single call."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        platforms = await conn.fetch("""
            SELECT platform, COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') AS today,
                   MAX(created_at) AS last_seen
            FROM cognitive_events
            GROUP BY platform ORDER BY total DESC
        """)

        moments_total = await conn.fetchval(
            "SELECT COUNT(*) FROM coherence_moments"
        )
        moments_24h = await conn.fetchval(
            "SELECT COUNT(*) FROM coherence_moments "
            "WHERE created_at > NOW() - INTERVAL '24 hours'"
        )

        by_type = await conn.fetch("""
            SELECT coherence_type, COUNT(*) AS count,
                   AVG(confidence) AS avg_confidence
            FROM coherence_moments
            GROUP BY coherence_type ORDER BY count DESC
        """)

        conf_dist = await conn.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE confidence >= 0.90) AS tier_90,
                COUNT(*) FILTER (WHERE confidence >= 0.80 AND confidence < 0.90) AS tier_80,
                COUNT(*) FILTER (WHERE confidence >= 0.70 AND confidence < 0.80) AS tier_70,
                COUNT(*) FILTER (WHERE confidence >= 0.60 AND confidence < 0.70) AS tier_60,
                COUNT(*) FILTER (WHERE confidence < 0.60) AS tier_low
            FROM coherence_moments
        """)

        embedded_count = await conn.fetchval(
            "SELECT COUNT(*) FROM embedding_cache"
        )

    return {
        "platforms": [
            {
                "platform": r["platform"],
                "total": r["total"],
                "today": r["today"],
                "last_seen": _dt_to_iso(r["last_seen"]),
            }
            for r in platforms
        ],
        "moments_total": moments_total or 0,
        "moments_24h": moments_24h or 0,
        "by_type": [
            {
                "coherence_type": r["coherence_type"],
                "count": r["count"],
                "avg_confidence": float(r["avg_confidence"] or 0),
            }
            for r in by_type
        ],
        "confidence_distribution": dict(conf_dist) if conf_dist else {},
        "embedded_count": embedded_count or 0,
    }


# ── Moments ──────────────────────────────────────────────


@router.get("/moments")
async def coherence_moments(
    limit: int = Query(100, ge=1, le=500),
    since_hours: int = Query(168, ge=1, le=8760),
    type: Optional[str] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
):
    """List coherence moments with optional filters."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT moment_id, detected_ns, event_ids, platforms,
                   coherence_type, confidence, description,
                   insight_summary, insight_category, insight_novelty,
                   time_window_s, metadata, created_at
            FROM coherence_moments
            WHERE created_at > NOW() - make_interval(hours => $1)
              AND ($2::text IS NULL OR coherence_type = $2)
              AND confidence >= $3
            ORDER BY created_at DESC
            LIMIT $4
            """,
            since_hours,
            type,
            min_confidence,
            limit,
        )

        total = await conn.fetchval(
            """
            SELECT COUNT(*) FROM coherence_moments
            WHERE created_at > NOW() - make_interval(hours => $1)
              AND ($2::text IS NULL OR coherence_type = $2)
              AND confidence >= $3
            """,
            since_hours,
            type,
            min_confidence,
        )

    return {
        "moments": [
            {
                "moment_id": r["moment_id"],
                "detected_at": _ns_to_iso(r["detected_ns"]),
                "event_ids": r["event_ids"],
                "platforms": r["platforms"],
                "coherence_type": r["coherence_type"],
                "confidence": float(r["confidence"]),
                "description": r["description"],
                "time_window_s": r["time_window_s"],
                "signals": _extract_signals(r),
                "insight_summary": r.get("insight_summary"),
                "insight_category": r.get("insight_category"),
                "insight_novelty": float(r["insight_novelty"]) if r.get("insight_novelty") else None,
                "created_at": _dt_to_iso(r["created_at"]),
            }
            for r in rows
        ],
        "total": total or 0,
    }


def _extract_signals(row) -> dict:
    """Extract signals from metadata or synthesize from type."""
    meta = row.get("metadata")
    if meta:
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        if isinstance(meta, dict) and "signals" in meta:
            return meta["signals"]

    # Synthesize based on type
    ctype = row.get("coherence_type", "")
    conf = float(row.get("confidence", 0))

    if ctype == "signature_match":
        return {
            "temporal": 1.0,
            "semantic": conf,
            "meta_cognitive": 0.5,
            "instinct_alignment": 0.5,
            "concept_overlap": conf,
        }
    elif ctype == "semantic_echo":
        return {
            "temporal": 0.3,
            "semantic": conf,
            "meta_cognitive": 0.2,
            "instinct_alignment": 0.3,
            "concept_overlap": conf * 0.7,
        }
    else:  # synchronicity or unknown
        return {
            "temporal": conf * 0.6,
            "semantic": conf * 0.8,
            "meta_cognitive": conf * 0.5,
            "instinct_alignment": conf * 0.4,
            "concept_overlap": conf * 0.3,
        }


# ── Network ──────────────────────────────────────────────


@router.get("/network")
async def coherence_network():
    """Platform-pair graph data for network visualization."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        # Platform node sizes
        nodes = await conn.fetch("""
            SELECT platform, COUNT(*) AS event_count
            FROM cognitive_events
            GROUP BY platform ORDER BY event_count DESC
        """)

        # Platform-pair coherence links
        links = await conn.fetch("""
            SELECT
                platforms[1] AS source,
                platforms[2] AS target,
                COUNT(*) AS count,
                AVG(confidence) AS avg_confidence,
                MAX(confidence) AS max_confidence
            FROM coherence_moments
            WHERE array_length(platforms, 1) = 2
            GROUP BY platforms[1], platforms[2]
            ORDER BY count DESC
        """)

    return {
        "nodes": [
            {
                "id": r["platform"],
                "event_count": r["event_count"],
                "label": r["platform"].replace("-", " ").title(),
            }
            for r in nodes
        ],
        "links": [
            {
                "source": r["source"],
                "target": r["target"],
                "count": r["count"],
                "avg_confidence": float(r["avg_confidence"] or 0),
                "max_confidence": float(r["max_confidence"] or 0),
            }
            for r in links
        ],
    }


# ── Pulse ────────────────────────────────────────────────


@router.get("/pulse")
async def coherence_pulse(hours: int = Query(24, ge=1, le=168)):
    """Real-time platform activity + recent coherence arcs."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        activity = await conn.fetch(
            """
            SELECT platform,
                   date_trunc('hour', created_at) +
                     INTERVAL '30 min' * floor(extract(minute from created_at) / 30)
                     AS bucket,
                   COUNT(*) AS event_count
            FROM cognitive_events
            WHERE created_at > NOW() - make_interval(hours => $1)
            GROUP BY platform, bucket
            ORDER BY bucket DESC
            """,
            hours,
        )

        arcs = await conn.fetch(
            """
            SELECT moment_id, platforms, confidence, coherence_type,
                   detected_ns, created_at
            FROM coherence_moments
            WHERE created_at > NOW() - make_interval(hours => $1)
            ORDER BY created_at DESC
            LIMIT 50
            """,
            hours,
        )

    return {
        "activity": [
            {
                "platform": r["platform"],
                "bucket": _dt_to_iso(r["bucket"]),
                "event_count": r["event_count"],
            }
            for r in activity
        ],
        "arcs": [
            {
                "moment_id": r["moment_id"],
                "platforms": r["platforms"],
                "confidence": float(r["confidence"]),
                "coherence_type": r["coherence_type"],
                "detected_at": _ns_to_iso(r["detected_ns"]),
                "created_at": _dt_to_iso(r["created_at"]),
            }
            for r in arcs
        ],
    }


# ── Signals ──────────────────────────────────────────────


@router.get("/moment/{moment_id}/signals")
async def moment_signals(moment_id: str):
    """Signal breakdown for a specific coherence moment."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT moment_id, coherence_type, confidence,
                   metadata, description, event_ids, platforms
            FROM coherence_moments WHERE moment_id = $1
            """,
            moment_id,
        )

    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Moment not found")

    return {
        "moment_id": row["moment_id"],
        "coherence_type": row["coherence_type"],
        "confidence": float(row["confidence"]),
        "description": row["description"],
        "platforms": row["platforms"],
        "signals": _extract_signals(dict(row)),
    }


# ── Insights ────────────────────────────────────────────


@router.get("/moment/{moment_id}/insight")
async def moment_insight(moment_id: str):
    """Get the extracted insight for a coherence moment.

    Returns the LLM-synthesized insight summary, category, and novelty score.
    If no insight exists yet, triggers extraction on-demand.
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT moment_id, coherence_type, confidence, description,
                   event_ids, platforms, insight_summary, insight_category,
                   insight_novelty
            FROM coherence_moments WHERE moment_id = $1
            """,
            moment_id,
        )

    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Moment not found")

    # If insight doesn't exist yet, extract on-demand
    if not row["insight_summary"]:
        from coherence_engine.insight_extractor import extract_insight_for_moment
        result = await extract_insight_for_moment(pool, moment_id)
        if result:
            return {
                "moment_id": moment_id,
                "insight_summary": result.summary,
                "insight_category": result.category,
                "insight_novelty": result.novelty,
                "coherence_type": row["coherence_type"],
                "confidence": float(row["confidence"]),
                "platforms": row["platforms"],
                "extracted_now": True,
            }

    return {
        "moment_id": row["moment_id"],
        "insight_summary": row["insight_summary"],
        "insight_category": row["insight_category"],
        "insight_novelty": float(row["insight_novelty"]) if row["insight_novelty"] else None,
        "coherence_type": row["coherence_type"],
        "confidence": float(row["confidence"]),
        "platforms": row["platforms"],
        "extracted_now": False,
    }


@router.get("/insights")
async def list_insights(
    limit: int = Query(50, ge=1, le=500),
    category: Optional[str] = Query(None),
    min_novelty: float = Query(0.0, ge=0.0, le=1.0),
):
    """List all moments that have extracted insights."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT moment_id, detected_ns, platforms, coherence_type,
                   confidence, insight_summary, insight_category,
                   insight_novelty, created_at
            FROM coherence_moments
            WHERE insight_summary IS NOT NULL
              AND ($1::text IS NULL OR insight_category = $1)
              AND COALESCE(insight_novelty, 0) >= $2
            ORDER BY COALESCE(insight_novelty, 0) DESC, confidence DESC
            LIMIT $3
            """,
            category,
            min_novelty,
            limit,
        )

    return {
        "insights": [
            {
                "moment_id": r["moment_id"],
                "detected_at": _ns_to_iso(r["detected_ns"]),
                "platforms": r["platforms"],
                "coherence_type": r["coherence_type"],
                "confidence": float(r["confidence"]),
                "insight_summary": r["insight_summary"],
                "insight_category": r["insight_category"],
                "insight_novelty": float(r["insight_novelty"]) if r["insight_novelty"] else None,
                "created_at": _dt_to_iso(r["created_at"]),
            }
            for r in rows
        ],
        "total": len(rows),
    }


# ── Capture (Chrome Extension) ──────────────────────────


from pydantic import BaseModel


class ExtensionEvent(BaseModel):
    """Event captured by the UCW Chrome extension."""
    platform: str
    content: str
    direction: str = "in"  # "in" (assistant) or "out" (user)
    url: Optional[str] = None
    topic: Optional[str] = None
    intent: Optional[str] = None
    concepts: Optional[list] = None
    session_hint: Optional[str] = None
    metadata: Optional[dict] = None


@router.post("/capture/extension")
async def capture_extension_event(event: ExtensionEvent):
    """
    Receive a cognitive event from the UCW Chrome extension.

    Stores the event in cognitive_events with full UCW semantic layers.
    Returns the created event_id for correlation.
    """
    import hashlib
    import time

    pool = await _get_pool()

    now_ns = time.time_ns()
    event_id = f"ext-{hashlib.sha256(f'{now_ns}{event.content[:50]}'.encode()).hexdigest()[:16]}"

    # Build session_id from hint or generate
    session_id = event.session_hint or f"ext-{event.platform}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H')}"

    # Build semantic layers
    data_layer = json.dumps({"content": event.content[:10000], "source_url": event.url})
    light_layer = json.dumps({
        "topic": event.topic or "",
        "intent": event.intent or "",
        "concepts": event.concepts or [],
        "summary": event.content[:200],
    })
    instinct_layer = json.dumps({
        "coherence_potential": 0.5,
        "gut_signal": "extension_capture",
    })

    # Compute coherence signature
    sig_text = f"{event.platform}:{event.topic or ''}:{event.content[:100]}"
    coherence_sig = hashlib.sha256(sig_text.encode()).hexdigest()[:32]

    async with pool.acquire() as conn:
        # Upsert session
        await conn.execute(
            """INSERT INTO cognitive_sessions
                   (session_id, platform, started_ns, status, event_count)
               VALUES ($1, $2, $3, 'active', 1)
               ON CONFLICT (session_id) DO UPDATE SET
                   event_count = cognitive_sessions.event_count + 1,
                   last_event_ns = $3""",
            session_id, event.platform, now_ns,
        )

        # Insert event
        await conn.execute(
            """INSERT INTO cognitive_events
                   (event_id, session_id, timestamp_ns, platform,
                    direction, method,
                    data_layer, light_layer, instinct_layer,
                    coherence_sig)
               VALUES ($1, $2, $3, $4, $5, 'extension',
                       $6::jsonb, $7::jsonb, $8::jsonb, $9)""",
            event_id, session_id, now_ns, event.platform,
            event.direction,
            data_layer, light_layer, instinct_layer,
            coherence_sig,
        )

    return {
        "status": "captured",
        "event_id": event_id,
        "session_id": session_id,
        "platform": event.platform,
        "timestamp_ns": now_ns,
    }


@router.post("/capture/extension/batch")
async def capture_extension_batch(events: list[ExtensionEvent]):
    """Batch capture multiple events from the Chrome extension."""
    results = []
    for event in events[:50]:  # Cap at 50 per batch
        try:
            result = await capture_extension_event(event)
            results.append(result)
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return {
        "captured": len([r for r in results if r.get("status") == "captured"]),
        "errors": len([r for r in results if r.get("status") == "error"]),
        "results": results,
    }


# ── Session Coherence ──────────────────────────────────


@router.get("/session-coherence")
async def session_coherence(
    hours: int = Query(168, ge=1, le=8760),
    min_similarity: float = Query(0.55, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
):
    """Find cross-platform session coherence pairs."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
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
              AND a.session_embedding IS NOT NULL
              AND b.session_embedding IS NOT NULL
              AND a.platform != b.platform
              AND a.started_ns > EXTRACT(EPOCH FROM (NOW() - make_interval(hours => $1))) * 1e9
              AND 1 - (a.session_embedding <=> b.session_embedding) >= $2
            ORDER BY similarity DESC
            LIMIT $3
            """,
            hours,
            min_similarity,
            limit,
        )

    return {
        "pairs": [
            {
                "session_a": r["session_a"],
                "session_b": r["session_b"],
                "platform_a": r["platform_a"],
                "platform_b": r["platform_b"],
                "summary_a": r["summary_a"],
                "summary_b": r["summary_b"],
                "similarity": float(r["similarity"]),
            }
            for r in rows
        ],
        "total": len(rows),
    }


@router.get("/concept-evolution")
async def concept_evolution(
    concept: Optional[str] = Query(None),
    min_versions: int = Query(2, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    """Get concept evolution chains from the knowledge graph."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        if concept:
            rows = await conn.fetch(
                """SELECT concept, version, definition, first_seen_ns,
                          last_seen_ns, platform, session_id, evolved_from
                   FROM concept_versions
                   WHERE concept = $1
                   ORDER BY version ASC""",
                concept.lower(),
            )
            return {
                "concept": concept.lower(),
                "versions": [
                    {
                        "version": r["version"],
                        "definition": r["definition"],
                        "platform": r["platform"],
                        "session_id": r["session_id"],
                        "evolved_from": r["evolved_from"],
                        "first_seen": _ns_to_iso(r["first_seen_ns"]),
                        "last_seen": _ns_to_iso(r["last_seen_ns"]),
                    }
                    for r in rows
                ],
                "total_versions": len(rows),
            }
        else:
            rows = await conn.fetch(
                """SELECT concept, COUNT(*) AS version_count,
                          MAX(version) AS latest_version,
                          COUNT(DISTINCT platform) AS platform_count
                   FROM concept_versions
                   GROUP BY concept
                   HAVING COUNT(*) >= $1
                   ORDER BY version_count DESC
                   LIMIT $2""",
                min_versions, limit,
            )
            return {
                "evolving_concepts": [
                    {
                        "concept": r["concept"],
                        "version_count": r["version_count"],
                        "latest_version": r["latest_version"],
                        "platform_count": r["platform_count"],
                    }
                    for r in rows
                ],
                "total": len(rows),
            }


@router.get("/breakthroughs")
async def list_breakthroughs(
    limit: int = Query(20, ge=1, le=100),
):
    """List detected cognitive breakthroughs."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT breakthrough_id, detected_at, breakthrough_type,
                   title, narrative, evidence_moment_ids, platforms,
                   concepts, novelty_score, impact_score
            FROM cognitive_breakthroughs
            ORDER BY detected_at DESC
            LIMIT $1
            """,
            limit,
        )

    return {
        "breakthroughs": [
            {
                "breakthrough_id": r["breakthrough_id"],
                "detected_at": _dt_to_iso(r["detected_at"]),
                "type": r["breakthrough_type"],
                "title": r["title"],
                "narrative": r["narrative"],
                "evidence_moment_ids": r["evidence_moment_ids"],
                "platforms": r["platforms"],
                "concepts": r["concepts"],
                "novelty_score": float(r["novelty_score"]) if r["novelty_score"] else None,
                "impact_score": float(r["impact_score"]) if r["impact_score"] else None,
            }
            for r in rows
        ],
        "total": len(rows),
    }
