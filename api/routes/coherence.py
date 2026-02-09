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
