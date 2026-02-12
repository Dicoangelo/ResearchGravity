"""
Webhook Receiver Server — FastAPI on port 3848.

Receives, verifies, normalizes, and stores webhook events as cognitive_events.
Coherence engine auto-detects new events via PostgreSQL LISTEN/NOTIFY.
"""

import json
import logging
import time
from typing import Optional

import asyncpg
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse

from . import config as cfg
from .security import verify_provider_signature
from .handlers import get_handler, list_handlers, register_handlers
from .normalizer import WebhookNormalizer
from .audit import AuditTrail

from capture.normalizer import BaseNormalizer
from capture.dedup import DeduplicationEngine
from capture.quality import score_event

log = logging.getLogger("webhook.server")

app = FastAPI(
    title="UCW Webhook Receiver",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

# ── Shared state ──────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None
_dedup: Optional[DeduplicationEngine] = None
_webhook_normalizer: Optional[WebhookNormalizer] = None
_base_normalizer: Optional[BaseNormalizer] = None
_audit: Optional[AuditTrail] = None
_start_time: float = 0.0
_stats = {
    "events_received": 0,
    "events_stored": 0,
    "events_rejected": 0,
    "events_deduped": 0,
    "events_garbage": 0,
}


# ── Lifecycle ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _pool, _dedup, _webhook_normalizer, _base_normalizer, _audit, _start_time

    _start_time = time.time()

    _pool = await asyncpg.create_pool(
        cfg.PG_DSN,
        min_size=cfg.PG_MIN_POOL,
        max_size=cfg.PG_MAX_POOL,
        command_timeout=30,
    )

    _dedup = DeduplicationEngine()
    await _dedup.initialize(_pool)

    _webhook_normalizer = WebhookNormalizer()
    _base_normalizer = BaseNormalizer()

    _audit = AuditTrail(_pool)
    await _audit.ensure_table()

    register_handlers(cfg.ENABLED_PROVIDERS)

    log.info(
        f"Webhook server started — port {cfg.WEBHOOK_PORT}, "
        f"providers: {cfg.ENABLED_PROVIDERS}"
    )


@app.on_event("shutdown")
async def shutdown():
    if _pool:
        await _pool.close()
    log.info("Webhook server shutdown")


# ── Main webhook endpoint ─────────────────────────────────

@app.post("/webhook/{provider}")
async def receive_webhook(provider: str, request: Request):
    t0 = time.time()

    # 1. Get handler
    handler = get_handler(provider)
    if not handler:
        raise HTTPException(404, f"Unknown provider: {provider}")

    # 2. Read raw body (needed for HMAC verification)
    body = await request.body()

    # 3. Verify signature
    sig_valid = verify_provider_signature(provider, request.headers, body)
    if not sig_valid:
        elapsed_ms = int((time.time() - t0) * 1000)
        await _audit.log_received(
            provider=provider,
            event_type="unknown",
            delivery_id=request.headers.get("x-github-delivery", ""),
            signature_valid=False,
            events_parsed=0,
            events_stored=0,
            processing_time_ms=elapsed_ms,
            error_message="Invalid signature",
        )
        _stats["events_rejected"] += 1
        raise HTTPException(401, "Invalid webhook signature")

    # 4. Handle Slack URL verification challenge
    if provider == "slack":
        try:
            payload = json.loads(body)
            if payload.get("type") == "url_verification":
                return JSONResponse({"challenge": payload["challenge"]})
        except (json.JSONDecodeError, KeyError):
            pass

    # 5. Parse events via handler
    try:
        events = await handler.handle(request.headers, body)
    except Exception as exc:
        elapsed_ms = int((time.time() - t0) * 1000)
        await _audit.log_received(
            provider=provider,
            event_type="parse_error",
            delivery_id=request.headers.get("x-github-delivery", ""),
            signature_valid=True,
            events_parsed=0,
            events_stored=0,
            processing_time_ms=elapsed_ms,
            error_message=str(exc),
        )
        log.error(f"Handler error for {provider}: {exc}")
        raise HTTPException(500, f"Handler error: {exc}")

    if not events:
        elapsed_ms = int((time.time() - t0) * 1000)
        await _audit.log_received(
            provider=provider,
            event_type="empty",
            delivery_id=request.headers.get("x-github-delivery", ""),
            signature_valid=True,
            events_parsed=0,
            events_stored=0,
            processing_time_ms=elapsed_ms,
        )
        return {"status": "ok", "received": 0, "stored": 0}

    # 6. Normalize to CapturedEvent
    captured_events = _webhook_normalizer.normalize_batch(provider, events)
    event_type = events[0].event_type if events else "unknown"

    # 7. Dedup + quality gate + store
    stored = 0
    for captured in captured_events:
        content_hash = _base_normalizer.content_hash(captured.content)

        if _dedup.is_duplicate(
            captured.event_id, content_hash,
            captured.session_id, captured.platform,
        ):
            _stats["events_deduped"] += 1
            continue

        qs, mode = score_event(captured.content, captured.role, captured.platform)
        if mode == "garbage":
            _stats["events_garbage"] += 1
            continue

        captured.quality_score = qs
        captured.cognitive_mode = mode

        row = _base_normalizer.to_cognitive_event(captured)
        row["quality_score"] = qs
        row["cognitive_mode"] = mode

        try:
            await _store_event(row)
            _dedup.mark_seen(
                captured.event_id, content_hash,
                captured.session_id, captured.platform,
            )
            stored += 1
            _stats["events_stored"] += 1
        except Exception as exc:
            log.error(f"Store error: {exc}")

    _stats["events_received"] += len(events)

    # 8. Audit trail
    elapsed_ms = int((time.time() - t0) * 1000)
    await _audit.log_received(
        provider=provider,
        event_type=event_type,
        delivery_id=request.headers.get("x-github-delivery", ""),
        signature_valid=True,
        events_parsed=len(events),
        events_stored=stored,
        processing_time_ms=elapsed_ms,
    )

    return {"status": "ok", "received": len(events), "stored": stored}


# ── Relay endpoint (from Supabase Edge Function) ─────────

@app.post("/webhook/relay/{provider}")
async def receive_relay(provider: str, request: Request):
    """Accept forwarded webhooks from the Supabase relay."""
    body = await request.body()

    if not verify_provider_signature("relay", request.headers, body):
        raise HTTPException(401, "Invalid relay secret")

    # Reconstruct as if it came directly — forward to main endpoint
    handler = get_handler(provider)
    if not handler:
        raise HTTPException(404, f"Unknown provider: {provider}")

    # The original provider signature was already verified by the relay
    # Just parse, normalize, and store
    events = await handler.handle(request.headers, body)
    captured_events = _webhook_normalizer.normalize_batch(provider, events)

    stored = 0
    for captured in captured_events:
        content_hash = _base_normalizer.content_hash(captured.content)
        if _dedup.is_duplicate(
            captured.event_id, content_hash,
            captured.session_id, captured.platform,
        ):
            continue

        qs, mode = score_event(captured.content, captured.role, captured.platform)
        if mode == "garbage":
            continue

        captured.quality_score = qs
        captured.cognitive_mode = mode
        row = _base_normalizer.to_cognitive_event(captured)
        row["quality_score"] = qs
        row["cognitive_mode"] = mode

        try:
            await _store_event(row)
            _dedup.mark_seen(
                captured.event_id, content_hash,
                captured.session_id, captured.platform,
            )
            stored += 1
        except Exception as exc:
            log.error(f"Relay store error: {exc}")

    return {"status": "ok", "relayed": True, "received": len(events), "stored": stored}


# ── Health / Status ───────────────────────────────────────

@app.get("/webhook/health")
async def health():
    pool_ok = _pool is not None and not _pool._closed if _pool else False
    return {
        "status": "healthy" if pool_ok else "degraded",
        "pool": {"min": cfg.PG_MIN_POOL, "max": cfg.PG_MAX_POOL, "ok": pool_ok},
        "handlers": list(list_handlers().keys()),
        "uptime_s": int(time.time() - _start_time),
    }


@app.get("/webhook/status")
async def status():
    audit_stats = await _audit.stats() if _audit else {}
    return {
        "uptime_s": int(time.time() - _start_time),
        "providers": list_handlers(),
        "stats": _stats,
        "dedup": _dedup.stats if _dedup else {},
        "audit": audit_stats,
    }


# ── Test endpoint ─────────────────────────────────────────

@app.post("/webhook/test/{provider}")
async def test_webhook(provider: str):
    """Simulate a webhook delivery for testing."""
    handler = get_handler(provider)
    if not handler:
        raise HTTPException(404, f"Unknown provider: {provider}")

    # Build a minimal test payload per provider
    test_payloads = {
        "github": {
            "headers": {
                "x-github-event": "push",
                "x-github-delivery": "test-delivery-001",
            },
            "body": json.dumps({
                "ref": "refs/heads/main",
                "pusher": {"name": "test-user"},
                "repository": {"full_name": "Dicoangelo/test-repo"},
                "commits": [{"id": "abc12345", "message": "Test commit from webhook receiver"}],
            }).encode(),
        },
        "slack": {
            "headers": {},
            "body": json.dumps({
                "type": "event_callback",
                "team_id": "T_TEST",
                "event_id": "test-event-001",
                "event": {
                    "type": "message",
                    "user": "U_TEST",
                    "text": "Test message from webhook receiver",
                    "channel": "C_TEST",
                    "ts": str(time.time()),
                },
            }).encode(),
        },
        "generic": {
            "headers": {},
            "body": json.dumps({
                "type": "test",
                "text": "Test event from webhook receiver",
                "session_id": "test-session",
            }).encode(),
        },
    }

    test = test_payloads.get(provider, test_payloads["generic"])
    events = await handler.handle(test["headers"], test["body"])

    if not events:
        return {"status": "ok", "provider": provider, "events_parsed": 0, "note": "no events"}

    captured = _webhook_normalizer.normalize_batch(provider, events)

    stored = 0
    for ce in captured:
        content_hash = _base_normalizer.content_hash(ce.content)
        if _dedup.is_duplicate(ce.event_id, content_hash, ce.session_id, ce.platform):
            continue

        qs, mode = score_event(ce.content, ce.role, ce.platform)
        if mode == "garbage":
            continue

        ce.quality_score = qs
        ce.cognitive_mode = mode
        row = _base_normalizer.to_cognitive_event(ce)
        row["quality_score"] = qs
        row["cognitive_mode"] = mode

        await _store_event(row)
        _dedup.mark_seen(ce.event_id, content_hash, ce.session_id, ce.platform)
        stored += 1

    return {
        "status": "ok",
        "provider": provider,
        "events_parsed": len(events),
        "events_stored": stored,
        "sample_content": events[0].content[:200] if events else "",
    }


# ── Internal: store event ─────────────────────────────────

async def _store_event(row: dict) -> None:
    """INSERT a cognitive_event row — mirrors capture/manager.py:_store_event."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO cognitive_events (
                event_id, session_id, timestamp_ns, direction, stage,
                method, request_id, parent_event_id, turn,
                raw_bytes, parsed_json, content_length, error,
                data_layer, light_layer, instinct_layer,
                coherence_sig, platform, protocol,
                quality_score, cognitive_mode
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9,
                $10, $11, $12, $13, $14, $15, $16, $17, $18, $19,
                $20, $21
            )
            ON CONFLICT (event_id) DO NOTHING""",
            row["event_id"],
            row["session_id"],
            row["timestamp_ns"],
            row["direction"],
            row["stage"],
            row["method"],
            row.get("request_id"),
            row.get("parent_event_id"),
            row.get("turn", 0),
            row.get("raw_bytes"),
            row.get("parsed_json"),
            row.get("content_length", 0),
            row.get("error"),
            row.get("data_layer"),
            row.get("light_layer"),
            row.get("instinct_layer"),
            row.get("coherence_sig"),
            row["platform"],
            row.get("protocol", "webhook"),
            row.get("quality_score"),
            row.get("cognitive_mode"),
        )
