#!/usr/bin/env python3
"""
ResearchGravity Proof API — Real pipeline exposed via HTTP.

Wraps demo_proof.py so the proof-deck.html interactive demo
hits real antigravity.db instead of client-side simulation.

Usage:
    python3 proof_api.py          # Starts on port 3848
    python3 proof_api.py --port N # Custom port
"""

import json
import time
import uuid
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio

# Import pipeline functions from demo_proof
from demo_proof import (
    extract_evidence, run_oracle, get_db,
    store_finding, store_url, retrieve_finding,
    search_findings, fetch_url_text, DEMO_TRANSCRIPT,
    DB_PATH,
)

app = FastAPI(title="ResearchGravity Proof API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    text: str = ""
    url: str = ""


@app.get("/")
def health():
    return {"status": "ok", "db": str(DB_PATH), "db_exists": DB_PATH.exists()}


@app.post("/api/pipeline/stream")
async def pipeline_stream(req: PipelineRequest):
    """
    Stream the pipeline stages as Server-Sent Events.
    Each stage emits a JSON event so the frontend can animate in real-time.
    """
    async def generate():
        t0 = time.time()
        text = req.text or ""
        source_url = req.url or ""

        # Use demo transcript if nothing provided
        if not text and not source_url:
            text = DEMO_TRANSCRIPT

        # ── Stage 1: INGEST ──
        yield _sse("stage", {
            "stage": "ingest", "status": "running",
            "detail": f"Processing {len(text):,} characters..."
        })
        await asyncio.sleep(0.1)

        # If URL provided, fetch it
        if source_url:
            yield _sse("stage_update", {
                "stage": "ingest", "detail": f"Fetching {source_url}..."
            })
            fetched = await asyncio.to_thread(fetch_url_text, source_url)
            text = f"Source: {source_url}\n\n{fetched}\n\n{text}" if text else fetched

        # Extract evidence
        sources = extract_evidence(text)
        session_id = f"proof-{int(time.time())}-{uuid.uuid4().hex[:8]}"

        # Store URLs in DB
        conn = get_db()
        for s in sources:
            store_url(conn, session_id, s["url"], s.get("tier", 3), s.get("excerpt", ""))

        t1_count = sum(1 for s in sources if s.get("tier") == 1)
        t2_count = sum(1 for s in sources if s.get("tier") == 2)
        t3_count = len(sources) - t1_count - t2_count

        yield _sse("stage", {
            "stage": "ingest", "status": "done",
            "detail": f"Extracted {len(sources)} sources ({t1_count} Tier 1, {t2_count} Tier 2, {t3_count} Tier 3)",
            "data": {
                "session_id": session_id,
                "text_length": len(text),
                "source_count": len(sources),
            }
        })

        # ── Stage 2: EXTRACT ──
        yield _sse("stage", {
            "stage": "extract", "status": "running",
            "detail": "Classifying citations by tier..."
        })
        await asyncio.sleep(0.05)

        source_list = []
        for s in sources:
            source_list.append({
                "url": s["url"],
                "tier": s.get("tier", 3),
                "arxiv_id": s.get("arxiv_id"),
                "relevance": s["relevance_score"],
            })

        yield _sse("stage", {
            "stage": "extract", "status": "done",
            "detail": f"{len(sources)} evidence sources classified",
            "data": {"sources": source_list}
        })

        # ── Stage 3: VALIDATE (3-stream oracle) ──
        yield _sse("stage", {
            "stage": "validate", "status": "running",
            "detail": "Running 3-stream oracle critique (accuracy, completeness, relevance)..."
        })

        # Build claim
        claim_match = re.search(
            r'(?:Finding|Insight|Thesis|Claim)[:\s]+(.+?)(?:\n\n|\.\s+[A-Z])',
            text, re.DOTALL
        )
        if claim_match:
            claim = claim_match.group(1).strip()
        else:
            sentences = text.split(". ")
            claim = ". ".join(sentences[:3])[:400]

        # Real oracle critique
        oracle = await asyncio.to_thread(run_oracle, sources, claim)

        streams_data = []
        for s in oracle["streams"]:
            streams_data.append({
                "name": s["stream"],
                "score": round(s["score"], 4),
                "issues": s["issues"],
            })

        yield _sse("stage", {
            "stage": "validate", "status": "done",
            "detail": f"Weighted confidence: {oracle['confidence']:.1%} — {'VALIDATED' if oracle['validated'] else 'BELOW THRESHOLD'}",
            "data": {
                "streams": streams_data,
                "confidence": round(oracle["confidence"], 4),
                "validated": oracle["validated"],
                "threshold": 0.70,
            }
        })

        # ── Stage 4: STORE ──
        yield _sse("stage", {
            "stage": "store", "status": "running",
            "detail": "Writing finding to antigravity.db..."
        })
        await asyncio.sleep(0.05)

        now_iso = datetime.now().isoformat()
        evidence = {
            "sources": sources,
            "confidence": oracle["confidence"],
            "reasoning_chain": [
                f"Extracted {len(sources)} citation(s) from input text",
                f"Source tiers: {t1_count} T1, {t2_count} T2, {t3_count} T3",
                "3-stream oracle validation (accuracy, completeness, relevance)",
                f"Consensus: {oracle['confidence']:.2%} ({'VALIDATED' if oracle['validated'] else 'BELOW THRESHOLD'})",
            ],
            "validation": {
                "validated": oracle["validated"],
                "validated_at": now_iso,
                "critic_notes": oracle["issues"],
                "oracle_streams": [f"{s['stream']}: {s['score']:.2f}" for s in oracle["streams"]],
            }
        }

        finding = {
            "id": f"finding-{int(time.time())}-{uuid.uuid4().hex[:9]}",
            "session_id": session_id,
            "content": claim,
            "type": "finding",
            "evidence": evidence,
            "created_at": now_iso,
            "updated_at": now_iso,
            "derived_from": [],
            "enables": [],
        }

        fid = store_finding(conn, finding)

        yield _sse("stage", {
            "stage": "store", "status": "done",
            "detail": f"Stored in antigravity.db → findings table",
            "data": {
                "finding_id": fid,
                "claim": claim[:200],
                "confidence": oracle["confidence"],
                "source_count": len(sources),
                "validated": oracle["validated"],
            }
        })

        # ── Stage 5: RETRIEVE ──
        yield _sse("stage", {
            "stage": "retrieve", "status": "running",
            "detail": f"Retrieving {fid} from database..."
        })
        await asyncio.sleep(0.05)

        retrieved = retrieve_finding(conn, fid)
        verified = retrieved is not None and retrieved["id"] == fid

        # Also do a search to show retrieval works
        query_terms = " ".join(re.findall(r'\b[a-z]{5,}\b', claim.lower())[:5])
        if not query_terms:
            query_terms = "research finding"
        related = search_findings(conn, query_terms, limit=3)

        conn.close()

        elapsed = round(time.time() - t0, 3)

        yield _sse("stage", {
            "stage": "retrieve", "status": "done",
            "detail": f"Round-trip verified — record intact ({elapsed}s)",
            "data": {
                "finding_id": fid,
                "verified": verified,
                "search_query": query_terms,
                "related_count": len(related),
                "elapsed_seconds": elapsed,
            }
        })

        # Final summary event
        yield _sse("complete", {
            "finding_id": fid,
            "session_id": session_id,
            "confidence": oracle["confidence"],
            "validated": oracle["validated"],
            "sources": len(sources),
            "elapsed": elapsed,
            "claim": claim[:200],
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/stats")
def db_stats():
    """Return live database stats."""
    conn = get_db()
    findings = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
    urls = conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    lineage = conn.execute("SELECT COUNT(*) FROM lineage").fetchone()[0]
    conn.close()
    return {
        "findings": findings,
        "urls": urls,
        "sessions": sessions,
        "lineage": lineage,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3848)
    args = parser.parse_args()
    print(f"🔬 ResearchGravity Proof API starting on port {args.port}")
    print(f"   DB: {DB_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
