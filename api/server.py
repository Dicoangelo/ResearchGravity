#!/usr/bin/env python3
"""
Agent Core API Server - FastAPI implementation.

Provides REST endpoints for:
- Session management
- Finding queries with evidence
- Semantic search (with Qdrant vector search)
- Context pack selection
- Reinvigoration support
- Graph intelligence (lineage, related concepts)
- UCW pack ingestion
- Storage statistics

Phase 3a: Now uses SQLite + Qdrant for concurrent-safe storage.

Usage:
    python3 -m api.server --port 3847
    uvicorn api.server:app --port 3847 --reload
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Storage engine (Phase 3a)
STORAGE_AVAILABLE = False
try:
    from storage import StorageEngine, get_engine
    from storage.ucw_ingestion import UCWIngestionPipeline, ingest_ucw_trade
    STORAGE_AVAILABLE = True
except ImportError:
    pass

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Agent Core API",
        description="REST API for Antigravity Chief of Staff - Knowledge service for OS-App, CareerCoach, and ResearchGravity",
        version="2.1.0",  # Upgraded with security
    )

    # Security imports
    try:
        from api.security import (
            validate_session_id,
            validate_project_id,
            limiter,
            RequestLoggingMiddleware,
            RATE_LIMIT_DEFAULT,
            RATE_LIMIT_SEARCH,
            RATE_LIMIT_WRITE,
            get_current_user,
            optional_auth,
            create_access_token,
        )
        from fastapi import Request, Depends
        SECURITY_AVAILABLE = True
        AUTH_AVAILABLE = True

        # Rate limit decorators (conditional on limiter availability)
        def rate_limit_default(func):
            if limiter:
                return limiter.limit(RATE_LIMIT_DEFAULT)(func)
            return func

        def rate_limit_search(func):
            if limiter:
                return limiter.limit(RATE_LIMIT_SEARCH)(func)
            return func

        def rate_limit_write(func):
            if limiter:
                return limiter.limit(RATE_LIMIT_WRITE)(func)
            return func

    except ImportError:
        SECURITY_AVAILABLE = False
        AUTH_AVAILABLE = False
        print("Warning: Security module not available")
        # Define dummy fallbacks
        Request = None
        Depends = None
        def rate_limit_default(func): return func
        def rate_limit_search(func): return func
        def rate_limit_write(func): return func
        async def get_current_user(): return {"type": "anonymous"}
        async def optional_auth(): return None

    # Add request logging middleware
    if SECURITY_AVAILABLE:
        app.add_middleware(RequestLoggingMiddleware)

    # Add rate limiting (if slowapi available)
    if SECURITY_AVAILABLE and limiter:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods instead of "*"
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],  # Expose request ID header
        max_age=3600,
    )

    # Include intelligence routes (V2)
    try:
        from api.routes.intelligence import router as intelligence_router
        app.include_router(intelligence_router)
    except ImportError:
        print("Warning: Intelligence routes not available")

# Data directories
AGENT_CORE_DIR = Path.home() / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
MEMORY_DIR = Path.home() / ".claude" / "memory"


# ============================================================
# Pydantic Models (only if FastAPI available)
# ============================================================

if FASTAPI_AVAILABLE:
    class SessionSummary(BaseModel):
        id: str
        topic: Optional[str] = None
        status: str = "archived"
        project: Optional[str] = None
        url_count: int = 0
        finding_count: int = 0
        created_at: Optional[str] = None

    class FindingResponse(BaseModel):
        id: str
        session_id: str
        content: str
        type: str
        confidence: float = 0.0
        sources: list = []
        needs_review: bool = False

    class SearchQuery(BaseModel):
        query: str
        category: str = "all"
        limit: int = 10
        min_confidence: float = 0.3
        project: Optional[str] = None

    class SearchResult(BaseModel):
        content: str
        category: str
        similarity: float
        session_id: Optional[str] = None
        tags: list = []

    class NewFinding(BaseModel):
        content: str
        type: str = "finding"
        project: Optional[str] = None
        tags: list = []
        source_url: Optional[str] = None

    class PackSelection(BaseModel):
        project: Optional[str] = None
        pattern: Optional[str] = None
        limit: int = 5

    class UCWIngestRequest(BaseModel):
        wallet_id: str
        packs: List[dict]
        validate: bool = False

    class BulkFindingsRequest(BaseModel):
        findings: List[dict]
        source: str = "api"

    class SemanticSearchRequest(BaseModel):
        query: str
        limit: int = 10
        min_score: float = 0.4
        collections: Optional[List[str]] = None  # ['findings', 'sessions', 'packs']

    # Phase 5: Meta-Learning Prediction Models
    class PredictionRequest(BaseModel):
        intent: str
        cognitive_state: Optional[dict] = None
        available_research: Optional[List[str]] = None
        track_prediction: bool = False  # Whether to store for calibration

    class ErrorPredictionRequest(BaseModel):
        intent: str
        include_preventable_only: bool = True

    class OptimalTimeRequest(BaseModel):
        intent: str
        current_hour: Optional[int] = None

    class PredictionOutcomeUpdate(BaseModel):
        prediction_id: str
        actual_quality: float
        actual_outcome: str
        session_id: str


# ============================================================
# Storage Engine (Phase 3a)
# ============================================================

_storage_engine: Optional['StorageEngine'] = None


async def get_storage():
    """Get initialized storage engine."""
    global _storage_engine
    if _storage_engine is None and STORAGE_AVAILABLE:
        _storage_engine = await get_engine()
    return _storage_engine


if FASTAPI_AVAILABLE:
    @app.on_event("startup")
    async def startup_event():
        """Initialize storage engine on startup."""
        if STORAGE_AVAILABLE:
            try:
                global _storage_engine
                _storage_engine = await get_engine()
                health = await _storage_engine.health_check()
                print(f"Storage engine initialized: SQLite=✓ Qdrant={'✓' if health.get('qdrant') else '✗'}")
            except Exception as e:
                print(f"Storage engine initialization warning: {e}")
                print("Falling back to file-based storage")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up storage engine."""
        if _storage_engine:
            await _storage_engine.close()


# ============================================================
# Helper Functions
# ============================================================

def load_json_file(path: Path) -> dict | list:
    """Safely load JSON file."""
    if not path.exists():
        return {} if path.suffix == ".json" else []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {} if path.suffix == ".json" else []


def get_session_metadata(session_id: str) -> dict:
    """Get session metadata from session.json."""
    session_dir = SESSIONS_DIR / session_id
    session_file = session_dir / "session.json"

    if not session_file.exists():
        return {"id": session_id}

    data = load_json_file(session_file)

    # Count URLs and findings
    urls_file = session_dir / "urls_captured.json"
    findings_file = session_dir / "findings_captured.json"

    urls = load_json_file(urls_file) if urls_file.exists() else []
    findings = load_json_file(findings_file) if findings_file.exists() else []

    return {
        "id": session_id,
        "topic": data.get("topic", session_id[:50]),
        "status": data.get("status", "archived"),
        "project": data.get("implementation_project"),
        "url_count": len(urls) if isinstance(urls, list) else 0,
        "finding_count": len(findings) if isinstance(findings, list) else 0,
        "created_at": data.get("started_at"),
    }


def get_evidenced_findings(session_id: str) -> list:
    """Get findings with evidence for a session."""
    session_dir = SESSIONS_DIR / session_id

    # Prefer evidenced findings
    evidenced_file = session_dir / "findings_evidenced.json"
    if evidenced_file.exists():
        return load_json_file(evidenced_file)

    # Fall back to regular findings
    findings_file = session_dir / "findings_captured.json"
    if findings_file.exists():
        findings = load_json_file(findings_file)
        # Convert to standard format
        return [
            {
                "id": f"finding-{i}-{session_id[:8]}",
                "session_id": session_id,
                "content": f.get("text", ""),
                "type": f.get("type", "finding"),
                "evidence": {"sources": [], "confidence": 0.0},
                "needs_review": True
            }
            for i, f in enumerate(findings)
        ]

    return []


# ============================================================
# API Endpoints
# ============================================================

if FASTAPI_AVAILABLE:
    @app.get("/")
    async def root():
        """API root - health check."""
        return {
            "service": "Agent Core API",
            "version": "2.1.0",
            "status": "healthy",
            "auth_enabled": AUTH_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }

    # ============================================================
    # Authentication Endpoints
    # ============================================================

    class TokenRequest(BaseModel):
        """Request for JWT token generation."""
        client_id: str
        scope: str = "read"  # read, write, admin

    class TokenResponse(BaseModel):
        """JWT token response."""
        access_token: str
        token_type: str = "bearer"
        expires_in: int = 86400  # 24 hours

    if AUTH_AVAILABLE:
        @app.post("/api/auth/token", response_model=TokenResponse)
        async def generate_token(request: TokenRequest):
            """
            Generate a JWT access token.

            Requires valid client credentials (set via RG_API_KEY env var for validation).
            For development, tokens are generated freely.

            Returns:
                JWT token valid for 24 hours
            """
            # In production, validate client_id against a database
            # For now, generate token for any valid request
            token_data = {
                "sub": request.client_id,
                "scope": request.scope,
                "type": "access_token"
            }

            token = create_access_token(token_data)

            return TokenResponse(
                access_token=token,
                token_type="bearer",
                expires_in=86400
            )

        @app.get("/api/auth/me")
        async def get_current_user_info(user: dict = Depends(get_current_user)):
            """
            Get information about the current authenticated user.

            Requires valid JWT token or API key.
            """
            return {
                "authenticated": True,
                "user": user
            }

    @app.get("/api/sessions", response_model=list[SessionSummary])
    async def list_sessions(
        limit: int = Query(20, ge=1, le=100),
        project: Optional[str] = None,
        status: Optional[str] = None
    ):
        """List all sessions with metadata."""
        if not SESSIONS_DIR.exists():
            return []

        sessions = []
        for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue

            metadata = get_session_metadata(session_dir.name)

            # Filter by project
            if project and metadata.get("project") != project:
                continue

            # Filter by status
            if status and metadata.get("status") != status:
                continue

            sessions.append(SessionSummary(**metadata))

            if len(sessions) >= limit:
                break

        return sessions

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get detailed session information."""
        # Validate session ID to prevent path traversal
        if SECURITY_AVAILABLE:
            session_id = validate_session_id(session_id)

        session_dir = SESSIONS_DIR / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        metadata = get_session_metadata(session_id)
        findings = get_evidenced_findings(session_id)
        urls = load_json_file(session_dir / "urls_captured.json")
        lineage = load_json_file(session_dir / "lineage.json")

        return {
            **metadata,
            "findings": findings,
            "urls": urls if isinstance(urls, list) else [],
            "lineage": lineage if isinstance(lineage, dict) else {},
        }

    @app.get("/api/findings", response_model=list[FindingResponse])
    async def search_findings(
        type: Optional[str] = None,
        project: Optional[str] = None,
        needs_review: Optional[bool] = None,
        limit: int = Query(50, ge=1, le=200)
    ):
        """Search findings across all sessions."""
        all_findings = []

        if not SESSIONS_DIR.exists():
            return []

        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue

            # Check project filter
            if project:
                session_data = load_json_file(session_dir / "session.json")
                if session_data.get("implementation_project") != project:
                    continue

            findings = get_evidenced_findings(session_dir.name)

            for f in findings:
                # Apply filters
                if type and f.get("type") != type:
                    continue
                if needs_review is not None and f.get("needs_review") != needs_review:
                    continue

                all_findings.append(FindingResponse(
                    id=f.get("id", "unknown"),
                    session_id=f.get("session_id", session_dir.name),
                    content=f.get("content", ""),
                    type=f.get("type", "finding"),
                    confidence=f.get("evidence", {}).get("confidence", 0.0),
                    sources=f.get("evidence", {}).get("sources", []),
                    needs_review=f.get("needs_review", False)
                ))

                if len(all_findings) >= limit:
                    return all_findings

        return all_findings

    @app.post("/api/findings")
    async def create_finding(finding: NewFinding):
        """Log a new finding (for real-time capture from OS-App/CareerCoach)."""
        # Get current session or create ad-hoc entry
        now = datetime.now()

        # Store in memory knowledge.json
        knowledge_file = MEMORY_DIR / "knowledge.json"
        knowledge = load_json_file(knowledge_file)

        if not isinstance(knowledge, dict):
            knowledge = {"facts": [], "decisions": [], "patterns": []}

        # Determine category
        category = "facts"
        if finding.type in ["decision", "choice"]:
            category = "decisions"
        elif finding.type in ["pattern", "observation"]:
            category = "patterns"

        entry = {
            "id": len(knowledge.get(category, [])),
            "content": finding.content,
            "type": finding.type,
            "tags": finding.tags,
            "source": finding.source_url,
            "project": finding.project,
            "timestamp": now.isoformat() + "Z"
        }

        if category not in knowledge:
            knowledge[category] = []
        knowledge[category].append(entry)

        knowledge_file.write_text(json.dumps(knowledge, indent=2))

        return {
            "status": "created",
            "id": entry["id"],
            "category": category
        }

    @app.post("/api/search/semantic", response_model=list[SearchResult])
    @rate_limit_search
    async def semantic_search(request: Request, search: SearchQuery):
        """Semantic search across knowledge base."""
        # Try to use the memory API if available
        try:
            sys.path.insert(0, str(Path.home() / ".claude" / "kernel"))
            from memory_api import VectorMemory
            mem = VectorMemory()

            results = mem.query(
                search.query,
                category=search.category,
                limit=search.limit,
                min_similarity=search.min_confidence
            )

            return [
                SearchResult(
                    content=r.get("content", ""),
                    category=r.get("category", "unknown"),
                    similarity=r.get("similarity", 0.0),
                    session_id=r.get("session_id"),
                    tags=r.get("tags", [])
                )
                for r in results
            ]
        except Exception:
            # Fallback to simple keyword search
            knowledge_file = MEMORY_DIR / "knowledge.json"
            knowledge = load_json_file(knowledge_file)

            results = []
            query_lower = search.query.lower()

            categories = [search.category] if search.category != "all" else ["facts", "decisions", "patterns"]

            for cat in categories:
                items = knowledge.get(cat, [])
                for item in items:
                    content = item.get("content", "").lower()
                    # Simple keyword matching
                    if query_lower in content or any(kw in content for kw in query_lower.split()):
                        results.append(SearchResult(
                            content=item.get("content", ""),
                            category=cat,
                            similarity=0.5,  # Fixed score for keyword match
                            tags=item.get("tags", [])
                        ))

                        if len(results) >= search.limit:
                            return results

            return results

    @app.get("/api/packs")
    async def list_packs():
        """List available context packs."""
        packs_dir = AGENT_CORE_DIR / "packs"
        if not packs_dir.exists():
            return []

        packs = []
        for pack_file in packs_dir.glob("*.json"):
            try:
                data = json.loads(pack_file.read_text())
                packs.append({
                    "id": pack_file.stem,
                    "type": data.get("type", "unknown"),
                    "tokens": data.get("tokens", 0),
                    "sessions": len(data.get("sessions", [])),
                    "created_at": data.get("created_at")
                })
            except:
                continue

        return packs

    @app.post("/api/packs/select")
    async def select_packs(selection: PackSelection):
        """Select relevant context packs for a session."""
        packs_dir = AGENT_CORE_DIR / "packs"
        if not packs_dir.exists():
            return {"packs": [], "total_tokens": 0}

        selected = []
        total_tokens = 0

        for pack_file in packs_dir.glob("*.json"):
            try:
                data = json.loads(pack_file.read_text())

                # Filter by project
                if selection.project:
                    pack_projects = data.get("projects", [])
                    if selection.project not in pack_projects and data.get("project") != selection.project:
                        continue

                # Filter by pattern
                if selection.pattern:
                    pack_patterns = data.get("patterns", [])
                    if selection.pattern not in pack_patterns and data.get("pattern") != selection.pattern:
                        continue

                tokens = data.get("tokens", 0)
                selected.append({
                    "id": pack_file.stem,
                    "type": data.get("type"),
                    "tokens": tokens,
                    "content": data.get("content", "")[:500] + "..."  # Preview
                })
                total_tokens += tokens

                if len(selected) >= selection.limit:
                    break
            except:
                continue

        return {
            "packs": selected,
            "total_tokens": total_tokens,
            "count": len(selected)
        }

    # ============================================================
    # Storage-Backed Endpoints (Phase 3a)
    # ============================================================

    @app.get("/api/v2/search")
    @rate_limit_search
    async def semantic_search_v2(
        request: Request,
        query: str = Query(..., min_length=1),
        limit: int = Query(10, ge=1, le=50),
        min_score: float = Query(0.4, ge=0.0, le=1.0),
        collections: Optional[str] = Query(None, description="Comma-separated: findings,sessions,packs")
    ):
        """
        Semantic search using vector embeddings (Qdrant).

        This endpoint uses the new storage engine for true semantic search,
        not just keyword matching. Requires Qdrant to be running.
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(
                status_code=503,
                detail="Storage engine not available. Run migration first."
            )

        coll_list = collections.split(",") if collections else None

        try:
            results = await storage.semantic_search(
                query=query,
                limit=limit,
                min_score=min_score,
                collections=coll_list
            )
            return {
                "query": query,
                "results": results,
                "engine": "qdrant" if storage._qdrant_enabled else "fts"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v2/findings/batch")
    @rate_limit_write
    async def store_findings_batch(
        http_request: Request,
        request: BulkFindingsRequest
    ):
        """
        Store multiple findings in a single transaction.

        Optimized for agent production - handles concurrent writes safely.
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(
                status_code=503,
                detail="Storage engine not available"
            )

        try:
            count = await storage.store_findings_batch(
                request.findings,
                source=request.source
            )
            return {
                "status": "success",
                "stored": count,
                "source": request.source
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v2/ucw/ingest")
    async def ingest_ucw_packs(request: UCWIngestRequest):
        """
        Ingest packs from a UCW (Universal Cognitive Wallet) trade.

        Handles:
        - Bulk import with transaction safety
        - Provenance tracking
        - Optional validation via Writer-Critic
        - Deduplication
        """
        if not STORAGE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Storage engine not available"
            )

        try:
            result = await ingest_ucw_trade(
                wallet_id=request.wallet_id,
                packs=request.packs,
                validate=request.validate
            )
            return result.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/stats")
    async def get_storage_stats():
        """
        Get storage statistics.

        Returns counts for sessions, findings, packs, and UCW imports.
        """
        storage = await get_storage()
        if not storage:
            # Return file-based stats as fallback
            sessions_count = len(list(SESSIONS_DIR.iterdir())) if SESSIONS_DIR.exists() else 0
            return {
                "engine": "file-based",
                "sessions": sessions_count,
                "storage_available": False
            }

        try:
            stats = await storage.get_stats()
            health = await storage.health_check()
            return {
                "engine": "storage-triad",
                "health": health,
                **stats
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/health")
    async def storage_health():
        """Check storage engine health."""
        storage = await get_storage()
        if not storage:
            return {
                "status": "degraded",
                "sqlite": False,
                "qdrant": False,
                "message": "Storage engine not initialized"
            }

        health = await storage.health_check()
        status = "healthy" if all(health.values()) else "degraded"

        return {
            "status": status,
            **health,
            "message": "Qdrant disabled" if not health.get("qdrant") else "All systems operational"
        }

    @app.post("/api/v2/sessions")
    async def store_session_v2(session: dict):
        """Store a session using the storage engine."""
        storage = await get_storage()
        if not storage:
            raise HTTPException(status_code=503, detail="Storage engine not available")

        try:
            session_id = await storage.store_session(session)
            return {"status": "created", "session_id": session_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/sessions")
    async def list_sessions_v2(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        project: Optional[str] = None
    ):
        """List sessions from storage engine."""
        storage = await get_storage()
        if not storage:
            # Fall back to file-based
            return await list_sessions(limit=limit, project=project)

        try:
            sessions = await storage.list_sessions(
                limit=limit,
                offset=offset,
                project=project
            )
            return sessions
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ============================================================
    # Graph Intelligence Endpoints (Phase 6)
    # ============================================================

    # Import graph module if available
    GRAPH_AVAILABLE = False
    try:
        from graph import ConceptGraph, get_related_sessions, get_research_lineage, get_concept_network
        GRAPH_AVAILABLE = True
    except ImportError:
        pass

    @app.get("/api/v2/graph/stats")
    async def graph_stats():
        """Get knowledge graph statistics."""
        if not GRAPH_AVAILABLE:
            raise HTTPException(status_code=503, detail="Graph module not available")

        graph = ConceptGraph()
        stats = await graph.get_stats()
        return stats

    @app.get("/api/v2/graph/session/{session_id}")
    async def get_session_graph(session_id: str, depth: int = Query(2, ge=1, le=4)):
        """
        Get the knowledge subgraph centered on a session.

        Returns D3.js compatible format for visualization.
        """
        # Validate session ID to prevent path traversal
        if SECURITY_AVAILABLE:
            session_id = validate_session_id(session_id)

        if not GRAPH_AVAILABLE:
            raise HTTPException(status_code=503, detail="Graph module not available")

        graph = ConceptGraph()
        subgraph = await graph.get_session_graph(session_id, depth=depth)
        return subgraph.to_d3_format()

    @app.get("/api/v2/graph/related/{session_id}")
    async def related_sessions(session_id: str, limit: int = Query(10, ge=1, le=50)):
        """
        Find sessions related to a given session.

        Uses shared sources and lineage connections.
        """
        # Validate session ID to prevent path traversal
        if SECURITY_AVAILABLE:
            session_id = validate_session_id(session_id)

        if not GRAPH_AVAILABLE:
            raise HTTPException(status_code=503, detail="Graph module not available")

        related = await get_related_sessions(session_id, limit=limit)
        return {"session_id": session_id, "related": related}

    @app.get("/api/v2/graph/lineage/{session_id}")
    async def session_lineage(session_id: str, include_urls: bool = True):
        """
        Get the complete research lineage for a session.

        Returns ancestors (what it builds on) and descendants (what builds on it).
        """
        # Validate session ID to prevent path traversal
        if SECURITY_AVAILABLE:
            session_id = validate_session_id(session_id)

        if not GRAPH_AVAILABLE:
            raise HTTPException(status_code=503, detail="Graph module not available")

        lineage = await get_research_lineage(session_id, include_urls=include_urls)
        return lineage

    @app.get("/api/v2/graph/clusters")
    async def concept_clusters(min_size: int = Query(5, ge=2, le=50)):
        """
        Find concept clusters (groups of related sessions/findings).

        Returns connected components analysis.
        """
        if not GRAPH_AVAILABLE:
            raise HTTPException(status_code=503, detail="Graph module not available")

        graph = ConceptGraph()
        clusters = await graph.get_concept_clusters(min_size=min_size)
        return {"clusters": clusters, "total": len(clusters)}

    @app.get("/api/v2/graph/timeline")
    async def research_timeline(
        project: Optional[str] = None,
        limit: int = Query(50, ge=1, le=200)
    ):
        """Get chronological research timeline with lineage links."""
        if not GRAPH_AVAILABLE:
            raise HTTPException(status_code=503, detail="Graph module not available")

        graph = ConceptGraph()
        timeline = await graph.get_research_timeline(project=project, limit=limit)
        return {"timeline": timeline, "count": len(timeline)}

    @app.get("/api/v2/graph/network/{node_id}")
    async def concept_network(node_id: str, depth: int = Query(2, ge=1, le=4)):
        """
        Get a concept network centered on any node.

        node_id format: "session:xxx" or "finding:xxx" or just session_id
        """
        if not GRAPH_AVAILABLE:
            raise HTTPException(status_code=503, detail="Graph module not available")

        network = await get_concept_network(node_id, depth=depth)
        return network

    # ============================================================
    # Phase 5: Meta-Learning Prediction Endpoints
    # ============================================================

    @app.post("/api/v2/predict/session")
    @rate_limit_search  # Expensive computation, treat like search
    async def predict_session_outcome(http_request: Request, request: PredictionRequest):
        """
        Predict session outcome based on multi-dimensional correlation.

        Uses historical outcomes, cognitive states, research context, and error patterns
        to predict quality, success probability, and optimal timing.

        Returns:
            {
                "predicted_quality": float (1-5),
                "success_probability": float (0-1),
                "optimal_time": int (hour 0-23),
                "recommended_research": List[Dict],
                "potential_errors": List[Dict],
                "similar_sessions": List[Dict],
                "confidence": float (0-1),
                "signals": Dict,
                "prediction_id": str (if track_prediction=True)
            }
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(status_code=503, detail="Storage engine not available")

        try:
            # Import meta-learning engine
            from storage.meta_learning import MetaLearningEngine

            # Initialize engine
            engine = MetaLearningEngine()
            await engine.initialize()

            # Make prediction
            prediction = await engine.predict_session_outcome(
                intent=request.intent,
                cognitive_state=request.cognitive_state,
                available_research=request.available_research
            )

            # Optionally store for tracking
            prediction_id = None
            if request.track_prediction:
                prediction_id = await engine.store_prediction_for_tracking(
                    intent=request.intent,
                    prediction=prediction,
                    cognitive_state=request.cognitive_state
                )
                prediction["prediction_id"] = prediction_id

            await engine.close()
            return prediction

        except Exception as e:
            import traceback
            print(f"ERROR in predict_session_outcome: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.post("/api/v2/predict/errors")
    async def predict_errors(request: ErrorPredictionRequest):
        """
        Predict potential errors for a task.

        Searches error patterns database for relevant errors with prevention strategies.

        Returns:
            List of error patterns with:
            - error_type: str
            - context: str
            - solution: str
            - success_rate: float (prevention effectiveness)
            - severity: str ('high' or 'medium')
            - score: float (relevance to query)
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(status_code=503, detail="Storage engine not available")

        try:
            from storage.meta_learning import MetaLearningEngine

            engine = MetaLearningEngine()
            await engine.initialize()

            errors = await engine.predict_errors(
                intent=request.intent,
                include_preventable_only=request.include_preventable_only
            )

            await engine.close()
            return {"errors": errors, "count": len(errors)}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error prediction failed: {str(e)}")

    @app.post("/api/v2/predict/optimal-time")
    async def predict_optimal_time(request: OptimalTimeRequest):
        """
        Predict the optimal time to work on a task.

        Analyzes historical cognitive patterns and session outcomes to suggest
        the best time of day for the given task.

        Returns:
            {
                "optimal_hour": int (0-23),
                "is_optimal_now": bool,
                "wait_hours": int,
                "reasoning": str
            }
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(status_code=503, detail="Storage engine not available")

        try:
            from storage.meta_learning import MetaLearningEngine

            engine = MetaLearningEngine()
            await engine.initialize()

            result = await engine.predict_optimal_time(
                intent=request.intent,
                current_hour=request.current_hour
            )

            await engine.close()
            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Optimal time prediction failed: {str(e)}")

    @app.get("/api/v2/predict/accuracy")
    async def get_prediction_accuracy(days: int = Query(30, ge=1, le=365)):
        """
        Get prediction accuracy metrics.

        Analyzes tracked predictions vs actual outcomes to calculate
        calibration metrics.

        Returns:
            {
                "total_predictions": int,
                "accurate_predictions": int,
                "accuracy": float (0-1),
                "avg_quality_error": float,
                "success_prediction_rate": float,
                "period_days": int
            }
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(status_code=503, detail="Storage engine not available")

        try:
            from storage.meta_learning import MetaLearningEngine

            engine = MetaLearningEngine()
            await engine.initialize()

            accuracy = await engine.get_prediction_accuracy(days=days)

            await engine.close()
            return accuracy

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Accuracy calculation failed: {str(e)}")

    @app.post("/api/v2/predict/update-outcome")
    async def update_prediction_outcome(request: PredictionOutcomeUpdate):
        """
        Update a tracked prediction with actual outcome for calibration.

        Used to close the feedback loop: after a session completes,
        update the stored prediction with actual results.

        Returns:
            {"status": "updated", "prediction_id": str}
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(status_code=503, detail="Storage engine not available")

        try:
            from storage.meta_learning import MetaLearningEngine

            engine = MetaLearningEngine()
            await engine.initialize()

            await engine.update_prediction_with_outcome(
                prediction_id=request.prediction_id,
                actual_quality=request.actual_quality,
                actual_outcome=request.actual_outcome,
                session_id=request.session_id
            )

            await engine.close()
            return {"status": "updated", "prediction_id": request.prediction_id}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Outcome update failed: {str(e)}")

    @app.get("/api/v2/predict/multi-search")
    async def multi_vector_search(
        query: str = Query(..., description="Search query"),
        limit: int = Query(5, ge=1, le=20)
    ):
        """
        Perform multi-dimensional vector search across all dimensions.

        Searches outcomes, cognitive states, research findings, and error patterns
        in parallel for comprehensive context.

        Returns:
            {
                "outcomes": List[Dict],
                "cognitive": List[Dict],
                "research": List[Dict],
                "errors": List[Dict],
                "total_results": int
            }
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(status_code=503, detail="Storage engine not available")

        try:
            from storage.meta_learning import MetaLearningEngine

            engine = MetaLearningEngine()
            await engine.initialize()

            results = await engine.multi_vector_search(query=query, limit=limit)

            await engine.close()
            return results

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Multi-search failed: {str(e)}")

    @app.get("/api/v2/predict/calibrate-weights")
    async def calibrate_weights():
        """
        Get recommended correlation weights based on prediction accuracy.

        Analyzes recent performance to suggest optimal weighting for:
        - Outcome signal
        - Cognitive alignment
        - Research availability
        - Error probability

        Returns:
            {
                "outcome_weight": float,
                "cognitive_weight": float,
                "research_weight": float,
                "error_weight": float,
                "recommended_update": bool
            }
        """
        storage = await get_storage()
        if not storage:
            raise HTTPException(status_code=503, detail="Storage engine not available")

        try:
            from storage.meta_learning import MetaLearningEngine

            engine = MetaLearningEngine()
            await engine.initialize()

            weights = await engine.calibrate_weights()

            await engine.close()
            return weights

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

    # ============================================================
    # Original Endpoints (backward compatibility)
    # ============================================================

    @app.get("/api/reinvigorate/{session_id}")
    async def get_reinvigoration_context(session_id: str):
        """Get full context for session reinvigoration."""
        # Validate session ID to prevent path traversal
        if SECURITY_AVAILABLE:
            session_id = validate_session_id(session_id)

        session_dir = SESSIONS_DIR / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        # Load all session data
        metadata = get_session_metadata(session_id)
        findings = get_evidenced_findings(session_id)
        urls = load_json_file(session_dir / "urls_captured.json")
        lineage = load_json_file(session_dir / "lineage.json")

        # Load transcript excerpt
        transcript_file = session_dir / "full_transcript.txt"
        transcript_excerpt = ""
        if transcript_file.exists():
            content = transcript_file.read_text()
            # Get last 5000 chars as recent context
            transcript_excerpt = content[-5000:] if len(content) > 5000 else content

        # Build reinvigoration context
        context = f"""## SESSION REINVIGORATION: {session_id}

### Session Info
- Topic: {metadata.get('topic', 'Unknown')}
- Project: {metadata.get('project', 'None')}
- URLs: {metadata.get('url_count', 0)}
- Findings: {metadata.get('finding_count', 0)}

### Key Findings ({len(findings)} total)
"""
        for f in findings[:10]:
            confidence = f.get("evidence", {}).get("confidence", 0)
            context += f"- [{f.get('type', 'finding')}] {f.get('content', '')[:200]}... (conf: {confidence:.2f})\n"

        context += f"""

### URLs Captured ({len(urls)} total)
"""
        for u in urls[:10] if isinstance(urls, list) else []:
            context += f"- Tier {u.get('tier', 3)}: {u.get('url', '')[:80]}\n"

        if transcript_excerpt:
            context += f"""

### Recent Transcript
```
{transcript_excerpt[-2000:]}
```
"""

        return {
            "session_id": session_id,
            "metadata": metadata,
            "findings_count": len(findings),
            "urls_count": len(urls) if isinstance(urls, list) else 0,
            "context_block": context,
            "lineage": lineage
        }

    # ============================================================
    # Graph Intelligence Endpoints
    # ============================================================

    @app.get("/api/graph/concepts")
    async def get_related_concepts(
        query: str = Query(..., description="Concept to find related items for"),
        depth: int = Query(2, ge=1, le=3, description="Relationship depth"),
        limit: int = Query(20, ge=1, le=50, description="Max results")
    ):
        """
        Find concepts related to a query term.

        Searches across:
        - Session topics
        - Finding types and content
        - URL categories
        - Paper keywords

        Returns a graph of related concepts with relationship types.
        """
        concepts = []
        edges = []
        seen_concepts = set()

        query_lower = query.lower()

        if not SESSIONS_DIR.exists():
            return {"query": query, "concepts": [], "edges": []}

        # Scan sessions for related concepts
        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue

            # Load session data
            session_data = load_json_file(session_dir / "session.json")
            topic = session_data.get("topic", "").lower()
            session_id = session_dir.name

            # Check if session topic matches query
            if query_lower in topic or any(w in topic for w in query_lower.split()):
                if session_id not in seen_concepts:
                    seen_concepts.add(session_id)
                    concepts.append({
                        "id": session_id,
                        "label": session_data.get("topic", session_id[:30]),
                        "type": "session",
                        "relevance": 0.8
                    })
                    edges.append({
                        "source": query,
                        "target": session_id,
                        "relation": "researched_in"
                    })

            # Check findings
            findings = get_evidenced_findings(session_id)
            for f in findings:
                content = f.get("content", "").lower()
                finding_type = f.get("type", "finding")

                if query_lower in content:
                    finding_id = f.get("id", f"finding-{len(concepts)}")
                    if finding_id not in seen_concepts:
                        seen_concepts.add(finding_id)
                        concepts.append({
                            "id": finding_id,
                            "label": f.get("content", "")[:50] + "...",
                            "type": finding_type,
                            "session_id": session_id,
                            "relevance": f.get("evidence", {}).get("confidence", 0.5)
                        })
                        edges.append({
                            "source": query,
                            "target": finding_id,
                            "relation": "found_as"
                        })

                        # Connect finding to session
                        if session_id in seen_concepts:
                            edges.append({
                                "source": session_id,
                                "target": finding_id,
                                "relation": "contains"
                            })

            # Check URLs for paper references
            urls = load_json_file(session_dir / "urls_captured.json")
            if isinstance(urls, list):
                for u in urls:
                    url = u.get("url", "")
                    if "arxiv.org" in url:
                        # Extract arXiv ID
                        import re
                        match = re.search(r'(\d{4}\.\d{4,5})', url)
                        if match:
                            arxiv_id = match.group(1)
                            paper_id = f"paper-{arxiv_id}"
                            if paper_id not in seen_concepts:
                                seen_concepts.add(paper_id)
                                concepts.append({
                                    "id": paper_id,
                                    "label": f"arXiv:{arxiv_id}",
                                    "type": "paper",
                                    "url": url,
                                    "relevance": 0.7
                                })
                                edges.append({
                                    "source": session_id,
                                    "target": paper_id,
                                    "relation": "cites"
                                })

            if len(concepts) >= limit:
                break

        return {
            "query": query,
            "concepts": concepts[:limit],
            "edges": edges,
            "depth": depth
        }

    @app.get("/api/graph/lineage/{session_id}")
    async def get_session_lineage(
        session_id: str,
        include_findings: bool = Query(True, description="Include findings as nodes"),
        include_papers: bool = Query(True, description="Include cited papers")
    ):
        """
        Get the research lineage graph for a session.

        Returns nodes (session, findings, papers, concepts) and edges
        showing relationships between them.
        """
        session_dir = SESSIONS_DIR / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        nodes = []
        edges = []

        # Root node: the session
        session_data = load_json_file(session_dir / "session.json")
        nodes.append({
            "id": session_id,
            "label": session_data.get("topic", session_id[:30]),
            "type": "session",
            "isRoot": True
        })

        # Add findings
        if include_findings:
            findings = get_evidenced_findings(session_id)
            for f in findings:
                finding_id = f.get("id", f"finding-{len(nodes)}")
                nodes.append({
                    "id": finding_id,
                    "label": f.get("content", "")[:50] + "...",
                    "type": f.get("type", "finding"),
                    "confidence": f.get("evidence", {}).get("confidence", 0.0)
                })
                edges.append({
                    "source": session_id,
                    "target": finding_id,
                    "relation": "produced"
                })

                # Add sources as edges
                sources = f.get("evidence", {}).get("sources", [])
                for s in sources:
                    if "arxiv_id" in s and include_papers:
                        paper_id = f"paper-{s['arxiv_id']}"
                        # Check if paper node exists
                        if not any(n["id"] == paper_id for n in nodes):
                            nodes.append({
                                "id": paper_id,
                                "label": f"arXiv:{s['arxiv_id']}",
                                "type": "paper",
                                "url": s.get("url")
                            })
                        edges.append({
                            "source": finding_id,
                            "target": paper_id,
                            "relation": "cites"
                        })

        # Add papers from URLs
        if include_papers:
            urls = load_json_file(session_dir / "urls_captured.json")
            if isinstance(urls, list):
                import re
                for u in urls:
                    url = u.get("url", "")
                    if "arxiv.org" in url:
                        match = re.search(r'(\d{4}\.\d{4,5})', url)
                        if match:
                            paper_id = f"paper-{match.group(1)}"
                            if not any(n["id"] == paper_id for n in nodes):
                                nodes.append({
                                    "id": paper_id,
                                    "label": f"arXiv:{match.group(1)}",
                                    "type": "paper",
                                    "url": url,
                                    "tier": u.get("tier", 3)
                                })
                                edges.append({
                                    "source": session_id,
                                    "target": paper_id,
                                    "relation": "references"
                                })

        # Load existing lineage if available
        lineage = load_json_file(session_dir / "lineage.json")
        if lineage:
            # Add lineage connections
            for parent in lineage.get("parents", []):
                edges.append({
                    "source": parent,
                    "target": session_id,
                    "relation": "builds_on"
                })
            for child in lineage.get("children", []):
                edges.append({
                    "source": session_id,
                    "target": child,
                    "relation": "enables"
                })

        return {
            "session_id": session_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }

    @app.get("/api/graph/sessions")
    async def get_sessions_graph(
        limit: int = Query(30, ge=1, le=100),
        project: Optional[str] = None
    ):
        """
        Get a graph of all sessions with connections.

        Connections are based on:
        - Shared topics/keywords
        - Shared paper citations
        - Explicit lineage links
        """
        nodes = []
        edges = []
        paper_sessions = {}  # Track which sessions cite which papers

        if not SESSIONS_DIR.exists():
            return {"nodes": [], "edges": []}

        # First pass: collect sessions and paper citations
        for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True)[:limit]:
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            session_data = load_json_file(session_dir / "session.json")

            # Filter by project
            if project and session_data.get("implementation_project") != project:
                continue

            topic = session_data.get("topic", session_id[:30])

            # Add session node
            nodes.append({
                "id": session_id,
                "label": topic[:40],
                "type": "session",
                "project": session_data.get("implementation_project"),
                "status": session_data.get("status", "archived")
            })

            # Track paper citations
            urls = load_json_file(session_dir / "urls_captured.json")
            if isinstance(urls, list):
                import re
                for u in urls:
                    url = u.get("url", "")
                    if "arxiv.org" in url:
                        match = re.search(r'(\d{4}\.\d{4,5})', url)
                        if match:
                            paper_id = match.group(1)
                            if paper_id not in paper_sessions:
                                paper_sessions[paper_id] = []
                            paper_sessions[paper_id].append(session_id)

            # Check lineage
            lineage = load_json_file(session_dir / "lineage.json")
            if lineage:
                for parent in lineage.get("parents", []):
                    edges.append({
                        "source": parent,
                        "target": session_id,
                        "relation": "builds_on"
                    })

        # Second pass: create edges for shared papers
        for paper_id, sessions in paper_sessions.items():
            if len(sessions) > 1:
                # Create edges between sessions that share this paper
                for i, s1 in enumerate(sessions):
                    for s2 in sessions[i+1:]:
                        edges.append({
                            "source": s1,
                            "target": s2,
                            "relation": "shares_reference",
                            "paper": paper_id
                        })

        return {
            "nodes": nodes,
            "edges": edges,
            "shared_papers": len([p for p, s in paper_sessions.items() if len(s) > 1])
        }


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent Core API Server")
    parser.add_argument("--port", type=int, default=3847, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    import uvicorn
    print(f"Starting Agent Core API on http://{args.host}:{args.port}")
    print("Docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
