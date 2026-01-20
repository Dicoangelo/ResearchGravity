#!/usr/bin/env python3
"""
Agent Core API Server - FastAPI implementation.

Provides REST endpoints for:
- Session management
- Finding queries with evidence
- Semantic search
- Context pack selection
- Reinvigoration support

Usage:
    python3 -m api.server --port 3847
    uvicorn api.server:app --port 3847 --reload
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        version="1.0.0",
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            "version": "1.0.0",
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
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
    async def semantic_search(search: SearchQuery):
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
        except Exception as e:
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

    @app.get("/api/reinvigorate/{session_id}")
    async def get_reinvigoration_context(session_id: str):
        """Get full context for session reinvigoration."""
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
