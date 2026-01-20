"""
Storage Triad - Phase 3a Implementation

Provides concurrent-safe storage for the Antigravity Chief of Staff:
- SQLite: Relational data (sessions, findings, packs, provenance)
- Qdrant: Vector embeddings for semantic search

Usage:
    from storage import StorageEngine

    engine = StorageEngine()
    await engine.initialize()

    # Store a session
    await engine.store_session(session_data)

    # Semantic search
    results = await engine.semantic_search("multi-agent orchestration", limit=10)

    # Bulk pack import (UCW)
    await engine.ingest_packs(packs, source="ucw_trade")
"""

from .sqlite_db import SQLiteDB, get_db
from .qdrant_db import QdrantDB, get_qdrant
from .engine import StorageEngine, get_engine
from .migrate import migrate_from_json

__all__ = [
    'SQLiteDB',
    'QdrantDB',
    'StorageEngine',
    'get_db',
    'get_qdrant',
    'get_engine',
    'migrate_from_json',
]
