#!/usr/bin/env python3
"""
CPB Precision Mode - ResearchGravity Adapter

Provides connection to ResearchGravity's knowledge base with graceful degradation:
1. MCP Server (preferred) - Full semantic search capability
2. REST API (port 3847) - API access to stored knowledge
3. Direct file access (~/.agent-core/) - Fallback to files
4. Degraded mode - Proceed with warning

Features:
- Context enrichment from learnings, packs, and research index
- Semantic search over research sessions
- Citation lookup and verification
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Optional aiohttp import
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

from .precision_config import PRECISION_CONFIG

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# CONNECTION MODES
# =============================================================================

class ConnectionMode(str, Enum):
    """Available connection modes to ResearchGravity."""
    MCP = 'mcp'              # MCP Server (preferred)
    REST = 'rest'            # REST API
    FILE = 'file'            # Direct file access
    DEGRADED = 'degraded'    # No connection


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
HOME = Path.home()
AGENT_CORE_DIR = HOME / ".agent-core"
SESSIONS_DIR = AGENT_CORE_DIR / "sessions"
CONTEXT_PACKS_DIR = AGENT_CORE_DIR / "context-packs"
LEARNINGS_FILE = AGENT_CORE_DIR / "memory" / "learnings.md"
CONFIG_FILE = AGENT_CORE_DIR / "config.json"

# API endpoints
RG_API_BASE = "http://localhost:3847"
MCP_ENDPOINT = "http://localhost:3847/mcp"

# Timeouts
MCP_TIMEOUT = 5.0
REST_TIMEOUT = 10.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RGContext:
    """Context retrieved from ResearchGravity."""
    learnings: str = ""
    packs: List[Dict[str, Any]] = field(default_factory=list)
    research_index: Dict[str, Any] = field(default_factory=dict)
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    connection_mode: ConnectionMode = ConnectionMode.DEGRADED
    warnings: List[str] = field(default_factory=list)

    def to_enriched_context(self, budget: int = 15000) -> str:
        """
        Combine all context into enriched string within budget.

        Args:
            budget: Maximum characters for context

        Returns:
            Formatted context string
        """
        parts = []
        remaining = budget

        # Add learnings first (most valuable)
        if self.learnings and remaining > 0:
            learnings_truncated = self.learnings[:min(len(self.learnings), remaining // 2)]
            parts.append(f"## Recent Learnings\n{learnings_truncated}")
            remaining -= len(learnings_truncated)

        # Add relevant sessions
        if self.sessions and remaining > 1000:
            session_text = self._format_sessions(remaining // 3)
            if session_text:
                parts.append(f"## Research Sessions\n{session_text}")
                remaining -= len(session_text)

        # Add context packs
        if self.packs and remaining > 500:
            pack_text = self._format_packs(remaining // 4)
            if pack_text:
                parts.append(f"## Context Packs\n{pack_text}")

        return "\n\n".join(parts) if parts else ""

    def _format_sessions(self, budget: int) -> str:
        """Format sessions within budget."""
        lines = []
        remaining = budget

        for session in self.sessions[:5]:  # Max 5 sessions
            session_id = session.get('id', 'unknown')
            topic = session.get('topic', session.get('title', 'Untitled'))
            findings = session.get('findings', [])

            line = f"- [{session_id}] {topic}"
            if findings:
                top_findings = findings[:3]
                for f in top_findings:
                    content = f.get('content', '')[:100]
                    line += f"\n  - {content}..."

            if len(line) > remaining:
                break

            lines.append(line)
            remaining -= len(line)

        return "\n".join(lines)

    def _format_packs(self, budget: int) -> str:
        """Format context packs within budget."""
        lines = []
        remaining = budget

        for pack in self.packs[:3]:  # Max 3 packs
            name = pack.get('name', 'unnamed')
            description = pack.get('description', '')[:100]
            line = f"- **{name}**: {description}"

            if len(line) > remaining:
                break

            lines.append(line)
            remaining -= len(line)

        return "\n".join(lines)


@dataclass
class SearchResult:
    """Result from semantic search."""
    content: str
    source: str
    relevance_score: float
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RESEARCHGRAVITY ADAPTER
# =============================================================================

class RGAdapter:
    """
    Adapter for connecting to ResearchGravity's knowledge base.

    Implements graceful degradation across connection modes:
    1. MCP Server (preferred)
    2. REST API (port 3847)
    3. Direct file access
    4. Degraded mode (proceed with warning)
    """

    def __init__(self):
        self._connection_mode: Optional[ConnectionMode] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._config: Dict[str, Any] = {}

    async def initialize(self) -> ConnectionMode:
        """
        Initialize adapter and determine best connection mode.

        Returns:
            The connection mode being used
        """
        # Try modes in order of preference
        mode = await self._try_mcp()
        if mode:
            self._connection_mode = ConnectionMode.MCP
            logger.info("Connected via MCP server")
            return self._connection_mode

        mode = await self._try_rest()
        if mode:
            self._connection_mode = ConnectionMode.REST
            logger.info("Connected via REST API")
            return self._connection_mode

        if self._check_file_access():
            self._connection_mode = ConnectionMode.FILE
            logger.info("Using direct file access")
            return self._connection_mode

        self._connection_mode = ConnectionMode.DEGRADED
        logger.warning("Operating in degraded mode - no RG connection")
        return self._connection_mode

    async def _try_mcp(self) -> bool:
        """Try to connect via MCP server."""
        if not HAS_AIOHTTP:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{RG_API_BASE}/",
                    timeout=aiohttp.ClientTimeout(total=MCP_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if MCP endpoints are available
                        return data.get('status') == 'ok' or 'version' in data
        except Exception as e:
            logger.debug(f"MCP connection failed: {e}")
        return False

    async def _try_rest(self) -> bool:
        """Try to connect via REST API."""
        if not HAS_AIOHTTP:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{RG_API_BASE}/api/sessions",
                    timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"REST connection failed: {e}")
        return False

    def _check_file_access(self) -> bool:
        """Check if file access is available."""
        return AGENT_CORE_DIR.exists() and SESSIONS_DIR.exists()

    async def close(self):
        """Close any open connections."""
        if self._session:
            await self._session.close()
            self._session = None

    # =========================================================================
    # CONTEXT RETRIEVAL
    # =========================================================================

    async def get_context(
        self,
        query: str,
        limit: int = 10,
        include_learnings: bool = True,
        include_packs: bool = True,
        include_sessions: bool = True
    ) -> RGContext:
        """
        Get enriched context from ResearchGravity for a query.

        Args:
            query: Query to contextualize
            limit: Maximum items to retrieve
            include_learnings: Include learnings.md
            include_packs: Include context packs
            include_sessions: Include relevant sessions

        Returns:
            RGContext with all available context
        """
        if self._connection_mode is None:
            await self.initialize()

        context = RGContext(connection_mode=self._connection_mode)

        if self._connection_mode == ConnectionMode.DEGRADED:
            context.warnings.append("Operating in degraded mode - no RG context available")
            return context

        # Gather context based on connection mode
        tasks = []

        if include_learnings:
            tasks.append(self._get_learnings())

        if include_packs:
            tasks.append(self._get_relevant_packs(query, limit))

        if include_sessions:
            tasks.append(self._search_sessions(query, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        idx = 0
        if include_learnings:
            if isinstance(results[idx], str):
                context.learnings = results[idx]
            elif isinstance(results[idx], Exception):
                context.warnings.append(f"Failed to load learnings: {results[idx]}")
            idx += 1

        if include_packs:
            if isinstance(results[idx], list):
                context.packs = results[idx]
            elif isinstance(results[idx], Exception):
                context.warnings.append(f"Failed to load packs: {results[idx]}")
            idx += 1

        if include_sessions:
            if isinstance(results[idx], list):
                context.sessions = results[idx]
            elif isinstance(results[idx], Exception):
                context.warnings.append(f"Failed to search sessions: {results[idx]}")

        return context

    async def _get_learnings(self) -> str:
        """Get learnings content."""
        if self._connection_mode in (ConnectionMode.FILE, ConnectionMode.DEGRADED):
            return self._get_learnings_from_file()

        # Try API first
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{RG_API_BASE}/api/learnings",
                    timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('content', '')
        except Exception:
            pass

        # Fall back to file
        return self._get_learnings_from_file()

    def _get_learnings_from_file(self) -> str:
        """Get learnings from file."""
        if LEARNINGS_FILE.exists():
            try:
                return LEARNINGS_FILE.read_text()[:20000]  # Limit size
            except Exception as e:
                logger.warning(f"Failed to read learnings file: {e}")
        return ""

    async def _get_relevant_packs(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Get context packs relevant to query."""
        if self._connection_mode in (ConnectionMode.FILE, ConnectionMode.DEGRADED):
            return self._get_packs_from_file(query, limit)

        # Try API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{RG_API_BASE}/api/packs/select",
                    json={"query": query, "limit": limit},
                    timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('packs', [])
        except Exception:
            pass

        return self._get_packs_from_file(query, limit)

    def _get_packs_from_file(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Get context packs from file system."""
        packs = []

        if not CONTEXT_PACKS_DIR.exists():
            return packs

        query_lower = query.lower()

        for pack_file in CONTEXT_PACKS_DIR.glob("*.json"):
            try:
                with open(pack_file) as f:
                    pack = json.load(f)

                # Simple relevance check
                name = pack.get('name', '').lower()
                description = pack.get('description', '').lower()
                tags = ' '.join(pack.get('tags', [])).lower()

                if any(word in name or word in description or word in tags
                       for word in query_lower.split() if len(word) > 3):
                    packs.append(pack)

                if len(packs) >= limit:
                    break

            except Exception as e:
                logger.debug(f"Failed to read pack {pack_file}: {e}")

        return packs

    async def _search_sessions(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for relevant sessions."""
        if self._connection_mode == ConnectionMode.MCP:
            return await self._search_sessions_mcp(query, limit)
        elif self._connection_mode == ConnectionMode.REST:
            return await self._search_sessions_rest(query, limit)
        else:
            return self._search_sessions_file(query, limit)

    async def _search_sessions_mcp(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search sessions via MCP (semantic search)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{RG_API_BASE}/api/search/semantic",
                    json={"query": query, "limit": limit, "collection": "sessions"},
                    timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [])
        except Exception as e:
            logger.debug(f"MCP session search failed: {e}")

        return await self._search_sessions_rest(query, limit)

    async def _search_sessions_rest(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search sessions via REST API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{RG_API_BASE}/api/sessions",
                    params={"search": query, "limit": limit},
                    timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('sessions', data) if isinstance(data, dict) else data
        except Exception as e:
            logger.debug(f"REST session search failed: {e}")

        return self._search_sessions_file(query, limit)

    def _search_sessions_file(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search sessions via direct file access."""
        sessions = []

        if not SESSIONS_DIR.exists():
            return sessions

        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 3]

        for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue

            metadata_file = session_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Check relevance
                topic = metadata.get('topic', '').lower()
                tags = ' '.join(metadata.get('tags', [])).lower()

                if any(word in topic or word in tags for word in query_words):
                    session_data = {
                        'id': session_dir.name,
                        **metadata
                    }

                    # Load findings if available
                    findings_file = session_dir / "findings_captured.json"
                    if findings_file.exists():
                        try:
                            with open(findings_file) as f:
                                findings = json.load(f)
                                session_data['findings'] = findings if isinstance(findings, list) else findings.get('findings', [])
                        except Exception:
                            pass

                    sessions.append(session_data)

                    if len(sessions) >= limit:
                        break

            except Exception as e:
                logger.debug(f"Failed to read session {session_dir}: {e}")

        return sessions

    # =========================================================================
    # SEMANTIC SEARCH
    # =========================================================================

    async def search_learnings(
        self,
        query: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search learnings for relevant content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of search results
        """
        results = []

        if self._connection_mode in (ConnectionMode.MCP, ConnectionMode.REST):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{RG_API_BASE}/api/search/semantic",
                        json={"query": query, "limit": limit, "collection": "learnings"},
                        timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            for item in data.get('results', []):
                                results.append(SearchResult(
                                    content=item.get('content', ''),
                                    source=item.get('source', 'learnings'),
                                    relevance_score=item.get('score', item.get('relevance_score', 0.5)),
                                    session_id=item.get('session_id'),
                                    metadata=item.get('metadata', {})
                                ))
                            return results
            except Exception as e:
                logger.debug(f"Semantic search failed: {e}")

        # Fallback to simple search
        learnings = self._get_learnings_from_file()
        if learnings:
            # Simple keyword matching
            query_words = [w.lower() for w in query.split() if len(w) > 3]

            for line in learnings.split('\n'):
                if any(word in line.lower() for word in query_words):
                    results.append(SearchResult(
                        content=line,
                        source='learnings.md',
                        relevance_score=0.5
                    ))
                    if len(results) >= limit:
                        break

        return results

    async def select_context_packs(
        self,
        query: str,
        budget: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Select context packs within budget.

        Args:
            query: Query to contextualize
            budget: Maximum characters for pack content

        Returns:
            List of selected packs
        """
        packs = await self._get_relevant_packs(query, limit=10)

        # Select within budget
        selected = []
        remaining = budget

        for pack in packs:
            pack_size = len(json.dumps(pack))
            if pack_size <= remaining:
                selected.append(pack)
                remaining -= pack_size
            else:
                break

        return selected

    # =========================================================================
    # CITATION HELPERS
    # =========================================================================

    async def get_source_by_id(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a source/citation by ID.

        Args:
            source_id: Source identifier (arXiv ID, session ID, etc.)

        Returns:
            Source data if found
        """
        # Check if it's an arXiv ID
        if source_id.startswith('arXiv:') or '.' in source_id:
            return await self._get_arxiv_source(source_id)

        # Check if it's a session ID
        session_dir = SESSIONS_DIR / source_id
        if session_dir.exists():
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        return json.load(f)
                except Exception:
                    pass

        return None

    async def _get_arxiv_source(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get arXiv paper info."""
        # Clean up ID
        arxiv_id = arxiv_id.replace('arXiv:', '').strip()

        # Try to find in logged URLs
        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue

            urls_file = session_dir / "urls_captured.json"
            if urls_file.exists():
                try:
                    with open(urls_file) as f:
                        urls = json.load(f)
                        urls = urls if isinstance(urls, list) else urls.get('urls', [])

                        for url_entry in urls:
                            url = url_entry.get('url', '') if isinstance(url_entry, dict) else str(url_entry)
                            if arxiv_id in url:
                                return {
                                    'arxiv_id': arxiv_id,
                                    'url': url,
                                    'session': session_dir.name,
                                    **({k: v for k, v in url_entry.items() if k != 'url'} if isinstance(url_entry, dict) else {})
                                }
                except Exception:
                    pass

        # Return basic info
        return {
            'arxiv_id': arxiv_id,
            'url': f'https://arxiv.org/abs/{arxiv_id}'
        }

    # =========================================================================
    # STATUS & INFO
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            'connection_mode': self._connection_mode.value if self._connection_mode else 'uninitialized',
            'agent_core_exists': AGENT_CORE_DIR.exists(),
            'sessions_count': len(list(SESSIONS_DIR.iterdir())) if SESSIONS_DIR.exists() else 0,
            'learnings_exists': LEARNINGS_FILE.exists(),
            'packs_count': len(list(CONTEXT_PACKS_DIR.glob("*.json"))) if CONTEXT_PACKS_DIR.exists() else 0,
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

rg_adapter = RGAdapter()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def get_context(query: str, **kwargs) -> RGContext:
    """Get context from ResearchGravity."""
    return await rg_adapter.get_context(query, **kwargs)


async def search_learnings(query: str, limit: int = 10) -> List[SearchResult]:
    """Search learnings."""
    return await rg_adapter.search_learnings(query, limit)


async def select_context_packs(query: str, budget: int = 10000) -> List[Dict[str, Any]]:
    """Select context packs within budget."""
    return await rg_adapter.select_context_packs(query, budget)
