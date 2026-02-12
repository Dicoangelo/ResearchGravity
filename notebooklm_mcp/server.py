"""
NotebookLM MCP Server — HTTP/RPC API + Cognitive Intelligence + UCW Capture

Extends the raw MCP server pattern with:
  - HTTP/RPC client for NotebookLM batchexecute API
  - Cognitive Intelligence Layer (GraphRAG, coherence, FSRS)
  - UCW capture infrastructure (Data/Light/Instinct layers)

Transport → Protocol → Router → Capture → Database
"""

import asyncio
import os
import signal
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_raw.config import Config
from mcp_raw.logger import get_logger
from mcp_raw.transport import RawStdioTransport
from mcp_raw.protocol import (
    validate_message,
    make_response,
    make_error,
    ProtocolError,
    INTERNAL_ERROR,
)
from mcp_raw.router import Router
from mcp_raw.capture import CaptureEngine
from mcp_raw.db import CaptureDB
from mcp_raw.database import CognitiveDatabase
from mcp_raw.ucw_bridge import extract_layers, coherence_signature
from mcp_raw.embeddings import EmbeddingPipeline

from .config_notebooklm import NotebookLMConfig
from .api.client import NotebookLMAPIClient
from .api.cognitive import CognitiveLayer

log = get_logger("notebooklm_server")

# Protocol methods that should NEVER trigger embedding (fast-path)
_PROTOCOL_METHODS = frozenset({
    "initialize", "initialized", "notifications/initialized",
    "tools/list", "resources/list", "ping",
    "notifications/cancelled",
})


class UCWBridgeAdapter:
    """Adapts ucw_bridge module functions to the enrich(event) interface."""

    def enrich(self, event):
        data, light, instinct = extract_layers(event.parsed, event.direction)
        event.data_layer = data
        event.light_layer = light
        event.instinct_layer = instinct
        event.coherence_signature = coherence_signature(
            light.get("intent", ""),
            light.get("topic", ""),
            event.timestamp_ns,
            data.get("content", ""),
        )


class NotebookLMMCPServer:
    """
    NotebookLM MCP Server with HTTP/RPC API + Cognitive Intelligence.

    Usage:
        server = NotebookLMMCPServer()
        server.register_tools(TOOLS, handle_tool)
        await server.run()
    """

    def __init__(self):
        # Use NotebookLM config
        NotebookLMConfig.ensure_dirs()

        # Core components (reuse mcp_raw infrastructure)
        self._capture = CaptureEngine()
        self._transport = RawStdioTransport(on_capture=self._capture.capture)
        self._router = Router()
        self._db = None  # CaptureDB (SQLite) or CognitiveDatabase (PostgreSQL)
        self._pg_db: Optional[CognitiveDatabase] = None
        self._embedding_pipeline: Optional[EmbeddingPipeline] = None
        self._running = False
        self._db_ready = False
        self._db_init_task: Optional[asyncio.Task] = None
        self._handshake_complete = False

        # NotebookLM API client + Cognitive layer (replaces browser_controller)
        self._api_client: Optional[NotebookLMAPIClient] = None
        self._cognitive: Optional[CognitiveLayer] = None

    # ── tool/resource registration ─────────────────────────────────

    def register_tools(self, tools_list, handler):
        """Register a tools module with the router."""
        self._router.register_tools_module(tools_list, handler)

    def register_resources(self, resources, handler):
        """Register a resources provider with the router."""
        self._router.register_resources(resources, handler)

    @property
    def capture_engine(self) -> CaptureEngine:
        return self._capture

    @property
    def db(self) -> Optional[CaptureDB]:
        return self._db

    @property
    def api_client(self) -> Optional[NotebookLMAPIClient]:
        return self._api_client

    @property
    def cognitive(self) -> Optional[CognitiveLayer]:
        return self._cognitive

    # ── API client initialization ───────────────────────────────────

    def _init_api_client(self):
        """Initialize the NotebookLM API client from env cookies."""
        cookies = os.environ.get("NOTEBOOKLM_COOKIES", "")
        if cookies:
            try:
                self._api_client = NotebookLMAPIClient(cookies=cookies)
                log.info("NotebookLM API client initialized from NOTEBOOKLM_COOKIES")
            except Exception as exc:
                log.warning(f"Failed to init API client: {exc}")
                self._api_client = None
        else:
            log.info("No NOTEBOOKLM_COOKIES set — use save_auth_tokens tool to authenticate")

    def _init_cognitive_layer(self):
        """Initialize the Cognitive Intelligence Layer."""
        if self._api_client and self._pg_db and self._pg_db.available:
            try:
                self._cognitive = CognitiveLayer(
                    self._api_client,
                    self._pg_db._pool,
                    self._embedding_pipeline,
                )
                log.info("Cognitive Intelligence Layer active (GraphRAG, coherence, FSRS)")
            except Exception as exc:
                log.warning(f"Cognitive layer init failed: {exc}")
                self._cognitive = None
        elif self._api_client:
            # Cognitive layer without DB (no enrichment, but API still works)
            self._cognitive = CognitiveLayer(self._api_client)
            log.info("Cognitive layer active (API only, no DB enrichment)")

    def _inject_api_into_tools(self):
        """Inject API client and cognitive layer into tool module."""
        try:
            from .tools import notebooklm_tools
            if self._api_client:
                notebooklm_tools.set_api_client(self._api_client)
            if self._cognitive:
                notebooklm_tools.set_cognitive(self._cognitive)
        except Exception as exc:
            log.debug(f"Tool injection skipped: {exc}")

    # ── lazy database initialization ───────────────────────────────

    async def _init_db_background(self):
        """Initialize database in background (don't block handshake)."""
        try:
            # Try PostgreSQL first, fall back to SQLite
            self._pg_db = CognitiveDatabase()
            pg_ok = await self._pg_db.initialize()
            if pg_ok:
                self._db = self._pg_db
                log.info("Using PostgreSQL backend")
            else:
                self._pg_db = None
                self._db = CaptureDB()
                await self._db.initialize()
                log.info("Using SQLite backend (PostgreSQL unavailable)")

            # Wire capture to DB
            self._capture.set_db_sink(self._db)

            # Wire real-time embedding (non-blocking callback)
            if self._pg_db and self._pg_db.available:
                self._embedding_pipeline = EmbeddingPipeline(self._pg_db._pool)
                self._capture.on_event(self._auto_embed)
                log.info("Real-time embedding pipeline active")

            self._db_ready = True
            log.info(f"Database ready — platform=notebooklm")

            # Now that DB is ready, initialize cognitive layer
            self._init_cognitive_layer()
            self._inject_api_into_tools()

        except Exception as exc:
            log.error(f"DB initialization failed: {exc}", exc_info=True)

    async def _auto_embed(self, event):
        """Non-blocking embedding callback (fire-and-forget)."""
        if not self._embedding_pipeline:
            return

        # Skip everything during MCP handshake phase
        if not self._handshake_complete:
            return

        # Skip protocol handshake messages
        event_method = event.method or ""
        if event_method in _PROTOCOL_METHODS:
            return

        parent_method = (event.parsed or {}).get("method", "")
        if parent_method in _PROTOCOL_METHODS:
            return

        # Only embed meaningful events
        if event.direction == "out" and event.light_layer:
            asyncio.create_task(self._do_embed(event))

    async def _do_embed(self, event):
        """Background embedding task — errors are logged, never propagated."""
        try:
            await self._embedding_pipeline.embed_event(event)
        except Exception as exc:
            log.debug(f"Auto-embed skipped: {exc}")

    # ── server lifecycle ───────────────────────────────────────────

    async def run(self):
        """Main server loop."""
        log.info(f"NotebookLM MCP Server starting — version={NotebookLMConfig.SERVER_VERSION}")

        # Initialize transport FIRST — must be ready for MCP handshake
        await self._transport.start()

        # Initialize API client from env (non-blocking)
        self._init_api_client()

        # Start DB init in background (don't await — don't block handshake)
        self._db_init_task = asyncio.create_task(self._init_db_background())

        # Inject API client into tools immediately (cognitive layer injected after DB)
        self._inject_api_into_tools()

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                pass  # Windows

        self._running = True
        log.info(
            f"Server ready — tools={self._router.tool_count} "
            f"resources={self._router.resource_count} "
            f"(db initializing in background)"
        )

        try:
            while self._running:
                result = await self._transport.read_message()
                if result is None:
                    log.info("EOF on stdin — shutting down")
                    break

                raw_bytes, parsed = result
                await self._handle_message(parsed)

        except asyncio.CancelledError:
            log.info("Server cancelled")
        except Exception as exc:
            log.error(f"Server error: {exc}", exc_info=True)
        finally:
            await self.shutdown()

    async def _handle_message(self, msg: dict):
        """Process a single JSON-RPC message through the full pipeline."""
        request_id = msg.get("id")
        method = msg.get("method", "")

        # Track handshake completion
        if method in ("initialized", "notifications/initialized"):
            self._handshake_complete = True
            log.info("MCP handshake complete")

        try:
            # Validate protocol
            msg_type = validate_message(msg)

            # For tool calls, ensure DB is ready first
            if method == "tools/call":
                if self._db_init_task and not self._db_init_task.done():
                    await self._db_init_task

            # Route to handler
            result = await self._router.route(msg_type, msg)

            # Notifications get no response
            if result is None:
                return

            # Send success response
            response = make_response(request_id, result)
            await self._transport.write_message(response, request_id=request_id)

        except ProtocolError as exc:
            log.warning(f"Protocol error: {exc.message} (code={exc.code})")
            if request_id is not None:
                error_resp = make_error(request_id, exc.code, exc.message, exc.data)
                await self._transport.write_message(error_resp, request_id=request_id)

        except Exception as exc:
            log.error(f"Unhandled error: {exc}", exc_info=True)
            if request_id is not None:
                error_resp = make_error(request_id, INTERNAL_ERROR, str(exc))
                await self._transport.write_message(error_resp, request_id=request_id)

    async def shutdown(self):
        """Graceful shutdown."""
        if not self._running:
            return

        self._running = False
        log.info("Shutting down NotebookLM MCP Server")

        # Close API client
        if self._api_client:
            try:
                self._api_client.close()
            except Exception as exc:
                log.error(f"Error closing API client: {exc}")

        # Close database
        if self._db:
            try:
                await self._db.close()
            except Exception as exc:
                log.error(f"Error closing database: {exc}")

        # Close embedding pipeline
        if self._embedding_pipeline:
            try:
                await self._embedding_pipeline.close()
            except Exception as exc:
                log.error(f"Error closing embedding pipeline: {exc}")

        log.info("Shutdown complete")
