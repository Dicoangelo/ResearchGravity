"""
Raw MCP Server — Main Orchestrator

Ties together:
  Transport → Protocol → Router → Capture → Database

Flow:
  1. Transport reads raw bytes from stdin
  2. Protocol validates JSON-RPC 2.0
  3. Router dispatches to correct handler
  4. Capture engine records at every stage
  5. UCW Bridge enriches with semantic layers
  6. Database persists with perfect fidelity
  7. Transport writes response to stdout

Startup strategy (fast init for MCP handshake):
  - Transport starts IMMEDIATELY (respond to initialize within ms)
  - Database connects in BACKGROUND (lazy — first tool call triggers if needed)
  - SBERT model pre-warms in BACKGROUND (non-blocking)
  - Embedding callbacks are NON-BLOCKING (fire-and-forget tasks)
"""

import asyncio
import signal
from typing import Optional

from .config import Config
from .logger import get_logger
from .transport import RawStdioTransport
from .protocol import (
    validate_message,
    make_response,
    make_error,
    ProtocolError,
    PARSE_ERROR,
    INTERNAL_ERROR,
)
from .router import Router
from .capture import CaptureEngine
from .db import CaptureDB
from .database import CognitiveDatabase
from .ucw_bridge import extract_layers, coherence_signature
from .embeddings import EmbeddingPipeline

log = get_logger("server")

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


class RawMCPServer:
    """
    Main server orchestrator.

    Usage:
        server = RawMCPServer()
        # register tools before running
        server.register_tools(TOOLS, handle_tool)
        await server.run()
    """

    def __init__(self):
        Config.ensure_dirs()

        # Core components
        self._capture = CaptureEngine()
        self._transport = RawStdioTransport(on_capture=self._capture.capture)
        self._router = Router()
        self._db = None  # CaptureDB (SQLite) or CognitiveDatabase (PostgreSQL)
        self._pg_db: Optional[CognitiveDatabase] = None
        self._embedding_pipeline: Optional[EmbeddingPipeline] = None
        self._running = False
        self._db_ready = False  # True once DB is connected (lazy init)
        self._db_init_task: Optional[asyncio.Task] = None
        self._handshake_complete = False  # True after initialized notification

    # ── tool/resource registration (call before run) ─────────────

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

    # ── lazy database initialization ───────────────────────────────

    async def _init_db_background(self):
        """
        Initialize database in background. Called as a fire-and-forget task
        so the MCP handshake is never blocked by DB connection time.
        """
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

            # Inject shared DB into tool modules
            self._inject_db()

            self._db_ready = True
            log.info(
                f"Database ready — session={self._db.session_id}"
            )

            # Pre-warm SBERT model in background (non-blocking)
            asyncio.get_event_loop().run_in_executor(None, self._prewarm_sbert)

        except Exception as exc:
            log.error(f"Background DB init failed: {exc}", exc_info=True)
            # Fall back to SQLite on any error
            if not self._db:
                try:
                    self._db = CaptureDB()
                    await self._db.initialize()
                    self._capture.set_db_sink(self._db)
                    self._inject_db()
                    self._db_ready = True
                    log.info("Fallback to SQLite after error")
                except Exception as exc2:
                    log.error(f"SQLite fallback also failed: {exc2}")

    async def _ensure_db_ready(self):
        """Wait for DB to be ready (called before tool execution)."""
        if self._db_ready:
            return
        if self._db_init_task and not self._db_init_task.done():
            await self._db_init_task

    @staticmethod
    def _prewarm_sbert():
        """Pre-load SBERT model in a thread so it's ready for first real embed."""
        try:
            from .embeddings import _get_model
            _get_model()
            log.info("SBERT model pre-warmed")
        except Exception as exc:
            log.debug(f"SBERT pre-warm skipped: {exc}")

    # ── main loop ────────────────────────────────────────────────

    async def run(self):
        """Start the server and process messages until EOF or signal."""
        log.info(f"Starting {Config.SERVER_NAME} v{Config.SERVER_VERSION}")

        # Wire UCW bridge immediately (no I/O, instant)
        self._capture.set_ucw_bridge(UCWBridgeAdapter())

        # Initialize transport FIRST — must be ready for MCP handshake
        await self._transport.start()

        # Start database initialization in background (non-blocking)
        self._db_init_task = asyncio.create_task(self._init_db_background())

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
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

        try:
            # Validate protocol
            msg_type = validate_message(msg)

            # For tool calls, ensure DB is ready first
            if method == "tools/call":
                await self._ensure_db_ready()

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

    async def _auto_embed(self, event):
        """
        Callback: auto-embed each captured event in real-time.

        IMPORTANT: This is non-blocking — skips protocol handshake messages
        and schedules embedding as a background task so it never delays
        MCP responses.
        """
        if not self._embedding_pipeline:
            return

        # Skip everything during MCP handshake phase (initialize → tools/list → resources/list)
        # This is the critical guard: SBERT model loading takes ~6 seconds and would
        # block the initialize response, causing Claude Code CLI to timeout
        if not self._handshake_complete:
            return

        # Skip protocol handshake messages — they don't need embedding
        event_method = event.method or ""
        if event_method in _PROTOCOL_METHODS:
            return

        # Also skip if the parent request was a protocol message
        parent_method = (event.parsed or {}).get("method", "")
        if parent_method in _PROTOCOL_METHODS:
            return

        # Only embed meaningful events (tool calls, responses with content)
        if event.direction == "out" and event.light_layer:
            # Fire-and-forget: schedule as background task, never block the response
            asyncio.create_task(self._do_embed(event))

    async def _do_embed(self, event):
        """Background embedding task — errors are logged, never propagated."""
        try:
            await self._embedding_pipeline.embed_event(event)
        except Exception as exc:
            log.debug(f"Auto-embed skipped: {exc}")

    def _inject_db(self):
        """Inject shared DB instance into tool modules that need it."""
        try:
            from mcp_raw.tools import ucw_tools, coherence_tools
            if hasattr(ucw_tools, 'set_db'):
                ucw_tools.set_db(self._db)
            if hasattr(coherence_tools, 'set_db'):
                coherence_tools.set_db(self._db)
            log.info("DB injected into tool modules")
        except ImportError:
            pass

    async def shutdown(self):
        """Graceful shutdown — flush captures, close DB."""
        if not self._running:
            return
        self._running = False

        log.info(
            f"Shutting down — captured {self._capture.event_count} events, "
            f"{self._capture.turn_count} turns"
        )

        # Cancel background DB init if still running
        if self._db_init_task and not self._db_init_task.done():
            self._db_init_task.cancel()
            try:
                await self._db_init_task
            except (asyncio.CancelledError, Exception):
                pass

        await self._transport.close()
        if self._pg_db:
            await self._pg_db.close()
        elif self._db:
            await self._db.close()

        log.info("Server stopped")
