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

    # ── main loop ────────────────────────────────────────────────

    async def run(self):
        """Start the server and process messages until EOF or signal."""
        log.info(f"Starting {Config.SERVER_NAME} v{Config.SERVER_VERSION}")

        # Initialize database — try PostgreSQL first, fall back to SQLite
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

        # Wire components
        self._capture.set_ucw_bridge(UCWBridgeAdapter())
        self._capture.set_db_sink(self._db)

        # Wire real-time embedding (auto-embed every captured event)
        if self._pg_db and self._pg_db.available:
            self._embedding_pipeline = EmbeddingPipeline(self._pg_db._pool)
            self._capture.on_event(self._auto_embed)
            log.info("Real-time embedding pipeline active")

        # Inject shared DB into tool modules
        self._inject_db()

        # Initialize transport
        await self._transport.start()

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
            f"session={self._db.session_id}"
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

        try:
            # Validate protocol
            msg_type = validate_message(msg)

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
        """Callback: auto-embed each captured event in real-time."""
        if not self._embedding_pipeline:
            return
        # Only embed meaningful events (tool calls, responses with content)
        if event.direction == "out" and event.light_layer:
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

        await self._transport.close()
        if self._pg_db:
            await self._pg_db.close()
        elif self._db:
            await self._db.close()

        log.info("Server stopped")
