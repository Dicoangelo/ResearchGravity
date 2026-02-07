"""
Raw STDIO Transport — Perfect byte-level capture

Reads from stdin, writes to stdout.
NEVER pollutes stdout with logs.
Every byte captured before processing.
"""

import sys
import json
import asyncio
import time
from typing import Optional, Tuple, Dict, Any, Callable, Awaitable

from .config import Config
from .logger import get_logger

log = get_logger("transport")


class RawStdioTransport:
    """Raw STDIO transport with perfect byte capture"""

    def __init__(self, on_capture: Callable[..., Awaitable[None]]):
        self.on_capture = on_capture
        self.running = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._stdout = None  # raw stdout buffer for synchronous writes

    async def start(self):
        """Initialize async stdin reader and direct stdout writer"""
        loop = asyncio.get_event_loop()

        # Async reader for stdin
        self._reader = asyncio.StreamReader(limit=2**16)
        protocol = asyncio.StreamReaderProtocol(self._reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        # Direct stdout — synchronous writes are safe and reliable
        # connect_write_pipe fails when stdout is not a proper pipe
        # (e.g., when Claude Code CLI spawns the process)
        self._stdout = sys.stdout.buffer
        self.running = True
        log.info("Transport initialized")

    async def read_message(self) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """
        Read one JSON-RPC message from stdin.
        Returns (raw_bytes, parsed_dict) or None on EOF.
        """
        if not self._reader:
            raise RuntimeError("Transport not started")

        try:
            raw_bytes = await self._reader.readline()
            if not raw_bytes:
                return None  # EOF

            ts = time.time_ns()

            try:
                parsed = json.loads(raw_bytes)
            except json.JSONDecodeError as exc:
                log.error(f"JSON parse error: {exc}")
                await self.on_capture(
                    raw_bytes=raw_bytes, parsed={},
                    timestamp_ns=ts, direction="in",
                    error=f"JSON parse error: {exc}",
                )
                return None

            # Capture inbound message
            await self.on_capture(
                raw_bytes=raw_bytes, parsed=parsed,
                timestamp_ns=ts, direction="in",
            )

            return raw_bytes, parsed

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error(f"Read error: {exc}")
            return None

    async def write_message(self, message: Dict[str, Any], *, request_id: Optional[int] = None):
        """Write a JSON-RPC message to stdout"""
        if self._stdout is None:
            raise RuntimeError("Transport not started")

        raw_text = json.dumps(message, separators=(",", ":")) + "\n"
        raw_bytes = raw_text.encode("utf-8")

        ts = time.time_ns()

        # Capture outbound BEFORE sending
        await self.on_capture(
            raw_bytes=raw_bytes, parsed=message,
            timestamp_ns=ts, direction="out",
            parent_protocol_id=str(request_id) if request_id is not None else None,
        )

        self._stdout.write(raw_bytes)
        self._stdout.flush()

    async def close(self):
        self.running = False
        log.info("Transport closed")
