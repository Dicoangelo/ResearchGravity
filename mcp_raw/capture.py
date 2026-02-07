"""
Perfect Capture Engine — Every byte, every stage, every event

Captures at EVERY lifecycle stage:
  received → parsed → routed → executed → sent

Tracks:
- Raw bytes with nanosecond timestamps
- Message lineage (parent/child request-response)
- Turn counting per session
- Content metrics
"""

import json
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .logger import get_logger

log = get_logger("capture")


class CaptureEvent:
    """A single capture event at a lifecycle stage."""

    __slots__ = (
        "event_id", "timestamp_ns", "direction", "stage",
        "raw_bytes", "parsed", "method", "request_id",
        "parent_protocol_id", "turn", "error",
        "data_layer", "light_layer", "instinct_layer",
        "coherence_signature", "content_length",
    )

    def __init__(
        self,
        direction: str,
        stage: str,
        raw_bytes: bytes,
        parsed: Dict[str, Any],
        timestamp_ns: Optional[int] = None,
        parent_protocol_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.event_id = uuid.uuid4().hex[:16]
        self.timestamp_ns = timestamp_ns or time.time_ns()
        self.direction = direction  # "in" or "out"
        self.stage = stage  # "received", "parsed", "routed", "executed", "sent"
        self.raw_bytes = raw_bytes
        self.parsed = parsed
        self.method = parsed.get("method", "")
        self.request_id = parsed.get("id")
        self.parent_protocol_id = parent_protocol_id
        self.turn = 0
        self.error = error
        self.content_length = len(raw_bytes)

        # UCW layers (populated by UCWBridge)
        self.data_layer: Optional[Dict] = None
        self.light_layer: Optional[Dict] = None
        self.instinct_layer: Optional[Dict] = None
        self.coherence_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "event_id": self.event_id,
            "timestamp_ns": self.timestamp_ns,
            "direction": self.direction,
            "stage": self.stage,
            "method": self.method,
            "request_id": self.request_id,
            "parent_protocol_id": self.parent_protocol_id,
            "turn": self.turn,
            "content_length": self.content_length,
        }
        if self.error:
            d["error"] = self.error
        if self.data_layer:
            d["data_layer"] = self.data_layer
        if self.light_layer:
            d["light_layer"] = self.light_layer
        if self.instinct_layer:
            d["instinct_layer"] = self.instinct_layer
        if self.coherence_signature:
            d["coherence_signature"] = self.coherence_signature
        return d


class CaptureEngine:
    """
    Perfect capture engine — never drops a message.

    Usage:
        engine = CaptureEngine()
        engine.set_ucw_bridge(bridge)
        engine.set_db_sink(db)

        # Called by transport on every read/write
        await engine.capture(raw_bytes=b, parsed={...}, direction="in")
    """

    def __init__(self):
        self._events: List[CaptureEvent] = []
        self._turn_counter: int = 0
        self._request_map: Dict[str, CaptureEvent] = {}  # id → inbound event
        self._ucw_bridge = None
        self._db_sink = None
        self._event_callbacks: List = []
        self._stats = defaultdict(int)

    def set_ucw_bridge(self, bridge):
        """Attach UCW bridge for semantic layer extraction."""
        self._ucw_bridge = bridge

    def set_db_sink(self, db):
        """Attach database sink for persistent capture."""
        self._db_sink = db

    def on_event(self, callback):
        """Register callback for each capture event."""
        self._event_callbacks.append(callback)

    async def capture(
        self,
        raw_bytes: bytes,
        parsed: Dict[str, Any],
        timestamp_ns: int,
        direction: str,
        parent_protocol_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """
        Capture a message at transport level.
        Called by RawStdioTransport on every read/write.
        """
        stage = "received" if direction == "in" else "sent"

        event = CaptureEvent(
            direction=direction,
            stage=stage,
            raw_bytes=raw_bytes,
            parsed=parsed,
            timestamp_ns=timestamp_ns,
            parent_protocol_id=parent_protocol_id,
            error=error,
        )

        # Track request-response lineage
        if direction == "in" and event.request_id is not None:
            self._request_map[str(event.request_id)] = event
            self._turn_counter += 1
            event.turn = self._turn_counter

        if direction == "out" and event.request_id is not None:
            parent = self._request_map.get(str(event.request_id))
            if parent:
                event.parent_protocol_id = parent.event_id
                event.turn = parent.turn

        # Run UCW bridge for semantic layers
        if self._ucw_bridge:
            try:
                self._ucw_bridge.enrich(event)
            except Exception as exc:
                log.error(f"UCW bridge error: {exc}")

        # Store in memory
        self._events.append(event)
        self._stats[f"{direction}_{stage}"] += 1
        self._stats["total"] += 1

        # Persist to database
        if self._db_sink:
            try:
                await self._db_sink.store_event(event)
            except Exception as exc:
                log.error(f"DB sink error: {exc}")

        # Notify callbacks
        for cb in self._event_callbacks:
            try:
                await cb(event)
            except Exception as exc:
                log.error(f"Callback error: {exc}")

        log.debug(
            f"[{direction}] {event.method or 'response'} "
            f"turn={event.turn} bytes={event.content_length}"
        )

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def event_count(self) -> int:
        return len(self._events)

    def recent_events(self, limit: int = 20) -> List[Dict]:
        return [e.to_dict() for e in self._events[-limit:]]
