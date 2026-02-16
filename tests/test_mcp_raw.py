#!/usr/bin/env python3
"""
Integration Test — Raw MCP Server Pipeline

Tests the full flow WITHOUT stdin/stdout (unit-tests the components directly):
  Protocol → Router → Capture → UCW Bridge → Database

Run: python3 test_mcp_raw.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure mcp_raw is importable
sys.path.insert(0, str(Path(__file__).parent))

from mcp_raw.config import Config
from mcp_raw.protocol import (
    validate_message,
    make_response,
    make_error,
    initialize_result,
    tools_list_result,
    tool_result_content,
    text_content,
    ProtocolError,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
)
from mcp_raw.router import Router
from mcp_raw.capture import CaptureEngine, CaptureEvent
from mcp_raw.db import CaptureDB
from mcp_raw.ucw_bridge import extract_layers, coherence_signature
from mcp_raw.server import UCWBridgeAdapter


passed = 0
failed = 0


def test(name):
    def decorator(fn):
        async def wrapper():
            global passed, failed
            try:
                await fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1
        return wrapper
    return decorator


# ── Protocol Tests ──────────────────────────────────────────────

@test("validate_message: request")
async def test_validate_request():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    assert validate_message(msg) == "request"


@test("validate_message: notification")
async def test_validate_notification():
    msg = {"jsonrpc": "2.0", "method": "initialized"}
    assert validate_message(msg) == "notification"


@test("validate_message: response")
async def test_validate_response():
    msg = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    assert validate_message(msg) == "response"


@test("validate_message: error response")
async def test_validate_error():
    msg = {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "fail"}}
    assert validate_message(msg) == "error"


@test("validate_message: rejects missing jsonrpc")
async def test_validate_rejects_bad():
    try:
        validate_message({"id": 1, "method": "x"})
        assert False, "Should have raised"
    except ProtocolError as e:
        assert e.code == INVALID_REQUEST


@test("make_response: correct structure")
async def test_make_response():
    r = make_response(42, {"tools": []})
    assert r["jsonrpc"] == "2.0"
    assert r["id"] == 42
    assert r["result"] == {"tools": []}


@test("make_error: correct structure")
async def test_make_error():
    r = make_error(42, -32601, "not found")
    assert r["error"]["code"] == -32601
    assert r["error"]["message"] == "not found"


@test("initialize_result: MCP format")
async def test_initialize_result():
    r = initialize_result("test", "1.0", "2024-11-05")
    assert r["protocolVersion"] == "2024-11-05"
    assert r["serverInfo"]["name"] == "test"
    assert "tools" in r["capabilities"]


# ── UCW Bridge Tests ────────────────────────────────────────────

@test("extract_layers: inbound tool call")
async def test_ucw_tool_call():
    msg = {
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "search_learnings", "arguments": {"query": "MCP protocol"}},
    }
    data, light, instinct = extract_layers(msg, "in")
    assert "search_learnings" in data["content"]
    assert light["intent"] in ("search", "execute", "retrieve", "explore")
    assert light["topic"] in ("mcp_protocol", "research", "general")
    assert isinstance(instinct["coherence_potential"], float)


@test("extract_layers: outbound response")
async def test_ucw_response():
    msg = {
        "jsonrpc": "2.0", "id": 1,
        "result": {
            "content": [{"type": "text", "text": "Found 3 results about cognitive architecture"}]
        },
    }
    data, light, instinct = extract_layers(msg, "out")
    assert "cognitive" in data["content"].lower() or "Found" in data["content"]
    assert isinstance(instinct["gut_signal"], str)


@test("coherence_signature: deterministic")
async def test_coherence_sig():
    ts = 1707350400_000_000_000  # Fixed timestamp
    sig1 = coherence_signature("search", "research", ts, "test content")
    sig2 = coherence_signature("search", "research", ts, "test content")
    assert sig1 == sig2
    assert len(sig1) == 64  # SHA-256 hex


@test("coherence_signature: different bucket = different sig")
async def test_coherence_sig_buckets():
    ts1 = 1707350400_000_000_000
    ts2 = ts1 + 6 * 60 * 1_000_000_000  # 6 minutes later (different bucket)
    sig1 = coherence_signature("search", "research", ts1, "content")
    sig2 = coherence_signature("search", "research", ts2, "content")
    assert sig1 != sig2


@test("UCWBridgeAdapter.enrich: populates all layers")
async def test_bridge_adapter():
    adapter = UCWBridgeAdapter()
    event = CaptureEvent(
        direction="in",
        stage="received",
        raw_bytes=b'{"jsonrpc":"2.0","method":"tools/call","params":{"name":"test"}}',
        parsed={"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "test"}},
    )
    adapter.enrich(event)
    assert event.data_layer is not None
    assert event.light_layer is not None
    assert event.instinct_layer is not None
    assert event.coherence_signature is not None


# ── Capture Engine Tests ────────────────────────────────────────

@test("CaptureEngine: basic capture")
async def test_capture_basic():
    engine = CaptureEngine()
    await engine.capture(
        raw_bytes=b'{"jsonrpc":"2.0","id":1,"method":"ping"}',
        parsed={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        timestamp_ns=time.time_ns(),
        direction="in",
    )
    assert engine.event_count == 1
    assert engine.turn_count == 1


@test("CaptureEngine: turn counting")
async def test_capture_turns():
    engine = CaptureEngine()
    # Inbound request
    await engine.capture(
        raw_bytes=b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}',
        parsed={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        timestamp_ns=time.time_ns(),
        direction="in",
    )
    # Outbound response
    await engine.capture(
        raw_bytes=b'{"jsonrpc":"2.0","id":1,"result":{}}',
        parsed={"jsonrpc": "2.0", "id": 1, "result": {}},
        timestamp_ns=time.time_ns(),
        direction="out",
        parent_protocol_id="1",
    )
    assert engine.turn_count == 1  # One turn (request+response)
    assert engine.event_count == 2


@test("CaptureEngine: UCW bridge integration")
async def test_capture_with_bridge():
    engine = CaptureEngine()
    engine.set_ucw_bridge(UCWBridgeAdapter())
    await engine.capture(
        raw_bytes=b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search"}}',
        parsed={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "search"}},
        timestamp_ns=time.time_ns(),
        direction="in",
    )
    events = engine.recent_events(1)
    assert events[0].get("data_layer") is not None
    assert events[0].get("light_layer") is not None
    assert events[0].get("coherence_signature") is not None


# ── Router Tests ────────────────────────────────────────────────

@test("Router: initialize")
async def test_router_initialize():
    router = Router()
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "test-client"},
    }}
    result = await router.route("request", msg)
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == Config.SERVER_NAME


@test("Router: tools/list empty")
async def test_router_tools_list_empty():
    router = Router()
    msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    result = await router.route("request", msg)
    assert result["tools"] == []


@test("Router: tools/list with registered tools")
async def test_router_tools_list():
    router = Router()
    tools = [{"name": "test_tool", "description": "A test", "inputSchema": {"type": "object"}}]

    async def handler(name, args):
        return tool_result_content([text_content("ok")])

    router.register_tools_module(tools, handler)
    msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    result = await router.route("request", msg)
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "test_tool"


@test("Router: tools/call dispatch")
async def test_router_tools_call():
    router = Router()
    tools = [{"name": "greet", "description": "Greet", "inputSchema": {"type": "object"}}]

    async def handler(name, args):
        return tool_result_content([text_content(f"Hello {args.get('name', 'world')}")])

    router.register_tools_module(tools, handler)
    msg = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "greet", "arguments": {"name": "UCW"},
    }}
    result = await router.route("request", msg)
    assert result["content"][0]["text"] == "Hello UCW"


@test("Router: unknown method raises ProtocolError")
async def test_router_unknown():
    router = Router()
    msg = {"jsonrpc": "2.0", "id": 4, "method": "nonexistent/method", "params": {}}
    try:
        await router.route("request", msg)
        assert False, "Should have raised"
    except ProtocolError as e:
        assert e.code == METHOD_NOT_FOUND


@test("Router: notification returns None")
async def test_router_notification():
    router = Router()
    msg = {"jsonrpc": "2.0", "method": "initialized"}
    result = await router.route("notification", msg)
    assert result is None


@test("Router: ping returns empty dict")
async def test_router_ping():
    router = Router()
    msg = {"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {}}
    result = await router.route("request", msg)
    assert result == {}


# ── Database Tests ──────────────────────────────────────────────

@test("CaptureDB: initialize and store event")
async def test_db_store():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = CaptureDB(db_path=Path(tmp) / "test.db")
        await db.initialize()
        assert db.session_id is not None

        event = CaptureEvent(
            direction="in",
            stage="received",
            raw_bytes=b'{"test": true}',
            parsed={"test": True},
        )
        event.data_layer = {"content": "test", "tokens_est": 1}
        event.light_layer = {"intent": "test", "topic": "testing", "concepts": [], "summary": "test"}
        event.instinct_layer = {"coherence_potential": 0.5, "emergence_indicators": [], "gut_signal": "routine"}

        await db.store_event(event)

        stats = await db.get_session_stats()
        assert stats["event_count"] == 1

        await db.close()


@test("CaptureDB: get_all_stats")
async def test_db_all_stats():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = CaptureDB(db_path=Path(tmp) / "test2.db")
        await db.initialize()

        stats = await db.get_all_stats()
        assert stats["total_sessions"] == 1
        assert stats["current_session"] == db.session_id

        await db.close()


# ── Full Pipeline Test ──────────────────────────────────────────

@test("Full pipeline: capture → bridge → db round-trip")
async def test_full_pipeline():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = CaptureDB(db_path=Path(tmp) / "pipeline.db")
        await db.initialize()

        engine = CaptureEngine()
        engine.set_ucw_bridge(UCWBridgeAdapter())
        engine.set_db_sink(db)

        # Simulate an inbound tool call
        msg = {
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "search_learnings", "arguments": {"query": "sovereign cognitive architecture"}},
        }
        raw = json.dumps(msg).encode()

        await engine.capture(
            raw_bytes=raw,
            parsed=msg,
            timestamp_ns=time.time_ns(),
            direction="in",
        )

        # Verify capture
        assert engine.event_count == 1
        events = engine.recent_events(1)
        assert events[0]["data_layer"] is not None
        assert events[0]["light_layer"] is not None
        assert events[0]["coherence_signature"] is not None

        # Verify DB
        stats = await db.get_session_stats()
        assert stats["event_count"] == 1

        await db.close()


# ── Runner ──────────────────────────────────────────────────────

async def run_all():
    global passed, failed
    print("\n" + "=" * 60)
    print("  UCW Raw MCP Server — Integration Tests")
    print("=" * 60 + "\n")

    all_globals = list(globals().values())
    tests = [v for v in all_globals if callable(v) and asyncio.iscoroutinefunction(v) and getattr(v, "__name__", "") not in ("run_all", "main")]

    for test_fn in tests:
        await test_fn()

    print(f"\n{'=' * 60}")
    total = passed + failed
    if failed == 0:
        print(f"  ALL {total} TESTS PASSED")
    else:
        print(f"  {passed}/{total} passed, {failed} FAILED")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_all())
    sys.exit(0 if ok else 1)
