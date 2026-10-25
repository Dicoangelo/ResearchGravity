"""Read/evaluate/export MCP tools using ResearchGravity's existing router.

Only explicitly submitted evidence enters this server. It does not initialize
UCW capture, query private history, discover credentials, or execute workloads.
"""
import asyncio
import json
import sys

from mcp_raw.protocol import ProtocolError, make_error, make_response, validate_message
from .handoff import export_handoff
from .planner import evaluate_workload

TOOLS = [
    {"name": "basix_evaluate_workload", "description": "Evaluate image-upscale options against constraints and save a local evidence receipt. Never executes or provisions anything.",
     "inputSchema": {"type": "object", "properties": {"workload": {"type": "object"}, "offers": {"type": "array", "items": {"type": "object"}}, "evidence": {"type": "array", "items": {"type": "object"}}}, "required": ["workload", "offers", "evidence"], "additionalProperties": False}},
    {"name": "basix_get_decision", "description": "Read and integrity-check one saved BASIX decision.",
     "inputSchema": {"type": "object", "properties": {"decision_id": {"type": "string", "pattern": "^[a-f0-9]{64}$"}}, "required": ["decision_id"], "additionalProperties": False}},
    {"name": "basix_export_handoff", "description": "Export only the selected decision, evidence, limitations and next action for another AI tool. No private-corpus sweep.",
     "inputSchema": {"type": "object", "properties": {"decision_id": {"type": "string", "pattern": "^[a-f0-9]{64}$"}}, "required": ["decision_id"], "additionalProperties": False}},
]


def make_router(store):
    # Router uses the existing file logger at import time. Keep those logs
    # inside this explicitly selected workspace instead of opening ~/.ucw.
    from mcp_raw.config import Config
    Config.UCW_DIR = store.root / "mcp-logs"
    Config.LOG_DIR = Config.UCW_DIR
    Config.LOG_FILE = Config.LOG_DIR / "mcp.log"
    Config.ERROR_LOG = Config.LOG_DIR / "errors.log"
    from mcp_raw.router import Router

    router = Router(server_name="basix-researchgravity", server_version="0.1.0")

    async def handle(name, args):
        try:
            if not isinstance(args, dict):
                raise ValueError("Arguments must be an object")
            if name == "basix_evaluate_workload":
                if set(args) != {"workload", "offers", "evidence"}:
                    raise ValueError("Expected workload, offers and evidence only")
                result = evaluate_workload(args["workload"], args["offers"], args["evidence"])
                store.save(result)
            elif name in {"basix_get_decision", "basix_export_handoff"}:
                if set(args) != {"decision_id"}:
                    raise ValueError("Expected decision_id only")
                result = store.get(args["decision_id"])
                if name == "basix_export_handoff":
                    result = export_handoff(result)
            else:
                raise ValueError("Unknown BASIX tool")
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, allow_nan=False)}]}
        except (ValueError, KeyError, TypeError, OSError):
            return {"isError": True, "content": [{"type": "text", "text": "BASIX request or persisted receipt is invalid/unavailable; no execution occurred."}]}

    router.register_tools_module(TOOLS, handle)
    return router


async def serve_stdio(store):
    router = make_router(store)
    while True:
        raw = await asyncio.to_thread(sys.stdin.buffer.readline, 512_001)
        if not raw:
            break
        if len(raw) > 512_000 or not raw.endswith(b"\n"):
            response = make_error(None, -32600, "Message exceeds limit or lacks newline")
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            break
        msg = None
        try:
            msg = json.loads(raw)
            kind = validate_message(msg)
            if kind not in {"request", "notification"}:
                continue
            # Notifications never invoke tools or mutate receipt state.
            if kind == "notification" and msg.get("method") not in {"notifications/initialized", "initialized", "notifications/cancelled"}:
                continue
            result = await router.route(kind, msg)
            if kind == "notification":
                continue
            response = make_response(msg["id"], result)
        except (json.JSONDecodeError, UnicodeError):
            response = make_error(None, -32700, "Invalid JSON")
        except ProtocolError as exc:
            response = make_error(msg.get("id") if isinstance(msg, dict) else None, exc.code, exc.message)
        except (ValueError, KeyError, TypeError, AttributeError):
            response = make_error(msg.get("id") if isinstance(msg, dict) else None, -32602, "Invalid request parameters")
        sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        sys.stdout.flush()
