"""
JSON-RPC 2.0 Protocol — MCP message construction and validation

Handles:
- Request/response/notification/error construction
- Message validation
- ID tracking for request-response correlation
"""

from typing import Any, Dict, List, Optional, Union

JSONRPC_VERSION = "2.0"


class ProtocolError(Exception):
    """JSON-RPC protocol error"""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def validate_message(msg: Dict[str, Any]) -> str:
    """
    Validate a JSON-RPC 2.0 message.
    Returns message type: 'request', 'notification', 'response', 'error', or raises ProtocolError.
    """
    if not isinstance(msg, dict):
        raise ProtocolError(INVALID_REQUEST, "Message must be a JSON object")

    if msg.get("jsonrpc") != JSONRPC_VERSION:
        raise ProtocolError(INVALID_REQUEST, f"Expected jsonrpc={JSONRPC_VERSION}")

    has_id = "id" in msg
    has_method = "method" in msg
    has_result = "result" in msg
    has_error = "error" in msg

    if has_method:
        return "request" if has_id else "notification"
    elif has_result and has_id:
        return "response"
    elif has_error and has_id:
        return "error"
    else:
        raise ProtocolError(INVALID_REQUEST, "Cannot determine message type")


def make_response(request_id: Union[int, str], result: Any) -> Dict[str, Any]:
    """Build a JSON-RPC success response."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "result": result,
    }


def make_error(
    request_id: Optional[Union[int, str]],
    code: int,
    message: str,
    data: Any = None,
) -> Dict[str, Any]:
    """Build a JSON-RPC error response."""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": error,
    }


def make_notification(method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a JSON-RPC notification (no id, no response expected)."""
    msg = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# --- MCP-specific message builders ---

def initialize_result(
    server_name: str,
    server_version: str,
    protocol_version: str,
    capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the MCP initialize result."""
    return {
        "protocolVersion": protocol_version,
        "capabilities": capabilities or {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
        },
        "serverInfo": {
            "name": server_name,
            "version": server_version,
        },
    }


def tools_list_result(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the MCP tools/list result."""
    return {"tools": tools}


def tool_result_content(
    content: List[Dict[str, Any]],
    is_error: bool = False,
) -> Dict[str, Any]:
    """Build the MCP tools/call result."""
    result = {"content": content}
    if is_error:
        result["isError"] = True
    return result


def text_content(text: str) -> Dict[str, Any]:
    """Build a text content block."""
    return {"type": "text", "text": text}


def resources_list_result(resources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the MCP resources/list result."""
    return {"resources": resources}


def resource_read_result(contents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the MCP resources/read result."""
    return {"contents": contents}
