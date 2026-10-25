"""
UCW Raw MCP Server — Sovereign Cognitive Capture Infrastructure.

Public classes are loaded on demand. Importing the protocol or configuration
alone must not initialize capture dependencies or open personal log files.
"""

__version__ = "1.0.0"

_EXPORTS = {
    "RawMCPServer": "server",
    "CaptureEngine": "capture",
    "CaptureEvent": "capture",
    "CaptureDB": "db",
    "CognitiveDatabase": "database",
    "Router": "router",
    "Config": "config",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    from importlib import import_module

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
