"""Input validation helpers for MCP tool parameters."""


def clamp_int(val, default: int, lo: int, hi: int) -> int:
    """Clamp an integer parameter to safe range."""
    try:
        v = int(val) if val is not None else default
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def clamp_float(val, default: float, lo: float, hi: float) -> float:
    """Clamp a float parameter to safe range."""
    try:
        v = float(val) if val is not None else default
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))
