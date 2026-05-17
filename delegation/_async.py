"""Loop-aware coroutine runner.

Prevents a recurring bug class: `asyncio.run(coro)` called from a
context that already has a running event loop (the async delegation
pipeline, pytest-asyncio). `asyncio.run()` raises RuntimeError, the
coroutine is created-but-never-awaited (RuntimeWarning), and a broad
`except` silently degrades the LLM path to a heuristic/keyword
fallback — so LLM decomposition / semantic similarity / trust scoring
never actually run in production whenever something drives the
pipeline under a loop.

`run_async` runs the coroutine to completion whether or not a loop is
already running.

(A near-identical helper predates this in verifier.py; that file is
intentionally left untouched here to avoid blast radius on working,
tested code. New code and the fixed sites should use this module.)
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Optional


def run_async(coro: Awaitable[Any], timeout: Optional[float] = None) -> Any:
    """Run ``coro`` to completion from a sync OR async context.

    timeout: optional ceiling on the cross-thread wait used only when a
    loop is already running. Default ``None`` preserves the callers'
    prior unbounded ``asyncio.run`` behavior — their own inner
    ``asyncio.wait_for`` still bounds the actual LLM calls.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run directly.
        return asyncio.run(coro)
    # A loop is already running — run on a fresh loop in a worker
    # thread so we never call asyncio.run() inside a live loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=timeout)
