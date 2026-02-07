"""
Capture CLI — Cross-platform live capture commands.

Usage:
    python3 -m capture start          # Start continuous capture
    python3 -m capture poll-once      # Single poll cycle, then exit
    python3 -m capture status         # Show adapter health + stats
    python3 -m capture list-adapters  # List available adapters
"""

import asyncio
import json
import sys
import logging

from .manager import CaptureManager
from . import config as cfg


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("capture")


def _build_manager() -> CaptureManager:
    """Create manager and register all configured adapters."""
    manager = CaptureManager()

    # ChatGPT adapter (always available — watches export dir)
    try:
        from .chatgpt.adapter import ChatGPTAdapter
        manager.register(ChatGPTAdapter())
    except Exception as exc:
        log.warning(f"ChatGPT adapter unavailable: {exc}")

    # Cursor adapter (available if data dir exists)
    try:
        from .cursor.adapter import CursorAdapter
        manager.register(CursorAdapter())
    except Exception as exc:
        log.warning(f"Cursor adapter unavailable: {exc}")

    # Grok adapter (available if API key set)
    try:
        from .grok.adapter import GrokAdapter
        adapter = GrokAdapter()
        manager.register(adapter)
    except Exception as exc:
        log.warning(f"Grok adapter unavailable: {exc}")

    return manager


async def cmd_start():
    """Start continuous capture with all configured adapters."""
    print("Starting cross-platform capture...")
    print(f"  Database: {cfg.PG_DSN}")
    print(f"  ChatGPT export: {cfg.CHATGPT_EXPORT_PATH}")
    print(f"  Cursor data: {cfg.CURSOR_DATA_DIR}")
    print(f"  Grok API: {'configured' if cfg.GROK_API_KEY else 'not configured'}")
    print()

    manager = _build_manager()
    await manager.start()


async def cmd_poll_once():
    """Single poll cycle across all adapters."""
    print("Running single poll cycle...")
    manager = _build_manager()
    results = await manager.poll_all()

    print("\nResults:")
    total = 0
    for platform, count in results.items():
        print(f"  {platform}: {count} events captured")
        total += count
    print(f"\n  Total: {total} events")


async def cmd_status():
    """Show adapter health and capture stats."""
    manager = _build_manager()

    # Quick initialize to check DB
    try:
        import asyncpg
        pool = await asyncpg.create_pool(
            cfg.PG_DSN, min_size=1, max_size=2, command_timeout=10,
        )

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT platform, COUNT(*) as cnt
                   FROM cognitive_events
                   WHERE platform IN ('chatgpt', 'cursor', 'grok')
                   GROUP BY platform"""
            )
            rows = await conn.fetch(
                """SELECT platform, COUNT(*) as cnt
                   FROM cognitive_events
                   WHERE platform IN ('chatgpt', 'cursor', 'grok')
                   GROUP BY platform"""
            )

        await pool.close()

        print("Capture Status")
        print("=" * 50)
        print(f"\nDatabase: {cfg.PG_DSN}")
        print(f"\nEvents by platform:")
        for r in rows:
            print(f"  {r['platform']}: {r['cnt']}")
        if not rows:
            print("  (no capture events yet)")

    except Exception as exc:
        print(f"Database unavailable: {exc}")

    print(f"\nConfiguration:")
    print(f"  ChatGPT export: {cfg.CHATGPT_EXPORT_PATH}")
    print(f"  ChatGPT poll interval: {cfg.CHATGPT_POLL_INTERVAL_S}s")
    print(f"  Cursor data: {cfg.CURSOR_DATA_DIR}")
    print(f"  Cursor poll interval: {cfg.CURSOR_POLL_INTERVAL_S}s")
    print(f"  Grok API: {'configured' if cfg.GROK_API_KEY else 'not configured'}")
    print(f"  Grok poll interval: {cfg.GROK_POLL_INTERVAL_S}s")
    print(f"  Dedup window: {cfg.DEDUP_WINDOW_HOURS}h")


def cmd_list_adapters():
    """List available adapters and their configuration."""
    print("Available Capture Adapters")
    print("=" * 50)

    adapters = [
        ("ChatGPT", "chatgpt", "Export-diff polling + OpenAI API", cfg.CHATGPT_EXPORT_PATH),
        ("Cursor", "cursor", "Workspace file watcher", cfg.CURSOR_DATA_DIR),
        ("Grok/X", "grok", "X API polling", "API key" if cfg.GROK_API_KEY else "not configured"),
    ]

    for name, platform, method, source in adapters:
        print(f"\n  {name} ({platform})")
        print(f"    Method: {method}")
        print(f"    Source: {source}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m capture <command>")
        print()
        print("Commands:")
        print("  start          Start continuous capture")
        print("  poll-once      Single poll cycle, then exit")
        print("  status         Show adapter health + stats")
        print("  list-adapters  List available adapters")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "start":
        asyncio.run(cmd_start())
    elif cmd == "poll-once":
        asyncio.run(cmd_poll_once())
    elif cmd == "status":
        asyncio.run(cmd_status())
    elif cmd == "list-adapters":
        cmd_list_adapters()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
