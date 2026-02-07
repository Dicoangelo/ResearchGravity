"""
Coherence Engine — CLI Entry Point

Commands:
  start     Run daemon in foreground (poll mode)
  oneshot   Process all events once and exit
  status    Show engine status
"""

import asyncio
import logging
import sys

from .daemon import CoherenceDaemon
from . import config as cfg


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def cmd_start():
    """Run the coherence daemon in foreground."""
    daemon = CoherenceDaemon()
    await daemon.run()


async def cmd_oneshot():
    """Process all events once and exit."""
    daemon = CoherenceDaemon()
    moments = await daemon.oneshot()
    print(f"Coherence scan complete: {moments} moments detected")


async def cmd_status():
    """Show current coherence engine status."""
    import asyncpg

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=1, max_size=2)

    async with pool.acquire() as conn:
        events = await conn.fetchval("SELECT COUNT(*) FROM cognitive_events")
        embedded = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
        moments = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")
        platforms = await conn.fetch(
            "SELECT platform, COUNT(*) as cnt FROM cognitive_events GROUP BY platform"
        )
        recent_moments = await conn.fetch(
            """SELECT coherence_type, confidence, description, platforms
               FROM coherence_moments
               ORDER BY detected_ns DESC LIMIT 5"""
        )

    await pool.close()

    print("=" * 60)
    print("  UCW COHERENCE ENGINE STATUS")
    print("=" * 60)
    print(f"  Events:     {events:,}")
    print(f"  Embedded:   {embedded:,}")
    print(f"  Moments:    {moments:,}")
    print()
    print("  Platforms:")
    for p in platforms:
        print(f"    {p['platform']:20s}  {p['cnt']:,} events")
    print()

    if recent_moments:
        print("  Recent Coherence Moments:")
        for m in recent_moments:
            plats = " <-> ".join(m["platforms"])
            print(f"    [{m['confidence']:.0%}] {m['coherence_type']}")
            print(f"          {plats}")
            print(f"          {m['description'][:80]}")
            print()
    else:
        print("  No coherence moments detected yet.")
        print("  Run: python3 -m coherence_engine oneshot")

    print("=" * 60)


def main():
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python3 -m coherence_engine <command>")
        print()
        print("Commands:")
        print("  start     Run daemon (foreground, polls every 10s)")
        print("  oneshot   One-shot scan of all embedded events")
        print("  status    Show engine status")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "start":
        asyncio.run(cmd_start())
    elif cmd == "oneshot":
        asyncio.run(cmd_oneshot())
    elif cmd == "status":
        asyncio.run(cmd_status())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


main()
