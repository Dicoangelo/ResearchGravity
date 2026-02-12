"""
Webhook CLI — Real-time webhook receiver for the UCW ecosystem.

Usage:
    python3 -m webhook start        Start webhook server (foreground)
    python3 -m webhook status       Show event stats from audit trail
    python3 -m webhook test         Send test events to running server
    python3 -m webhook providers    List configured providers
    python3 -m webhook poller       Start queue poller (foreground)
    python3 -m webhook poll-once    Process pending queue items once
    python3 -m webhook queue        Show queue status
"""

import asyncio
import sys
import logging

from . import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("webhook")


async def cmd_start():
    """Start the webhook server."""
    import uvicorn
    from .handlers import register_handlers

    register_handlers(cfg.ENABLED_PROVIDERS)

    print(f"UCW Webhook Receiver v1.0.0")
    print(f"  Host: {cfg.WEBHOOK_HOST}:{cfg.WEBHOOK_PORT}")
    print(f"  Providers: {', '.join(cfg.ENABLED_PROVIDERS)}")
    print(f"  Database: {cfg.PG_DSN}")
    print(f"  Endpoint: POST /webhook/{{provider}}")
    print()

    config = uvicorn.Config(
        "webhook.server:app",
        host=cfg.WEBHOOK_HOST,
        port=cfg.WEBHOOK_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def cmd_status():
    """Show webhook server status from audit trail."""
    import asyncpg

    pool = await asyncpg.create_pool(cfg.PG_DSN, min_size=1, max_size=2)

    try:
        async with pool.acquire() as conn:
            # Check if table exists
            exists = await conn.fetchval(
                """SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'webhook_events'
                )"""
            )

            if not exists:
                print("Webhook audit table does not exist yet.")
                print("Start the server first: python3 -m webhook start")
                return

            total = await conn.fetchval("SELECT COUNT(*) FROM webhook_events")
            by_provider = await conn.fetch(
                """SELECT provider, status, COUNT(*) as cnt
                   FROM webhook_events
                   GROUP BY provider, status ORDER BY provider"""
            )
            recent = await conn.fetch(
                """SELECT provider, event_type, status, processing_time_ms, created_at
                   FROM webhook_events ORDER BY created_at DESC LIMIT 10"""
            )

            # Also check cognitive_events for webhook platforms
            webhook_events = await conn.fetchval(
                """SELECT COUNT(*) FROM cognitive_events
                   WHERE platform LIKE '%-webhook'"""
            )
    finally:
        await pool.close()

    print("=" * 55)
    print("  UCW WEBHOOK STATUS")
    print("=" * 55)
    print(f"  Audit entries:     {total}")
    print(f"  Cognitive events:  {webhook_events} (from webhooks)")
    print(f"  Port:              {cfg.WEBHOOK_PORT}")
    print(f"  Providers:         {', '.join(cfg.ENABLED_PROVIDERS)}")
    print()

    if by_provider:
        print("  Deliveries by provider/status:")
        for r in by_provider:
            print(f"    {r['provider']:18s} {r['status']:10s} {r['cnt']}")
        print()

    if recent:
        print("  Recent deliveries:")
        for r in recent:
            ms = f"{r['processing_time_ms']}ms" if r['processing_time_ms'] else "—"
            print(
                f"    [{r['created_at']:%H:%M:%S}] "
                f"{r['provider']:10s} {r['event_type']:18s} "
                f"-> {r['status']:8s} ({ms})"
            )

    print("=" * 55)


async def cmd_test():
    """Send test webhook events to the running server."""
    try:
        import httpx
    except ImportError:
        print("httpx not installed. Install with: pip install httpx")
        print("Or test manually: curl -X POST http://localhost:3848/webhook/test/github")
        return

    base_url = f"http://{cfg.WEBHOOK_HOST}:{cfg.WEBHOOK_PORT}"

    async with httpx.AsyncClient(timeout=10) as client:
        # Health check first
        try:
            resp = await client.get(f"{base_url}/webhook/health")
            data = resp.json()
            print(f"Server: {data.get('status', 'unknown')} (uptime: {data.get('uptime_s', 0)}s)")
        except Exception as e:
            print(f"Server not reachable at {base_url}: {e}")
            print("Start the server first: python3 -m webhook start")
            return

        # Test each provider
        for provider in cfg.ENABLED_PROVIDERS:
            try:
                resp = await client.post(f"{base_url}/webhook/test/{provider}")
                data = resp.json()
                stored = data.get("events_stored", 0)
                parsed = data.get("events_parsed", 0)
                sample = data.get("sample_content", "")[:80]
                print(f"  {provider:10s} parsed={parsed} stored={stored} | {sample}")
            except Exception as e:
                print(f"  {provider:10s} FAILED: {e}")


def cmd_providers():
    """List configured providers and their secret status."""
    secrets_status = {
        "github": bool(cfg.GITHUB_WEBHOOK_SECRET),
        "slack": bool(cfg.SLACK_SIGNING_SECRET),
        "stripe": bool(cfg.STRIPE_WEBHOOK_SECRET),
        "generic": True,
    }
    print("Webhook Providers")
    print("=" * 55)
    for provider in ["github", "slack", "stripe", "generic"]:
        enabled = provider in cfg.ENABLED_PROVIDERS
        has_secret = secrets_status.get(provider, False)
        status = "ENABLED" if enabled else "disabled"
        secret = "configured" if has_secret else "NOT SET"
        print(f"  {provider:15s} [{status:8s}] secret: {secret}")
    print()
    print(f"Relay secret: {'configured' if cfg.RELAY_SHARED_SECRET else 'NOT SET'}")
    print(f"Endpoint: POST http://{cfg.WEBHOOK_HOST}:{cfg.WEBHOOK_PORT}/webhook/{{provider}}")


async def cmd_poller():
    """Start the queue poller (foreground daemon)."""
    from .poller import poll_loop
    await poll_loop()


async def cmd_poll_once():
    """Process pending queue items once and exit."""
    from .poller import poll_once_cli
    await poll_once_cli()


async def cmd_queue():
    """Show queue status."""
    from .poller import queue_status
    await queue_status()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m webhook <command>")
        print()
        print("Commands:")
        print("  start       Start webhook server (foreground)")
        print("  status      Show event stats from audit trail")
        print("  test        Send test events to running server")
        print("  providers   List configured providers and secrets")
        print("  poller      Start queue poller daemon (foreground)")
        print("  poll-once   Process pending queue items once")
        print("  queue       Show queue status")
        sys.exit(1)

    cmd = sys.argv[1]
    commands = {
        "start": cmd_start,
        "status": cmd_status,
        "test": cmd_test,
        "providers": cmd_providers,
        "poller": cmd_poller,
        "poll-once": cmd_poll_once,
        "queue": cmd_queue,
    }

    handler = commands.get(cmd)
    if not handler:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    import inspect
    if inspect.iscoroutinefunction(handler):
        asyncio.run(handler())
    else:
        handler()


if __name__ == "__main__":
    main()
