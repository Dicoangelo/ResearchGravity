#!/usr/bin/env python3
"""
Test MCP handshake timing — verifies initialize response comes back fast.

This simulates what Claude Code CLI does:
  1. Send initialize
  2. Send initialized notification
  3. Send tools/list
  4. Measure time to get each response

The server MUST respond to initialize within 2 seconds for Claude Code CLI.
"""

import asyncio
import json
import subprocess
import sys
import time


async def test_handshake():
    """Test that the MCP handshake completes within timeout."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "mcp_raw",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/Users/dicoangelo/researchgravity",
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/Users/dicoangelo",
            "PYTHONPATH": "/Users/dicoangelo/researchgravity",
            "UCW_DATABASE_URL": "postgresql://localhost:5432/ucw_cognitive",
        },
    )

    results = {}

    try:
        # Step 1: Send initialize
        t0 = time.monotonic()
        init_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "timing-test", "version": "1.0"},
            },
        }) + "\n"
        proc.stdin.write(init_msg.encode())
        await proc.stdin.drain()

        # Read initialize response
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
        t1 = time.monotonic()
        init_time = t1 - t0
        init_resp = json.loads(line)
        results["initialize"] = {
            "time_ms": round(init_time * 1000, 1),
            "ok": "result" in init_resp,
            "server": init_resp.get("result", {}).get("serverInfo", {}).get("name", "?"),
        }
        print(f"  initialize: {init_time*1000:.1f}ms {'PASS' if init_time < 2.0 else 'FAIL (>2s)'}")

        # Step 2: Send initialized notification (no response expected)
        notif_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }) + "\n"
        proc.stdin.write(notif_msg.encode())
        await proc.stdin.drain()

        # Small delay to let server process notification
        await asyncio.sleep(0.05)

        # Step 3: Send tools/list
        t2 = time.monotonic()
        tools_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }) + "\n"
        proc.stdin.write(tools_msg.encode())
        await proc.stdin.drain()

        # Read tools/list response
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
        t3 = time.monotonic()
        tools_time = t3 - t2
        tools_resp = json.loads(line)
        tool_count = len(tools_resp.get("result", {}).get("tools", []))
        results["tools_list"] = {
            "time_ms": round(tools_time * 1000, 1),
            "tool_count": tool_count,
        }
        print(f"  tools/list: {tools_time*1000:.1f}ms — {tool_count} tools {'PASS' if tools_time < 2.0 else 'FAIL (>2s)'}")

        total_time = t3 - t0
        results["total"] = {"time_ms": round(total_time * 1000, 1)}
        print(f"  total handshake: {total_time*1000:.1f}ms {'PASS' if total_time < 3.0 else 'FAIL (>3s)'}")

    except asyncio.TimeoutError:
        print("  FAIL: Timeout waiting for response!")
        results["error"] = "timeout"
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    return results


async def main():
    print("\n" + "=" * 60)
    print("  MCP Handshake Timing Test")
    print("=" * 60 + "\n")

    results = await test_handshake()

    print(f"\n{'=' * 60}")
    init_time = results.get("initialize", {}).get("time_ms", 99999)
    total_time = results.get("total", {}).get("time_ms", 99999)

    if init_time < 2000 and total_time < 3000 and "error" not in results:
        print("  PASS: Handshake fast enough for Claude Code CLI")
        print(f"  (initialize: {init_time}ms, total: {total_time}ms)")
    else:
        print("  FAIL: Handshake too slow!")
        if "error" in results:
            print(f"  Error: {results['error']}")
        else:
            print(f"  (initialize: {init_time}ms, total: {total_time}ms)")
            print("  Claude Code CLI requires < 2000ms for initialize")
    print("=" * 60 + "\n")

    return "error" not in results and init_time < 2000


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
