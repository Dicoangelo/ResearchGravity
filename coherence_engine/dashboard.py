"""
Coherence Engine — TUI Dashboard

Live terminal dashboard showing coherence status, moments,
and emergence signals. Uses rich for rendering.

Usage:
    python3 -m coherence_engine dashboard
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import asyncpg

from . import config as cfg

import logging

log = logging.getLogger("coherence.dashboard")

# Try rich, fall back to plain ANSI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class CoherenceDashboard:
    """
    Live TUI dashboard for coherence monitoring.

    Shows:
    - Platform activity (events per platform, last seen)
    - Coherence moments (recent, by type, confidence distribution)
    - Emergence signals (active concept clusters, meta-cognitive events)
    - Live feed of new detections
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self._pool = pool
        self._owns_pool = pool is None
        self._running = False

    async def _ensure_pool(self):
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                cfg.PG_DSN, min_size=1, max_size=3, command_timeout=15
            )
            self._owns_pool = True

    async def get_data(self) -> Dict:
        """Fetch all dashboard data in one pass."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            # Platform stats
            platforms = await conn.fetch("""
                SELECT platform, COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') AS today,
                       MAX(created_at) AS last_seen
                FROM cognitive_events
                GROUP BY platform ORDER BY total DESC
            """)

            # Coherence moments summary
            moments_total = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")
            moments_24h = await conn.fetchval(
                "SELECT COUNT(*) FROM coherence_moments WHERE created_at > NOW() - INTERVAL '24 hours'"
            )

            # By type
            by_type = await conn.fetch("""
                SELECT coherence_type, COUNT(*) AS cnt, AVG(confidence) AS avg_conf
                FROM coherence_moments GROUP BY coherence_type ORDER BY cnt DESC
            """)

            # Confidence distribution
            conf_dist = await conn.fetch("""
                SELECT
                    COUNT(*) FILTER (WHERE confidence >= 0.90) AS tier_90,
                    COUNT(*) FILTER (WHERE confidence >= 0.80 AND confidence < 0.90) AS tier_80,
                    COUNT(*) FILTER (WHERE confidence >= 0.70 AND confidence < 0.80) AS tier_70,
                    COUNT(*) FILTER (WHERE confidence >= 0.60 AND confidence < 0.70) AS tier_60,
                    COUNT(*) FILTER (WHERE confidence < 0.60) AS tier_low
                FROM coherence_moments
            """)

            # Top 5 recent moments with event content
            recent = await conn.fetch("""
                SELECT moment_id, confidence, coherence_type, platforms, description,
                       event_ids, created_at
                FROM coherence_moments
                ORDER BY created_at DESC LIMIT 5
            """)

            # Emergence signals (last 24h)
            emergence = await conn.fetch("""
                SELECT instinct_layer->>'gut_signal' AS gut,
                       COUNT(*) AS cnt
                FROM cognitive_events
                WHERE created_at > NOW() - INTERVAL '24 hours'
                  AND instinct_layer->>'gut_signal' IS NOT NULL
                GROUP BY gut ORDER BY cnt DESC
            """)

            # Top topics in moments
            top_topics = await conn.fetch("""
                SELECT ce.light_layer->>'topic' AS topic, COUNT(DISTINCT cm.moment_id) AS cnt
                FROM coherence_moments cm
                CROSS JOIN LATERAL unnest(cm.event_ids) AS eid
                JOIN cognitive_events ce ON ce.event_id = eid
                WHERE ce.light_layer->>'topic' IS NOT NULL
                GROUP BY topic ORDER BY cnt DESC LIMIT 8
            """)

            # Embedded count
            embedded = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")

        return {
            "platforms": [dict(p) for p in platforms],
            "moments_total": moments_total,
            "moments_24h": moments_24h,
            "by_type": [dict(t) for t in by_type],
            "confidence_dist": dict(conf_dist[0]) if conf_dist else {},
            "recent_moments": [dict(r) for r in recent],
            "emergence": [dict(e) for e in emergence],
            "top_topics": [dict(t) for t in top_topics],
            "embedded": embedded,
        }

    def render_plain(self, data: Dict) -> str:
        """Render dashboard as plain ANSI text."""
        now = datetime.now().strftime("%H:%M:%S")

        lines = []
        lines.append(f"\033[1;36m{'=' * 70}\033[0m")
        lines.append(f"\033[1;37m  UCW COHERENCE DASHBOARD          {now}\033[0m")
        lines.append(f"\033[1;36m{'=' * 70}\033[0m")
        lines.append("")

        # Platforms
        lines.append("\033[1;33m  PLATFORMS\033[0m")
        for p in data["platforms"]:
            last = p["last_seen"]
            if last:
                ago = datetime.now(last.tzinfo) - last if hasattr(last, 'tzinfo') and last.tzinfo else timedelta(0)
                ago_str = f"{ago.seconds // 60}m ago" if ago.total_seconds() < 3600 else f"{ago.seconds // 3600}h ago"
            else:
                ago_str = "never"
            status = "\033[32m●\033[0m" if p["today"] > 0 else "\033[31m○\033[0m"
            lines.append(f"    {status} {p['platform']:20s} {p['total']:>8,} total  {p['today']:>6,} today  ({ago_str})")
        lines.append("")

        # Moments summary
        lines.append("\033[1;33m  COHERENCE MOMENTS\033[0m")
        lines.append(f"    Total: {data['moments_total']}  |  Last 24h: {data['moments_24h']}  |  Embedded: {data['embedded']:,}")
        lines.append("")

        for t in data["by_type"]:
            lines.append(f"    {t['coherence_type']:20s} {t['cnt']:>4} moments  (avg {t['avg_conf']:.0%})")
        lines.append("")

        # Confidence distribution
        cd = data["confidence_dist"]
        if cd:
            lines.append("\033[1;33m  CONFIDENCE DISTRIBUTION\033[0m")
            tiers = [
                ("90-100%", cd.get("tier_90", 0), "\033[32m"),
                ("80-89%",  cd.get("tier_80", 0), "\033[32m"),
                ("70-79%",  cd.get("tier_70", 0), "\033[33m"),
                ("60-69%",  cd.get("tier_60", 0), "\033[33m"),
                ("<60%",    cd.get("tier_low", 0), "\033[31m"),
            ]
            max_bar = max(t[1] for t in tiers) or 1
            for label, count, color in tiers:
                bar_len = int(count / max_bar * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                lines.append(f"    {label:>8s} {color}{bar}\033[0m {count}")
            lines.append("")

        # Top topics
        if data["top_topics"]:
            lines.append("\033[1;33m  TOP COHERENCE TOPICS\033[0m")
            for t in data["top_topics"]:
                lines.append(f"    {t['topic']:25s} {t['cnt']:>4} moments")
            lines.append("")

        # Emergence signals
        if data["emergence"]:
            lines.append("\033[1;33m  EMERGENCE SIGNALS (24h)\033[0m")
            for e in data["emergence"]:
                lines.append(f"    {e['gut']:25s} {e['cnt']:>6}")
            lines.append("")

        # Recent moments
        if data["recent_moments"]:
            lines.append("\033[1;33m  LATEST MOMENTS\033[0m")
            for m in data["recent_moments"]:
                desc = (m["description"] or "")[:80]
                plats = " <-> ".join(m["platforms"]) if m["platforms"] else "?"
                lines.append(
                    f"    \033[1m{m['confidence']:.0%}\033[0m {m['coherence_type']:15s} "
                    f"{plats:35s} {desc}"
                )
            lines.append("")

        lines.append(f"\033[1;36m{'=' * 70}\033[0m")
        lines.append("  [q]uit  [r]efresh                   ")

        return "\n".join(lines)

    def render_rich(self, data: Dict) -> Layout:
        """Render dashboard using rich library."""
        console = Console()
        layout = Layout()

        # Header
        header = Panel(
            Text("UCW COHERENCE DASHBOARD", style="bold cyan", justify="center"),
            style="cyan",
        )

        # Platforms table
        plat_table = Table(title="Platforms", show_header=True)
        plat_table.add_column("Status", width=3)
        plat_table.add_column("Platform", width=18)
        plat_table.add_column("Total", justify="right")
        plat_table.add_column("Today", justify="right")

        for p in data["platforms"]:
            status = "[green]●[/green]" if p["today"] > 0 else "[red]○[/red]"
            plat_table.add_row(status, p["platform"], f"{p['total']:,}", f"{p['today']:,}")

        # Moments table
        mom_table = Table(title="Recent Moments", show_header=True)
        mom_table.add_column("Conf", width=5)
        mom_table.add_column("Type", width=15)
        mom_table.add_column("Platforms", width=30)
        mom_table.add_column("Description", width=50)

        for m in data["recent_moments"]:
            plats = " <-> ".join(m["platforms"]) if m["platforms"] else "?"
            desc = (m["description"] or "")[:50]
            conf = f"{m['confidence']:.0%}"
            mom_table.add_row(conf, m["coherence_type"], plats, desc)

        return plat_table, mom_table

    async def run_live(self, refresh_s: int = None):
        """Run the live dashboard loop."""
        refresh = refresh_s or cfg.DASHBOARD_REFRESH_S
        self._running = True

        print("\033[2J\033[H")  # Clear screen
        print("Loading dashboard...")

        try:
            while self._running:
                data = await self.get_data()
                output = self.render_plain(data)
                print("\033[2J\033[H")  # Clear screen
                print(output)
                await asyncio.sleep(refresh)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            if self._owns_pool and self._pool:
                await self._pool.close()

    async def snapshot(self) -> str:
        """Get a single dashboard snapshot as text."""
        data = await self.get_data()
        return self.render_plain(data)
