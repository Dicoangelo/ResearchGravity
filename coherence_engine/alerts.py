"""
Coherence Engine — Alert System

Multi-channel alerting for coherence events:
  1. Log file (always)
  2. Desktop notification (macOS, high confidence)
  3. Database record (always, via scorer)
  4. Webhook (optional)
"""

import json
import os
import subprocess
from datetime import datetime
from typing import Optional

from . import config as cfg
from .scorer import CoherenceMoment

import logging

log = logging.getLogger("coherence.alerts")


class AlertSystem:
    """
    Multi-channel alerting for coherence detections.

    Fires alerts through all enabled channels when a
    CoherenceMoment exceeds the minimum confidence threshold.
    """

    def __init__(self):
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._alert_count = 0

    async def notify(self, moment: CoherenceMoment):
        """Send alert through all enabled channels."""
        if moment.confidence < cfg.MIN_ALERT_CONFIDENCE:
            return

        self._alert_count += 1

        # Always log
        self._log_moment(moment)

        # Desktop notification for high confidence
        if moment.confidence >= cfg.HIGH_CONFIDENCE_THRESHOLD and cfg.DESKTOP_NOTIFICATIONS:
            self._desktop_notify(moment)

        # Webhook if configured
        if cfg.WEBHOOK_URL:
            await self._webhook_notify(moment)

    def _log_moment(self, moment: CoherenceMoment):
        """Append to coherence log file."""
        ts = datetime.fromtimestamp(moment.detected_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S")
        platforms = " <-> ".join(moment.platforms)
        entry = (
            f"[{ts}] {moment.coherence_type.upper()} "
            f"| {platforms} | confidence={moment.confidence:.2f} "
            f"| {moment.description}\n"
        )
        try:
            with open(cfg.LOG_FILE, "a") as f:
                f.write(entry)
        except Exception as e:
            log.error(f"Failed to write coherence log: {e}")

        log.info(
            f"COHERENCE: {moment.coherence_type} "
            f"{platforms} conf={moment.confidence:.2f}"
        )

    def _desktop_notify(self, moment: CoherenceMoment):
        """macOS desktop notification via osascript."""
        title = f"UCW Coherence: {moment.coherence_type}"
        platforms = " <-> ".join(moment.platforms)
        body = (
            f"{platforms}\\n"
            f"Confidence: {moment.confidence:.0%}\\n"
            f"{moment.description[:100]}"
        )
        try:
            subprocess.run(
                [
                    "osascript", "-e",
                    f'display notification "{body}" with title "{title}"',
                ],
                capture_output=True,
                timeout=5,
            )
        except Exception as e:
            log.warning(f"Desktop notification failed: {e}")

    async def _webhook_notify(self, moment: CoherenceMoment):
        """POST coherence moment to webhook URL."""
        try:
            import aiohttp

            payload = {
                "moment_id": moment.moment_id,
                "coherence_type": moment.coherence_type,
                "confidence": moment.confidence,
                "platforms": moment.platforms,
                "description": moment.description,
                "time_window_s": moment.time_window_s,
                "signals": moment.signals,
            }
            async with aiohttp.ClientSession() as session:
                await session.post(
                    cfg.WEBHOOK_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                )
        except ImportError:
            log.warning("aiohttp not installed, webhook disabled")
        except Exception as e:
            log.warning(f"Webhook failed: {e}")

    @property
    def alert_count(self) -> int:
        return self._alert_count
