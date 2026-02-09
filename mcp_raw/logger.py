"""
File-only logger — NEVER writes to stdout (would corrupt MCP protocol)
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import Config


def _secure_handler(log_path: Path, level: int, fmt: str) -> RotatingFileHandler:
    """Create a rotating file handler with restricted permissions."""
    handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    # Restrict permissions — log files may contain conversation data
    try:
        os.chmod(log_path, 0o600)
    except OSError:
        pass  # Best-effort; file may not exist yet on first call

    return handler


def get_logger(name: str) -> logging.Logger:
    """Get a logger that writes to file only"""
    Config.ensure_dirs()

    logger = logging.getLogger(f"ucw.{name}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fh = _secure_handler(
        Config.LOG_FILE,
        logging.DEBUG,
        "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )
    logger.addHandler(fh)

    # Separate error log
    eh = _secure_handler(
        Config.ERROR_LOG,
        logging.ERROR,
        "%(asctime)s %(levelname)-5s [%(name)s] %(message)s\n%(exc_info)s",
    )
    logger.addHandler(eh)

    # Never propagate to root (which might have stdout handlers)
    logger.propagate = False

    return logger
