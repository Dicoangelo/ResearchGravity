"""Structured logging configuration for UCW daemons."""

import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.expanduser("~/.ucw/logs")


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure rotating file + console logging for a UCW daemon."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on re-init
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler (10 MB max, keep 5 backups)
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, f"{name}.log"),
        maxBytes=10_000_000,
        backupCount=5,
    )
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Console handler (WARNING+ only to keep stderr clean)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.WARNING)
    logger.addHandler(ch)

    # Suppress noisy libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("tqdm").setLevel(logging.CRITICAL)

    # Suppress tqdm via env
    os.environ["TQDM_DISABLE"] = "1"

    return logger
