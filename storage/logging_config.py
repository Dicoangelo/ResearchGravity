"""
Centralized Logging Configuration for ResearchGravity

Provides structured logging with:
1. JSON format for production (machine-readable)
2. Console format for development (human-readable)
3. Request context (request_id, session_id)
4. Performance metrics (duration, counts)

Usage:
    from storage.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Processing session", extra={"session_id": "abc123", "count": 5})
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from contextvars import ContextVar

# Context variables for request tracking
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='')
session_id_ctx: ContextVar[str] = ContextVar('session_id', default='')


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context variables
        if request_id := request_id_ctx.get():
            log_data["request_id"] = request_id
        if session_id := session_id_ctx.get():
            log_data["session_id"] = session_id

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add location for errors
        if record.levelno >= logging.ERROR:
            log_data["location"] = f"{record.pathname}:{record.lineno}"

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        # Color the level name
        color = self.COLORS.get(record.levelname, '')
        level = f"{color}{record.levelname:8}{self.RESET}"

        # Build prefix with context
        prefix_parts = []
        if request_id := request_id_ctx.get():
            prefix_parts.append(f"[{request_id}]")
        if session_id := session_id_ctx.get():
            prefix_parts.append(f"[{session_id[:12]}]")
        prefix = " ".join(prefix_parts)

        # Format message
        msg = record.getMessage()

        # Add extra fields inline
        extras = []
        if hasattr(record, 'extra_fields'):
            for k, v in record.extra_fields.items():
                extras.append(f"{k}={v}")
        extra_str = f" ({', '.join(extras)})" if extras else ""

        # Build final message
        timestamp = datetime.now().strftime("%H:%M:%S")
        name = record.name.split('.')[-1][:15]  # Short module name

        return f"{timestamp} {level} {name:15} {prefix} {msg}{extra_str}"


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds context and extra fields."""

    def process(self, msg, kwargs):
        extra = kwargs.get('extra', {})

        # Store extra fields for formatters
        if extra:
            kwargs['extra'] = {'extra_fields': extra}

        return msg, kwargs


def setup_logging(
    level: str = None,
    json_format: bool = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Default from RG_LOG_LEVEL env.
        json_format: Use JSON format. Default from RG_LOG_JSON env or False.
        log_file: Optional file path for logging.
    """
    # Determine settings from environment or defaults
    level = level or os.environ.get('RG_LOG_LEVEL', 'INFO')
    if json_format is None:
        json_format = os.environ.get('RG_LOG_JSON', '').lower() == 'true'

    # Get root logger for researchgravity
    root_logger = logging.getLogger('researchgravity')
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('qdrant_client').setLevel(logging.WARNING)


def get_logger(name: str) -> ContextLogger:
    """
    Get a logger with context support.

    Args:
        name: Logger name (typically __name__)

    Returns:
        ContextLogger instance
    """
    # Ensure it's under researchgravity namespace
    if not name.startswith('researchgravity'):
        name = f"researchgravity.{name.split('.')[-1]}"

    logger = logging.getLogger(name)
    return ContextLogger(logger, {})


# Auto-setup on import if not already configured
if not logging.getLogger('researchgravity').handlers:
    setup_logging()
