"""Logging configuration for the application."""
import logging
import sys
from typing import Optional
from config import settings


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up application logging."""

    log_level = level or settings.LOG_LEVEL
    log_format = format_string or settings.LOG_FORMAT

    # Create logger
    logger = logging.getLogger("crypto_dashboard")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"crypto_dashboard.{name}")


# Initialize default logger
logger = setup_logging()