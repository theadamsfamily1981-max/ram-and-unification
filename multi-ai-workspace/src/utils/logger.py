"""Logging utilities for Multi-AI Workspace."""

import logging
import sys
from typing import Optional

# Global logger configuration
_logger_initialized = False
_log_level = logging.INFO


def setup_logger(level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up global logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    global _logger_initialized, _log_level

    _log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

    _logger_initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    if not _logger_initialized:
        setup_logger()

    logger = logging.getLogger(name)
    logger.setLevel(_log_level)
    return logger
