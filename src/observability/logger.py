"""Minimal stderr logger utilities."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger that writes to stderr.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
