"""Utility functions for I/O operations."""

__all__ = [
    "resolve_path",
]

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_path(path: Path | str) -> Path:
    """Expand user and resolve path."""
    return Path(path).expanduser().resolve()