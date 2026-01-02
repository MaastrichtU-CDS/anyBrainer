"""Utility functions for transforms."""

__all__ = [
    "assign_key",
]

import logging

logger = logging.getLogger(__name__)


def assign_key(data, key):
    try:
        return data[key]
    except Exception as e:
        msg = f"key {key} not found in data"
        logger.error(msg)
        raise ValueError(msg) from e
