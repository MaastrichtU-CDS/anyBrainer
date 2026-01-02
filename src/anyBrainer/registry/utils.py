"""Utility functions/classes for the registry."""

import logging
from typing import Any
from collections.abc import Callable as ABCCallable

logger = logging.getLogger(__name__)


def check_allowed_types(kind: str, obj: object, allowed_types: tuple[Any, ...]) -> None:
    """Raise TypeError if `obj` is not allowed in the bucket `kind`."""
    # Check if callables are allowed
    if ABCCallable in allowed_types and callable(obj):
        return

    allowed_no_callable = tuple(t for t in allowed_types if t is not ABCCallable)

    if allowed_no_callable:
        if isinstance(obj, type) and issubclass(obj, allowed_no_callable):
            return

    msg = (
        f"{obj!r} cannot be registered in {kind}. " f"Allowed base(s): {allowed_types}"
    )
    logger.error(msg)
    raise TypeError(msg)
