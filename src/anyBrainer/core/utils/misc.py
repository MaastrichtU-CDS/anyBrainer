"""General-purpose utility functions.

Other anyBrainer.core.utils modules are allowed to import from this
module; not vice versa.
"""

__all__ = [
    "ensure_tuple_dim",
    "callable_name",
]

import logging
from typing import Any, Sequence


logger = logging.getLogger(__name__)


def ensure_tuple_dim(
    value: Any | Sequence[Any],
    dim: int,
) -> tuple[Any, ...]:
    """Ensure that a value is a tuple of length `dim`."""
    if isinstance(value, (list, tuple)):
        if len(value) == dim:
            return tuple(value)
        else:
            msg = f"Expected tuple of length {dim}, got {len(value)}"
            logger.error(msg)
            raise ValueError(msg)
    elif isinstance(value, int | float):
        return (value,) * dim
    else:
        msg = f"Expected tuple or list, got {type(value)}"
        logger.error(msg)
        raise ValueError(msg)


def callable_name(obj: Any | None) -> str:
    """Get name of a callable or a descriptive string if None/non-callable."""
    if obj is None:
        return "None"
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__class__.__name__
    return str(obj)
