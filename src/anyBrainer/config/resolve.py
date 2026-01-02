"""Resolve settings into Callable objects."""

__all__ = [
    "resolve_fn",
    "resolve_metric",
    "resolve_transform",
]

import logging
from typing import Callable, Any, cast

from monai.transforms.compose import Compose

from anyBrainer.registry import get, RegistryKind as RK
from anyBrainer.factories.unit import UnitFactory

logger = logging.getLogger(__name__)


def resolve_fn(
    fn: str | Callable | None,
) -> Callable | Any | None:
    """Get function from config."""
    if isinstance(fn, str):
        return cast(Callable, get(RK.UTIL, fn))
    if callable(fn) or fn is None:
        return fn
    msg = f"Unsupported input type: {type(fn)}"
    logger.error(msg)
    raise ValueError(msg)


def resolve_metric(
    obj: str | dict[str, Any] | Callable | None,
) -> Callable | None:
    """Get function or class from config."""
    if isinstance(obj, str):
        return cast(Callable, get(RK.UTIL, obj))
    if isinstance(obj, dict):
        return UnitFactory.get_metric_from_kwargs(obj)
    if callable(obj) or obj is None:
        return obj
    msg = f"Unsupported input type: {type(obj)}"
    logger.error(msg)
    raise ValueError(msg)


def resolve_transform(
    transform: dict[str, Any] | str | list[Callable] | Callable | None,
) -> list[Callable] | list[Compose] | None:
    """Directly returns the list of transforms."""
    if isinstance(transform, dict):
        return UnitFactory.get_transformslist_from_kwargs(transform)
    if isinstance(transform, str):
        return cast(Callable, get(RK.TRANSFORM, transform))()
    if transform is None:
        return None
    if callable(transform):
        return [transform]
    if isinstance(transform, list):
        return [t for t in transform if callable(t)]  # Filters out non-callables
    msg = f"Unsupported input type: {type(transform)}"
    logger.error(msg)
    raise ValueError(msg)
