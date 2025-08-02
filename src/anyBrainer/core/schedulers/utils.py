"""Utility functions for schedulers."""

import logging

logger = logging.getLogger(__name__)


def resolve_num_list_arg(
    arg_value: int | float | list[int] | list[float], 
    n_groups: int,
    arg_name: str,
) -> list[int] | list[float]:
    """Resolve single-number or list of numbers args into a list."""
    if isinstance(arg_value, (int, float)):
        return [arg_value] * n_groups # type: ignore
    
    if len(arg_value) != n_groups:
        msg = f"[{arg_name}] Expected {n_groups} values, got {len(arg_value)}."
        logger.error(msg)
        raise ValueError(msg)
    
    return arg_value