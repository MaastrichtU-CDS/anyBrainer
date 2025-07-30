"""Utility functions for creating and running models."""

from typing import Sequence, Any

import logging

logger = logging.getLogger(__name__)


def get_mlp_head_args(
    in_dim: int,
    out_dim: int,
    num_hidden_layers: int,
    dropout: float | Sequence[float],
    hidden_dim: int | Sequence[int],
    activation: str | Sequence[str],
    activation_kwargs: dict[str, Any] | Sequence[dict[str, Any]]
) -> tuple[list[float], list[int], list[str], list[dict[str, Any]]]:
    """Validate and normalize MLP head arguments."""
    
    # Normalize dropout
    if isinstance(dropout, float):
        dropouts = [dropout] * (num_hidden_layers + 1)
    elif isinstance(dropout, Sequence):
        dropouts = list(dropout)
        if len(dropouts) != num_hidden_layers + 1:
            msg = f"Expected {num_hidden_layers + 1} dropout values, got {len(dropouts)}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"dropout must be float or sequence of floats, got {type(dropout)}"
        logger.error(msg)
        raise TypeError(msg)

    # Normalize hidden_dim
    if isinstance(hidden_dim, int):
        hidden_dims_core = [hidden_dim] * num_hidden_layers
    elif isinstance(hidden_dim, Sequence):
        hidden_dims_core = list(hidden_dim)
        if len(hidden_dims_core) != num_hidden_layers:
            msg = f"Expected {num_hidden_layers} hidden_dim values, got {len(hidden_dims_core)}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"hidden_dim must be int or sequence of ints, got {type(hidden_dim)}"
        logger.error(msg)
        raise TypeError(msg)
    
    hidden_dims = [in_dim] + hidden_dims_core + [out_dim]

    # Normalize activation
    if isinstance(activation, str):
        activations = [activation] * (num_hidden_layers + 1)
    elif isinstance(activation, Sequence):
        activations = list(activation)
        if len(activations) != num_hidden_layers + 1:
            msg = f"Expected {num_hidden_layers + 1} activation values, got {len(activations)}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"activation must be str or sequence of str, got {type(activation)}"
        logger.error(msg)
        raise TypeError(msg)
    
    # Normalize activation_kwargs
    if isinstance(activation_kwargs, dict):
        activation_kwargs = [activation_kwargs] * (num_hidden_layers + 1)
    elif isinstance(activation_kwargs, Sequence):
        activation_kwargs = list(activation_kwargs)
        if len(activation_kwargs) != num_hidden_layers + 1:
            msg = f"Expected {num_hidden_layers + 1} activation_kwargs values, got {len(activation_kwargs)}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"activation_kwargs must be dict or sequence of dict, got {type(activation_kwargs)}"
        logger.error(msg)
        raise TypeError(msg)
    
    return dropouts, hidden_dims, activations, activation_kwargs