"""Classification losses."""

__all__ = [
    "WeightedCrossEntropyLoss",
]

import logging
import warnings
from typing import Any

import torch
import torch.nn as nn

from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.engines.utils import dict_get_as_tensor

logger = logging.getLogger(__name__)

_DEPRECATED_MSG = (
    "WeightedCrossEntropyLoss is deprecated. Prefer CrossEntropyLoss "
    "(or any other loss that accepts `weight`) with a YAML list for `weight`; "
    "LossMixin converts list weights to tensors before instantiation."
)


@register(RK.LOSS)
class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Deprecated: use ``CrossEntropyLoss`` with a YAML ``weight`` list instead.

    ``LossMixin`` converts list ``weight`` values to tensors for any loss that
    accepts that argument, so a dedicated wrapper is no longer needed.

    Kept for backward compatibility; emits ``DeprecationWarning`` on init.
    """

    def __init__(
        self,
        weight: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        warnings.warn(_DEPRECATED_MSG, DeprecationWarning, stacklevel=2)
        logger.warning(_DEPRECATED_MSG)
        if isinstance(weight, tuple):
            weight = list(weight)
        super().__init__(weight=dict_get_as_tensor(weight), **kwargs)
