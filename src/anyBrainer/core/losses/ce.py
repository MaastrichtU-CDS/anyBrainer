"""Classification losses."""

__all__ = [
    "WeightedCrossEntropyLoss",
]

from typing import Any

import torch
import torch.nn as nn

from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.engines.utils import dict_get_as_tensor


@register(RK.LOSS)
class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    """`nn.CrossEntropyLoss` that accepts YAML lists for `weight`.

    Uses the same list-to-tensor helper as `CLwAuxModel` (`dict_get_as_tensor`).
    All other `CrossEntropyLoss` kwargs are forwarded unchanged.
    """

    def __init__(
        self,
        weight: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(weight, tuple):
            weight = list(weight)
        super().__init__(weight=dict_get_as_tensor(weight), **kwargs)
