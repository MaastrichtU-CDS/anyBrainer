"""Schedulers for learning rate."""

__all__ = [
    "CosineAnnealingWithWarmup",
]

import logging
from typing import cast

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.schedulers.utils import resolve_num_list_arg

logger = logging.getLogger(__name__)


@register(RK.LR_SCHEDULER)
class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int | list[int],
        total_iters: int | list[int],
        eta_min: float | list[float] = 0.0,
        last_epoch: int = -1,
        **kwargs,
    ):
        """"
        Cosine annealing with warmup scheduler.

        If `warmup_iters` and `total_iters` are lists, they must be of the same length as
        the number of optimizer parameter groups, and each value will be used for the
        corresponding group -> order matters.

        Args:
            optimizer: PyTorch optimizer.
            warmup_iters: Number of warmup iterations.
            total_iters: Total number of iterations (warmup + cosine decay).
            eta_min: Minimum LR after cosine decay.
            last_epoch: The index of last epoch (or -1 if starting).
        """
        n_groups = len(optimizer.param_groups)

        self.warmup_iters = resolve_num_list_arg(
            warmup_iters, n_groups, f"[CosineAnnealingWithWarmup] `warmup_iters`"
        )
        self.total_iters = resolve_num_list_arg(
            total_iters, n_groups, f"[CosineAnnealingWithWarmup] `total_iters`"
        )
        self.eta_min = resolve_num_list_arg(
            eta_min, n_groups, f"[CosineAnnealingWithWarmup] `eta_min`"
        )

        super().__init__(optimizer, last_epoch)

        logger.info("[CosineAnnealingWithWarmup] Initialized with per-group config:\n" +
                    "\n".join(
                        f"Group {i}: warmup_iters={w}, eta_min={e}" 
                        for i, (w, e) in enumerate(zip(self.warmup_iters, self.eta_min))
                    ))

    def get_lr(self):
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            warmup = self.warmup_iters[i]
            eta_min = self.eta_min[i]

            if self.last_epoch < warmup:
                # Linear warmup
                lr = base_lr * (self.last_epoch + 1) / float(warmup)
            else:
                # Cosine decay
                progress = (self.last_epoch - warmup) / float(self.total_iters[i] - warmup)
                progress = min(max(progress, 0.0), 1.0)  # clamp to [0, 1]
                lr = eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

            lrs.append(lr)
        return lrs