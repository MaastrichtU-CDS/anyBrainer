"""Schedulers for learning rate."""

__all__ = [
    "CosineAnnealingWithWarmup",
    "PolyLRWithWarmup",
]

import logging

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
        start_iter: int | list[int] = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        """Cosine annealing with warmup scheduler.

        If `warmup_iters` and `total_iters` are lists, they must be of the same length as
        the number of optimizer parameter groups, and each value will be used for the
        corresponding group -> order matters.

        Args:
            optimizer: PyTorch optimizer.
            warmup_iters: Number of warmup iterations.
            total_iters: Total number of iterations (warmup + cosine decay).
            eta_min: Minimum LR after cosine decay.
            start_iter: The iteration to start the scheduler.
            last_epoch: The index of last epoch (or -1 if starting).
        """
        n_groups = len(optimizer.param_groups)

        self.warmup_iters = resolve_num_list_arg(
            warmup_iters, n_groups, "[CosineAnnealingWithWarmup] `warmup_iters`"
        )
        self.total_iters = resolve_num_list_arg(
            total_iters, n_groups, "[CosineAnnealingWithWarmup] `total_iters`"
        )
        self.eta_min = resolve_num_list_arg(
            eta_min, n_groups, "[CosineAnnealingWithWarmup] `eta_min`"
        )
        self.start_iter = resolve_num_list_arg(
            start_iter, n_groups, "[CosineAnnealingWithWarmup] `start_iter`"
        )

        super().__init__(optimizer, last_epoch)

        logger.info(
            "[CosineAnnealingWithWarmup] Initialized with per-group config:\n"
            + "\n".join(
                f"Group {i}: warmup_iters={w}, eta_min={e}"
                for i, (w, e) in enumerate(zip(self.warmup_iters, self.eta_min))
            )
        )

    def get_lr(self):
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            start_iter = self.start_iter[i]
            warmup = self.warmup_iters[i]
            eta_min = self.eta_min[i]
            total_iters = self.total_iters[i]

            if self.last_epoch < start_iter:
                # Phase 1: Inactive
                lr = 0.0
            else:
                # Effective step after start
                effective_epoch = self.last_epoch - start_iter
                if effective_epoch < warmup:
                    # Phase 2: Linear warm-up
                    lr = base_lr * (effective_epoch + 1) / float(warmup)
                else:
                    # Phase 3: Cosine decay
                    decay_iters = total_iters - start_iter - warmup
                    progress = (effective_epoch - warmup) / float(decay_iters)
                    progress = min(max(progress, 0.0), 1.0)  # clamp to [0, 1]
                    lr = eta_min + (base_lr - eta_min) * 0.5 * (
                        1.0 + math.cos(math.pi * progress)
                    )

            lrs.append(lr)

        return lrs


@register(RK.LR_SCHEDULER)
class PolyLRWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int | list[int],
        total_iters: int | list[int],
        exponent: float | list[float] = 0.9,
        start_iter: int | list[int] = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        """Polynomial LR decay with linear warmup (nnU-Net style).

        If list args are provided, they must match the number of optimizer
        parameter groups; order matters.

        Phases per group:
            1. ``last_epoch < start_iter`` -> LR = 0
            2. Linear warmup -> ``base_lr * (effective_step + 1) / warmup_iters``
            3. Poly decay -> ``base_lr * (1 - progress) ** exponent``

        Args:
            optimizer: PyTorch optimizer.
            warmup_iters: Number of warmup iterations.
            total_iters: Total number of iterations (warmup + poly decay).
            exponent: Polynomial decay exponent (nnU-Net default: 0.9).
            start_iter: Iteration to start the scheduler.
            last_epoch: Index of last epoch (or -1 if starting).
        """
        n_groups = len(optimizer.param_groups)

        self.warmup_iters = resolve_num_list_arg(
            warmup_iters, n_groups, "[PolyLRWithWarmup] `warmup_iters`"
        )
        self.total_iters = resolve_num_list_arg(
            total_iters, n_groups, "[PolyLRWithWarmup] `total_iters`"
        )
        self.exponent = resolve_num_list_arg(
            exponent, n_groups, "[PolyLRWithWarmup] `exponent`"
        )
        self.start_iter = resolve_num_list_arg(
            start_iter, n_groups, "[PolyLRWithWarmup] `start_iter`"
        )

        super().__init__(optimizer, last_epoch)

        logger.info(
            "[PolyLRWithWarmup] Initialized with per-group config:\n"
            + "\n".join(
                f"Group {i}: warmup_iters={w}, exponent={e}"
                for i, (w, e) in enumerate(zip(self.warmup_iters, self.exponent))
            )
        )

    def get_lr(self):
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            start_iter = self.start_iter[i]
            warmup = self.warmup_iters[i]
            exponent = self.exponent[i]
            total_iters = self.total_iters[i]

            if self.last_epoch < start_iter:
                lr = 0.0
            else:
                effective_epoch = self.last_epoch - start_iter
                if effective_epoch < warmup:
                    lr = base_lr * (effective_epoch + 1) / float(warmup)
                else:
                    decay_iters = total_iters - start_iter - warmup
                    progress = (effective_epoch - warmup) / float(decay_iters)
                    progress = min(max(progress, 0.0), 1.0)
                    lr = base_lr * (1.0 - progress) ** exponent

            lrs.append(lr)

        return lrs
