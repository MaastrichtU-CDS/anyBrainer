"""Schedulers for learning rate."""

__all__ = [
    "CosineAnnealingWithWarmup",
]

import logging

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

from anyBrainer.registry import register, RegistryKind as RK

logger = logging.getLogger(__name__)


@register(RK.LR_SCHEDULER)
class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        total_iters: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        **kwargs,
    ):
        """"
        Args:
            optimizer: PyTorch optimizer.
            warmup_iters: Number of warmup iterations.
            total_iters: Total number of iterations (warmup + cosine decay).
            eta_min: Minimum LR after cosine decay.
            last_epoch: The index of last epoch (or -1 if starting).
        """
        if total_iters <= warmup_iters:
            msg = "total_iters must be greater than warmup_iters."
            logger.error(msg)
            raise ValueError(msg)

        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

        logger.info(f"\nInitialized CosineAnnealingWithWarmup scheduler with:\n"
                    f"warmup_iters={self.warmup_iters}\n"
                    f"total_iters={self.total_iters}\n"
                    f"eta_min={self.eta_min}\n"
                    f"last_step={self.last_epoch}.")

    def get_lr(self):
        """Get current learning rate for all model params."""
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup_iters: # warm-up
                lr = base_lr * (self.last_epoch + 1) / float(self.warmup_iters)
            else: # cosine decay for total_iters - warmup_iters iterations
                prog = (self.last_epoch - self.warmup_iters) / float(
                        self.total_iters - self.warmup_iters)
                prog = min(max(prog, 0.0), 1.0) # clamp
                lr = self.eta_min + (base_lr - self.eta_min) * 0.5 * (
                        1.0 + math.cos(math.pi * prog))
            lrs.append(lr)
        return lrs