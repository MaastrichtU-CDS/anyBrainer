"""Schedulers for LR and more."""

__all__ = [
    "CosineAnnealingWithWarmup",
    "StepwiseParameterScheduler",
]

import logging
from abc import ABC, abstractmethod

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


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


class ParameterScheduler(ABC):
    """
    A general-purpose scheduler for interpolating any scalar parameter value
    (e.g., loss weight, EMA momentum) over training steps.
    """

    @abstractmethod
    def get_value(self, current_step: int) -> dict[str, float]:
        """Get the value of the scheduler at the current step."""
        pass


class StepwiseParameterScheduler(ParameterScheduler):
    """
    A general-purpose scheduler for interpolating any scalar parameter value
    (e.g., loss weight, EMA momentum) over training steps.

    Args:
        start_step (int): Step where scheduling starts.
        end_step (int): Step where scheduling ends.
        start_value (float): Value at `start_step`.
        end_value (float): Value at `end_step`.
        mode (str): Interpolation strategy: 'linear' (default), 'cosine', or 'constant'.
    """

    def __init__(
        self,
        param_name: str,
        *,
        start_step: int,
        end_step: int,
        start_value: float,
        end_value: float,
        mode: str = "linear",
    ):
        if end_step < start_step:
            msg = "end_step must be greater than or equal to start_step."
            logger.error(msg)
            raise ValueError(msg)
        if mode not in {"linear", "cosine", "constant"}:
            msg = f"Unsupported mode: {mode}"
            logger.error(msg)
            raise ValueError(msg)

        self.param_name = param_name
        self.start_step = start_step
        self.end_step = end_step
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.mode = mode

        logger.info(f"\nInitialized {self.param_name} scheduler with:\n"
                    f"start_step={self.start_step}\n"
                    f"end_step={self.end_step}\n"
                    f"start_value={self.start_value}\n"
                    f"end_value={self.end_value}\n"
                    f"mode={self.mode}.")

    def get_value(self, step: int) -> dict[str, float]:
        """Return the scheduled value at the given global step."""
        if step <= self.start_step:
            return {self.param_name: self.start_value}
        if step >= self.end_step:
            return {self.param_name: self.end_value}

        progress = (step - self.start_step) / (self.end_step - self.start_step)

        if self.mode == "linear":
            return {self.param_name: self.start_value + 
                    progress * (self.end_value - self.start_value)}
        elif self.mode == "cosine":
            # Cosine interpolation from start_value to end_value
            import math
            cos_progress = 0.5 * (1 - math.cos(math.pi * progress))
            return {self.param_name: self.start_value + cos_progress * 
                    (self.end_value - self.start_value)}
        elif self.mode == "constant":
            return {self.param_name: self.start_value}
        else:
            msg = f"Unsupported mode: {self.mode}"
            logger.error(msg)
            raise ValueError(msg)