"""Helper classes for scheduling parameter values."""

__all__ = [
    "StepwiseParameterScheduler",
]

import logging

from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.interfaces import ParameterScheduler

logger = logging.getLogger(__name__)
    

@register(RK.PARAM_SCHEDULER)
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

        logger.info(f"Initialized {self.param_name} scheduler with: "
                    f"start_step={self.start_step}, end_step={self.end_step}, "
                    f"start_value={self.start_value}, end_value={self.end_value}, "
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