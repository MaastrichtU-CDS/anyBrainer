"""Interfaces for schedulers."""

__all__ = [
    "ParameterScheduler",
]

from abc import ABC, abstractmethod

class ParameterScheduler(ABC):
    """
    A general-purpose scheduler for interpolating any scalar parameter value
    (e.g., loss weight, EMA momentum) over training steps.
    """

    @abstractmethod
    def get_value(self, current_step: int) -> dict[str, float]:
        """Get the value of the scheduler at the current step."""
        pass