"""Interface for workflows."""

from __future__ import annotations

__all__ = [
    "Workflow",
]

from abc import ABC, abstractmethod

class Workflow(ABC):
    """Interface for any workflow."""
    @abstractmethod
    def __init__(self, **extra):
        pass

    @abstractmethod
    def __call__(self):
        pass