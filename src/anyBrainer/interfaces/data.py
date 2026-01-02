"""Data-related interfaces.

Includes:
- DataExplorer: Abstract interface for dataset exploration
"""

__all__ = [
    "DataExplorer",
]

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Iterator


class DataExplorer(ABC):
    """Abstract interface for dataset exploration."""

    @abstractmethod
    def __init__(self, base_dir: Path):
        """Initialize the data explorer."""
        pass

    @abstractmethod
    def get_all_image_files(
        self,
        files_suffix: str | None = None,
        exts: Sequence[str] = (".nii.gz", ".nii"),
        as_list: bool = True,
        **kwargs,
    ) -> list[Path] | Iterator[Path]:
        """Return all requested files in the dataset."""
        pass
