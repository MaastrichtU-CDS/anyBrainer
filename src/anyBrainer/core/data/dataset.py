"""Contains datasets for all tasks.

Data should come as a list of dictionaries, including the .npy filepath.
"""

__all__ = [
    "CustomDataset",
]

from typing import Union, List, Callable, Sequence, Optional

# pyright: reportPrivateImportUsage=false
from monai.data import Dataset as MONAIDataset


class CustomDataset(MONAIDataset):
    """Template for building a custom PyTorch dataset inheriting from MONAI's.

    default dataset class that offers some benefits:
    - Ensures transforms are properly composed.
    - Ensures __get_item__() handles slices or list of indices.

    Args:
        data: list of sample-level dictionaries.
        transforms: composed transforms or sequence of transforms.
    """

    def __init__(
        self,
        data: List[dict],
        transforms: Optional[Union[Callable, Sequence[Callable]]] = None,
    ):
        super().__init__(data=data, transform=transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Can change for contrastive learning, e.g. generate two views with
        separate transforms, etc."""
        return super().__getitem__(idx)
