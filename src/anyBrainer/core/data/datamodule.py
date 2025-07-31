"""
PyTorch Lightning DataModules to control datasets, dataloaders, and 
splits alltogether. 

Includes:
- BaseDataModule: Base class for all datamodules
- MAEDataModule: DataModule for masked autoencoder
- ContrastiveDataModule: DataModule for contrastive learning

The datamodule is responsible for:
- Creating and managing datasets
- Creating and managing dataloaders
- Splitting data into train/val/test sets
- Setting random seeds for reproducibility
- Saving and loading state_dicts
"""

from __future__ import annotations

__all__ = [
    "MAEDataModule",
    "ContrastiveDataModule",
]

import logging
from pathlib import Path
from typing import Callable, Literal, Any, TYPE_CHECKING, cast
from collections import defaultdict, Counter

import lightning.pytorch as pl
import numpy as np
import torch
from tqdm import tqdm

# pyright: reportPrivateImportUsage=false
from monai.data import Dataset as MONAIDataset
from monai.data import DataLoader as MONAIDataLoader
from monai.data.utils import set_rnd, list_data_collate

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from anyBrainer.registry import register
from anyBrainer.core.data.utils import (
    check_data_dir_exists,
    parse_filename_nested_nifti,
    get_summary_msg,
    resolve_transform,
    resolve_fn,
)
from anyBrainer.core.utils import (
    split_data_by_subjects, 
    resolve_path,
    make_worker_init_fn,
)
from anyBrainer.core.transforms import(
    get_mae_train_transforms,
    get_mae_val_transforms,
    get_contrastive_train_transforms,
    get_contrastive_val_transforms,
    get_predict_transforms,
)
from anyBrainer.core.data.explorer import DataHandler
from anyBrainer.registry import RegistryKind as RK

STAGE_SEED_OFFSET = {
    "fit": 0,
    "validate": 1000,
    "test": 2000,
    "predict": 3000,
}
STAGE_LOG_PREFIX = {
    "fit": "train",
    "validate": "val",
    "test": "test",
    "predict": "predict",
}
MAX_SEED = 2**32 - 1

logger = logging.getLogger(__name__)


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule class.
    
    Args: 
        data_dir: data directory
        data_handler_kwargs: kwargs for DataHandler
        batch_size: batch size for dataloaders
        num_workers: number of workers for dataloaders
        dataloader_kwargs: kwargs for DataLoader
        train_val_test_split: train/val/test split
        worker_logging_fn: function to log worker information
        worker_seeding_fn: function to seed worker
        collate_fn: function to collate data
        seed: random seed for reproducibility
        random_state: random state for reproducibility
        train_transforms: transforms for train
        val_transforms: transforms for val
        test_transforms: transforms for test
        predict_transforms: transforms for predict
        predict_on_test: whether to predict on test set
    """
    def __init__(
        self,
        data_dir: Path | str,
        data_handler_kwargs: dict[str, Any] | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        dataloader_kwargs: dict[str, Any] | None = None,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        worker_logging_fn: Callable | str | None = None,
        worker_seeding_fn: Callable | str | None = set_rnd,
        collate_fn: Callable | str | None = list_data_collate,
        seed: int | None = None,
        random_state: np.random.RandomState | None = None,
        train_transforms: dict[str, Any] | str | list[Callable] | None = None,
        val_transforms: dict[str, Any] | str | list[Callable] | None = None,
        test_transforms: dict[str, Any] | str | list[Callable] | None = None,
        predict_transforms: dict[str, Any] | str | list[Callable] | None = None,
        predict_on_test: bool = False,
    ):
        super().__init__()
        self.data_dir = resolve_path(data_dir)

        if data_handler_kwargs is None:
            data_handler_kwargs = {}
        self.data_handler_kwargs = data_handler_kwargs

        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        self.dataloader_kwargs = dataloader_kwargs

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.seed = seed

        self.set_random_state(seed, random_state)

        self.worker_logging_fn = resolve_fn(worker_logging_fn)
        self.worker_seeding_fn = resolve_fn(worker_seeding_fn)
        self.collate_fn = resolve_fn(collate_fn)

        self.train_transforms = resolve_transform(train_transforms)
        self.val_transforms = resolve_transform(val_transforms)
        self.test_transforms = resolve_transform(test_transforms)
        self.predict_transforms = resolve_transform(predict_transforms)

        self.predict_on_test = predict_on_test

        # Will be populated in setup()
        self.train_data: list[Any] | None = None
        self.val_data: list[Any] | None = None
        self.test_data: list[Any] | None = None
        self.predict_data: list[Any] | None = None

        # Will get updated by trainer.fit()
        self._current_epoch = 0

        logger.info(f"[{self.__class__.__name__}] Datamodule initialized with following settings: "
                    f"data_dir: {self.data_dir}, batch_size: {self.batch_size}, "
                    f"num_workers: {self.num_workers}, train_val_test_split: {self.train_val_test_split}, "
                    f"seed: {self.seed}, random_state: {self.R}, "
                    f"worker_logging_fn: {self.worker_logging_fn}, "
                    f"worker_seeding_fn: {self.worker_seeding_fn}, "
                    f"collate_fn: {self.collate_fn}")
    
    def set_random_state(
        self, 
        seed: int | None = None, 
        state: np.random.RandomState | None = None,
    ) -> pl.LightningDataModule:
        """
        Set the random state locally, to control the randomness.
        Similar to monai.utils.Randomizable.set_random_state.
        """
        if seed is not None:
            self.R = np.random.RandomState(seed)
            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                logger.error(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
                raise TypeError(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self
    
    def seed_dataloaders(
        self, 
        stage: Literal["fit", "validate", "test", "predict"],
    ) -> tuple[Callable, torch.Generator | None]:
        """
        Make a worker init function and a torch generator for epoch-aware determinism.

        Different handling for each stage (training, validation, testing, prediction).
        Assumes that the Trainer uses the UpdateDatamoduleEpoch callback, as well as 
        reload_dataloaders_every_n_epochs is set to 1 (default setting in the 
        TrainWorkflow).

        Override for custom seeding logic.
        """
        stage_offset = STAGE_SEED_OFFSET.get(stage, 3000) * self.num_workers
        
        epoch_offset = self._current_epoch * self.num_workers if stage == "fit" else 0

        seed = ((self.seed + stage_offset + epoch_offset) % MAX_SEED
                if self.seed is not None else None)
    
        return (
            make_worker_init_fn(
                seed=seed,
                setup_logging_fn=self.worker_logging_fn,
                seeding_fn=self.worker_seeding_fn,
                loader=STAGE_LOG_PREFIX[stage], # type: ignore
            ), 
            torch.Generator().manual_seed(seed) if seed is not None else None
        )
        
    def prepare_data(self) -> None:
        """
        One time only (downloading or preprocessing), not intended for 
        assigning any state.

        Current implementation is trivial; just checks if data_dir exists.

        Override for custom prepare_data logic.
        """
        check_data_dir_exists(self.data_dir)
    
    def create_data_list(self) -> list[Any]:
        """
        Create list[Path] using the DataHandler kwargs.

        Override for custom data list creation. In general, users are encouraged
        to create new DataExplorer subclasses for custom data structures instead of
        modifying this method.
        """
        logger.info(f"Creating data list from {self.data_dir}")
        logger.info(f"This may take a while...")

        explorer = DataHandler(self.data_dir, **self.data_handler_kwargs)
        
        return cast(list[Path], explorer.get_all_nifti_files(as_list=True))

    def process_data_list(self, data_list: list[Path]) -> list[Any]:
        """
        Process raw data list[Path] to create custom list of data entries; 
        e.g., list[dict[str, Any]] for typical dictionary-based transforms.

        The output of this method is used to create the dataset.

        Current implementation is trivial; just returns the raw data list.
        If list[Path] is not the desired format, override the build_data_list()
        method that creates the data_list input. 

        Override for custom data list processing.
        """
        return data_list
    
    def split_dataset(self, all_data: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Split data into train/val/test sets.

        Override for custom split logic.
        """
        return split_data_by_subjects(
            all_data, 
            train_val_test_split=self.train_val_test_split,
            seed=self.seed,
        )
    
    def setup(self, stage: str) -> None:
        """
        Assign train/val datasets for use in dataloaders. 

        Responsibilities:
        - Split data into train/val/test sets.
        - Assign to self.train_data, self.val_data, self.test_data, self.predict_data.
        """
        raw_data = self.create_data_list()
        all_data = self.process_data_list(raw_data)

        # Split data
        train_data, val_data, test_data = self.split_dataset(all_data)
        
        if stage == "fit":
            self.train_data = train_data
            self.val_data = val_data
        elif stage == "validate":
            self.val_data = val_data
        elif stage == "test":
            self.test_data = test_data
        elif stage == "predict":
            self.predict_data = test_data if self.predict_on_test else all_data
    
    def train_dataloader(self) -> DataLoader:
        """
        Create train dataloader. 

        Uses train_data created in setup(), together with train_transforms.
        Uses seed_dataloaders() to create a worker_init_fn and a torch generator.

        Override for custom train_dataloader logic.
        """
        if self.train_data is None:
            msg = "train_data is None. Make sure setup('fit') was called."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.debug(f"Epoch {self._current_epoch}: Creating new DataLoader")
        
        # Create dataset with transforms
        dataset = MONAIDataset(
            data=self.train_data, 
            transform=self.train_transforms
        )
        worker_init_fn, generator = self.seed_dataloaders("fit")

        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create val dataloader. 

        Uses val_data created in setup(), together with val_transforms.
        Uses seed_dataloaders() to create a worker_init_fn and a torch generator.

        Override for custom val_dataloader logic.
        """
        if self.val_data is None:
            msg = "val_data is None. Make sure setup('validate') was called."
            logger.error(msg)
            raise RuntimeError(msg)
            
        dataset = MONAIDataset(
            data=self.val_data, 
            transform=self.val_transforms
        )
        worker_init_fn, generator = self.seed_dataloaders("validate")

        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            **self.dataloader_kwargs,
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader. 

        Uses test_data created in setup(), together with test_transforms.
        Uses seed_dataloaders() to create a worker_init_fn and a torch generator.

        Override for custom test_dataloader logic.
        """
        if self.test_data is None:
            msg = "test_data is None. Make sure setup('test') was called."
            logger.error(msg)
            raise RuntimeError(msg)
            
        dataset = MONAIDataset(
            data=self.test_data, 
            transform=self.test_transforms
        )
        worker_init_fn, generator = self.seed_dataloaders("test")

        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            **self.dataloader_kwargs,
        )
    
    def predict_dataloader(self) -> DataLoader:
        """
        Create predict dataloader. 

        Uses predict_data created in setup(), together with predict_transforms.
        Uses seed_dataloaders() to create a worker_init_fn and a torch generator.

        Override for custom predict_dataloader logic.
        """
        if self.predict_data is None:
            msg = "predict_data is None. Make sure setup('predict') was called."
            logger.error(msg)
            raise RuntimeError(msg)
            
        dataset = MONAIDataset(
            data=self.predict_data, 
            transform=self.predict_transforms
        )
        worker_init_fn, generator = self.seed_dataloaders("predict")

        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            **self.dataloader_kwargs,
        )
    
    def state_dict(self) -> dict[str, Any]:
        """
        Lightning will call it automatically when saving a checkpoint - 
        only if defined.

        Override for custom state_dict logic.
        """
        state = {
            "train_val_test_split": self.train_val_test_split,
            "datamodule_base_seed": self.seed,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Lightning will call it automatically when loading a checkpoint - 
        only if defined.

        Override for custom load_state_dict logic.
        """
        self.train_val_test_split = state_dict.get(
            "train_val_test_split", self.train_val_test_split
        )
        self.seed = state_dict.get("datamodule_base_seed")
        self._current_epoch = state_dict.get("epoch", 0)

        logger.info(f"Loaded from checkpoint: "
                    f"train_val_test_split: {self.train_val_test_split}, "
                    f"base_seed: {self.seed}, "
                    f"current_epoch: {self._current_epoch}")


@register(RK.DATAMODULE)
class MAEDataModule(BaseDataModule):
    """
    DataModule for brain MRI foundation model training using masked autoencoder.
    
    The data_dir should contain .npy files with naming pattern 
    sub_x_ses_y_modalityname_count_if_more_than_one.npy
    
    Args: 
        masks_dir: Directory containing brain masks with naming pattern 
            sub_x_ses_y_mask.npy
        **base_module_kwargs: kwargs for BaseDataModule
            All other keyword arguments are passed to the `BaseDataModule` constructor.
            Refer to `BaseDataModule` for supported options such as `data_dir`, `batch_size`,
            `num_workers`, and preprocessing transforms.
    """
    def __init__(
        self,
        *,
        masks_dir: Path | str | None = None,
        **base_module_kwargs,
    ):
        super().__init__(**base_module_kwargs)
        self.masks_dir = resolve_path(masks_dir) if masks_dir is not None else None
        
    def process_data_list(self, data_list: list[Path]) -> list[Any]:
        """
        Create list[dict] for masked autoencoder setup.

        Each scan is a separate entry. A brain mask is optionally exctracted
        to restrict the calculation of loss. 
        """
        list_of_dicts = []
        subjects = set()
        sessions = set()
        modality_counts = Counter()
        
        for file_path in data_list:
            metadata = parse_filename_nested_nifti(file_path)
            subjects.add(metadata['sub_id'])
            sessions.add(f"{metadata['sub_id']}_ses_{metadata['ses_id']}")
            modality_counts[metadata['modality']] += 1

            data_entry = {
                'img': metadata['file_name'],
                'sub_id': metadata['sub_id'],
                'ses_id': metadata['ses_id'],
                'mod': metadata['modality'],
            }

            if self.masks_dir is not None:
                brain_mask_path = (
                    self.masks_dir / file_path.relative_to(self.data_dir).parent / "mask.npy"
                )
            if self.masks_dir is not None and brain_mask_path.exists():
                data_entry['brain_mask'] = str(brain_mask_path)
            else:
                logger.warning(f"Mask file {brain_mask_path} does not exist")
    
            list_of_dicts.append(data_entry)
        
        logger.info(get_summary_msg(subjects, sessions, modality_counts))
        return list_of_dicts


@register(RK.DATAMODULE)
class ContrastiveDataModule(BaseDataModule):
    """
    DataModule for brain MRI foundation model training using contrastive learning.

    This subclass organizes scans into session-level groups so that multiple
    modalities or time points from the same session can be used as positive
    pairs in contrastive setups.

    The `data_dir` should contain `.npy` files with naming pattern:
    `sub-<x>_ses-<y>_<modality>[_<index_if_multiple>].npy`

    See `BaseDataModule` for all initialization parameters.
    """
    def process_data_list(self, data_list: list[Path]) -> list[Any]:
        """
        Create data list of dicts for contrastive - group by session.
        """
        session_groups = defaultdict(list)
        subjects = set()
        sessions = set()
        modality_counts = Counter()

        # Group by session
        for file_path in data_list:
            metadata = parse_filename_nested_nifti(file_path)
            subjects.add(metadata['sub_id'])
            sessions.add(f"{metadata['sub_id']}_ses_{metadata['ses_id']}")
            modality_counts[metadata['modality']] += 1

            session_key = f"{metadata['sub_id']}_ses_{metadata['ses_id']}"
            session_groups[session_key].append(metadata)
        
        list_of_dicts = []

        for session_key, session_files in tqdm(session_groups.items(), desc="Creating data list"):
            if len(session_files) == 0:
                continue
            
            session_entry = {
                'sub_id': session_files[0]['sub_id'],
                'ses_id': session_files[0]['ses_id'],
                'count': len(session_files),
            }
            
            # Add each scan and modality with img_i key
            for i, file_metadata in enumerate(session_files):
                session_entry[f"img_{i}"] = file_metadata['file_name']
                session_entry[f"mod_{i}"] = file_metadata['modality']
            
            list_of_dicts.append(session_entry)
        
        logger.info(get_summary_msg(subjects, sessions, modality_counts))
        
        return list_of_dicts