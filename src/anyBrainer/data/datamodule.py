"""
PyTorch DataModule to control datasets, dataloaders, and splits alltogether. 
"""

__all__ = [
    "MAEDataModule",
    "ContrastiveDataModule",
]

import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import lightning as L
import numpy as np
from tqdm import tqdm

# pyright: reportPrivateImportUsage=false
from monai.data import Dataset as MONAIDataset
from monai.data import DataLoader as MONAIDataLoader

from anyBrainer.utils.utils import resolve_path
from anyBrainer.data.utils import (
    trivial_check_nested_nifti_dataset,
    parse_filename_nested_nifti,
    split_data_by_subjects,
)
from anyBrainer.data.explorer import (
    GenericNiftiDataExplorer,
)
from anyBrainer.transforms import(
    get_mae_train_transforms,
    get_mae_val_transforms,
    get_contrastive_train_transforms,
    get_contrastive_val_transforms,
)
from anyBrainer.transforms import (
    DeterministicCompose,
)

logger = logging.getLogger(__name__)


class BaseDataModule(L.LightningDataModule):
    """
    Base DataModule class. 
    
    Args: 
        data_dir: data directory
        batch_size: batch size for dataloaders
        num_workers: number of workers for dataloaders
        train_val_test_split: train/val/test split
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        data_dir: Path | str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        seed: int | None = None,
        random_state: np.random.RandomState | None = None,
    ):
        super().__init__()
        self.data_dir = resolve_path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.seed = seed
        self.set_random_state(seed, random_state)

        # Will be populated in setup()
        self.train_data = None
        self.val_data = None  
        self.test_data = None
        self.predict_data = None

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
        self.predict_transforms = None

    def set_random_state(
        self, 
        seed: int | None = None, 
        state: np.random.RandomState | None = None,
    ) -> L.LightningDataModule:
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
    
    def prepare_data(self):
        """
        One time only (downloading or preprocessing), not intended for 
        assigning any state. 
        """
        raise NotImplementedError()
        
    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders
        
        Args: 
            stage: separate setup logic for trainer.{fit,validate,test,predict}
        """
        raise NotImplementedError()
    
    def train_dataloader(self):
        """Create train dataloader with dataset from setup()"""
        raise NotImplementedError()
    
    def val_dataloader(self):
        """Used by both trainer.fit() and trainer.validate()"""
        raise NotImplementedError()
    
    def test_dataloader(self):
        """Used by trainer.test()"""
        raise NotImplementedError()
    
    def predict_dataloader(self):
        """Inference-only, no labels. Used by trainer.predict()"""
        raise NotImplementedError()
    
    def state_dict(self):
        """
        Lightning will call it automatically when saving a checkpoint - 
        only if defined.
        """
        state = {
            "train_val_test_split": self.train_val_test_split,
            "rng": self.R.get_state()
        }
        return state

    def load_state_dict(self, state_dict):
        """
        Lightning will call it automatically when loading a checkpoint - 
        only if defined.
        """
        self.train_val_test_split = state_dict.get(
            "train_val_test_split", self.train_val_test_split
        )
        
        rng_state = state_dict.get("rng", None)
        if rng_state is not None:
            self.R.set_state(rng_state)


class MAEDataModule(BaseDataModule):
    """
    DataModule for brain MRI foundation model training using masked autoencoder.
    
    The data_dir should contain .npy files with naming pattern 
    sub_x_ses_y_modalityname_count_if_more_than_one.npy
    
    Args: 
        data_dir: Directory containing .npy files with naming pattern 
                 sub_x_ses_y_modalityname_count_if_more_than_one.npy
        masks_dir: Directory containing brain masks with naming pattern 
                 sub_x_ses_y_mask.npy
        transform_config: Configuration for transforms
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio)
        seed: Random seed for reproducible splits
    """
    def __init__(
        self,
        data_dir: Path | str,
        masks_dir: Path | str,
        transform_config: Dict[str, Any],  # Now required - loaded from YAML
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        seed: int | None = None,
        random_state: np.random.RandomState | None = None,
    ):
        super().__init__(data_dir, batch_size, num_workers, train_val_test_split, seed, random_state)
        self.masks_dir = resolve_path(masks_dir)
        
        # Get transforms
        self.train_transforms = DeterministicCompose(
            get_mae_train_transforms(), 
            master_seed=seed
        )
        self.val_transforms = DeterministicCompose(
            get_mae_val_transforms(), 
            master_seed=seed
        )
        self.test_transforms = DeterministicCompose(
            get_mae_val_transforms(), 
            master_seed=seed
        )
    
    def prepare_data(self):
        """
        One time only (downloading or preprocessing), not intended for 
        assigning any state. 

        Check if data_dir contains the .npy files and log basic statistics.
        """
        trivial_check_nested_nifti_dataset(self.data_dir)
        
    def _create_data_list(self) -> List[Dict]:
        """
        Create list[dict] for masked autoencoder setup.

        Each scan is a separate entry. A brain mask is optionally exctracted
        to restrict the calculation of loss. 
        """
        logger.info(f"Creating data list from {self.data_dir}")
        logger.info(f"This may take a while...")

        explorer = GenericNiftiDataExplorer(self.data_dir)

        data_list = []
        subjects = set()
        sessions = set()
        modalities = set()
        
        for file_path in explorer.get_all_image_files(as_list=True, exts=(".npy")):
            metadata = parse_filename_nested_nifti(file_path)
            subjects.add(metadata['sub_id'])
            sessions.add(f"{metadata['sub_id']}_ses_{metadata['ses_id']}")
            modalities.add(metadata['modality'])

            data_entry = {
                'file_name': metadata['file_name'],
                'brain_mask': str(brain_mask_path) if brain_mask_path else None,
                'sub_id': metadata['sub_id'],
                'ses_id': metadata['ses_id'],
                'modality': metadata['modality'],
            }

            brain_mask_path = (
                self.masks_dir / file_path.relative_to(self.data_dir).parent / "mask.npy"
            )
            if brain_mask_path.exists():
                data_entry['brain_mask'] = str(brain_mask_path)
            else:
                logger.warning(f"Mask file {brain_mask_path} does not exist")
    
            data_list.append(data_entry)
        
        logger.info(f"Dataset contains {len(subjects)} subjects, "
                    f"{len(sessions)} sessions, {len(modalities)} modalities")
        logger.info(f"Data collection completed")
        
        return data_list
        
    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders
        
        Args: 
            stage: separate setup logic for trainer.{fit,validate,test,predict}
        """
        # Create list of data entries
        all_data = self._create_data_list()
        
        # Split data
        train_data, val_data, test_data = split_data_by_subjects(
            all_data, 
            train_val_test_split=self.train_val_test_split,
            seed=self.seed, # use seed instead of rng for continuing experiments
        )
        
        if stage == "fit":
            self.train_data = train_data
            self.val_data = val_data
        elif stage == "validate":
            self.val_data = val_data
        elif stage == "test":
            self.test_data = test_data
        elif stage == "predict":
            self.predict_data = all_data  # Use all data for prediction
    
    def train_dataloader(self):
        """Create train dataloader with dataset from setup()"""
        if self.train_data is None:
            logger.error("train_data is None. Make sure setup('fit') was called.")
            raise RuntimeError("train_data is None. Make sure setup('fit') was called.")
        
        # Create dataset with transforms
        dataset = MONAIDataset(data=self.train_data, transform=self.train_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def val_dataloader(self):
        """Used by both trainer.fit() and trainer.validate()"""
        if self.val_data is None:
            logger.error("val_data is None. Make sure setup('validate') was called.")
            raise RuntimeError("val_data is None. Make sure setup() was called.")
            
        dataset = MONAIDataset(data=self.val_data, transform=self.val_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def test_dataloader(self):
        """Used by trainer.test()"""
        if self.test_data is None:
            logger.error("test_data is None. Make sure setup('test') was called.")
            raise RuntimeError("test_data is None. Make sure setup('test') was called.")
            
        dataset = MONAIDataset(data=self.test_data, transform=self.test_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def predict_dataloader(self):
        """Inference-only, no labels. Used by trainer.predict()"""
        if self.predict_data is None:
            logger.error("predict_data is None. Make sure setup('predict') was called.")
            raise RuntimeError("predict_data is None. Make sure setup('predict') was called.")
            
        dataset = MONAIDataset(data=self.predict_data, transform=self.predict_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )


class ContrastiveDataModule(BaseDataModule):
    """
    DataModule for brain MRI foundation model training using contrastive learning.

    The data_dir should contain .npy files with naming pattern 
    sub_x_ses_y_modalityname_count_if_more_than_one.npy
    
    Args: 
        data_dir: Directory containing .npy files with naming pattern 
                 sub_x_ses_y_modalityname_count_if_more_than_one.npy
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio)
        seed: Random seed for reproducible splits
    """
    def __init__(
        self,
        data_dir: str,
        transform_config: Dict[str, Any],
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        seed: int | None = None,
        random_state: np.random.RandomState | None = None,
    ):
        super().__init__(data_dir, batch_size, num_workers, train_val_test_split, seed, random_state)
        
        # Get transforms
        self.train_transforms = DeterministicCompose(
            get_contrastive_train_transforms(), 
            master_seed=seed
        )
        self.val_transforms = DeterministicCompose(
            get_contrastive_val_transforms(), 
            master_seed=seed
        )
        self.test_transforms = DeterministicCompose(
            get_contrastive_val_transforms(), 
            master_seed=seed
        )
    
    def prepare_data(self):
        """
        One time only (downloading or preprocessing), not intended for 
        assigning any state. 

        Check if data_dir contains the .npy files and log basic statistics.
        """
        trivial_check_nested_nifti_dataset(self.data_dir)
    
    def _create_data_list(self) -> List[Dict]:
        """Create data list for contrastive - group by session."""
        logger.info(f"Grouping data by session from {self.data_dir}")
        logger.info(f"This may take a while...")

        explorer = GenericNiftiDataExplorer(self.data_dir)

        session_groups = defaultdict(list)
        subjects = set()
        sessions = set()
        modalities = set()

        # Group by session
        for file_path in explorer.get_all_image_files(as_list=True, exts=(".npy")):
            metadata = parse_filename_nested_nifti(file_path)
            subjects.add(metadata['sub_id'])
            sessions.add(f"{metadata['sub_id']}_ses_{metadata['ses_id']}")
            modalities.add(metadata['modality'])

            session_key = f"{metadata['sub_id']}_ses_{metadata['ses_id']}"
            session_groups[session_key].append(metadata)
        
        data_list = []

        for session_key, session_files in tqdm(session_groups.items(), desc="Creating data list"):
            if len(session_files) == 0:
                continue
            
            session_entry = {
                'sub_id': session_files[0]['sub_id'],
                'ses_id': session_files[0]['ses_id'],
                'total': len(session_files),
            }
            
            # Add each scan and modality with img_i key
            for i, file_metadata in enumerate(session_files):
                session_entry[f"img_{i}"] = file_metadata['file_name']
                session_entry[f"mod_{i}"] = file_metadata['modality']
            
            data_list.append(session_entry)
        
        logger.info(f"Dataset contains {len(subjects)} subjects, "
                    f"{len(sessions)} sessions, {len(modalities)} modalities")
        logger.info(f"Data collection completed")
        
        return data_list
        
    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders
        
        Args: 
            stage: separate setup logic for trainer.{fit,validate,test,predict}
        """
        # Create list of data entries
        all_data = self._create_data_list()
        
        # Split data
        train_data, val_data, test_data = split_data_by_subjects(
            all_data, 
            train_val_test_split=self.train_val_test_split,
            seed=self.seed, # use seed instead of rng for continuing experiments
        )
        
        if stage == "fit":
            self.train_data = train_data
            self.val_data = val_data
        elif stage == "validate":
            self.val_data = val_data
        elif stage == "test":
            self.test_data = test_data
        elif stage == "predict":
            self.predict_data = all_data  # Use all data for prediction
    
    def train_dataloader(self):
        """Create train dataloader with dataset from setup()"""
        if self.train_data is None:
            logger.error("train_data is None. Make sure setup('fit') was called.")
            raise RuntimeError("train_data is None. Make sure setup('fit') was called.")
        
        # Create dataset with transforms
        dataset = MONAIDataset(data=self.train_data, transform=self.train_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def val_dataloader(self):
        """Used by both trainer.fit() and trainer.validate()"""
        if self.val_data is None:
            logger.error("val_data is None. Make sure setup('validate') was called.")
            raise RuntimeError("val_data is None. Make sure setup() was called.")
            
        dataset = MONAIDataset(data=self.val_data, transform=self.val_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def test_dataloader(self):
        """Used by trainer.test()"""
        if self.test_data is None:
            logger.error("test_data is None. Make sure setup('test') was called.")
            raise RuntimeError("test_data is None. Make sure setup('test') was called.")
            
        dataset = MONAIDataset(data=self.test_data, transform=self.test_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def predict_dataloader(self):
        """Inference-only, no labels. Used by trainer.predict()"""
        if self.predict_data is None:
            logger.error("predict_data is None. Make sure setup('predict') was called.")
            raise RuntimeError("predict_data is None. Make sure setup('predict') was called.")
            
        dataset = MONAIDataset(data=self.predict_data, transform=self.predict_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )