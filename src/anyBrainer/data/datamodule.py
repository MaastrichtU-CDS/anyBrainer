"""
PyTorch DataModule to control datasets, dataloaders, and splits alltogether. 

Will use MONAI's dataloader due to enhanced data collating behavior through 
monai.data.utils.list_data_collate and handling of seeds for determinism.
TODO: check how dataloader RNG can be restored for continuing training, 
usually through the worker_init_fn. 
"""

__all__ = [
    "MAEDataModule",
    "ContrastiveDataModule",
]

import logging
import re
from pathlib import Path
from typing import List, Dict, Literal, Optional, Any, Sequence
from collections import defaultdict

import lightning as L
import torch
import numpy as np

# pyright: reportPrivateImportUsage=false
from monai.data import Dataset as MONAIDataset
from monai.data import DataLoader as MONAIDataLoader

from anyBrainer.utils.utils import resolve_path

from anyBrainer.transforms.managers import (
    ContrastiveTransformManager, 
    MAETransformManager,
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
    R: np.random.RandomState = np.random.RandomState()

    def __init__(
        self,
        data_dir: Path | str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        self.data_dir = resolve_path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

        # Will be populated in setup()
        self.train_data = None
        self.val_data = None  
        self.test_data = None
        self.predict_data = None

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
        self.train_val_test_split = state_dict.get("train_val_test_split", self.train_val_test_split)
        self.R = state_dict.get("rng", self.R)


class MAEDataModule(BaseDataModule):
    """
    DataModule for brain MRI foundation model training using masked autoencoder.
    
    The data_dir should contain .npy files with naming pattern 
    sub_x_ses_y_modalityname_count_if_more_than_one.npy

    #TODO: get mask paths
    
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
    ):
        super().__init__(data_dir, batch_size, num_workers, train_val_test_split)
        self.masks_dir = resolve_path(masks_dir)
        
        # Create transform manager
        self.transform_manager = MAETransformManager(transform_config)
    
    def prepare_data(self):
        """
        One time only (downloading or preprocessing), not intended for 
        assigning any state. 

        Check if data_dir contains the .npy files and log basic statistics.
        """
        data_path = Path(self.data_dir)
        if not data_path.exists():
            logger.error(f"Data directory {self.data_dir} does not exist")
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        logger.info(f"Collecting data from {self.data_dir}")
        
        npy_files = list(data_path.glob("*.npy"))
        if len(npy_files) == 0:
            logger.error(f"No .npy files found in {self.data_dir}")
            raise FileNotFoundError(f"No .npy files found in {self.data_dir}")
            
        logger.info(f"Found {len(npy_files)} .npy files in {self.data_dir}")
        
        # Parse filenames to get basic statistics
        subjects = set()
        sessions = set()
        modalities = set()
        
        for file_path in npy_files:
            metadata = self._parse_filename(file_path)
            if metadata:
                subjects.add(metadata['sub_id'])
                sessions.add(f"{metadata['sub_id']}_ses_{metadata['ses_id']}")
                modalities.add(metadata['modality'])
        
        logger.info(f"Dataset contains {len(subjects)} subjects, "
                    f"{len(sessions)} sessions, {len(modalities)} modalities")
        
    def _parse_filename(self, file_path: Path | str) -> Optional[Dict]:
        """
        Parse filename with pattern: sub_x_ses_y_ModalityName_CountIfMoreThanOne.npy
        
        Returns:
            Dict with keys: sub_id, ses_id, modality, count, filepath
        """
        # Remove .npy extension
        base_name = Path(file_path).name.replace('.npy', '')
        
        # Pattern: sub_(\d+)_ses_(\d+)_(.+)
        pattern = r'sub_(\d+)_ses_(\d+)_(.+)'
        match = re.match(pattern, base_name)
        
        if not match:
            logger.warning(f"Could not parse filename in: {file_path}")
            return None
            
        sub_id = match.group(1)
        ses_id = match.group(2)
        modality_part = match.group(3)
        
        # Check if modality has count (ends with _number)
        count_pattern = r'(.+)_(\d+)$'
        count_match = re.match(count_pattern, modality_part)
        
        if count_match:
            modality = count_match.group(1)
            count = int(count_match.group(2))
            modality_suffix = f"{modality}_{count}"
        else:
            modality = modality_part
            count = 1
            modality_suffix = modality
            
        return {
            'sub_id': sub_id,
            'ses_id': ses_id,
            'modality': modality,
            'count': count,
            'modality_suffix': modality_suffix,
            'filepath': str(file_path)
        }
    
    def _create_data_list(self) -> List[Dict]:
        """
        Create list[dict] for masked autoencoder setup.
        Each scan is a separate entry.
        """
        data_list = []
        
        for file_path in self.data_dir.glob("*.npy"):
            metadata = self._parse_filename(file_path)
            if metadata:
                data_entry = {
                    'img': metadata['filepath'],
                    'sub_id': metadata['sub_id'],
                    'ses_id': metadata['ses_id'],
                    'modality_suffix': metadata['modality_suffix'],
                    'modality': metadata['modality'],
                    'count': metadata['count']
                }
                data_list.append(data_entry)
                
        return data_list
    
    def _split_data(self, data_list: List[Dict]) -> tuple:
        """
        Split data into train/val/test based on subjects for proper separation.
        """
        current_state = self.R.get_state()
        logger.info(f"Splitting data into train/val/test based on subjects for "
                    f"masked autoencoder with current state: {current_state[1][:5]}") # pyright: ignore[reportArgumentType]
        
        # Get unique subjects
        subjects = list(set([item['sub_id'] for item in data_list]))
        
        # Shuffle subjects for random split
        self.R.shuffle(subjects)
        
        # Calculate split indices
        n_subjects = len(subjects)
        train_end = int(n_subjects * self.train_val_test_split[0])
        val_end = train_end + int(n_subjects * self.train_val_test_split[1])
        
        train_subjects = set(subjects[:train_end])
        val_subjects = set(subjects[train_end:val_end])
        test_subjects = set(subjects[val_end:])
        
        # Split data based on subject assignment
        train_data = [item for item in data_list if item['sub_id'] in train_subjects]
        val_data = [item for item in data_list if item['sub_id'] in val_subjects]
        test_data = [item for item in data_list if item['sub_id'] in test_subjects]
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
        
    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders
        
        Args: 
            stage: separate setup logic for trainer.{fit,validate,test,predict}
        """
        # Create list of data entries
        all_data = self._create_data_list()
        
        # Split data
        train_data, val_data, test_data = self._split_data(all_data)
        
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
        
        train_transforms = self.transform_manager.get_train_transforms()

        # Create dataset with transforms
        dataset = MONAIDataset(data=self.train_data, transform=train_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Used by both trainer.fit() and trainer.validate()"""
        if self.val_data is None:
            logger.error("val_data is None. Make sure setup('validate') was called.")
            raise RuntimeError("val_data is None. Make sure setup() was called.")
            
        val_transforms = self.transform_manager.get_val_transforms()

        dataset = MONAIDataset(data=self.val_data, transform=val_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Used by trainer.test()"""
        if self.test_data is None:
            logger.error("test_data is None. Make sure setup('test') was called.")
            raise RuntimeError("test_data is None. Make sure setup('test') was called.")
            
        test_transforms = self.transform_manager.get_val_transforms()

        dataset = MONAIDataset(data=self.test_data, transform=test_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        """Inference-only, no labels. Used by trainer.predict()"""
        if self.predict_data is None:
            logger.error("predict_data is None. Make sure setup('predict') was called.")
            raise RuntimeError("predict_data is None. Make sure setup('predict') was called.")
            
        predict_transforms = self.transform_manager.get_val_transforms()

        dataset = MONAIDataset(data=self.predict_data, transform=predict_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
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
        transform_config: Dict[str, Any],  # Now required - loaded from YAML
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
    ):
        super().__init__(data_dir, batch_size, num_workers, train_val_test_split)
        
        # Create transform manager
        self.transform_manager = ContrastiveTransformManager(transform_config)
    
    def prepare_data(self):
        """
        One time only (downloading or preprocessing), not intended for 
        assigning any state. 

        Check if data_dir contains the .npy files and log basic statistics.
        """
        data_path = Path(self.data_dir)
        if not data_path.exists():
            logger.error(f"Data directory {self.data_dir} does not exist")
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        logger.info(f"Collecting data from {self.data_dir}")
        
        npy_files = list(data_path.glob("*.npy"))
        if len(npy_files) == 0:
            logger.error(f"No .npy files found in {self.data_dir}")
            raise FileNotFoundError(f"No .npy files found in {self.data_dir}")
            
        logger.info(f"Found {len(npy_files)} .npy files in {self.data_dir}")
        
        # Parse filenames to get basic statistics
        subjects = set()
        sessions = set()
        modalities = set()
        
        for file_path in npy_files:
            metadata = self._parse_filename(file_path)
            if metadata:
                subjects.add(metadata['sub_id'])
                sessions.add(f"{metadata['sub_id']}_ses_{metadata['ses_id']}")
                modalities.add(metadata['modality'])
        
        logger.info(f"Dataset contains {len(subjects)} subjects, "
                    f"{len(sessions)} sessions, {len(modalities)} modalities")
        
    def _parse_filename(self, file_path: Path | str) -> Optional[Dict]:
        """
        Parse filename with pattern: sub_x_ses_y_ModalityName_CountIfMoreThanOne.npy
        
        Returns:
            Dict with keys: sub_id, ses_id, modality, count, filepath
        """
        # Remove .npy extension
        base_name = Path(file_path).name.replace('.npy', '')
        
        # Pattern: sub_(\d+)_ses_(\d+)_(.+)
        pattern = r'sub_(\d+)_ses_(\d+)_(.+)'
        match = re.match(pattern, base_name)
        
        if not match:
            logger.warning(f"Could not parse filename in: {file_path}")
            return None
            
        sub_id = match.group(1)
        ses_id = match.group(2)
        modality_part = match.group(3)
        
        # Check if modality has count (ends with _number)
        count_pattern = r'(.+)_(\d+)$'
        count_match = re.match(count_pattern, modality_part)
        
        if count_match:
            modality = count_match.group(1)
            count = int(count_match.group(2))
            modality_suffix = f"{modality}_{count}"
        else:
            modality = modality_part
            count = 1
            modality_suffix = modality
            
        return {
            'sub_id': sub_id,
            'ses_id': ses_id,
            'modality': modality,
            'count': count,
            'modality_suffix': modality_suffix,
            'filepath': str(file_path)
        }
    
    def _create_data_list(self) -> List[Dict]:
        """Create data list for contrastive - group by session."""
        session_groups = defaultdict(list)
        
        for file_path in self.data_dir.glob("*.npy"):
            metadata = self._parse_filename(file_path)
            if metadata:
                session_key = f"{metadata['sub_id']}_ses_{metadata['ses_id']}"
                session_groups[session_key].append(metadata)
        
        data_list = []
        for session_key, session_files in session_groups.items():
            if len(session_files) == 0:
                continue
                
            session_entry = {
                'sub_id': session_files[0]['sub_id'],
                'ses': session_files[0]['ses_id'],
            }
            
            # Add each scan with img_i key
            for i, file_metadata in enumerate(session_files):
                session_entry[f"img_{i}"] = file_metadata['filepath']
            
            data_list.append(session_entry)
            
        return data_list
    
    def _split_data(self, data_list: List[Dict]) -> tuple:
        """
        Split data into train/val/test based on subjects for proper separation.
        """
        current_state = self.R.get_state()
        logger.info(f"Splitting data into train/val/test based on subjects for "
                    f"contrastive learning with current state: {current_state[1][:5]}") # pyright: ignore[reportArgumentType]
        
        # Get unique subjects
        subjects = list(set([item['sub_id'] for item in data_list]))
        
        # Shuffle subjects for random split
        self.R.shuffle(subjects)
        
        # Calculate split indices
        n_subjects = len(subjects)
        train_end = int(n_subjects * self.train_val_test_split[0])
        val_end = train_end + int(n_subjects * self.train_val_test_split[1])
        
        train_subjects = set(subjects[:train_end])
        val_subjects = set(subjects[train_end:val_end])
        test_subjects = set(subjects[val_end:])
        
        # Split data based on subject assignment
        train_data = [item for item in data_list if item['sub_id'] in train_subjects]
        val_data = [item for item in data_list if item['sub_id'] in val_subjects]
        test_data = [item for item in data_list if item['sub_id'] in test_subjects]
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
        
    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders
        
        Args: 
            stage: separate setup logic for trainer.{fit,validate,test,predict}
        """
        # Create list of data entries
        all_data = self._create_data_list()
        
        # Split data
        train_data, val_data, test_data = self._split_data(all_data)
        
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
        
        train_transforms = self.transform_manager.get_train_transforms()

        # Create dataset with transforms
        dataset = MONAIDataset(data=self.train_data, transform=train_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Used by both trainer.fit() and trainer.validate()"""
        if self.val_data is None:
            logger.error("val_data is None. Make sure setup('validate') was called.")
            raise RuntimeError("val_data is None. Make sure setup() was called.")
            
        val_transforms = self.transform_manager.get_val_transforms()

        dataset = MONAIDataset(data=self.val_data, transform=val_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Used by trainer.test()"""
        if self.test_data is None:
            logger.error("test_data is None. Make sure setup('test') was called.")
            raise RuntimeError("test_data is None. Make sure setup('test') was called.")
            
        test_transforms = self.transform_manager.get_val_transforms()

        dataset = MONAIDataset(data=self.test_data, transform=test_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        """Inference-only, no labels. Used by trainer.predict()"""
        if self.predict_data is None:
            logger.error("predict_data is None. Make sure setup('predict') was called.")
            raise RuntimeError("predict_data is None. Make sure setup('predict') was called.")
            
        predict_transforms = self.transform_manager.get_val_transforms()

        dataset = MONAIDataset(data=self.predict_data, transform=predict_transforms)
        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )