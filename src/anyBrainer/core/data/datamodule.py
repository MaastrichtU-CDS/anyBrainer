"""PyTorch Lightning DataModules to control datasets, dataloaders, and splits
alltogether.

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
    "ClassificationDataModule",
]

import logging
from pathlib import Path
from itertools import product
from typing import Callable, Literal, Any, TYPE_CHECKING, cast, Sequence
from collections import Counter

import lightning.pytorch as pl
import numpy as np
import torch
from tqdm import tqdm

# pyright: reportPrivateImportUsage=false
from monai.data import Dataset as MONAIDataset
from monai.data import DataLoader as MONAIDataLoader
from monai.data.utils import set_rnd, pad_list_data_collate

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from anyBrainer.registry import register
from anyBrainer.config import resolve_fn, resolve_transform
from anyBrainer.core.data.utils import (
    check_data_dir_exists,
    parse_filename_nested_nifti,
    group_data,
    get_summary_msg,
    get_summary_msg_w_labels,
    read_label_from_txt,
)
from anyBrainer.core.utils import (
    split_data_by_subjects,
    resolve_path,
    make_worker_init_fn,
    callable_name,
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
    """Base DataModule class."""

    def __init__(
        self,
        data_dir: Path | str,
        *,
        data_handler_kwargs: dict[str, Any] | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        extra_dataloader_kwargs: dict[str, Any] | None = None,
        train_val_test_split: Sequence[float] = (0.7, 0.15, 0.15),
        val_mode: Literal["single", "repeated"] = "single",
        n_splits: int | None = None,
        current_split: int = 0,
        worker_logging_fn: Callable | str | None = None,
        worker_seeding_fn: Callable | str | None = set_rnd,
        collate_fn: Callable | str | None = pad_list_data_collate,
        seed: int | None = None,
        random_state: np.random.RandomState | None = None,
        train_transforms: dict[str, Any] | str | list[Callable] | None = None,
        val_transforms: dict[str, Any] | str | list[Callable] | None = None,
        test_transforms: dict[str, Any] | str | list[Callable] | None = None,
        predict_transforms: dict[str, Any] | str | list[Callable] | None = None,
        predict_on_test: bool = False,
    ):
        """Initializes the datamodule.

        Resolves any *_fn args and transforms into callable objects.

        Args:
        - data_dir: data directory
        - data_handler_kwargs: kwargs for DataHandler
        - batch_size: batch size for dataloaders
        - num_workers: number of workers for dataloaders
        - dataloader_kwargs: kwargs for DataLoader
        - train_val_test_split: train/val/test split
        - worker_logging_fn: function to log worker information
        - worker_seeding_fn: function to seed worker
        - collate_fn: function to collate data
        - seed: random seed for reproducibility
        - random_state: random state for reproducibility
        - train_transforms: transforms for train
        - val_transforms: transforms for val
        - test_transforms: transforms for test
        - predict_transforms: transforms for predict
        - predict_on_test: whether to predict on test set
        """
        super().__init__()
        self.data_dir = resolve_path(data_dir)
        self.data_handler_kwargs = data_handler_kwargs or {}
        extra_dataloader_kwargs = extra_dataloader_kwargs or {}
        self.extra_dataloader_kwargs = extra_dataloader_kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        if len(train_val_test_split) != 3:
            msg = "train_val_test_split must be a tuple of 3 floats."
            logger.error(msg)
            raise ValueError(msg)
        self.train_val_test_split = tuple(train_val_test_split)
        if val_mode != "single" and n_splits is None:
            msg = "n_splits must be provided when val_mode is not 'single'."
            logger.error(msg)
            raise ValueError(msg)
        self.val_mode = val_mode
        self.n_splits = n_splits
        self.predict_on_test = predict_on_test
        self.seed = seed
        self.set_random_state(seed, random_state)

        self.worker_logging_fn = resolve_fn(worker_logging_fn)
        self.worker_seeding_fn = resolve_fn(worker_seeding_fn)
        self.collate_fn = resolve_fn(collate_fn)

        self.train_transforms = resolve_transform(train_transforms)
        self.val_transforms = resolve_transform(val_transforms)
        self.test_transforms = resolve_transform(test_transforms)
        self.predict_transforms = resolve_transform(predict_transforms)

        # Will be populated in setup()
        self.train_data: list[Any] | None = None
        self.val_data: list[Any] | None = None
        self.test_data: list[Any] | None = None
        self.predict_data: list[Any] | None = None

        # Will get updated by trainer.fit()
        self._current_epoch = 0
        self._current_split = current_split

        logger.info(
            f"[{self.__class__.__name__}] Datamodule initialized with following settings: "
            f"data_dir: {self.data_dir}, batch_size: {self.batch_size}, "
            f"num_workers: {self.num_workers}, train_val_test_split: {self.train_val_test_split}, "
            f"seed: {self.seed}, random_state: {self.R}, "
            f"worker_logging_fn: {callable_name(self.worker_logging_fn)}, "
            f"worker_seeding_fn: {callable_name(self.worker_seeding_fn)}, "
            f"collate_fn: {callable_name(self.collate_fn)}, "
            f"val_mode: {self.val_mode}, n_splits: {self.n_splits}, "
            f"predict_on_test: {self.predict_on_test}"
        )

    def set_random_state(
        self,
        seed: int | None = None,
        state: np.random.RandomState | None = None,
    ) -> pl.LightningDataModule:
        """Set the random state locally, to control the randomness.

        Similar to monai.utils.Randomizable.set_random_state.
        """
        if seed is not None:
            self.R = np.random.RandomState(seed)
            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                logger.error(
                    f"state must be None or a np.random.RandomState but is {type(state).__name__}."
                )
                raise TypeError(
                    f"state must be None or a np.random.RandomState but is {type(state).__name__}."
                )
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self

    def seed_dataloaders(
        self,
        stage: Literal["fit", "validate", "test", "predict"],
    ) -> tuple[Callable, torch.Generator | None]:
        """Make a worker init function and a torch generator for epoch-aware
        determinism.

        Different handling for each stage (training, validation,
        testing, prediction). Assumes that the Trainer uses the
        UpdateDatamoduleEpoch callback, as well as
        reload_dataloaders_every_n_epochs is set to 1 (default setting
        in the TrainWorkflow).

        Override for custom seeding logic.
        """
        stage_offset = STAGE_SEED_OFFSET.get(stage, 3000) * self.num_workers

        epoch_offset = self._current_epoch * self.num_workers if stage == "fit" else 0

        seed = (
            (self.seed + stage_offset + epoch_offset) % MAX_SEED
            if self.seed is not None
            else None
        )

        return (
            make_worker_init_fn(
                seed=seed,
                setup_logging_fn=self.worker_logging_fn,
                seeding_fn=self.worker_seeding_fn,
                loader=STAGE_LOG_PREFIX[stage],  # type: ignore
            ),
            torch.Generator().manual_seed(seed) if seed is not None else None,
        )

    def prepare_data(self) -> None:
        """One time only (downloading or preprocessing), not intended for
        assigning any state.

        Current implementation is trivial; just checks if data_dir
        exists.

        Override for custom prepare_data logic.
        """
        check_data_dir_exists(self.data_dir)

    def create_data_list(self) -> list[Any]:
        """Create list[Path] using the DataHandler kwargs.

        Override for custom data list creation. In general, users are
        encouraged to create new DataExplorer subclasses for custom data
        structures instead of modifying this method.
        """
        logger.info(f"Creating data list from {self.data_dir}")
        logger.info("This may take a while...")

        explorer = DataHandler(self.data_dir, **self.data_handler_kwargs)

        return cast(list[Path], explorer.get_all_nifti_files(as_list=True))

    def process_data_list(self, data_list: list[Path]) -> list[Any]:
        """Process raw data list[Path] to create custom list of data entries;
        e.g., list[dict[str, Any]] for typical dictionary-based transforms.

        The output of this method is used to create the dataset.

        Current implementation is trivial; just returns the raw data
        list. If list[Path] is not the desired format, override the
        build_data_list() method that creates the data_list input.

        Override for custom data list processing.
        """
        return data_list

    def split_dataset(
        self, all_data: list[Any]
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Split data into train/val/test sets.

        Override for custom split logic.
        """
        if self.val_mode == "single":
            return split_data_by_subjects(
                all_data,
                train_val_test_split=self.train_val_test_split,
                seed=self.seed,
            )
        elif self.val_mode == "repeated":
            logger.info(
                f"[{self.__class__.__name__}] Splitting data into train/val/test sets "
                f"for the {self._current_split + 1}th time."
            )
            if self.seed is None:
                _seed = None
                _rng = self.R
            else:
                _seed = self.seed + self._current_split
                _rng = None

            return split_data_by_subjects(
                all_data,
                train_val_test_split=self.train_val_test_split,
                seed=_seed,
                random_state=_rng,
            )
        else:
            msg = f"[{self.__class__.__name__}] Invalid val_mode: {self.val_mode}"
            logger.error(msg)
            raise ValueError(msg)

    def setup(self, stage: str) -> None:
        """Assign train/val datasets for use in dataloaders.

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

        self._current_split += 1  # update for next split

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader.

        Uses train_data created in setup(), together with
        train_transforms. Uses seed_dataloaders() to create a
        worker_init_fn and a torch generator.

        Override for custom train_dataloader logic.
        """
        if self.train_data is None:
            msg = "train_data is None. Make sure setup('fit') was called."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.debug(f"Epoch {self._current_epoch}: Creating new DataLoader")

        # Create dataset with transforms
        dataset = MONAIDataset(data=self.train_data, transform=self.train_transforms)
        worker_init_fn, generator = self.seed_dataloaders("fit")

        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            **self.extra_dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """Create val dataloader.

        Uses val_data created in setup(), together with val_transforms.
        Uses seed_dataloaders() to create a worker_init_fn and a torch
        generator.

        Override for custom val_dataloader logic.
        """
        if self.val_data is None:
            msg = "val_data is None. Make sure setup('validate') was called."
            logger.error(msg)
            raise RuntimeError(msg)

        dataset = MONAIDataset(data=self.val_data, transform=self.val_transforms)
        worker_init_fn, generator = self.seed_dataloaders("validate")

        return MONAIDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader.

        Uses test_data created in setup(), together with
        test_transforms. Uses seed_dataloaders() to create a
        worker_init_fn and a torch generator.

        Override for custom test_dataloader logic.
        """
        if self.test_data is None:
            msg = "test_data is None. Make sure setup('test') was called."
            logger.error(msg)
            raise RuntimeError(msg)

        dataset = MONAIDataset(data=self.test_data, transform=self.test_transforms)
        worker_init_fn, generator = self.seed_dataloaders("test")

        return MONAIDataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create predict dataloader.

        Uses predict_data created in setup(), together with
        predict_transforms. Uses seed_dataloaders() to create a
        worker_init_fn and a torch generator.

        Override for custom predict_dataloader logic.
        """
        if self.predict_data is None:
            msg = "predict_data is None. Make sure setup('predict') was called."
            logger.error(msg)
            raise RuntimeError(msg)

        dataset = MONAIDataset(
            data=self.predict_data, transform=self.predict_transforms
        )
        worker_init_fn, generator = self.seed_dataloaders("predict")

        return MONAIDataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def state_dict(self) -> dict[str, Any]:
        """Lightning will call it automatically when saving a checkpoint - only
        if defined.

        Override for custom state_dict logic.
        """
        state = {
            "train_val_test_split": self.train_val_test_split,
            "datamodule_base_seed": self.seed,
            "current_split": self._current_split,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Lightning will call it automatically when loading a checkpoint -
        only if defined.

        Override for custom load_state_dict logic.
        """
        self.train_val_test_split = state_dict.get(
            "train_val_test_split", self.train_val_test_split
        )
        self.seed = state_dict.get("datamodule_base_seed")
        self._current_split = state_dict.get("current_split", 0)
        self._current_epoch = state_dict.get("epoch", 0)

        logger.info(
            f"Loaded from checkpoint: "
            f"train_val_test_split: {self.train_val_test_split}, "
            f"base_seed: {self.seed}, "
            f"current_epoch: {self._current_epoch}, "
            f"current_split: {self._current_split}"
        )


@register(RK.DATAMODULE)
class MAEDataModule(BaseDataModule):
    """DataModule for brain MRI foundation model training using masked
    autoencoder.

    The data_dir should contain .npy files with naming pattern
    sub_x_ses_y_modalityname_count_if_more_than_one.npy
    """

    def __init__(
        self,
        *,
        masks_dir: Path | str | None = None,
        **base_module_kwargs,
    ):
        """
        Args:
        - masks_dir: Directory containing brain masks with naming pattern
            sub_x_ses_y_mask.npy
        - **base_module_kwargs: kwargs for BaseDataModule
            All other keyword arguments are passed to the `BaseDataModule` constructor.
            Refer to `BaseDataModule` for supported options such as `data_dir`, `batch_size`,
            `num_workers`, and preprocessing transforms.
        """
        super().__init__(**base_module_kwargs)
        self.masks_dir = resolve_path(masks_dir) if masks_dir is not None else None

    def process_data_list(self, data_list: list[Path]) -> list[Any]:
        """Create list[dict] for masked autoencoder setup.

        Each scan is a separate entry. A brain mask is optionally
        exctracted to restrict the calculation of loss.
        """
        list_of_dicts = []
        subjects = set()
        sessions = set()
        modality_counts = Counter()

        for file_path in data_list:
            metadata = parse_filename_nested_nifti(file_path)
            subjects.add(metadata["sub_id"])
            sessions.add(f"{metadata['sub_id']}_ses_{metadata['ses_id']}")
            modality_counts[metadata["modality"]] += 1

            data_entry = {
                "img": metadata["file_name"],
                "sub_id": metadata["sub_id"],
                "ses_id": metadata["ses_id"],
                "mod": metadata["modality"],
            }

            if self.masks_dir is not None:
                brain_mask_path = (
                    self.masks_dir
                    / file_path.relative_to(self.data_dir).parent
                    / "mask.npy"
                )
            if self.masks_dir is not None and brain_mask_path.exists():
                data_entry["brain_mask"] = str(brain_mask_path)
            else:
                logger.warning(f"Mask file {brain_mask_path} does not exist")

            list_of_dicts.append(data_entry)

        logger.info(get_summary_msg(subjects, sessions, modality_counts))
        return list_of_dicts


@register(RK.DATAMODULE)
class ContrastiveDataModule(BaseDataModule):
    """DataModule for brain MRI foundation model training using contrastive
    learning.

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
        grouped_data = group_data(group_by="session", data_list=data_list)

        list_of_dicts = []

        for session_files in tqdm(
            grouped_data["grouped_data"].values(), desc="Creating data list"
        ):
            if len(session_files) == 0:
                continue

            session_entry = {
                "sub_id": session_files[0]["sub_id"],
                "ses_id": session_files[0]["ses_id"],
                "count": len(session_files),
            }

            # Add each scan and modality with img_i key
            for i, file_metadata in enumerate(session_files):
                session_entry[f"img_{i}"] = file_metadata["file_name"]
                session_entry[f"mod_{i}"] = file_metadata["modality"]

            list_of_dicts.append(session_entry)

        logger.info(
            get_summary_msg(
                grouped_data["subjects"],
                grouped_data["sessions"],
                grouped_data["modality_counts"],
            )
        )
        return list_of_dicts


class MultimodalDataModule(BaseDataModule):
    """DataModule for any multimodal task.

    Creates data list of dicts grouped by session.

    For each session, builds all ordered combinations of acquisitions
    for each input channel. Each channel position k has an allowed set
    of modalities (`modalities_per_ch[k]`). Each acquisition (img_i,
    mod_i) is treated as unique; multiple acquisitions of the same
    modality produce multiple unique candidates and therefore multiple
    dataset entries.
    """

    def __init__(
        self,
        modalities_per_ch: Sequence[set[str]],
        distinct_modalities: bool = False,
        distinct_acquisitions: bool = True,
        **base_module_kwargs,
    ):
        """
        Args:
            modalities_per_ch: list of allowed modality sets per channel position.
                Example for 2-channel model:
                    [ {"T1w","FLAIR","T2w"}, {"T1w","FLAIR","T2w"} ]
                This will create all ordered 2-tuples from available acquisitions matching each set.
            distinct_modalities: if True, disallow same modality appearing in two channels.
            distinct_acquisitions: if True, disallow same acquisition appearing in two channels.
        """
        super().__init__(**base_module_kwargs)

        if len(modalities_per_ch) == 0:
            msg = "`modalities_per_ch` must be a non-empty sequence"
            logger.error(msg)
            raise ValueError(msg)

        if not all(isinstance(s, set) for s in modalities_per_ch):
            msg = "`modalities_per_ch` must be a sequence of sets"
            logger.error(msg)
            raise ValueError(msg)

        self.modalities_per_ch = modalities_per_ch
        self.distinct_modalities = distinct_modalities
        self.distinct_acquisitions = distinct_acquisitions

    def process_data_list(self, data_list: list[Path]) -> list[Any]:
        """Create data list of dicts for multimodal pretraining.

        Returns a list of dicts, one per generated combination of modalities,
        each containing (for n channels):
            {
                "sub_id": ...,
                "ses_id": ...,
                "ch1": <filepath>,
                ...
                "chN": <filepath>,
                "mod1": <modality string>,
                ...
                "modN": <modality string>,
            }
        """
        grouped_data = group_data(group_by="session", data_list=data_list)

        list_of_dicts: list[dict[str, Any]] = []

        for session_files in tqdm(
            grouped_data["grouped_data"].values(), desc="Creating data list"
        ):
            if not session_files:
                continue

            # Get per-channel lists of eligible files for the given session
            # (as indices to access metadata)
            candidates: list[list[int]] = []
            for allowed in self.modalities_per_ch:
                idxs = [
                    i
                    for i, md in enumerate(session_files)
                    if md.get("modality") in allowed
                ]
                candidates.append(idxs)

            # Skip sessions that cannot satisfy all channels
            if any(len(idxs) == 0 for idxs in candidates):
                continue

            # Enumerate all ordered combinations
            for comb in product(*candidates):
                # Prevent re-using the same acquisition/file across channels
                if self.distinct_acquisitions and len(set(comb)) != len(
                    self.modalities_per_ch
                ):
                    continue

                # Prevent repeating modalities across channels
                if self.distinct_modalities:
                    mods = [session_files[i]["modality"] for i in comb]
                    if len(set(mods)) != len(self.modalities_per_ch):
                        continue

                # Build dataset entry
                first = session_files[comb[0]]
                entry: dict[str, Any] = {
                    "sub_id": first["sub_id"],
                    "ses_id": first["ses_id"],
                }

                for k, i in enumerate(comb, start=1):
                    md = session_files[i]
                    entry[f"ch{k}"] = md["file_name"]
                    entry[f"mod{k}"] = md["modality"]

                list_of_dicts.append(entry)

        logger.info(
            get_summary_msg(
                grouped_data["subjects"],
                grouped_data["sessions"],
                grouped_data["modality_counts"],
            )
        )
        return list_of_dicts


@register(RK.DATAMODULE)
class ClassificationDataModule(BaseDataModule):
    """DataModule for any classification task.

    The `labels_dir` should be either a directory with the same structure as `data_dir`
    or `data_dir` itself. The label is retrieved from `labels_filename`.

    Optionally attaches segmentation masks to positive samples, if `get_seg_masks` is True.
    The segmentation mask is retrieved from `seg_dir` and `seg_filename`.
    """

    def __init__(
        self,
        *,
        labels_dir: Path | str | None = None,
        labels_filename: str = "label.txt",
        expected_labels: list[str] | list[int] = ["0", "1"],
        strict: bool = False,
        modalities: list[str] | None = None,
        get_seg_masks: bool = False,
        seg_dir: Path | str | None = None,
        seg_filename: str = "seg.npy",
        **base_module_kwargs,
    ):
        """
        Args:
        - labels_dir: Directory containing labels with naming pattern
            sub_x/ses_y/label.txt or any other structure mirroring the data_dir.
        - labels_filename: Name of the label file
        - expected_labels: Expected labels in the label file.
            If list of two ints, it will be interpreted as a range of values.
        - strict: Whether to raise an error when label does not match expected labels.
            If False, it skips the session; can be used for label filtering.
        - modalities: List of modalities to include. If None, all modalities are included.
        - get_seg_masks: Whether to attach segmentation masks to positive samples.
        - seg_dir: Directory containing segmentation masks with naming pattern
          any structure mirroring the data_dir.
        - seg_filename: Name of the segmentation mask file
        See `BaseDataModule` for all initialization parameters.
        """
        super().__init__(**base_module_kwargs)

        self.labels_dir = (
            resolve_path(labels_dir) if labels_dir is not None else self.data_dir
        )
        self.labels_filename = labels_filename

        if len(expected_labels) == 2 and all(
            isinstance(l, int) for l in expected_labels
        ):
            logger.info(f"Interpreting expected_labels as range: {expected_labels}")
            start, end = cast(tuple[int, int], expected_labels)
            self.expected_labels = [str(i) for i in range(start, end + 1)]
        else:
            self.expected_labels = [str(i) for i in expected_labels]

        self.strict = strict
        self.modalities = modalities

        self.get_seg_masks = get_seg_masks
        self.seg_dir = resolve_path(seg_dir) if seg_dir is not None else self.data_dir
        self.seg_filename = seg_filename

        # Will get populated in setup()
        self.train_label_mean: float | None = None
        self.train_label_std: float | None = None

    def process_data_list(self, data_list: list[Path]) -> list[Any]:
        """Create data list of session-level entries for classification.

        Skips sessions with no label file.
        """
        grouped_data = group_data(
            group_by="session", data_list=data_list
        )  # list(sessions(list(files+metadata)))

        list_of_dicts = []
        used_subjects = set()
        used_sessions = set()
        used_modality_counts = Counter()
        label_counts = Counter()
        seg_counts = Counter()

        for session, session_files in tqdm(
            grouped_data["grouped_data"].items(), desc="Creating data list"
        ):
            # Filter by modality
            if self.modalities is not None:
                session_files = [
                    f for f in session_files if f["modality"] in self.modalities
                ]
                if not session_files:
                    logger.warning(
                        f"No valid modalities found for session " f"{session}; skipping"
                    )
                    continue

            file_metadata = session_files[0]
            label_path = (
                self.labels_dir
                / file_metadata["file_name"].relative_to(self.data_dir).parent
                / self.labels_filename
            )

            if label_path.exists():
                label = read_label_from_txt(
                    label_path, expected_labels=self.expected_labels, strict=self.strict
                )
                if label is None:
                    logger.warning(
                        f"Label of session {session} does not match "
                        f"expected labels; skipping"
                    )
                    continue
            else:
                logger.warning(
                    f"Label file {label_path} not found for session "
                    f"{session}; skipping"
                )
                continue

            # Build the session entry
            session_entry = {
                "sub_id": file_metadata["sub_id"],
                "ses_id": file_metadata["ses_id"],
                "count": len(session_files),
                "label": label,
            }

            for f in session_files:
                session_entry[f"{f['modality']}"] = f["file_name"]
                used_modality_counts[f["modality"]] += 1

            # Optionally get segmentation masks; allows missing
            if self.get_seg_masks:
                seg_path = (
                    self.seg_dir
                    / file_metadata["file_name"].relative_to(self.data_dir).parent
                    / self.seg_filename
                )
                if seg_path.exists():
                    session_entry["seg"] = seg_path
                    seg_counts[label] += 1

            used_subjects.add(file_metadata["sub_id"])
            used_sessions.add(f"{file_metadata['sub_id']}_{file_metadata['ses_id']}")
            label_counts[label] += 1

            list_of_dicts.append(session_entry)

        msg = get_summary_msg_w_labels(
            used_subjects,
            used_sessions,
            used_modality_counts,
            label_counts,
        )
        if self.get_seg_masks:
            msg += "\n  - Segmentation mask counts:"
            for label, count in seg_counts.items():
                msg += f"\n    - {label}: {count} files "

        logger.info(msg)

        return list_of_dicts


@register(RK.DATAMODULE)
class SegmentationDataModule(BaseDataModule):
    """DataModule for any segmentation task.

    The `seg_dir` should be either a directory with the same structure as `data_dir`
    or `data_dir` itself. The segmentation mask is retrieved from `seg_filename`.
    """

    def __init__(
        self,
        *,
        seg_dir: Path | str | None = None,
        seg_filename: str = "seg.npy",
        modalities: list[str] | None = None,
        **base_module_kwargs,
    ):
        """
        Args:
        - seg_dir: Directory containing segmentation masks with naming pattern
            sub_x/ses_y/seg.npy or any other structure mirroring the data_dir.
        - seg_filename: Name of the segmentation mask file
        - modalities: List of modalities to include. If None, all modalities are included.

        See `BaseDataModule` for all initialization parameters.
        """
        super().__init__(**base_module_kwargs)
        self.seg_dir = resolve_path(seg_dir) if seg_dir is not None else self.data_dir
        self.seg_filename = seg_filename
        self.modalities = modalities

    def process_data_list(self, data_list: list[Path]) -> list[Any]:
        """Create data list of session-level entries for segmentation.

        Skips sessions with no segmentation mask file.
        """
        grouped_data = group_data(
            group_by="session", data_list=data_list
        )  # list(sessions(list(files+metadata)))

        list_of_dicts = []
        used_subjects = set()
        used_sessions = set()
        used_modality_counts = Counter()

        for session, session_files in tqdm(
            grouped_data["grouped_data"].items(), desc="Creating data list"
        ):

            session_files = cast(list, session_files).copy()

            # Filter by modality
            if self.modalities is not None:
                session_files = [
                    f for f in session_files if f["modality"] in self.modalities
                ]
                if not session_files:
                    logger.warning(
                        f"No valid modalities found for session " f"{session}; skipping"
                    )
                    continue

            file_metadata = session_files[0]
            seg_path = (
                self.seg_dir
                / file_metadata["file_name"].relative_to(self.data_dir).parent
                / self.seg_filename
            )

            if not seg_path.exists():
                logger.warning(
                    f"Segmentation mask file {seg_path} not found for session "
                    f"{session}; skipping"
                )
                continue

            # Build the session entry
            session_entry = {
                "sub_id": file_metadata["sub_id"],
                "ses_id": file_metadata["ses_id"],
                "count": len(session_files),
                "seg": seg_path,
            }

            for i, f in enumerate(session_files):  # remove seg from img_* scans
                if self.seg_filename in cast(Path, f["file_name"]).name:
                    session_files.pop(i)

            for f in session_files:
                session_entry[f"{f['modality']}"] = f["file_name"]
                used_modality_counts[f["modality"]] += 1

            used_subjects.add(file_metadata["sub_id"])
            used_sessions.add(f"{file_metadata['sub_id']}_{file_metadata['ses_id']}")

            list_of_dicts.append(session_entry)

        logger.info(
            get_summary_msg(
                used_subjects,
                used_sessions,
                used_modality_counts,
            )
        )

        return list_of_dicts
