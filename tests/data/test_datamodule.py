"""Tests datamodule methods."""

from pathlib import Path

import pytest
import torch

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    LoadImage,
)
from monai.data import DataLoader, Dataset

from anyBrainer.core.data.explorer import GenericNiftiDataExplorer
from anyBrainer.core.data import (
    MAEDataModule,
    ContrastiveDataModule,
)
from anyBrainer.core.utils import make_worker_init_fn


@pytest.fixture(autouse=True)
def mock_data_explorer(monkeypatch):
    """Monkey-patch DataModule so every attempt to read a folder returns a
    synthetic list of files."""

    def _dummy_call(self, *args, **kwargs):
        # 10 subjects, 11 sessions, 15 images
        return [
            Path("/Users/project/dataset/sub_1/ses_1/t1.npy"),
            Path("/Users/project/dataset/sub_1/ses_1/t1_2.npy"),
            Path("/Users/project/dataset/sub_1/ses_1/t2.npy"),
            Path("/Users/project/dataset/sub_1/ses_1/dwi.npy"),
            Path("/Users/project/dataset/sub_1/ses_2/t1.npy"),
            Path("/Users/project/dataset/sub_12/ses_1/flair.npy"),
            Path("/Users/project/dataset/sub_123/ses_1/dwi.npy"),
            Path("/Users/project/dataset/sub_123/ses_1/dwi_2.npy"),
            Path("/Users/project/dataset/sub_1234/ses_2/dwi.npy"),
            Path("/Users/project/dataset/sub_1235/ses_1/t1.npy"),
            Path("/Users/project/dataset/sub_2235/ses_11/dwi.npy"),
            Path("/Users/project/dataset/sub_3235/ses_1/flair.npy"),
            Path("/Users/project/dataset/sub_4235/ses_1/t2.npy"),
            Path("/Users/project/dataset/sub_5235/ses_2/dwi.npy"),
            Path("/Users/project/dataset/sub_6235/ses_3/dwi.npy"),
        ]

    monkeypatch.setattr(
        GenericNiftiDataExplorer, "get_all_image_files", _dummy_call, raising=True
    )


@pytest.fixture(autouse=True)
def mock_load_image(monkeypatch):
    """Monkey-patch LoadImage so every attempt to read a file yields a
    synthetic 3-D volume instead of touching the disk."""

    def _dummy_call(self, *args, **kwargs):
        # Create data with the shape the pipeline expects
        gen = torch.Generator().manual_seed(42)
        img = torch.rand((1, 150, 150, 150), dtype=torch.float32, generator=gen)
        return img

    monkeypatch.setattr(LoadImage, "__call__", _dummy_call, raising=True)


data_settings = {
    "data_dir": "/Users/project/dataset",
    "batch_size": 2,
    "num_workers": 4,
    "train_val_test_split": (0.7, 0.2, 0.1),
    "seed": 12345,
}
transforms_mae = {
    "train_transforms": "get_mae_train_transforms",
    "val_transforms": "get_mae_val_transforms",
    "test_transforms": "get_mae_val_transforms",
    "predict_transforms": {
        "name": "get_predict_transforms",
        "keys": ["img"],
    },
}
transforms_contrastive = {
    "train_transforms": "get_contrastive_train_transforms",
    "val_transforms": "get_contrastive_val_transforms",
    "test_transforms": "get_contrastive_val_transforms",
    "predict_transforms": "get_predict_transforms",
}
transforms_classification = {
    "train_transforms": "get_classification_train_transforms",
    "val_transforms": "get_classification_val_transforms",
    "test_transforms": "get_classification_val_transforms",
    "predict_transforms": "get_predict_transforms",
}


class TestMAEDataModule:
    @pytest.fixture
    def data_module(self):
        return MAEDataModule(
            masks_dir="/Users/project/masks", **transforms_mae, **data_settings
        )

    def test_data_splits(self, data_module):
        data_module.setup(stage="fit")
        data_module.setup(stage="test")
        assert (
            len(data_module.train_data) == 11
        )  # specific to seed; sub_1 (5 scans), ...
        assert len(data_module.val_data) == 2  # specific to seed; ...
        assert (
            len(data_module.test_data) == 2
        )  # specific to seed; sub_123 (2 scans), ...

    def test_data_list(self, data_module):
        data_module.setup(stage="fit")
        for i in data_module.train_data:
            assert set(i.keys()) == {"img", "sub_id", "ses_id", "mod"}

    @pytest.mark.slow
    def test_train_loader_rng_locks(self, data_module):
        """Check that RNG changes in every iter."""
        # Datamodule
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())
        out = next(train_loader_iter)
        next_out = next(train_loader_iter)

        # Ensure outputs don't match between iterations
        assert not torch.equal(out["img"], next_out["img"])

    @pytest.mark.slow
    def test_train_loader_w_reference(self, data_module, ref_mae_train_transforms):
        """Check that dataloader transforms are deterministic."""
        # Datamodule
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())

        # Reference transforms
        ref_dataset = Dataset(
            data=data_module.train_data, transform=ref_mae_train_transforms
        )
        ref_loader_iter = iter(
            DataLoader(
                ref_dataset,
                batch_size=data_settings["batch_size"],
                num_workers=data_settings["num_workers"],
                shuffle=True,
                worker_init_fn=make_worker_init_fn(data_settings["seed"]),
                generator=torch.Generator().manual_seed(12345),
            )
        )
        for _ in range(2):
            out = next(train_loader_iter)
            ref_out = next(ref_loader_iter)

        # Ensure outputs match for each sample
        assert torch.equal(out["img"], ref_out["img"])


class TestContrastiveDataModule:
    @pytest.fixture
    def data_module(self):
        return ContrastiveDataModule(**transforms_contrastive, **data_settings)

    def test_data_splits(self, data_module):
        data_module.setup(stage="fit")
        data_module.setup(stage="test")
        assert (
            len(data_module.train_data) == 8
        )  # specific to seed; sub_1 (2 sessions), ...
        assert len(data_module.val_data) == 2  # specific to seed; ...
        assert len(data_module.test_data) == 1  # specific to seed; ...

    @pytest.mark.slow
    def test_train_loader_pair(self, data_module):
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())
        out = next(train_loader_iter)

        # Ensure keys and queries are different
        assert not torch.equal(out["key"], out["query"])

    @pytest.mark.slow
    def test_train_loader_rng_locks(self, data_module):
        """Check that RNG changes in every iter."""
        # Datamodule
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())
        out = next(train_loader_iter)
        next_out = next(train_loader_iter)

        # Ensure outputs don't match between iterations
        assert not torch.equal(out["key"], next_out["key"])

    @pytest.mark.slow
    def test_train_loader_w_reference(
        self, data_module, ref_contrastive_train_transforms
    ):
        """Check that dataloader transforms are deterministic."""
        # Datamodule
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())

        # Reference transforms
        ref_dataset = Dataset(
            data=data_module.train_data, transform=ref_contrastive_train_transforms
        )
        ref_loader_iter = iter(
            DataLoader(
                ref_dataset,
                batch_size=data_settings["batch_size"],
                num_workers=data_settings["num_workers"],
                shuffle=True,
                worker_init_fn=make_worker_init_fn(data_settings["seed"]),
                generator=torch.Generator().manual_seed(12345),
            )
        )
        for _ in range(2):
            out = next(train_loader_iter)
            ref_out = next(ref_loader_iter)

        # Ensure outputs match for each sample
        assert torch.allclose(out["key"], ref_out["key"])
        assert torch.allclose(out["query"], ref_out["query"])


@pytest.mark.slow
class TestEpochAwareDeterminism:
    @pytest.fixture
    def data_module(self):
        return MAEDataModule(
            masks_dir="/Users/project/masks", **transforms_mae, **data_settings
        )

    def test_different_seeds_per_stage(self, data_module):
        """Check that different stages have different seeds."""
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())
        out = next(train_loader_iter)

        data_module.setup(stage="validate")
        val_loader_iter = iter(data_module.val_dataloader())
        val_out = next(val_loader_iter)

        assert not torch.allclose(out["img"], val_out["img"])

    def test_different_seeds_per_epoch(self, data_module):
        """Check that different epochs have different seeds."""
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())
        out_0 = next(train_loader_iter)

        data_module._current_epoch = 1
        train_loader_iter = iter(data_module.train_dataloader())
        out_1 = next(train_loader_iter)

        assert not torch.allclose(out_0["img"], out_1["img"])

    def test_epoch_determinism_vs_ref(self, data_module, ref_mae_train_transforms):
        """Check that dataloader transforms are epoch-aware deterministic."""
        data_module.setup(stage="fit")
        data_module._current_epoch = 3
        train_loader_iter = iter(data_module.train_dataloader())
        out = next(train_loader_iter)

        ref_dataset = Dataset(
            data=data_module.train_data, transform=ref_mae_train_transforms
        )
        ref_seed = data_settings["seed"] + 3 * data_settings["num_workers"]
        ref_loader_iter = iter(
            DataLoader(
                ref_dataset,
                batch_size=data_settings["batch_size"],
                num_workers=data_settings["num_workers"],
                shuffle=True,
                worker_init_fn=make_worker_init_fn(ref_seed),
                generator=torch.Generator().manual_seed(ref_seed),
            )
        )
        ref_out = next(ref_loader_iter)

        assert torch.allclose(out["img"], ref_out["img"])

    def test_resumed_training_from_ckpt(self, data_module):
        """Check that dataloading resumed from a checkpoint has identical
        behavior with single training run."""
        data_module.setup(stage="fit")
        data_module._current_epoch = 4
        train_loader_iter = iter(data_module.train_dataloader())
        single_run_out = next(train_loader_iter)
        state = data_module.state_dict()
        state["epoch"] = 4

        new_data_module = MAEDataModule(
            masks_dir="/Users/project/masks", **transforms_mae, **data_settings
        )
        new_data_module.setup(stage="fit")
        new_data_module.load_state_dict(state)
        train_loader_iter = iter(new_data_module.train_dataloader())
        resumed_out = next(train_loader_iter)

        assert torch.allclose(single_run_out["img"], resumed_out["img"])
