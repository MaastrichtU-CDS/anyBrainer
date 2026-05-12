"""Unit tests for custom callbacks."""

import pytest
import lightning.pytorch as pl

from anyBrainer.core.engines.callbacks import (
    UpdateDatamoduleEpoch,
    LogLR,
    LogGradNorm,
    StepOutputsWriter,
)


@pytest.fixture(autouse=True)
def log_mock(monkeypatch):
    """Mock the log_dict method of the LightningModule."""

    def dummy_log(self, obj, *args, **kwargs):
        print(obj)

    monkeypatch.setattr(pl.LightningModule, "log", dummy_log)
    monkeypatch.setattr(pl.LightningModule, "log_dict", dummy_log)


class DummyModule(pl.LightningModule):
    def __init__(self, model_with_grads):
        super().__init__()
        self.model = model_with_grads


class DummyDataModule:
    def __init__(self):
        self._current_epoch = 0


class DummyTrainer:
    def __init__(self, epoch):
        self.current_epoch = epoch
        self.datamodule = DummyDataModule()


class TestUpdateDatamoduleEpoch:
    @pytest.mark.parametrize(
        "epoch,new_epoch",
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
    )
    def test_on_train_epoch_end(self, epoch, new_epoch, model_with_grads):
        """Test that the datamodule's epoch is updated correctly."""
        module = DummyModule(model_with_grads)
        trainer = DummyTrainer(epoch)
        callback = UpdateDatamoduleEpoch()
        callback.on_train_epoch_end(trainer, module)  # type: ignore
        assert trainer.datamodule._current_epoch == new_epoch


@pytest.mark.skip(reason="Not implemented")
class TestLogLR:
    def test_on_before_optimizer_step(self, model_with_grads):
        """Test that the learning rate is logged correctly."""
        module = DummyModule(model_with_grads)
        trainer = DummyTrainer(0)
        callback = LogLR()
        callback.on_before_optimizer_step(trainer, module, module.optimizers(), 0)  # type: ignore

    def test_optimizer_not_list(self, model_with_grads):
        """Test that the callback does not raise an error if the optimizer is
        not a list."""
        module = DummyModule(model_with_grads)
        trainer = DummyTrainer(0)
        callback = LogLR()
        with pytest.warns(UserWarning):
            callback.on_before_optimizer_step(trainer, module, module.optimizers(), 0)  # type: ignore


@pytest.mark.skip(reason="Not implemented")
class TestLogGradNorm:
    def test_on_train_batch_end(self, model_with_grads):
        """Test that the gradient norm is logged correctly."""
        module = DummyModule(model_with_grads)
        trainer = DummyTrainer(0)
        callback = LogGradNorm()
        callback.on_train_batch_end(trainer, module)  # type: ignore
        assert trainer.datamodule._current_epoch == 1


class _RankZeroTrainer:
    global_rank = 0


class TestStepOutputsWriter:
    def test_writes_test_csv(self, tmp_path):
        import torch

        mod = pl.LightningModule()
        cb = StepOutputsWriter(output_dir=tmp_path, save_predictions=False)
        tr = _RankZeroTrainer()
        cb.on_test_start(tr, mod)  # type: ignore
        outputs = {
            "per_sample_metrics": {"dice": torch.tensor([0.1, 0.2])},
        }
        batch = {"sub_id": [1, 2], "ses_id": [10, 20]}
        cb.on_test_batch_end(tr, mod, outputs, batch, batch_idx=0, dataloader_idx=0)  # type: ignore
        csv_path = tmp_path / "test_per_sample_metrics.csv"
        assert csv_path.is_file()
        text = csv_path.read_text()
        assert "dice" in text
        assert "0.1" in text and "0.2" in text
        assert "1" in text and "10" in text

    def test_skips_non_dict_outputs(self, tmp_path):
        mod = pl.LightningModule()
        cb = StepOutputsWriter(output_dir=tmp_path, save_predictions=False)
        tr = _RankZeroTrainer()
        cb.on_test_start(tr, mod)  # type: ignore
        cb.on_test_batch_end(tr, mod, None, {}, 0, 0)  # type: ignore
        assert not (tmp_path / "test_per_sample_metrics.csv").exists()

    def test_predict_saves_pt(self, tmp_path):
        import torch

        mod = pl.LightningModule()
        cb = StepOutputsWriter(output_dir=tmp_path, save_predictions=True)
        tr = _RankZeroTrainer()
        cb.on_predict_start(tr, mod)  # type: ignore
        outputs = {"pred": torch.zeros(2, 1), "batch_idx": 0}
        cb.on_predict_batch_end(tr, mod, outputs, {}, batch_idx=3, dataloader_idx=1)  # type: ignore
        pt = tmp_path / "predict" / "dl1_b3.pt"
        assert pt.is_file()
        loaded = torch.load(pt, map_location="cpu", weights_only=False)
        assert loaded["pred"].shape == (2, 1)
