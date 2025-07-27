"""Unit tests for custom callbacks."""

import logging 

import pytest
import lightning.pytorch as pl
import torch

from anyBrainer.engines.callbacks import (
    UpdateDatamoduleEpoch,
    LogLR,
    LogGradNorm,
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
    @pytest.mark.parametrize("epoch,new_epoch", [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ])
    def test_on_train_epoch_end(self, epoch, new_epoch, model_with_grads):
        """Test that the datamodule's epoch is updated correctly."""
        module = DummyModule(model_with_grads)
        trainer = DummyTrainer(epoch)
        callback = UpdateDatamoduleEpoch()
        callback.on_train_epoch_end(trainer, module) # type: ignore
        assert trainer.datamodule._current_epoch == new_epoch


@pytest.mark.skip(reason="Not implemented")
class TestLogLR:
    def test_on_before_optimizer_step(self, model_with_grads):
        """Test that the learning rate is logged correctly."""
        module = DummyModule(model_with_grads)
        trainer = DummyTrainer(0)
        callback = LogLR()
        callback.on_before_optimizer_step(trainer, module, module.optimizers(), 0) # type: ignore
    
    def test_optimizer_not_list(self, model_with_grads):
        """Test that the callback does not raise an error if the optimizer is not a list."""
        module = DummyModule(model_with_grads)
        trainer = DummyTrainer(0)
        callback = LogLR()
        with pytest.warns(UserWarning):
            callback.on_before_optimizer_step(trainer, module, module.optimizers(), 0) # type: ignore


@pytest.mark.skip(reason="Not implemented")
class TestLogGradNorm:
    def test_on_train_batch_end(self, model_with_grads):
        """Test that the gradient norm is logged correctly."""
        module = DummyModule(model_with_grads)
        trainer = DummyTrainer(0)
        callback = LogGradNorm()
        callback.on_train_batch_end(trainer, module) # type: ignore
        assert trainer.datamodule._current_epoch == 1