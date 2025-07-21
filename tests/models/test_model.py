"""Tests for Pytorch's Lightning Module."""

import pytest
import torch
from torch._functorch.vmap import out_dims_t
import torch.optim as optim
# pyright: reportPrivateImportUsage=false
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.models.model import CLwAuxModel
from anyBrainer.models.schedulers import (
    StepwiseParameterScheduler,
    CosineAnnealingWithWarmup,
)


@pytest.fixture(autouse=True)
def mock_swin_vit(monkeypatch):
    """
    Monkey-patch MONAI's SwinTransformer so every forward pass of the model
    yields a synthetic tensor that matches the bottleneck dimensions, given 
    a feature_size of 48 and 4 stages.
    """
    def _dummy_call(self, *args, **kwargs):
        # Create data with the shape the pipeline expects
        gen = torch.Generator().manual_seed(42)
        x = [torch.rand((8, 768, 4, 4, 4), dtype=torch.float32, generator=gen)]
        return x

    monkeypatch.setattr(SwinViT, "forward", _dummy_call, raising=True)

@pytest.fixture(scope="module")
def input_batch():
    """Input batch for the model."""
    return {
        "query": torch.randn(8, 1, 128, 128, 128),
        "key": torch.randn(8, 1, 128, 128, 128),
        "mod": ["t1", "t2", "flair", "dwi", "adc", "swi", "other", "t1"],
    }

swinv2cl_model_kwargs = {
    "in_channels": 1,
    "depths": (2, 2, 6, 2),
    "num_heads": (3, 6, 12, 24),
    "window_size": 7,
    "patch_size": 2,
    "use_v2": True,
    "feature_size": 48,
    "proj_dim": 128,
    "proj_hidden_dim": 2048,
    "proj_hidden_act": "gelu",
    "aux_mlp_head": True,
    "aux_mlp_num_classes": 7,
}

swinv2cl_optimizer_kwargs = {
    "name": "AdamW",
    "lr": 1e-4,
    "weight_decay": 1e-5,
}

swinv2cl_scheduler_kwargs = {
    "name": "CosineAnnealingWithWarmup",
    "warmup_iters": 1000,
    "total_iters": 10000,
    "interval": "step",
    "frequency": 1,
}

class TestCLwAuxModel:
    def test_model_forward_pass(self, input_tensor):
        """Test that the model is properly initialized."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            total_steps=10000,
        )
        proj, aux = model(input_tensor)
        assert proj.shape == (8, 128)
        assert aux.shape == (8, 7)
    
    def test_incorrect_model_kwargs(self):
        """Test that the model raises a ValueError if the model kwargs are incorrect."""
        with pytest.raises(Exception):
            CLwAuxModel(
                model_kwargs={"wrong_key": "wrong_value"},
                optimizer_kwargs=swinv2cl_optimizer_kwargs,
                lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
                total_steps=10000,
            )
    
    def test_incorrect_total_steps(self):
        """Test that the model raises a ValueError if the total steps are invalid."""
        with pytest.raises(ValueError):
            CLwAuxModel(
                model_kwargs=swinv2cl_model_kwargs,
                optimizer_kwargs=swinv2cl_optimizer_kwargs,
                lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
                total_steps=None, # type: ignore
            )

class TestConfigureOptimizers:
    def test_one_optimizer_one_scheduler(self, input_tensor):
        """Test that the model returns a dict with an optimizer and a scheduler."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            total_steps=10000,
        )
        out = model.configure_optimizers()
        assert isinstance(out, dict)
        assert "optimizer" in out
        assert "lr_scheduler" in out
        assert isinstance(out["optimizer"], optim.Optimizer)
        assert isinstance(out["lr_scheduler"], dict)
        assert "scheduler" in out["lr_scheduler"]
        assert isinstance(out["lr_scheduler"]["scheduler"], optim.lr_scheduler.LRScheduler)
    
    def test_one_optimizer_no_scheduler(self):
        """Test that the model returns an optimizer if no scheduler is provided."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            total_steps=10000,
        )
        out = model.configure_optimizers()
        assert isinstance(out, optim.Optimizer)

    def test_multiple_optimizers_no_scheduler(self):
        """Test that the model returns a list of optimizers if no scheduler is provided."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=[swinv2cl_optimizer_kwargs, swinv2cl_optimizer_kwargs],
            total_steps=10000,
        )
        out = model.configure_optimizers()
        assert isinstance(out, list)
        assert len(out) == 2
        assert isinstance(out[0], optim.Optimizer)
        assert isinstance(out[1], optim.Optimizer)
    
    def test_multiple_optimizers_one_scheduler(self):
        """Test that the model returns a list of optimizers and a list of scheduler dictionaries."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=[swinv2cl_optimizer_kwargs, swinv2cl_optimizer_kwargs],
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            total_steps=10000,
        )
        out = model.configure_optimizers()
        assert isinstance(out, tuple)
        assert len(out[0]) == 2
        assert len(out[1]) == 2
        assert isinstance(out[0][0], optim.Optimizer)
        assert isinstance(out[0][1], optim.Optimizer)
        assert isinstance(out[1][0], dict)
        assert isinstance(out[1][1], dict)
        assert "scheduler" in out[1][0]
        assert "scheduler" in out[1][1]
        assert isinstance(out[1][0]["scheduler"], optim.lr_scheduler.LRScheduler)
        assert isinstance(out[1][1]["scheduler"], optim.lr_scheduler.LRScheduler)
    
    def test_multiple_optimizers_multiple_schedulers(self):
        """Test that the model returns a list of optimizers and a list of scheduler dictionaries."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=[swinv2cl_optimizer_kwargs, swinv2cl_optimizer_kwargs],
            lr_scheduler_kwargs=[swinv2cl_scheduler_kwargs, swinv2cl_scheduler_kwargs],
            total_steps=10000,
        )
        out = model.configure_optimizers()
        assert isinstance(out, tuple)
        assert len(out[0]) == 2
        assert len(out[1]) == 2
        assert isinstance(out[0][0], optim.Optimizer)
        assert isinstance(out[0][1], optim.Optimizer)
        assert isinstance(out[1][0], dict)
        assert isinstance(out[1][1], dict)
        assert "scheduler" in out[1][0]
        assert "scheduler" in out[1][1]
        assert isinstance(out[1][0]["scheduler"], optim.lr_scheduler.LRScheduler)
        assert isinstance(out[1][1]["scheduler"], optim.lr_scheduler.LRScheduler)
    
    def test_multiple_schedulers_one_optimizer(self):
        """Test that the model returns a list of optimizers and a list of scheduler dictionaries."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=[swinv2cl_scheduler_kwargs, swinv2cl_scheduler_kwargs],
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_len_mismatch(self):
        """Test length mismatch between optimizer and scheduler."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=[swinv2cl_optimizer_kwargs],
            lr_scheduler_kwargs=[swinv2cl_scheduler_kwargs, swinv2cl_scheduler_kwargs],
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_not_found_optimizer(self):
        """Test that the model raises a ValueError if the optimizer name is not found."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs={"name": "WrongOptim", "lr": 1e-4, "weight_decay": 1e-5},
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_not_found_scheduler(self):
        """Test that the model raises a ValueError if the scheduler name is not found."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs={"name": "WrongScheduler", "interval": "step", "frequency": 1},
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_missing_scheduler_keys(self):
        """Test that the model raises a ValueError if the scheduler keys are missing."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs={"name": "CosineAnnealingWithWarmup", "frequency": 1},
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_incorrect_optimizer_kwargs(self):
        """Test that the model raises a Exception if the optimizer kwargs are incorrect."""
        model = CLwAuxModel(
                model_kwargs=swinv2cl_model_kwargs,
                optimizer_kwargs={"name": "AdamW", "wrong_key": 1e-4, "weight_decay": 1e-5},
                lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
                total_steps=10000,
            )
        with pytest.raises(Exception):
            model.configure_optimizers()

    
    def test_incorrect_lr_scheduler_kwargs(self):
        """Test that the model raises a ValueError if the lr scheduler kwargs are incorrect."""
        model = CLwAuxModel(
                model_kwargs=swinv2cl_model_kwargs,
                optimizer_kwargs=swinv2cl_optimizer_kwargs,
                lr_scheduler_kwargs={"name": "WrongScheduler", "interval": "step", "frequency": 1},
                total_steps=10000,
            )
        with pytest.raises(Exception):
            model.configure_optimizers()
    

class TestOnAfterBatchTransfer:
    def test_modality_to_onehot(self, input_batch):
        """Unit test that the model returns a one-hot encoded tensor."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            total_steps=10000,
        )
        output_batch = model.on_after_batch_transfer(input_batch, 0)
        assert "aux_labels" in output_batch
        assert output_batch["aux_labels"].shape == (8, 7)
        assert output_batch["aux_labels"].dtype == torch.float32
        assert input_batch["query"].device == output_batch["aux_labels"].device


class TestInitializations:
    def test_initialization_of_hparams(self):
        """
        Test that the model initializes with the correct hyperparameters.
        Ideally run with pytest -s to see the logger output.
        """
        model = CLwAuxModel(
                model_kwargs=swinv2cl_model_kwargs,
                optimizer_kwargs=swinv2cl_optimizer_kwargs,
                lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
                total_steps=10000,
                loss_kwargs={"temperature": 0.1, "top_k_negatives": 1000},
                loss_scheduler_kwargs={"loss_weight_step_start_ratio": 0.02, "loss_weight_step_end_ratio": 0.08},
                momentum_scheduler_kwargs={"momentum_start_value": 0.996, "momentum_end_value": 0.999},
        )
        assert model.hparams.loss_scheduler_kwargs["loss_weight_step_start_ratio"] == 0.02 # type: ignore
        assert model.hparams.loss_scheduler_kwargs["loss_weight_step_end_ratio"] == 0.08 # type: ignore
        assert model.hparams.momentum_scheduler_kwargs["momentum_start_value"] == 0.996 # type: ignore
        assert model.hparams.momentum_scheduler_kwargs["momentum_end_value"] == 0.999 # type: ignore
        assert model.hparams.loss_kwargs["temperature"] == 0.1 # type: ignore
        assert model.hparams.loss_kwargs["top_k_negatives"] == 1000 # type: ignore

    def test_initialization_of_model(self):
        """Test that the model is properly initialized."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            total_steps=10000,
        )
        model.summarize_model()
        

class TestTrainingStep:
    @pytest.fixture
    def model(self):
        """Model fixture."""
        return CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            total_steps=10000,
        )

    def test_key_encoder_matches_query_encoder(self, model, input_tensor):
        """Test that the key encoder matches the query encoder."""
        proj_1, _ = model.forward(input_tensor) # type: ignore
        proj_2, _ = model.key_encoder(input_tensor) # type: ignore
        assert proj_1.shape == proj_2.shape
        assert torch.allclose(proj_1, proj_2)

    def test_queue_update(self, model, input_tensor):
        """Test that the queue is updated correctly."""
        model._update_queue(input_tensor)
    
    def test_momentum_update(self, model, input_tensor):
        """Test that the momentum is updated correctly."""
        model._update_key_encoder()
    
    def test_loss_computation(self, model, input_tensor):
        """Test that the loss is computed correctly."""
        model._compute_loss(input_tensor)
    

class TestValTestSteps:
    pass