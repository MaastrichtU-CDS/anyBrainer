"""Tests for Pytorch's Lightning Module."""

import pytest
import torch
import torch.optim as optim
from torch import nn
# pyright: reportPrivateImportUsage=false
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.engines.models import CLwAuxModel
from anyBrainer.engines.utils import (
    pack_ids,
    get_sub_ses_tensors,
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

@pytest.fixture(autouse=True)
def log_mock(monkeypatch):
    """Mock the log_dict method of the LightningModule."""
    def dummy_log(self, obj, *args, **kwargs):
        print(obj)
    
    monkeypatch.setattr(CLwAuxModel, "log", dummy_log)
    monkeypatch.setattr(CLwAuxModel, "log_dict", dummy_log)

@pytest.fixture(scope="module")
def input_batch():
    """Input batch for the model."""
    return {
        "query": torch.randn(8, 1, 128, 128, 128),
        "key": torch.randn(8, 1, 128, 128, 128),
        "mod": ["t1", "t2", "flair", "dwi", "adc", "swi", "other", "t1"],
        "sub_id": ["sub_01", "sub_02", "sub_03", "sub_04", "sub_05", "sub_06", "sub_07", "sub_01"],
        "ses_id": ["ses_01", "ses_01", "ses_01", "ses_01", "ses_01", "ses_01", "ses_01", "ses_01"],
    }

@torch.no_grad()
def manual_param_step(model: nn.Module, lr: float = 1e-3, use_grads: bool = True):
    """
    Apply an in-place update to all trainable params of `model`.

    If `use_grads` is True, performs a simple SGD step:  p <- p - lr * grad
    Otherwise, adds a small random perturbation to each param (useful when no grads).

    Args:
        model: nn.Module whose params to update.
        lr:    learning rate / perturbation scale.
        use_grads: whether to use existing .grad tensors or random noise.
    """
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if use_grads and p.grad is not None:
            p.data.add_(p.grad, alpha=-lr)
        else:
            p.data.add_(torch.randn_like(p) * lr)


class TestCLwAuxModel:
    def test_model_forward_pass(
        self,
        input_tensor,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
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
    
    def test_incorrect_model_kwargs(
        self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
        """Test that the model raises a ValueError if the model kwargs are incorrect."""
        with pytest.raises(Exception):
            CLwAuxModel(
                model_kwargs={"wrong_key": "wrong_value"},
                optimizer_kwargs=swinv2cl_optimizer_kwargs,
                lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
                total_steps=10000,
            )
            

class TestConfigureOptimizers:
    def test_one_optimizer_one_scheduler(
        self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
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
    
    def test_one_optimizer_no_scheduler(
        self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
    ):
        """Test that the model returns an optimizer if no scheduler is provided."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            total_steps=10000,
        )
        out = model.configure_optimizers()
        assert isinstance(out, optim.Optimizer)

    def test_multiple_optimizers_no_scheduler(
        self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
    ):
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
    
    def test_multiple_optimizers_one_scheduler(
        self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
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
    
    def test_multiple_optimizers_multiple_schedulers(
        self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
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
    
    def test_multiple_schedulers_one_optimizer(
        self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
        """Test that the model returns a list of optimizers and a list of scheduler dictionaries."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=[swinv2cl_scheduler_kwargs, swinv2cl_scheduler_kwargs],
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_len_mismatch(self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
        """Test length mismatch between optimizer and scheduler."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=[swinv2cl_optimizer_kwargs],
            lr_scheduler_kwargs=[swinv2cl_scheduler_kwargs, swinv2cl_scheduler_kwargs],
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_not_found_optimizer(self,
        swinv2cl_model_kwargs,
    ):
        """Test that the model raises a ValueError if the optimizer name is not found."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs={"name": "WrongOptim", "lr": 1e-4, "weight_decay": 1e-5},
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_not_found_scheduler(self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
    ):
        """Test that the model raises a ValueError if the scheduler name is not found."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs={"name": "WrongScheduler", "interval": "step", "frequency": 1},
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_missing_scheduler_keys(self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
    ):
        """Test that the model raises a ValueError if the scheduler keys are missing."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs={"name": "CosineAnnealingWithWarmup", "frequency": 1},
            total_steps=10000,
        )
        with pytest.raises(ValueError):
            model.configure_optimizers()
    
    def test_incorrect_optimizer_kwargs(self,
        swinv2cl_model_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
        """Test that the model raises a Exception if the optimizer kwargs are incorrect."""
        model = CLwAuxModel(
                model_kwargs=swinv2cl_model_kwargs,
                optimizer_kwargs={"name": "AdamW", "wrong_key": 1e-4, "weight_decay": 1e-5},
                lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
                total_steps=10000,
            )
        with pytest.raises(Exception):
            model.configure_optimizers()

    
    def test_incorrect_lr_scheduler_kwargs(self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
    ):
        """Test that the model raises a ValueError if the lr scheduler kwargs are incorrect."""
        model = CLwAuxModel(
                model_kwargs=swinv2cl_model_kwargs,
                optimizer_kwargs=swinv2cl_optimizer_kwargs,
                lr_scheduler_kwargs={"name": "WrongScheduler", "interval": "step", "frequency": 1},
                total_steps=10000,
            )
        with pytest.raises(Exception):
            model.configure_optimizers()


class TestInitializations:
    def test_initialization_of_hparams(self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
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
                loss_scheduler_kwargs={"loss_weight_step_start": 1000, "loss_weight_step_end": 3000},
                momentum_scheduler_kwargs={"momentum_start_value": 0.996, "momentum_end_value": 0.999},
        )
        assert model.hparams.loss_scheduler_kwargs["loss_weight_step_start"] == 1000 # type: ignore
        assert model.hparams.loss_scheduler_kwargs["loss_weight_step_end"] == 3000 # type: ignore
        assert model.hparams.momentum_scheduler_kwargs["momentum_start_value"] == 0.996 # type: ignore
        assert model.hparams.momentum_scheduler_kwargs["momentum_end_value"] == 0.999 # type: ignore
        assert model.hparams.loss_kwargs["temperature"] == 0.1 # type: ignore
        assert model.hparams.loss_kwargs["top_k_negatives"] == 1000 # type: ignore

    def test_initialization_of_model(self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
        """Test that the model is properly initialized."""
        model = CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            total_steps=10000,
        )
        model.summarize()
        

class TestTrainingStep:
    @pytest.fixture
    def model(self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
        """Model fixture."""
        return CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            test_mode=True,
            total_steps=10000,
            momentum_scheduler_kwargs={
                "momentum_step_start": 0,
                "momentum_step_end": 2000,
                "momentum_start_value": 0.99,
                "momentum_end_value": 0.999
            },
        )

    def test_queue_update(self, model, input_batch):
        """Test that the queue is updated correctly."""
        input_batch["sub_id"], input_batch["ses_id"] = get_sub_ses_tensors(
            input_batch, input_batch["query"].device
        )
        for _ in range(3):
            proj, _ = model.forward(input_batch["query"]) # type: ignore
            model._update_queue(proj, pack_ids(input_batch["sub_id"], input_batch["ses_id"]))
        assert model.queue.shape == (16384, 128)
        assert model.queue_ids.shape == (16384,)
        assert model.queue_ptr.shape == (1,)
        assert model.queue_ptr[0] == 24
    
    def test_queue_update_with_duplicates(self, model, input_batch):
        """Test that the queue is updated correctly with duplicates."""
        input_batch["sub_id"], input_batch["ses_id"] = get_sub_ses_tensors(
            input_batch, input_batch["query"].device
        )
        for _ in range(3):
            proj, _ = model.forward(input_batch["query"]) # type: ignore
            model._update_queue(proj, pack_ids(input_batch["sub_id"], input_batch["ses_id"]))
        
        negatives = model._get_negatives(pack_ids(input_batch["sub_id"], input_batch["ses_id"]))
        assert negatives.shape == (0, 128)

    def test_key_encoder_matches_query_encoder(self, model, input_tensor):
        """Test that the key encoder matches the query encoder."""
        proj_1, _ = model.forward(input_tensor) # type: ignore
        proj_2, _ = model.key_encoder(input_tensor) # type: ignore
        assert proj_1.shape == proj_2.shape
        assert torch.equal(proj_1, proj_2)

    def test_key_encoder_no_grad(self, model):
        """Test that the key encoder has no gradients."""
        assert all(not p.requires_grad for p in model.key_encoder.parameters())
    
    def test_momentum_update(self, model, input_tensor):
        """Test that the momentum is updated correctly."""
        old_k = [p.clone() for p in model.key_encoder.parameters()]
        manual_param_step(model.model) # simulate online model update
        momentum = model.get_step_scheduler_values()[1]["momentum"]

        model._update_key_encoder()
        
        for pk_old, pq_new, pk_new in zip(
            old_k, model.model.parameters(), model.key_encoder.parameters()
        ):
            expected = pk_old * momentum + pq_new * (1 - momentum)
            assert torch.allclose(pk_new, expected)
    
    def test_loss_computation(self, model, input_batch):
        """Test that the loss is computed correctly."""
        input_batch["sub_id"], input_batch["ses_id"] = get_sub_ses_tensors(
            input_batch, input_batch["query"].device
        )
        for _ in range(3):
            proj_1, aux = model.forward(input_batch["query"]) # type: ignore
            proj_2, _ = model.key_encoder(input_batch["key"]) # type: ignore
            model._update_queue(proj_1, pack_ids(input_batch["sub_id"], input_batch["ses_id"]))

        aux_labels = torch.zeros(8, 7)
        aux_labels[:, 0] = 1
        loss, loss_dict = model._compute_loss(
            proj_1, proj_2, model.queue, aux, aux_labels
        )
        assert loss.shape == ()
        assert loss_dict.keys() == {"loss_info_nce", "loss_aux", "loss_weight", "pos_mean", 
                                    "neg_mean", "neg_entropy", "contrastive_acc"}
        assert loss_dict["loss_info_nce"].shape == ()
        assert loss_dict["loss_aux"].shape == ()
        assert loss_dict["pos_mean"].shape == ()
        assert loss_dict["neg_mean"].shape == ()
        assert loss_dict["neg_entropy"].shape == ()
        assert loss_dict["contrastive_acc"].shape == ()
    
    def test_complete_training_step(self, model, input_batch):
        """Test that the complete training step works."""
        for _ in range(3):
            model.on_after_batch_transfer(input_batch, 0) # type: ignore
            loss = model.training_step(input_batch, 0) # type: ignore
            assert loss.shape == ()
            
            
class TestValTestSteps:
    @pytest.fixture
    def model(self,
        swinv2cl_model_kwargs,
        swinv2cl_optimizer_kwargs,
        swinv2cl_scheduler_kwargs,
    ):
        """Model fixture."""
        return CLwAuxModel(
            model_kwargs=swinv2cl_model_kwargs,
            optimizer_kwargs=swinv2cl_optimizer_kwargs,
            lr_scheduler_kwargs=swinv2cl_scheduler_kwargs,
            total_steps=10000,
        )
    def test_val_step(self, model, input_batch):
        """Test that the validation step works."""
        model.on_after_batch_transfer(input_batch, 0)
        model.validation_step(input_batch, 0)
    
    def test_test_step(self, model, input_batch):
        """Test that the test step works."""
        model.on_after_batch_transfer(input_batch, 0)
        model.test_step(input_batch, 0)


@pytest.mark.skip(reason="Not implemented")
class TestPredictStep:
    pass