"""Define custom callbacks for Pytorch Lightning."""

__all__ = [
    "UpdateDatamoduleEpoch",
    "LogLR",
    "LogGradNorm",
]

import logging
from typing import Any

import lightning.pytorch as pl
import torch.optim as optim

from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.utils import (
    get_optimizer_lr,
    get_total_grad_norm,
)
from anyBrainer.core.engines.utils import (
    sync_dist_safe,
)

logger = logging.getLogger(__name__)


@register(RK.CALLBACK)
class UpdateDatamoduleEpoch(pl.Callback):
    """
    Callback to update the datamodule's _current_epoch attribute.
    """        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Sync the current epoch between the trainer and the datamodule."""
        if (trainer.datamodule is None or not hasattr(trainer.datamodule, "_current_epoch")): # pyright: ignore
            logger.warning("Datamodule does not have _current_epoch attribute. "
                           "This is likely due to a custom datamodule that does not "
                           "inherit from anyBrainer.data.BaseDataModule.")
            return
        
        trainer.datamodule._current_epoch = trainer.current_epoch + 1 # type: ignore
        logger.debug(f"Updated datamodule epoch to {trainer.current_epoch + 1}")


@register(RK.CALLBACK)
class LogLR(pl.Callback):
    """
    Callback to log the learning rate for all optimizers in the trainer.
    """
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: optim.Optimizer,
    ) -> None:
        """Log the learning rate."""
        if not isinstance(trainer.optimizers, list):
            logger.warning("Cannot log LR because pl_module.optimizers is not a list.")
            return
        
        pl_module.log_dict(
            get_optimizer_lr(trainer.optimizers),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=sync_dist_safe(pl_module),
        )


@register(RK.CALLBACK)
class LogGradNorm(pl.Callback):
    """
    Callback to log the gradient norm for all parameters in the model.
    """
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log the gradient norm."""
        pl_module.log(
            "train/grad_norm",
            get_total_grad_norm(pl_module),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=sync_dist_safe(pl_module),
        )