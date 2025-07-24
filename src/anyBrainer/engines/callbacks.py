"""Define custom callbacks for Pytorch Lightning."""

__all__ = [
    "UpdateDatamoduleEpoch",
]

import logging

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

class UpdateDatamoduleEpoch(pl.Callback):
    """
    Callback to update the datamodule's _current_epoch attribute.
    """        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Sync the current epoch between the trainer and the datamodule."""
        if not hasattr(pl_module, "_current_epoch"):
            logger.warning("Datamodule does not have _current_epoch attribute. "
                           "This is likely due to a custom datamodule that does not "
                           "inherit from anyBrainer.data.BaseDataModule.")
            return
        
        pl_module._current_epoch = trainer.current_epoch + 1 # type: ignore
        logger.debug(f"Updated datamodule epoch to {trainer.current_epoch + 1}")