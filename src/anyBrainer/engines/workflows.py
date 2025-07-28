"""Define workflows for training, validation, and testing."""

import logging
import time
from pathlib import Path
import resource
from typing import Any, cast
from dataclasses import dataclass

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from monai.data.utils import set_rnd
from monai.utils.misc import set_determinism
from typeguard import typechecked

from anyBrainer.utils import (
    create_save_dirs,
    load_model_from_ckpt,
    resolve_path,
)
from anyBrainer.log import LoggingManager
from anyBrainer.engines.utils import (
    unpack_settings_for_train_workflow,
)
from anyBrainer.engines.factory import (
    ModuleFactory,
    UnitFactory,
)

# Removed typechecked for now
@dataclass
class TrainingSettings:
    project: str
    experiment: str
    root_dir: Path
    seed: int
    worker_logs: bool
    dev_mode: bool
    disable_file_logs: bool
    enable_wandb: bool
    wandb_watch_enable: bool
    wandb_watch_kwargs: dict[str, Any]
    pl_datamodule_name: str
    data_dir: Path
    num_workers: int
    batch_size: int
    train_val_test_split: tuple[float, float, float] | list[float]
    train_transforms: dict[str, Any] | None
    val_transforms: dict[str, Any] | None
    test_transforms: dict[str, Any] | None
    predict_transforms: dict[str, Any] | None
    new_version: bool
    model_checkpoint: Path | None
    save_every_n_epochs: int
    save_last: bool
    pl_module_name: str
    pl_module_kwargs: dict[str, Any]
    pl_callback_kwargs: list[dict[str, Any]]
    pl_trainer_kwargs: dict[str, Any]

    def __post_init__(self):
        self.root_dir = resolve_path(self.root_dir)
        self.data_dir = resolve_path(self.data_dir)
        self.model_checkpoint = (
            resolve_path(self.model_checkpoint) 
            if self.model_checkpoint is not None else None
        )
        self.train_val_test_split = tuple(self.train_val_test_split) # type: ignore
        self.exp_dir: Path = self.root_dir / self.project / self.experiment

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


class TrainWorkflow:
    """
    Basic workflow for wrapping pytorch lightning trainer operations.
    """
    settings: TrainingSettings = cast(TrainingSettings, ...)
    logging_manager: LoggingManager = cast(LoggingManager, ...)
    main_logger: logging.Logger = cast(logging.Logger, ...)
    wandb_logger: WandbLogger = cast(WandbLogger, ...)
    datamodule: pl.LightningDataModule = cast(pl.LightningDataModule, ...)
    model: pl.LightningModule = cast(pl.LightningModule, ...)
    ckpt_path: Path | None = None
    callbacks: list[pl.Callback] = cast(list[pl.Callback], ...)
    trainer: pl.Trainer = cast(pl.Trainer, ...)

    def __init__(
        self,
        global_settings: dict[str, Any],
        *,
        pl_datamodule_settings: dict[str, Any],
        pl_module_settings: dict[str, Any],
        pl_callback_settings: list[dict[str, Any]],
        pl_trainer_settings: dict[str, Any],
        logging_settings: dict[str, Any] = {},
        ckpt_settings: dict[str, Any] = {},
        **kwargs,
    ):  
        self.settings = TrainingSettings(
            **unpack_settings_for_train_workflow(
                global_settings, 
                logging_settings, 
                pl_datamodule_settings,
                pl_module_settings,
                pl_callback_settings,
                pl_trainer_settings,
                ckpt_settings,
        ))

        self.__setup()
    
    def __setup(self):
        """
        Initializes the workflow components, including:
        Logging, Datamodule, Model, Callbacks, Trainer

        Also creates the target save directories. 
        """
        self.configure_environment()
        try:
            self.logging_manager, self.main_logger, self.wandb_logger = self.setup_logging()
        except ValueError:
            self.main_logger.exception("Error in setup_logging(); ensure returns match the docstring.")
            raise
        try:
            self.datamodule = self.setup_datamodule()
        except ValueError:
            self.main_logger.exception("Error in setup_datamodule(); ensure returns match the docstring.")
            raise
        try:
            self.model, self.ckpt_path = self.setup_model()
        except ValueError:
            self.main_logger.exception("Error in setup_model(); ensure returns match the docstring.")
            raise
        try:
            self.callbacks = self.setup_callbacks()
        except ValueError:
            self.main_logger.exception("Error in setup_callbacks(); ensure returns match the docstring.")
            raise
        try:
            self.trainer = self.setup_trainer()
        except ValueError:
            self.main_logger.exception("Error in setup_trainer(); ensure returns match the docstring.")
            raise
        self.finalize_setup()

        if self.wandb_logger is not None and self.settings.wandb_watch_enable:
            self.wandb_logger.watch(self.model, **self.settings.wandb_watch_kwargs)

        create_save_dirs(
            exp_dir=self.settings.exp_dir,
            new_version=self.settings.new_version,
            create_ckpt_dir=True,
        )
    
    def configure_environment(self):
        """
        Configures the environment for the workflow.

        Override for custom environment configuration.
        """
        set_determinism(seed=self.settings.seed)
        torch.set_float32_matmul_precision("high") # for H100
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_logging(self) -> tuple[LoggingManager, logging.Logger, WandbLogger]:
        """
        Returns logging configuration from self.settings.
        
        Will populate self.logging_manager, self.main_logger and self.wandb_logger.

        Override for custom logging configuration.
        """
        logging_config = {
            "name": "LoggingManager", 
            "logs_root": self.settings.exp_dir / "logs",
            "worker_logs": self.settings.worker_logs,
            "dev_mode": self.settings.dev_mode,
            "disable_file_logs": self.settings.disable_file_logs,
            "enable_wandb": self.settings.enable_wandb,
            "wandb_project": self.settings.project,
            "experiment": self.settings.experiment,
            "num_workers": self.settings.num_workers,
        }
        logging_manager = cast(
            LoggingManager, 
            ModuleFactory.get_logging_manager_instance_from_kwargs(
                logging_manager_kwargs=logging_config
            )
        )
        main_logger, wandb_logger = (
            cast(logging.Logger, logging_manager.main_logger), 
            cast(WandbLogger, logging_manager.wandb_logger)
        )
        return logging_manager, main_logger, wandb_logger

    def setup_datamodule(self) -> pl.LightningDataModule:
        """
        Returns datamodule configuration from self.settings.
        
        Will populate self.datamodule.

        Override for custom datamodule configuration.
        """
        datamodule_config = {
            "name": self.settings.pl_datamodule_name,
            "data_dir": self.settings.data_dir,
            "batch_size": self.settings.batch_size,
            "num_workers": self.settings.num_workers,
            "train_val_test_split": self.settings.train_val_test_split,
            "train_transforms": (UnitFactory.get_transformslist_from_kwargs(self.settings.train_transforms) 
                              if self.settings.train_transforms is not None else None),
            "val_transforms": (UnitFactory.get_transformslist_from_kwargs(self.settings.val_transforms) 
                              if self.settings.val_transforms is not None else None),
            "test_transforms": (UnitFactory.get_transformslist_from_kwargs(self.settings.test_transforms) 
                              if self.settings.test_transforms is not None else None),
            "predict_transforms": (UnitFactory.get_transformslist_from_kwargs(self.settings.predict_transforms) 
                              if self.settings.predict_transforms is not None else None),
            "worker_logging_fn": self.logging_manager.get_setup_worker_logging_fn(),
            "worker_seeding_fn": set_rnd,
            "seed": self.settings.seed,
        }
        return cast(
            pl.LightningDataModule,
            ModuleFactory.get_pl_datamodule_instance_from_kwargs(
                pl_datamodule_kwargs=datamodule_config
            )
        )

    def setup_model(self) -> tuple[pl.LightningModule, Path | None]:
        """
        Returns model configuration from settings.

        If not new version of the experiment or model checkpoint is provided,
        it attempts to load the model from the checkpoint. If successful, it 
        populates self.ckpt_path. Otherwise, it creates a new model instance.

        Will populate self.model and self.ckpt_path.

        Override for custom model configuration.
        """
        pl_module_config = {
            "name": self.settings.pl_module_name,
            **self.settings.pl_module_kwargs,
        }
        ckpt_path = (self.settings.model_checkpoint or 
                     self.settings.exp_dir / "checkpoints" / "last.ckpt")

        if not self.settings.new_version:
            model = load_model_from_ckpt(
                model_cls=cast(
                    type[pl.LightningModule],
                    ModuleFactory.get_pl_module_instance_from_kwargs(pl_module_config, cls_only=True)
                ),
                ckpt_path=ckpt_path,
            )

        if model is not None:
            return model, ckpt_path

        return (
            cast(pl.LightningModule, 
                 ModuleFactory.get_pl_module_instance_from_kwargs(pl_module_config)), 
            None
        )
        
    def setup_callbacks(self) -> list[pl.Callback]:
        """
        Returns callbacks configuration from settings.

        Will populate self.callbacks.

        Override for custom callbacks configuration.
        """
        callbacks_config = [
            {"name": "ModelCheckpoint",
                "dirpath": self.settings.exp_dir / "checkpoints",
                "filename": "{epoch:02d}",
                "every_n_epochs": self.settings.save_every_n_epochs,
                "save_last": self.settings.save_last},
            {"name": "UpdateDatamoduleEpoch"},
            {"name": "LogLR"},
            {"name": "LogGradNorm"},
        ]
        callbacks_config.extend(self.settings.pl_callback_kwargs)
        
        return cast(
            list[pl.Callback],
            UnitFactory.get_pl_callback_instances_from_kwargs(
                callback_kwargs=callbacks_config
            )
        )
    
    def setup_trainer(self) -> pl.Trainer:
        """
        Returns trainer configuration from settings.

        Will populate self.trainer.

        Override for custom trainer configuration.
        """
        trainer_config = {
            "logger": self.wandb_logger,
            "callbacks": self.callbacks,
            "reload_dataloaders_every_n_epochs": 1,
            **self.settings.pl_trainer_kwargs,
        }
        return pl.Trainer(**trainer_config)
    
    def finalize_setup(self):
        """
        Finalizes the setup of the workflow.

        Override for custom finalization.
        """
        pass

    def __call__(self):
        """
        Runs the training workflow.
        """
        self.main_logger.info("\n[TRAINING STARTED]")
        self.main_logger.info(self.train_start_summary())

        start_time = time.time()
        train_stats = self.train_run()
        duration = time.time() - start_time
        
        if self.wandb_logger is not None:
            self.wandb_logger.experiment.unwatch(self.model)

        self.main_logger.info(f"\n[TRAINING COMPLETED] in {duration/60:.1f} min")
        self.main_logger.info(self.train_end_summary(train_stats))

    def train_run(self) -> dict[str, Any]:
        """
        Runs the training workflow.

        Override for custom training run.
        """
        self.trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=self.ckpt_path)

        train_stats = {
            "peak_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_gpu_mem": (torch.cuda.max_memory_allocated() / 1024**2 
                             if torch.cuda.is_available() else None),
            "best_val_loss": self.trainer.callback_metrics.get("val/loss_best")
        }

        return train_stats
    
    def train_start_summary(self) -> str:
        """
        Formats the summary of the training workflow.

        Override for custom training start summary.
        """
        return (
            f"{'-'*60}"
            f"\n[EXPERIMENT DETAILS]\n"
            f"Experiment:       {self.settings.experiment}\n"
            f"Model:            {self.settings.pl_module_name}\n"
            f"Trainable Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}\n"
            f"Data:             {self.settings.pl_datamodule_name} @ {self.settings.data_dir}\n"
            f"Batch Size:       {self.settings.batch_size}\n"
            f"Epochs:           {self.trainer.max_epochs}\n"
            f"Optimizer:        {self.settings.pl_module_kwargs.get('optimizer_kwargs', 'unspecified')}\n"
            f"Loss:             {self.settings.pl_module_kwargs.get('loss_fn_kwargs', 'unspecified')}\n"
            f"WandB Project:    {self.settings.project} (enabled: {self.settings.enable_wandb})\n"
            f"Output Dir:       {self.settings.save_dir}\n"
            f"Device:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
            f"AMP enabled:      {self.trainer.precision == '16-mixed' or self.trainer.precision == 16}\n"
            f"{'-'*60}"
        )

    def train_end_summary(self, train_stats: dict[str, Any]) -> str:
        """
        Formats the summary of the training workflow.

        Override for custom training end summary.
        """
        peak_mem_mb = train_stats.get("peak_mem_mb")
        peak_gpu_mem = train_stats.get("peak_gpu_mem")
        best_val_loss = train_stats.get("best_val_loss")

        return (
            f"{'-'*60}"
            f"\n[TRAINING SUMMARY]\n"
            f"Peak Memory:      {peak_mem_mb:.02f} MB\n"
            f"Peak GPU Memory:  {peak_gpu_mem:.02f} MB\n" if peak_gpu_mem is not None else "Peak GPU Memory:  N/A\n"
            f"Best Val Loss:    {best_val_loss:.04f}\n" if best_val_loss is not None else "Best Val Loss:    N/A\n"
            f"{'-'*60}"
        )

    def close(self):
        """
        Close the workflow.

        Override for custom workflow closing.
        """
        self.logging_manager.close()

    fit = __call__
