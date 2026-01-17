"""Define workflows for training, validation, and testing."""

from __future__ import annotations

__all__ = [
    "TrainWorkflow",
    "CVWorkflow",
    "TestWorkflow",
    "SweepWorkflow",
    "CompositeWorkflow",
]

import logging
from typing import Any, cast, Literal, Sequence, Iterator
from itertools import product
import time
from pathlib import Path
import resource
from dataclasses import dataclass
from copy import deepcopy

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from anyBrainer.core.engines.callbacks import MetricAggregator
from monai.data.utils import set_rnd
from monai.utils.misc import set_determinism

from anyBrainer.core.utils import (
    create_save_dirs,
    resolve_path,
)
from anyBrainer.core.engines.utils import (
    unpack_settings_for_train_workflow,
)
from anyBrainer.registry import register, get
from anyBrainer.registry import RegistryKind as RK
from anyBrainer.factories import (
    ModuleFactory,
    UnitFactory,
)
from anyBrainer.interfaces import (
    Workflow,
    LoggingManager,
)


@dataclass
class TrainingSettings:
    """Training settings.

    Contains args that are primarily useful for orchestrating the components
    of the workflow and remove duplication.

    Note:
    Avoids storing args that are too specific for particular applications;
    e.g. input volume patch_size -> inevitable duplication in transforms and
    inferer kwargs.
    """

    project: str
    experiment: str
    root_dir: Path
    seed: int
    worker_logs: bool
    dev_mode: bool
    save_logs: bool
    enable_wandb: bool
    wandb_entity: str | None
    wandb_watch_enable: bool
    wandb_watch_kwargs: dict[str, Any]
    pl_datamodule_name: str
    data_dir: Path
    data_handler_kwargs: dict[str, Any]
    num_workers: int
    batch_size: int
    train_val_test_split: Sequence[float]
    val_mode: Literal["single", "repeated"]
    n_splits: int | None
    current_split: int
    extra_dataloader_kwargs: dict[str, Any]
    train_transforms: dict[str, Any] | str | None
    val_transforms: dict[str, Any] | str | None
    test_transforms: dict[str, Any] | str | None
    predict_transforms: dict[str, Any] | str | None
    new_version: bool
    model_checkpoint: Path | None
    save_every_n_epochs: int
    save_every_n_steps: int
    save_last: bool
    save_top_k: int
    pl_module_name: str
    pl_module_kwargs: dict[str, Any]
    pl_callback_kwargs: list[dict[str, Any]]
    pl_trainer_kwargs: dict[str, Any]
    extra_logging_kwargs: dict[str, Any]
    extra_pl_datamodule_kwargs: dict[str, Any]
    extra_ckpt_kwargs: dict[str, Any]

    def __post_init__(self):
        self.root_dir = resolve_path(self.root_dir)
        self.data_dir = resolve_path(self.data_dir)
        self.model_checkpoint = (
            resolve_path(self.model_checkpoint)
            if self.model_checkpoint is not None
            else None
        )
        self.exp_dir: Path = self.root_dir / self.project / self.experiment

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


@register(RK.WORKFLOW)
class TrainWorkflow(Workflow):
    """Basic workflow for wrapping pytorch lightning trainer operations."""

    settings: TrainingSettings
    logging_manager: LoggingManager
    main_logger: logging.Logger
    wandb_logger: WandbLogger | None
    datamodule: pl.LightningDataModule
    model: pl.LightningModule
    ckpt_path: Path | None
    callbacks: list[pl.Callback]
    trainer: pl.Trainer

    def __init__(
        self,
        global_settings: dict[str, Any],
        *,
        pl_datamodule_settings: dict[str, Any],
        pl_module_settings: dict[str, Any],
        pl_callback_settings: list[dict[str, Any]],
        pl_trainer_settings: dict[str, Any],
        logging_settings: dict[str, Any] | None = None,
        ckpt_settings: dict[str, Any] | None = None,
        validation_settings: dict[str, Any] | None = None,
        **extra,
    ):
        logging_settings = logging_settings or {}
        ckpt_settings = ckpt_settings or {}
        validation_settings = validation_settings or {}

        self.settings = TrainingSettings(
            **unpack_settings_for_train_workflow(
                global_settings,
                logging_settings,
                pl_datamodule_settings,
                pl_module_settings,
                pl_callback_settings,
                pl_trainer_settings,
                ckpt_settings,
            )
        )

    def __setup(self):
        """Initializes the workflow components, including: Logging, Datamodule,
        Model, Callbacks, Trainer.

        Also creates the target save directories.
        """
        create_save_dirs(
            exp_dir=self.settings.exp_dir,
            new_version=self.settings.new_version,
            create_ckpt_dir=True,
        )
        self.configure_environment()
        try:
            self.logging_manager, self.main_logger, self.wandb_logger = (
                self.setup_logging()
            )
        except ValueError:
            logging.exception(
                "[TrainWorkflow] Error in setup_logging(); ensure returns match the docstring."
            )
            raise
        try:
            self.datamodule = self.setup_datamodule()
        except ValueError:
            self.main_logger.exception(
                "[TrainWorkflow] Error in setup_datamodule(); ensure returns match the docstring."
            )
            raise
        try:
            self.model, self.ckpt_path = self.setup_model()
        except ValueError:
            self.main_logger.exception(
                "[TrainWorkflow] Error in setup_model(); ensure returns match the docstring."
            )
            raise
        try:
            self.callbacks = self.setup_callbacks()
        except ValueError:
            self.main_logger.exception(
                "[TrainWorkflow] Error in setup_callbacks(); ensure returns match the docstring."
            )
            raise
        try:
            self.trainer = self.setup_trainer()
        except ValueError:
            self.main_logger.exception(
                "[TrainWorkflow] Error in setup_trainer(); ensure returns match the docstring."
            )
            raise
        self.finalize_setup()

        if (
            self.wandb_logger is not None
            and self.settings.wandb_watch_enable
            and getattr(self.trainer, "is_global_zero", True)
        ):
            self.wandb_logger.watch(
                self.model, **(self.settings.wandb_watch_kwargs or {})
            )

    def configure_environment(self):
        """Configures the environment for the workflow.

        Override for custom environment configuration.
        """
        set_determinism(seed=self.settings.seed)
        torch.set_float32_matmul_precision("high")  # for H100
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_logging(
        self,
    ) -> tuple[LoggingManager, logging.Logger, WandbLogger | None]:
        """Returns logging configuration from self.settings.

        Will populate self.logging_manager, self.main_logger and
        self.wandb_logger.

        Override for custom logging configuration.
        """
        if self.settings.extra_logging_kwargs:
            logging.warning(
                "[TrainWorkflow] Identified extra_logging_kwargs; out-of-the-box "
                "TrainWorkflow cannot ensure they don't override explicitly "
                "set kwargs; unless custom logging manager is used."
            )

        logging_config = {
            "name": "DefaultLoggingManager",
            "logs_root": self.settings.exp_dir / "logs",
            "worker_logs": self.settings.worker_logs,
            "dev_mode": self.settings.dev_mode,
            "save_logs": self.settings.save_logs,
            "enable_wandb": self.settings.enable_wandb,
            "wandb_project": self.settings.project,
            "wandb_entity": self.settings.wandb_entity,
            "experiment": self.settings.experiment,
            "num_workers": self.settings.num_workers,
            **(self.settings.extra_logging_kwargs or {}),
        }
        logging_manager = ModuleFactory.get_logging_manager_instance_from_kwargs(
            logging_manager_kwargs=logging_config
        )
        main_logger, wandb_logger = (
            logging_manager.main_logger,
            logging_manager.wandb_logger,
        )
        return logging_manager, main_logger, wandb_logger

    def setup_datamodule(self) -> pl.LightningDataModule:
        """Returns datamodule configuration from self.settings.

        Will populate self.datamodule.

        Override for custom datamodule configuration.
        """
        if any(
            k
            in {
                "name",
                "data_dir",
                "data_handler_kwargs",
                "batch_size",
                "num_workers",
                "extra_dataloader_kwargs" "train_val_test_split",
                "val_mode",
                "n_splits",
                "current_split",
                "train_transforms",
                "val_transforms",
                "test_transforms",
                "predict_transforms",
                "worker_logging_fn",
                "worker_seeding_fn",
                "seed",
            }
            for k in self.settings.extra_pl_datamodule_kwargs.keys()
        ):
            logging.warning(
                "[TrainWorkflow] Identified arguments in `extra_pl_datamodule_kwargs` that are already "
                "set in the settings and will be overwritten."
            )

        datamodule_config = {
            "name": self.settings.pl_datamodule_name,
            "data_dir": self.settings.data_dir,
            "data_handler_kwargs": self.settings.data_handler_kwargs,
            "batch_size": self.settings.batch_size,
            "num_workers": self.settings.num_workers,
            "extra_dataloader_kwargs": self.settings.extra_dataloader_kwargs,
            "train_val_test_split": self.settings.train_val_test_split,
            "val_mode": self.settings.val_mode,
            "n_splits": self.settings.n_splits,
            "current_split": self.settings.current_split,
            "train_transforms": self.settings.train_transforms,
            "val_transforms": self.settings.val_transforms,
            "test_transforms": self.settings.test_transforms,
            "predict_transforms": self.settings.predict_transforms,
            "worker_logging_fn": self.logging_manager.get_setup_worker_logging_fn(),
            "worker_seeding_fn": set_rnd,
            "seed": self.settings.seed,
            **(self.settings.extra_pl_datamodule_kwargs or {}),
        }
        return ModuleFactory.get_pl_datamodule_instance_from_kwargs(
            pl_datamodule_kwargs=datamodule_config
        )

    def setup_model(self) -> tuple[pl.LightningModule, Path | None]:
        """Creates a new model instance from settings.

        If not `new_version` of the experiment or model checkpoint is provided,
        it attempts to find a checkpoint in the experiment directory. If not
        found, `ckpt_path` is set to None. Returns model instance and `ckpt_path`.

        Will populate self.model and self.ckpt_path.

        Override for custom model configuration.
        """
        model = ModuleFactory.get_pl_module_instance_from_kwargs(
            pl_module_kwargs=self.settings.pl_module_kwargs
        )
        if self.settings.new_version:
            return model, None

        ckpt_path = (
            self.settings.model_checkpoint
            or self.settings.exp_dir / "checkpoints" / "last.ckpt"
        )

        if not ckpt_path.exists():
            self.main_logger.warning(
                f"[TrainWorkflow] Checkpoint file {ckpt_path} does not exist; "
                "will create new model."
            )
            return model, None

        return model, ckpt_path

    def setup_callbacks(self) -> list[pl.Callback]:
        """Returns callbacks configuration from settings.

        Will populate self.callbacks.

        Override for custom callbacks configuration.
        """
        callbacks_config = [
            {
                "name": "ModelCheckpoint",
                "dirpath": self.settings.exp_dir / "checkpoints",
                "filename": "{epoch:02d}-{step}",
                "every_n_epochs": self.settings.save_every_n_epochs,
                "every_n_train_steps": self.settings.save_every_n_steps,
                "save_top_k": self.settings.save_top_k,
                "save_last": self.settings.save_last,
                **(self.settings.extra_ckpt_kwargs or {}),
            },
            {"name": "UpdateDatamoduleEpoch"},
            {"name": "LogLR"},
            {"name": "LogGradNorm"},
        ]

        # Allow passing already instantiated callbacks; e.g. for CVWorkflow
        cb_instantiated = []
        for cb in self.settings.pl_callback_kwargs:
            if isinstance(cb, dict):
                callbacks_config.append(cb)
            elif isinstance(cb, pl.Callback):
                cb_instantiated.append(cb)
            else:
                msg = f"[TrainWorkflow] Invalid callback type: {type(cb)}"
                logging.error(msg)
                raise ValueError(msg)

        callbacks = cast(
            list[pl.Callback],
            UnitFactory.get_pl_callback_instances_from_kwargs(
                callback_kwargs=callbacks_config
            ),
        )
        callbacks.extend(cb_instantiated)
        return callbacks

    def setup_trainer(self) -> pl.Trainer:
        """Returns trainer configuration from settings.

        Will populate self.trainer.

        Override for custom trainer configuration.
        """
        trainer_config = {
            "logger": self.wandb_logger,
            "callbacks": self.callbacks,
            "reload_dataloaders_every_n_epochs": 1,
            **(self.settings.pl_trainer_kwargs or {}),
        }
        return pl.Trainer(**trainer_config)

    def finalize_setup(self):
        """Finalizes the setup of the workflow.

        Override for custom finalization.
        """
        pass

    def __call__(self):
        """Runs the training workflow."""
        self.__setup()

        self.main_logger.info("\n[TrainWorkflow] TRAINING STARTED")
        self.main_logger.info(self.train_start_summary())

        start_time = time.time()
        train_stats = self.train_run()
        duration = time.time() - start_time

        if self.wandb_logger is not None:
            self.wandb_logger.experiment.unwatch(self.model)

        self.main_logger.info(
            f"\n[TrainWorkflow] TRAINING COMPLETED in {duration/60:.1f} min"
        )
        self.main_logger.info(self.train_end_summary(train_stats))

    def train_run(self) -> dict[str, Any]:
        """Runs the training workflow.

        Override for custom training run.
        """
        self.trainer.fit(
            self.model, datamodule=self.datamodule, ckpt_path=self.ckpt_path
        )

        train_stats = {
            "peak_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_gpu_mem": (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else None
            ),
            "best_val_loss": self.trainer.callback_metrics.get("val/loss_best"),
        }

        return train_stats

    def train_start_summary(self) -> str:
        """Formats the summary of the training workflow.

        Override for custom training start summary.
        """
        return (
            f"{'-'*60}"
            f"\n[TrainWorkflow] EXPERIMENT DETAILS\n"
            f"Experiment:       {self.settings.experiment}\n"
            f"Model:            {self.settings.pl_module_name}\n"
            f"Trainable Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}\n"
            f"Data:             {self.settings.pl_datamodule_name} @ {self.settings.data_dir}\n"
            f"Batch Size:       {self.settings.batch_size}\n"
            f"Epochs:           {self.trainer.max_epochs}\n"
            f"Optimizer:        {self.settings.pl_module_kwargs.get('optimizer_kwargs', 'unspecified')}\n"
            f"Loss:             {self.settings.pl_module_kwargs.get('loss_kwargs', 'unspecified')}\n"
            f"WandB Project:    {self.settings.project} (enabled: {self.settings.enable_wandb})\n"
            f"Output Dir:       {self.settings.exp_dir}\n"
            f"Device:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
            f"AMP enabled:      {self.trainer.precision == '16-mixed' or self.trainer.precision == 16}\n"
            f"{'-'*60}"
        )

    def train_end_summary(self, train_stats: dict[str, Any]) -> str:
        """Formats the summary of the training workflow.

        Override for custom training end summary.
        """
        peak_mem_mb = train_stats.get("peak_mem_mb")
        peak_gpu_mem = train_stats.get("peak_gpu_mem")
        best_val_loss = train_stats.get("best_val_loss")

        return (
            f"{'-'*60}"
            f"\n[TrainWorkflow] TRAINING SUMMARY\n"
            f"Peak Memory:      {peak_mem_mb:.02f} MB\n"
            f"Peak GPU Memory:  {peak_gpu_mem:.02f} MB\n"
            if peak_gpu_mem is not None
            else (
                "Peak GPU Memory:  N/A\n" f"Best Val Loss:    {best_val_loss:.04f}\n"
                if best_val_loss is not None
                else "Best Val Loss:    N/A\n" f"{'-'*60}"
            )
        )

    def close(self):
        """Close the workflow.

        Override for custom workflow closing.
        """
        torch.cuda.empty_cache()
        self.logging_manager.close()

    fit = __call__


@register(RK.WORKFLOW)
class CVWorkflow(Workflow):
    """Workflow for running cross-validation.

    Composes multiple `TrainWorkflow` instances, each running on a single split.
    Aggregates metrics from all splits.
    """

    def __init__(
        self, *, val_settings: dict[str, Any] | None = None, **train_workflow_kwargs
    ):
        """
        Args:
            val_settings: Validation settings, optionally containing:
                - `val_mode`: "single" or "repeated"
                - `n_splits`: Number of splits
                - `aggregate_metrics`: List of metric prefixes to aggregate
                - `run_test`: Whether to run trainer.test() on each split
            **train_workflow_kwargs: Keyword arguments for `TrainWorkflow`.
        """
        val_settings = val_settings or {}
        self.val_mode = val_settings.get("val_mode", "single")
        self.n_splits = val_settings.get("n_splits", 1)
        self.aggregate_metrics = val_settings.get("aggregate_metrics", "")
        self.run_test = val_settings.get("run_test", False)
        self.train_workflow_kwargs = train_workflow_kwargs
        self.start_idx = val_settings.get("start_idx", 0)
        self.seeds = val_settings.get("seeds")
        if self.seeds is not None:
            if not isinstance(self.seeds, list) or len(self.seeds) != self.n_splits:
                msg = (
                    f"[CVWorkflow] seeds must be a list of length {self.n_splits}; "
                    f"got {type(self.seeds)}"
                )
                logging.error(msg)
                raise ValueError(msg)

        self.aggregate_cb = UnitFactory.get_pl_callback_instances_from_kwargs(
            callback_kwargs=[
                {"name": "MetricAggregator", "prefix": self.aggregate_metrics}
            ]
        )[0]

        logging.info(
            f"[CVWorkflow] Initialized with the following settings: "
            f"val_mode: {self.val_mode}, n_splits: {self.n_splits}, "
            f"aggregate_metrics: {self.aggregate_metrics}, run_test: {self.run_test}"
        )

    def __call__(self):
        """Runs the cross-validation workflow."""
        for split_idx in range(self.start_idx, self.n_splits):
            logging.info(f"[CVWorkflow] Starting split {split_idx+1}/{self.n_splits}")
            self.run_split(split_idx)

    def run_split(self, split_idx: int) -> None:
        """Runs the training workflow for a single split."""
        workflow = TrainWorkflow(**deepcopy(self.train_workflow_kwargs))

        # Split-specific settings
        workflow.settings.experiment = (
            f"{workflow.settings.experiment}_split_{split_idx}"
        )
        workflow.settings.exp_dir = workflow.settings.exp_dir / f"split_{split_idx}"
        workflow.settings.val_mode = self.val_mode
        workflow.settings.n_splits = self.n_splits
        workflow.settings.current_split = split_idx
        workflow.settings.pl_callback_kwargs.append(self.aggregate_cb)  # type: ignore
        if self.seeds is not None:
            workflow.settings.seed = self.seeds[split_idx]

        # Run, test (optionally), and close
        workflow.fit()
        if self.run_test:
            workflow.trainer.test(workflow.model, datamodule=workflow.datamodule)

        if split_idx == self.n_splits - 1:
            self.aggregate_cb.final_summary()  # type: ignore[attr-defined]

        workflow.close()


@register(RK.WORKFLOW)
class TestWorkflow(Workflow):
    """Basic workflow for evaluating a model on a held-out test set."""

    def __init__(
        self,
        global_settings: dict[str, Any],
        *,
        pl_datamodule_settings: dict[str, Any],
        pl_module_settings: dict[str, Any],
        pl_callback_settings: list[dict[str, Any]],
        logging_settings: dict[str, Any] | None = None,
        **extra,
    ):
        logging_settings = logging_settings or {}

        to_be_overwritten = {
            "train_val_test_split",
            "val_mode",
            "n_splits",
            "current_split",
            "train_transforms",
            "val_transforms",
            "predict_transforms",
        }
        if any(k in to_be_overwritten for k in pl_module_settings.keys()):
            logging.warning(
                "[TestWorkflow] Identified arguments in `pl_module_settings` that are ignored by "
                f"TestWorkflow: {to_be_overwritten}."
            )

        self.settings = TrainingSettings(
            **unpack_settings_for_train_workflow(
                global_settings,
                logging_settings,
                pl_datamodule_settings,
                pl_module_settings,
                pl_callback_settings,
                {},
                {},
            )
        )

    def __setup(self):
        """Initializes the workflow components, including: Logging, Datamodule,
        Model, Callbacks, Trainer."""
        self.configure_environment()
        try:
            self.logging_manager, self.main_logger, self.wandb_logger = (
                self.setup_logging()
            )
        except ValueError:
            logging.exception(
                "[TestWorkflow] Error in setup_logging(); ensure returns match the docstring."
            )
            raise
        try:
            self.datamodule = self.setup_datamodule()
        except ValueError:
            self.main_logger.exception(
                "[TestWorkflow] Error in setup_datamodule(); ensure returns match the docstring."
            )
            raise
        try:
            self.model, self.ckpt_path = self.setup_model()
        except ValueError:
            self.main_logger.exception(
                "[TestWorkflow] Error in setup_model(); ensure returns match the docstring."
            )
            raise
        try:
            self.callbacks = self.setup_callbacks()
        except ValueError:
            self.main_logger.exception(
                "[TestWorkflow] Error in setup_callbacks(); ensure returns match the docstring."
            )
            raise
        try:
            self.trainer = self.setup_trainer()
        except ValueError:
            self.main_logger.exception(
                "[TestWorkflow] Error in setup_trainer(); ensure returns match the docstring."
            )
            raise
        self.finalize_setup()

        if (
            self.wandb_logger is not None
            and self.settings.wandb_watch_enable
            and getattr(self.trainer, "is_global_zero", True)
        ):
            self.wandb_logger.watch(
                self.model, **(self.settings.wandb_watch_kwargs or {})
            )

    def configure_environment(self):
        """Configures the environment for the workflow.

        Override for custom environment configuration.
        """
        set_determinism(seed=self.settings.seed)
        torch.set_float32_matmul_precision("high")  # for H100
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_logging(
        self,
    ) -> tuple[LoggingManager, logging.Logger, WandbLogger | None]:
        """Returns logging configuration from self.settings.

        Will populate self.logging_manager, self.main_logger and
        self.wandb_logger.

        Override for custom logging configuration.
        """
        if self.settings.extra_logging_kwargs:
            logging.warning(
                "[TestWorkflow] Identified extra_logging_kwargs; out-of-the-box "
                "TrainWorkflow cannot ensure they don't override explicitly "
                "set kwargs; unless custom logging manager is used."
            )

        logging_config = {
            "name": "DefaultLoggingManager",
            "logs_root": self.settings.exp_dir / "logs" / "test",
            "worker_logs": self.settings.worker_logs,
            "dev_mode": self.settings.dev_mode,
            "save_logs": self.settings.save_logs,
            "enable_wandb": self.settings.enable_wandb,
            "wandb_project": self.settings.project,
            "wandb_entity": self.settings.wandb_entity,
            "experiment": self.settings.experiment,
            "num_workers": self.settings.num_workers,
            **(self.settings.extra_logging_kwargs or {}),
        }
        logging_manager = ModuleFactory.get_logging_manager_instance_from_kwargs(
            logging_manager_kwargs=logging_config
        )
        main_logger, wandb_logger = (
            logging_manager.main_logger,
            logging_manager.wandb_logger,
        )
        return logging_manager, main_logger, wandb_logger

    def setup_datamodule(self) -> pl.LightningDataModule:
        """Returns datamodule configuration from self.settings.

        Will populate self.datamodule.

        Override for custom datamodule configuration.
        """
        non_extra_kwargs = {
            "name",
            "data_dir",
            "data_handler_kwargs",
            "batch_size",
            "num_workers",
            "extra_dataloader_kwargs",
            "test_transforms",
            "worker_logging_fn",
            "worker_seeding_fn",
            "seed",
        }
        if any(
            k in non_extra_kwargs
            for k in self.settings.extra_pl_datamodule_kwargs.keys()
        ):
            logging.warning(
                f"[TestWorkflow] Identified arguments in `extra_pl_datamodule_kwargs` that were "
                f" already set in the settings and will be overwritten: {non_extra_kwargs}."
            )

        datamodule_config = {
            "name": self.settings.pl_datamodule_name,
            "data_dir": self.settings.data_dir,
            "data_handler_kwargs": self.settings.data_handler_kwargs,
            "batch_size": self.settings.batch_size,
            "num_workers": self.settings.num_workers,
            "extra_dataloader_kwargs": self.settings.extra_dataloader_kwargs,
            "train_val_test_split": (0.0, 0.0, 1.0),
            "test_transforms": self.settings.test_transforms,
            "worker_logging_fn": self.logging_manager.get_setup_worker_logging_fn(),
            "worker_seeding_fn": set_rnd,
            "seed": self.settings.seed,
            **(self.settings.extra_pl_datamodule_kwargs or {}),
        }
        return ModuleFactory.get_pl_datamodule_instance_from_kwargs(
            pl_datamodule_kwargs=datamodule_config
        )

    def setup_model(self) -> tuple[pl.LightningModule, Path | None]:
        """Creates a new model instance from settings.

        If not `new_version` of the experiment or model checkpoint is provided,
        it attempts to find a checkpoint in the experiment directory. If not
        found, `ckpt_path` is set to None. Returns model instance and `ckpt_path`.

        Will populate self.model and self.ckpt_path.

        Override for custom model configuration.
        """
        model = ModuleFactory.get_pl_module_instance_from_kwargs(
            pl_module_kwargs=self.settings.pl_module_kwargs
        )
        if self.settings.new_version:
            return model, None

        ckpt_path = (
            self.settings.model_checkpoint
            or self.settings.exp_dir / "checkpoints" / "last.ckpt"
        )

        if not ckpt_path.exists():
            self.main_logger.warning(
                f"[TrainWorkflow] Checkpoint file {ckpt_path} does not exist; "
                "will create new model."
            )
            return model, None

        return model, ckpt_path

    def setup_callbacks(self) -> list[pl.Callback]:
        """Returns callbacks configuration from settings.

        Will populate self.callbacks.

        Override for custom callbacks configuration.
        """
        callbacks_config = [
            {
                "name": "MetricAggregator",
                "prefix": "test",
            }
        ]

        # Allow passing already instantiated callbacks; e.g. for CVWorkflow
        cb_instantiated = []
        for cb in self.settings.pl_callback_kwargs:
            if isinstance(cb, dict):
                callbacks_config.append(cb)
            elif isinstance(cb, pl.Callback):
                cb_instantiated.append(cb)
            else:
                msg = f"[TestWorkflow] Invalid callback type: {type(cb)}"
                logging.error(msg)
                raise ValueError(msg)

        callbacks = cast(
            list[pl.Callback],
            UnitFactory.get_pl_callback_instances_from_kwargs(
                callback_kwargs=callbacks_config
            ),
        )
        callbacks.extend(cb_instantiated)
        return callbacks

    def setup_trainer(self) -> pl.Trainer:
        """Returns a new trainer instance; will be only used for testing via
        `test()` method."""
        return pl.Trainer()

    def finalize_setup(self):
        """Finalizes the setup of the workflow.

        Override for custom finalization.
        """
        pass

    def __call__(self):
        """Runs the testing workflow."""
        self.__setup()

        self.main_logger.info("\n[TestWorkflow] TESTING STARTED")

        start_time = time.time()
        test_stats = self.test_run()
        duration = time.time() - start_time

        if self.wandb_logger is not None:
            self.wandb_logger.experiment.unwatch(self.model)

        self.main_logger.info(
            f"\n[TestWorkflow] TESTING COMPLETED in {duration/60:.1f} min"
        )
        self.main_logger.info(self.test_end_summary(test_stats))

    def test_run(self) -> dict[str, Any]:
        """Runs the testing workflow.

        Override for custom testing run.
        """
        self.trainer.test(self.model, datamodule=self.datamodule)

        test_stats = {
            "peak_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_gpu_mem": (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else None
            ),
        }

        for cb in self.callbacks:
            if isinstance(cb, MetricAggregator):
                cb.final_summary()

        return test_stats

    def test_start_summary(self) -> str:
        """Formats the summary of the training workflow.

        Override for custom training start summary.
        """
        return (
            f"{'-'*60}"
            f"\n[TestWorkflow] EXPERIMENT DETAILS\n"
            f"Experiment:       {self.settings.experiment}\n"
            f"Model:            {self.settings.pl_module_name}\n"
            f"Trainable Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}\n"
            f"Data:             {self.settings.pl_datamodule_name} @ {self.settings.data_dir}\n"
            f"Batch Size:       {self.settings.batch_size}\n"
            f"Metrics:          {getattr(self.model, 'metrics', 'unspecified')}\n"
            f"WandB Project:    {self.settings.project} (enabled: {self.settings.enable_wandb})\n"
            f"Output Dir:       {self.settings.exp_dir}\n"
            f"Device:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
            f"AMP enabled:      {self.trainer.precision == 'bf16' or self.trainer.precision == 16}\n"
            f"{'-'*60}"
        )

    def test_end_summary(self, test_stats: dict[str, Any]) -> str:
        """Formats the summary of the training workflow.

        Override for custom training end summary.
        """
        peak_mem_mb = test_stats.get("peak_mem_mb")
        peak_gpu_mem = test_stats.get("peak_gpu_mem")

        return (
            f"{'-'*60}"
            f"\n[TestWorkflow] TESTING SUMMARY\n"
            f"Peak Memory:      {peak_mem_mb:.02f} MB\n"
            f"Peak GPU Memory:  {peak_gpu_mem:.02f} MB\n"
            if peak_gpu_mem is not None
            else "Peak GPU Memory:  N/A\n" f"{'-'*60}"
        )

    def close(self):
        """Close the workflow.

        Override for custom workflow closing.
        """
        torch.cuda.empty_cache()
        self.logging_manager.close()

    test = __call__


@register(RK.WORKFLOW)
class CompositeWorkflow(Workflow):
    """Composes multiple workflows from a single config file."""

    def __init__(self, **config):
        wfs = [(k, v) for k, v in config.items() if k.endswith("Workflow")]
        if not wfs:
            msg = "[CompositeWorkflow] No workflows found in config"
            logging.error(msg)
            raise ValueError(msg)
        self.workflows = [
            get(RK.WORKFLOW, wf_name)(**wf_kwargs) for wf_name, wf_kwargs in wfs
        ]

    def __call__(self):
        for workflow in self.workflows:
            workflow()


@register(RK.WORKFLOW)
class SweepWorkflow(Workflow):
    """Basic workflow for sweeping over a list of settings.

    Loop over different values of a hyperparameter, at multiple levels.
    See example below and explanation:

    Example:
    ```
    sweep_settings:
        1:  # outer loop (level 1)
            - TrainWorkflow.pl_module_settings.model_kwargs.use_v2
        2:  # inner loop (level 2)
            - TrainWorkflow.pl_datamodule_settings.train_transforms.mask_size
            - TrainWorkflow.pl_datamodule_settings.train_transforms.other_param
    ```

    The above tells the workflow to:
    1. Read `TrainWorkflow.pl_datamodule_settings.train_transforms.mask_size` and
       `TrainWorkflow.pl_module_settings.model_kwargs.use_v2` settings as lists of
        hyperparameters to loop over.
    2. Loop over the above settings at two different levels: for each of the `v2` values (level `1`),
       loop over the `mask_size` values (level `2`).

    Notes:
        - This workflow wrapper is agnostic to the type of the individual workflows.
          A `CompositeWorkflow` is used to compose an arbitrary number and type of workflows. Users
          are expected to pass a dictionary of workflow names and their corresponding keyword arguments.

          See `CompositeWorkflow` documentation for more details.

        - Sweeping settings at the same level vary together and are hence expected to match in length.
        - Sweeping settings are specified as dotted paths to the workflow keyword arguments;
          e.g. `TrainWorkflow.pl_datamodule_settings.train_transforms.mask_size` in the example above.

    Future work (TODO):
        - Avoid repetitions `*Workflow` or other prefixing in the dotted paths across a given level.
        - Re-use experiment names and logging settings (+ save dirs) for different sweeps (applies
          also to `CVWorkflow` and `CompositeWorkflow`).
    """

    def __init__(self, *, sweep_settings: dict[int, list[str]], **workflow_kwargs):
        """
        Args:
            sweep_settings: List of dictionaries, each containing the settings and levels to sweep over.
            **workflows: Keyword arguments for the workflows to compose.
        """
        if not isinstance(sweep_settings, dict):
            msg = f"[SweepWorkflow] `sweep_settings` must be a dictionary, got {type(sweep_settings)}"
            logging.error(msg)
            raise TypeError(msg)

        self.sweep_settings = sweep_settings
        self.workflow_kwargs = workflow_kwargs
        self.iterables = self._parse_sweep_settings()

    def _parse_sweep_settings(self) -> dict[int, list[dict[str, Any]]]:
        """Parse the sweep settings into a dictionary of levels with
        corresponding setting names and values.

        The returned dictionary will have the following structure:
        ```
        {
            1: [
                {'path': 'setting_1', 'values': [value_1, value_2, ...]},
                {'path': 'setting_2', 'values': [value_1, value_2, ...]},
                ...
            ],
            2:  ...
            ...
        }
        ```
        """
        iterables: dict[int, list[dict[str, Any]]] = {}

        for level, paths in self.sweep_settings.items():
            level = int(level)  # YAML may parse as string
            iterables[level] = []

            for path in paths:
                values = self._get_nested_value(self.workflow_kwargs, path)

                if not isinstance(values, (list, tuple)):
                    msg = f"[SweepWorkflow] Setting at '{path}' must be a list for sweeping, got {type(values)}"
                    logging.error(msg)
                    raise ValueError(msg)

                iterables[level].append(
                    {
                        "path": path,
                        "values": list(values),
                    }
                )

            # Validate all paths at this level have same length
            lengths = [len(it["values"]) for it in iterables[level]]
            if len(set(lengths)) > 1:
                path_lengths = {
                    it["path"]: len(it["values"]) for it in iterables[level]
                }
                msg = f"[SweepWorkflow] Settings at level {level} have mismatched lengths: {path_lengths}"
                logging.error(msg)
                raise ValueError(msg)

        return iterables

    @staticmethod
    def _get_nested_value(d: dict, path: str) -> Any:
        """Get value from nested dict using dotted path."""
        for key in path.split("."):
            d = d[key]
        return d

    @staticmethod
    def _set_nested_value(d: dict, path: str, value: Any) -> None:
        """Set value in nested dict using dotted path."""
        keys = path.split(".")
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value

    def _generate_combinations(self) -> Iterator[tuple[dict[str, Any], str]]:
        """Generate all (kwargs, experiment_suffix) combinations."""
        levels = sorted(self.iterables.keys())

        # Build index ranges per level ([range(n_values_lvl1), range(n_values_lvl2), ...])
        level_ranges = []
        for level in levels:
            n_values = len(self.iterables[level][0]["values"])
            level_ranges.append(range(n_values))

        # Cartesian product of all level indices
        for indices in product(*level_ranges):  # e.g., 2 levels: (0, 1, 2) x (0, 1)
            kwargs = deepcopy(self.workflow_kwargs)
            suffix_parts = []

            for level_idx, value_idx in enumerate(
                indices
            ):  # ordered tuple of value indices for each level
                level = levels[
                    level_idx
                ]  # list of dicts with setting names and values for given level
                for setting in self.iterables[level]:
                    value = setting["values"][value_idx]
                    self._set_nested_value(kwargs, setting["path"], value)

                    # Build readable suffix (use last part of path)
                    short_name = setting["path"].split(".")[-1]
                    suffix_parts.append(f"{short_name}={value}")

            suffix = "_".join(suffix_parts)
            yield kwargs, suffix

    def __call__(self) -> None:
        """Run a `CompositeWorkflow` for each combination of settings."""
        combinations = list(self._generate_combinations())
        n_total = len(combinations)

        logging.info(f"[SweepWorkflow] Running {n_total} combinations")

        for i, (kwargs, suffix) in enumerate(combinations):
            logging.info(f"[SweepWorkflow] [{i+1}/{n_total}] Running: {suffix}")
            CompositeWorkflow(**kwargs)()

        logging.info(f"[SweepWorkflow] Completed all {n_total} combinations")
