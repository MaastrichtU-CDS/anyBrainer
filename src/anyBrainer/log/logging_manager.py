"""
Handles all logging logic for main and parallel processes.
"""

__all__ = [
    'LoggingManager',
]

from pathlib import Path
import logging
from logging.handlers import QueueListener
from multiprocessing import Queue
from dataclasses import dataclass
from functools import partial
import wandb

from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.utils.io import resolve_path
from anyBrainer.log.utils import setup_worker_logging, init_wandb_logger
from anyBrainer.log.utils import WandbFilter, WandbOnlyHandler


@dataclass
class LoggingSettings:
    logs_root: Path
    worker_logs: bool
    dev_mode: bool
    enable_wandb: bool
    experiment: str
    num_workers: int
    sync_timeout: float = 5.0
    wandb_project: str = "anyBrainer"
    save_logs: bool = True

    def __post_init__(self):
        self.logs_root = resolve_path(self.logs_root)


@register(RK.LOGGING_MANAGER)
class LoggingManager:
    """Handles all logging logic for main and parallel processes."""
    def __init__(self, **settings):
        self.settings = LoggingSettings(**settings)
        if self.settings.enable_wandb:
            self.wandb_logger = init_wandb_logger(
                project=self.settings.wandb_project,
                name=self.settings.experiment,
                dir=self.settings.logs_root
            )
        else:
            self.wandb_logger = None
        
        self.main_logger = self.setup_main_logging()
        self.log_queue, self.listener = self.setup_parallel_logging()

        self.main_logger.info(f"Logging manager initialized with settings: {self.settings}")
    
    def setup_main_logging(self) -> logging.Logger:
        """Sets up the main process logging handlers and loggers."""
        # Get the anyBrainer logger
        main_logger = logging.getLogger("anyBrainer")
        
        # Set the logging level
        level = logging.DEBUG if self.settings.dev_mode else logging.INFO
        main_logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicates
        main_logger.handlers.clear()
        
        # Create handlers
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        
        if self.settings.save_logs:
            self.settings.logs_root.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.settings.logs_root / "main.log")) # type: ignore
        
        # Add custom Wandb handler to the handlers list
        self._add_wandb_handler(handlers)
        
        # Create formatter
        base_format = "%(asctime)s | %(levelname)s | "
        format_tail = ("%(processName)s | %(message)s" 
                       if self.settings.dev_mode else "%(message)s")
        log_format = base_format + format_tail
        
        formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")
        
        # Apply formatter to handlers and add them to the logger
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(level)
            main_logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate logs
        main_logger.propagate = False

        return main_logger

    def setup_parallel_logging(self) -> tuple[Queue, QueueListener]:
        """Sets up multiprocessing-compatible logging using a Queue."""
        log_queue = Queue()

        handlers: list[logging.Handler] = [logging.StreamHandler()]

        if self.settings.save_logs:
            handlers.append(logging.FileHandler(self.settings.logs_root / "workers.log")) # type: ignore

        stream_fmt = (
            "%(asctime)s | %(levelname)s | %(processName)s | "
            "%(message)s"
        )

        for handler in handlers:
            handler.setFormatter(
                logging.Formatter(stream_fmt, "%Y-%m-%d %H:%M:%S")
            )
        
        # Add custom Wandb handler to the handlers list
        self._add_wandb_handler(handlers)

        listener = QueueListener(log_queue, *handlers)
        listener.start()

        return log_queue, listener
    
    def _add_wandb_handler(self, handlers: list[logging.Handler]) -> None:
        """Adds WandbOnlyHandler in-place to the handlers list."""
        wandb_handler = WandbOnlyHandler()
        wandb_filter = WandbFilter(
            wandb_logger=self.wandb_logger,
            num_expected_sync=self.settings.num_workers,
            sync_timeout=self.settings.sync_timeout
        )
        wandb_handler.addFilter(wandb_filter)
        handlers.append(wandb_handler)
    
    def setup_worker_logging(self):
        """Sets up worker logger"""
        setup_worker_logging(
            log_queue=self.log_queue, # type: ignore
            dev_mode=self.settings.dev_mode,
            worker_logs=self.settings.worker_logs
        )
    
    def get_setup_worker_logging_fn(self): 
        """Gets pickable function to setup logger during worker init"""
        return partial(setup_worker_logging, 
                       log_queue=self.log_queue, # type: ignore
                       dev_mode=self.settings.dev_mode,
                       worker_logs=self.settings.worker_logs)

    def stop_parallel_logging(self):
        """Stops the parallel logging listener."""
        self.listener.stop()
    
    def close(self):
        """Closes the logging manager."""
        self.stop_parallel_logging()
        if self.settings.enable_wandb:
            wandb.finish()