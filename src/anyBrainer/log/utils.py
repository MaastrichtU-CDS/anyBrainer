"""
Logging utility functions
"""

__all__ = [
    'setup_worker_logging',
    'get_safe_logger',
    'WandbFilter',
    'WandbOnlyHandler',
]

from pathlib import Path
import logging
from logging.handlers import QueueHandler
from multiprocessing import Queue
from threading import Lock
import time

import wandb
from lightning.pytorch.loggers import WandbLogger

from anyBrainer.utils.io import resolve_path

logger = logging.getLogger(__name__)


def setup_worker_logging(
        log_queue: Queue, 
        *,
        dev_mode: bool = False, 
        worker_logs: bool = True,
) -> None:
        """Call this inside the worker process."""
        if not log_queue:
            raise RuntimeError("Log queue not initialized. Call setup_parallel_logging in the main process.")
        if dev_mode: 
            level = logging.DEBUG
        elif worker_logs:
            level = logging.INFO
        else: 
            level = logging.WARNING
        
        root_logger = logging.getLogger("anyBrainer")
        root_logger.setLevel(level)
        root_logger.handlers = []
        root_logger.addHandler(QueueHandler(log_queue))

def get_safe_logger(
        name: str = "", 
        log_file: str | Path | None = None, 
        level=logging.INFO
) -> logging.Logger:
    """
    Returns a logger with default configuration if not already configured.
    If `name` is empty, root logger is returned (for global consistency).
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            log_path = resolve_path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

def init_wandb_logger(
        project: str,
        name: str,
        dir: str | Path,
) -> WandbLogger:
    """Initialize W&B and return logger."""
    wandb.init(
        project=project,
        name=name,
        dir=dir,
    )
    return WandbLogger(
        project=project,
        name=name,
        dir=dir,
    )


class WandbFilter(logging.Filter):
    """
    Filter to aggregate logs from multiple workers and sync them to W&B.

    Supports both sync and async modes.
    Sync mode: logs are aggregated and synced to W&B after a timeout or when the
    expected number of workers have logged.
    Async mode: logs are logged to W&B immediately.
    """
    def __init__(
        self,
        wandb_logger: WandbLogger | None = None,
        num_expected_sync: int = 0,
        sync_timeout: float = 5.0,
    ):
        super().__init__()
        self.wandb_logger = wandb_logger
        self.num_expected_sync = num_expected_sync
        self.sync_timeout = sync_timeout

        # Sync mode tracking
        self.aggregated_sync = {}
        self._sync_count = 0
        self._sync_first_time = None
        self._lock = Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if self.wandb_logger is None:
            return True

        if not hasattr(record, "wandb") or not isinstance(record.wandb, dict): # type: ignore
            return True

        data = dict(record.wandb) # type: ignore
        mode = data.pop("_wandb_mode", "async")

        if mode == "async":
            wandb.log(data)
            return True

        # Default: sync
        with self._lock:
            if self._sync_first_time is None:
                self._sync_first_time = time.time()

            self.aggregated_sync.update(data)
            self._sync_count += 1

            now = time.time()
            elapsed = now - self._sync_first_time

            if self._sync_count >= self.num_expected_sync:
                self._flush_sync_log()
            elif elapsed >= self.sync_timeout:
                logger.warning(f"[WandbFilter] Timeout after {elapsed:.1f}s — partial sync log to W&B")
                self._flush_sync_log()

        return True

    def _flush_sync_log(self):
        if self.aggregated_sync:
            wandb.log(self.aggregated_sync)

        self.aggregated_sync.clear()
        self._sync_count = 0
        self._sync_first_time = None
    

class WandbOnlyHandler(logging.Handler):
    """Handler that only logs to W&B."""
    def emit(self, record: logging.LogRecord) -> None:
        for filt in self.filters:
            filt.filter(record)