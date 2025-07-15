"""
Logging utility functions
"""

__all__ = [
    'setup_worker_logging',
    'get_safe_logger',
]

from pathlib import Path
import logging
from logging.handlers import QueueHandler
from multiprocessing import Queue

from anyBrainer.utils.utils import resolve_path

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
        root_logger = logging.getLogger()
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
    