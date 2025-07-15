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

from anyBrainer.utils.utils import resolve_path
from anyBrainer.log.utils import setup_worker_logging

@dataclass
class LoggingSettings:
    logs_root: Path | None
    worker_logs: bool
    dev_mode: bool


class LoggingManager:
    def __init__(self, settings: dict):
        self.settings = LoggingSettings(**settings)
        self.settings.logs_root = (
            resolve_path(self.settings.logs_root)
            if self.settings.logs_root else None
        )
        self.log_queue = None
        self.listener = None
        self.main_logger = None

    def setup_main_logging(self):
        """Sets up the main process logging handlers and loggers."""
        # Get the anyBrainer logger
        self.main_logger = logging.getLogger("anyBrainer")
        
        # Set the logging level
        level = logging.DEBUG if self.settings.dev_mode else logging.INFO
        self.main_logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicates
        self.main_logger.handlers.clear()
        
        # Create handlers
        handlers = [logging.StreamHandler()]
        
        if self.settings.logs_root:
            self.settings.logs_root.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.settings.logs_root / "main.log")) # type: ignore
        
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
            self.main_logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate logs
        self.main_logger.propagate = False

    def setup_parallel_logging(self):
        """Sets up multiprocessing-compatible logging using a Queue."""
        self.log_queue = Queue()

        handlers = [logging.StreamHandler()]
        if self.settings.logs_root:
            handlers.append(logging.FileHandler(self.settings.logs_root / "workers.log")) # type: ignore

        stream_fmt = (
            "%(asctime)s | %(levelname)s | %(processName)s | "
            "%(message)s"
        )

        for handler in handlers:
            handler.setFormatter(
                logging.Formatter(stream_fmt, "%Y-%m-%d %H:%M:%S")
            )

        self.listener = QueueListener(self.log_queue, *handlers)
        self.listener.start()
    
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
        if self.listener:
            self.listener.stop()
