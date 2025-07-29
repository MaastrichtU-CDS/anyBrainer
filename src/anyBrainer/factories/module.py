"""Factory to create module (PL module, PL datamodule, managers) instances."""

__all__ = [
    "ModuleFactory",
]

import logging
from typing import Any

import pytorch_lightning as pl

from anyBrainer.registry import get, RegistryKind as RK
from anyBrainer.log import LoggingManager

logger = logging.getLogger(__name__)


class ModuleFactory:
    @classmethod
    def get_pl_module_instance_from_kwargs(
        cls,
        pl_module_kwargs: dict[str, Any],
    ) -> pl.LightningModule:
        """
        Get pl module instance from anyBrainer.registry-PL_MODULE.

        Args:
            pl_module_kwargs: pl module kwargs, containing "name" key.

        Raises:
        - ValueError: If PL module name is not found in pl_module_kwargs.
        - ValueError: If requested PL module is not found in anyBrainer.registry-PL_MODULE.
        - Exception: If error occurs during PL module initialization.
        """
        if "name" not in pl_module_kwargs:
            msg = "PL module name not found in pl_module_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        pl_module_kwargs = pl_module_kwargs.copy()
        pl_module_name = pl_module_kwargs.pop("name")

        try:
            pl_module_cls = get(RK.PL_MODULE, pl_module_name)
        except AttributeError:
            msg = f"PL module '{pl_module_name}' not found in anyBrainer.registry-PL_MODULE."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            pl_module = pl_module_cls(**pl_module_kwargs) # type: ignore
        except Exception as e:
            msg = f"Error initializing PL module '{pl_module_name}': {e}"
            logger.exception(msg)
            raise
        
        return pl_module
    
    @classmethod
    def get_pl_datamodule_instance_from_kwargs(
        cls,
        pl_datamodule_kwargs: dict[str, Any],
    ) -> pl.LightningModule:
        """
        Get pl module instance from anyBrainer.registry-DATAMODULE.

        Args:
            pl_datamodule_kwargs: pl datamodule kwargs, containing "name" key.

        Raises:
        - ValueError: If PL datamodule name is not found in pl_datamodule_kwargs.
        - ValueError: If requested PL datamodule is not found in anyBrainer.registry-DATAMODULE.
        - Exception: If error occurs during PL datamodule initialization.
        """
        if "name" not in pl_datamodule_kwargs:
            msg = "PL datamodule name not found in pl_datamodule_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        pl_datamodule_kwargs = pl_datamodule_kwargs.copy()
        pl_datamodule_name = pl_datamodule_kwargs.pop("name")

        try:
            pl_datamodule_cls = get(RK.DATAMODULE, pl_datamodule_name)
        except AttributeError:
            msg = f"PL datamodule '{pl_datamodule_name}' not found in anyBrainer.registry-DATAMODULE."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            pl_datamodule = pl_datamodule_cls(**pl_datamodule_kwargs) # type: ignore
        except Exception as e:
            msg = f"Error initializing PL datamodule '{pl_datamodule_name}': {e}"
            logger.exception(msg)
            raise
        
        return pl_datamodule
    
    @classmethod
    def get_logging_manager_instance_from_kwargs(
        cls,
        logging_manager_kwargs: dict[str, Any],
    ) -> LoggingManager:
        """
        Get logging manager instance from anyBrainer.registry-LOGGING_MANAGER.

        Args:
            logging_manager_kwargs: logging manager kwargs, containing "name" key.

        Raises:
        - ValueError: If logging manager name is not found in logging_manager_kwargs.
        - ValueError: If requested logging manager is not found in anyBrainer.registry-LOGGING_MANAGER.
        - Exception: If error occurs during logging manager initialization.
        """
        if "name" not in logging_manager_kwargs:
            msg = "Logging manager name not found in logging_manager_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        logging_manager_kwargs = logging_manager_kwargs.copy()
        logging_manager_name = logging_manager_kwargs.pop("name")

        try:
            logging_manager_cls = get(RK.LOGGING_MANAGER, logging_manager_name)
        except AttributeError:
            msg = f"Logging manager '{logging_manager_name}' not found in anyBrainer.registry-LOGGING_MANAGER."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            logging_manager = logging_manager_cls(**logging_manager_kwargs) # type: ignore
        except Exception as e:
            msg = f"Error initializing logging manager '{logging_manager_name}': {e}"
            logger.exception(msg)
            raise
        
        return logging_manager