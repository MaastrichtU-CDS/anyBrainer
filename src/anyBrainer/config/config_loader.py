"""
Configuration loading utilities.
"""

__all__ = [
    "load_config",
    "merge_configs",
    "validate_config",
]

from asyncio.proactor_events import streams
import logging
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path | str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix not in ['.yaml', '.yml']:
        logger.error(f"Configuration file must be YAML format, got: {config_path.suffix}")
        raise ValueError(f"Configuration file must be YAML format, got: {config_path.suffix}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries recursively.
    Override config takes precedence over base config.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration keys are present.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ['mode', 'patch_size']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Required configuration key missing: {key}")
            raise ValueError(f"Required configuration key missing: {key}")
    
    # Validate mode
    valid_modes = ['masked_autoencoder', 'contrastive', 'supervised']
    if config['mode'] not in valid_modes:
        logger.error(f"Invalid mode: {config['mode']}. Must be one of {valid_modes}")
        raise ValueError(f"Invalid mode: {config['mode']}. Must be one of {valid_modes}")
    
    # Validate patch_size
    if not isinstance(config['patch_size'], (list, tuple)) or len(config['patch_size']) != 3:
        logger.error("patch_size must be a list or tuple of 3 integers")
        raise ValueError("patch_size must be a list or tuple of 3 integers")
    
    logger.debug("Configuration validated successfully")