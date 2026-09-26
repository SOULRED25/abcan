"""
abCAN-v2: Antibody-Antigen ΔΔG Prediction
Configuration module.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ─────────────────────────────────────────────
# Project root detection
# ─────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CONFIG_DIR.parent

DEFAULT_CONFIG_PATH = _CONFIG_DIR / "default_config.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML config. Defaults to config/default_config.yaml.
        
    Returns:
        Configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two configuration dictionaries (override takes precedence).
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load config and apply optional overrides.
    
    Args:
        config_path: Path to base config YAML.
        overrides: Dictionary of overrides to apply on top.
        
    Returns:
        Final merged configuration dictionary.
    """
    config = load_config(config_path)
    if overrides:
        config = merge_configs(config, overrides)
    return config


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
