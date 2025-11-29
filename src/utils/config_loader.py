"""Configuration loader utility"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Load and manage configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str = "config.yaml") -> Dict[str, Any]:
        """Load a configuration file
        
        Args:
            config_name: Name of the config file
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._configs[config_name] = config
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key
        
        Args:
            key: Dot-separated key path (e.g., 'data.raw_data_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._configs.get('config.yaml', {})
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files in the config directory
        
        Returns:
            Dictionary of all configurations
        """
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.name
            self._configs[config_name] = self.load_config(config_name)
        
        return self._configs
    
    @property
    def configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations"""
        return self._configs


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Quick function to load a configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

