"""
Configuration manager for GestureBind

Handles loading, saving, and managing application configurations.
"""

import os
import yaml
import logging
import json
from pathlib import Path

# Import path_manager functions for consistent path handling
try:
    from utils.path_manager import get_workspace_root, resolve_path, get_data_dir
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported path_manager for ConfigManager")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Couldn't import path_manager, using default paths")
    # Basic default implementation if path_manager is not available
    def get_workspace_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    def resolve_path(path, create_if_missing=False):
        if os.path.isabs(path):
            return path
        return os.path.join(get_workspace_root(), path)
    
    def get_data_dir(subdir=None, create=True):
        workspace_root = get_workspace_root()
        if subdir:
            path = os.path.join(workspace_root, "gesturebind", "data", subdir)
        else:
            path = os.path.join(workspace_root, "gesturebind", "data")
        
        if create and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

class ConfigManager:
    """
    Manages configuration settings for the application.
    """
    
    def __init__(self, config_path):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the configuration file
        """
        # Use path_manager's resolve_path to ensure consistent path resolution
        self.config_path = resolve_path(config_path)
        self.config = {}
        
        # Use a path within the workspace for user config instead of ~/.gesturebind
        self.user_config_dir = Path(get_data_dir("config", create=True))
        self.user_config_path = self.user_config_dir / "user_config.yaml"
        
        # Ensure user config directory exists
        self._ensure_user_config_dir()
    
    def load_config(self):
        """
        Load configuration from file.
        
        Returns:
            dict: Loaded configuration
        """
        try:
            # Load default configuration
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            logger.info(f"Loaded default configuration from {self.config_path}")
            
            # Load user configuration if it exists
            if self.user_config_path.exists():
                with open(self.user_config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                
                # Merge user config into default config
                self._deep_merge(self.config, user_config)
                logger.info(f"Merged user configuration from {self.user_config_path}")
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # If loading fails, use default empty config
            self.config = {}
        
        return self.config
    
    def save_config(self):
        """
        Save current configuration to user config file.
        
        Returns:
            bool: Success or failure
        """
        try:
            # Ensure user config directory exists
            self._ensure_user_config_dir()
            
            # Save user configuration
            with open(self.user_config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
            logger.info(f"Saved configuration to {self.user_config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def update_config(self, key_path, value):
        """
        Update a specific configuration value using a dot-separated path.
        
        Args:
            key_path (str): Dot-separated path to config value (e.g. "ui.theme")
            value: New value to set
            
        Returns:
            bool: Success or failure
        """
        try:
            keys = key_path.split('.')
            current = self.config
            
            # Navigate to the parent object of the target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
            logger.debug(f"Updated config: {key_path} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def get_value(self, key_path, default=None):
        """
        Get a configuration value using a dot-separated path.
        
        Args:
            key_path (str): Dot-separated path to config value (e.g. "ui.theme")
            default: Default value if key not found
            
        Returns:
            Value at the specified path or default
        """
        try:
            keys = key_path.split('.')
            current = self.config
            
            # Navigate through the keys
            for key in keys:
                if key not in current:
                    return default
                current = current[key]
            
            return current
            
        except Exception:
            return default
    
    def reset_to_default(self):
        """
        Reset configuration to default values.
        
        Returns:
            dict: Default configuration
        """
        try:
            # Load default configuration
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            logger.info("Reset configuration to defaults")
            return self.config
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return {}
    
    def export_config(self, export_path):
        """
        Export configuration to a file.
        
        Args:
            export_path (str): Path to export the configuration to
            
        Returns:
            bool: Success or failure
        """
        try:
            with open(export_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
            logger.info(f"Exported configuration to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, import_path):
        """
        Import configuration from a file.
        
        Args:
            import_path (str): Path to import the configuration from
            
        Returns:
            bool: Success or failure
        """
        try:
            with open(import_path, 'r') as f:
                imported_config = yaml.safe_load(f)
            
            # Validate configuration (simple check for required sections)
            required_sections = ["camera", "detection", "ui", "actions", "profiles"]
            for section in required_sections:
                if section not in imported_config:
                    logger.error(f"Imported configuration is missing required section: {section}")
                    return False
            
            # Update current configuration
            self.config = imported_config
            
            logger.info(f"Imported configuration from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
    
    def _ensure_user_config_dir(self):
        """Ensure user config directory exists"""
        if not self.user_config_dir.exists():
            self.user_config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created user config directory: {self.user_config_dir}")
    
    def _deep_merge(self, target, source):
        """
        Deep merge two dictionaries.
        
        Args:
            target (dict): Target dictionary to merge into
            source (dict): Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

def create_default_config(config_path):
    """
    Create a default configuration file at the specified path.
    
    Args:
        config_path (str): Path to create the default configuration file
    
    Returns:
        bool: Success or failure
    """
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Default configuration structure
        default_config = {
            "app": {
                "version": "1.0.0",
                "save_user_data_path": "~/gesturebind/data"
            },
            "camera": {
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "detection": {
                "engine": "mediapipe", # Options: mediapipe, yolov8
                "confidence_threshold": 0.7,
                "cooldown_ms": 500,
                "min_detection_duration_ms": 200
            },
            "ui": {
                "theme": "system",  # Options: system, light, dark
                "minimize_to_tray": True,
                "start_minimized": False,
                "overlay_feedback": True, 
                "overlay_timeout_ms": 1500,
                "show_preview": True,
                "show_fps": True
            },
            "actions": {
                "default_profile": "default",
                "enable_shortcuts": True
            },
            "profiles": {
                "default": {
                    "open_palm": "keyboard:space",
                    "fist": "keyboard:escape",
                    "pointing": "mouse:mouse_click",
                    "peace_sign": "media:play_pause"
                },
                "media": {
                    "open_palm": "media:play_pause",
                    "fist": "media:mute",
                    "thumbs_up": "media:volume_up",
                    "thumbs_down": "media:volume_down"
                },
                "presentation": {
                    "pointing_right": "keyboard:right",
                    "pointing_left": "keyboard:left",
                    "open_palm": "keyboard:f5"
                }
            }
        }
        
        # Write configuration to file
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
        
        logger.info(f"Created default configuration at {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating default configuration: {e}")
        return False