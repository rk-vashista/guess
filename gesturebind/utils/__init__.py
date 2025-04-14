"""
Utility functions for GestureBind application
"""

import os
import sys
import logging

# Fix imports by adding the project root to sys.path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use only relative imports for internal modules
try:
    # Try relative import first
    from .path_manager import (
        get_workspace_root,
        resolve_path,
        get_data_dir,
        get_gesture_data_path,
        get_gesture_data_file,
        ensure_workspace_paths
    )
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Error importing path_manager: {e}")
    
    # Define fallback minimal implementations if needed
    def get_workspace_root():
        """Get the absolute path to the workspace root directory"""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    def resolve_path(path, create_if_missing=False):
        """Basic path resolver"""
        return path
    
    def get_data_dir(subdir=None, create=True):
        """Get data directory"""
        root = get_workspace_root()
        if subdir:
            return os.path.join(root, "data", subdir)
        return os.path.join(root, "data")
    
    def get_gesture_data_path():
        """Get gesture data path"""
        return get_data_dir("gestures", True)
    
    def get_gesture_data_file():
        """Get gesture data file path"""
        return os.path.join(get_data_dir("default", True), "gesture_data.json")
    
    def ensure_workspace_paths():
        """Ensure all workspace paths exist"""
        get_data_dir(create=True)
        get_data_dir("gestures", create=True)
        get_data_dir("models", create=True)
        get_data_dir("default", create=True)

from .config_manager import ConfigManager
from .logger import setup_logger, set_log_level, get_recent_logs

__all__ = [
    'ConfigManager',
    'setup_logger',
    'set_log_level',
    'get_recent_logs',
    'get_workspace_root',
    'resolve_path',
    'get_data_dir',
    'get_gesture_data_path',
    'get_gesture_data_file',
    'ensure_workspace_paths'
]