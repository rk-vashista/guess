"""
Path manager for GestureBind application

Provides consistent path resolution across the application to ensure
all data is stored within the workspace directory.
"""

import os
import logging
import sys

# Ensure the module can be imported from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logger = logging.getLogger(__name__)

# Get the absolute path to the workspace root directory
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Make sure we use the correct data path in the current workspace, not another location
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "gesturebind", "data")

# Log the actual paths being used
logger.info(f"Workspace root: {WORKSPACE_ROOT}")
logger.info(f"Data root: {DATA_ROOT}")

def get_workspace_root():
    """Get the absolute path to the workspace root directory"""
    return WORKSPACE_ROOT

def resolve_path(path, create_if_missing=False):
    """
    Resolve a path to be relative to the workspace root.
    If the path is already absolute and within the workspace, it's returned as is.
    
    Args:
        path: Path to resolve (can be absolute or relative)
        create_if_missing: Whether to create the directory if it doesn't exist
    
    Returns:
        Absolute path within the workspace
    """
    # Expand home directory if present
    if path.startswith("~"):
        path = os.path.expanduser(path)

    # If path starts with ./, it's workspace-relative
    if path.startswith("./"):
        abs_path = os.path.normpath(os.path.join(WORKSPACE_ROOT, path[2:]))
    # If path is absolute
    elif os.path.isabs(path):
        # Check if it's already within the workspace root
        if path.startswith(WORKSPACE_ROOT):
            abs_path = os.path.normpath(path) # Keep it as is
        else:
            # External absolute path, try to make it relative to data dir
            # Extract the last components of the path
            path_components = path.split(os.sep)
            # Try to find 'gesturebind' to make it relative
            for i in range(len(path_components) - 1, 0, -1):
                if path_components[i] == "gesturebind":
                    rel_path = os.path.join(*path_components[i:])
                    abs_path = os.path.normpath(os.path.join(WORKSPACE_ROOT, rel_path))
                    logger.warning(f"Converting external absolute path '{path}' to workspace path '{abs_path}'")
                    break
            else:
                # If 'gesturebind' not found, place it under data/external
                rel_path = path_components[-1]
                abs_path = os.path.normpath(os.path.join(DATA_ROOT, "external", rel_path))
                logger.warning(f"External absolute path '{path}' could not be mapped cleanly. Placing under '{abs_path}'")
    else:
        # Relative path (not starting with ./), assume relative to workspace root
        abs_path = os.path.normpath(os.path.join(WORKSPACE_ROOT, path))
    
    # Create directory if requested and it doesn't exist
    if create_if_missing:
        directory = abs_path
        if os.path.splitext(abs_path)[1]:  # If it has an extension, it's probably a file
            directory = os.path.dirname(abs_path)
        
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
    
    return abs_path

def get_data_dir(subdir=None, create=True):
    """
    Get the path to a data directory within the workspace
    
    Args:
        subdir: Optional subdirectory within the data directory
        create: Whether to create the directory if it doesn't exist
    
    Returns:
        Absolute path to the data directory
    """
    if subdir:
        path = os.path.join(DATA_ROOT, subdir)
    else:
        path = DATA_ROOT
    
    if create and not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created data directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create data directory {path}: {e}")
    
    return path

def get_models_dir():
    """Get the path to the models directory within the workspace"""
    return get_data_dir("models", create=True)

def get_gesture_data_path():
    """Get the path to the gesture data directory"""
    return get_data_dir("gestures", create=True)

def get_gesture_data_file():
    """Get the path to the main gesture data file"""
    return os.path.join(get_data_dir("default", create=True), "gesture_data.json")

def ensure_workspace_paths():
    """Ensure all required data directories exist within the workspace"""
    # Create main data directories
    get_data_dir(create=True)
    get_data_dir("gestures", create=True)
    get_data_dir("default", create=True)
    get_models_dir() # Ensure models directory is created
