#!/usr/bin/env python3
"""
Main entry point for the GestureBind application.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging before any other imports
from utils.logger import setup_logger
logger = setup_logger()

# Import path_manager for consistent path resolution
try:
    from utils.path_manager import get_workspace_root, resolve_path
    logger.info("Successfully imported path_manager for application entry point")
except ImportError:
    logger.warning("Couldn't import path_manager, using default paths")
    # Simple default implementation if path_manager is not available
    def get_workspace_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    def resolve_path(path, create_if_missing=False):
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)

# Now import UI components
from ui.main_window import MainWindow
from utils.config_manager import ConfigManager

# Import PyQt5 components
from PyQt5.QtWidgets import QApplication, QStyleFactory
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

def set_dark_theme(app):
    """Apply dark theme to the application"""
    app.setStyle("Fusion")
    
    # Create dark palette
    dark_palette = QPalette()
    
    # Base colors
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    
    # Link colors
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    # Apply palette
    app.setPalette(dark_palette)

def set_light_theme(app):
    """Apply light theme to the application"""
    app.setStyle("Fusion")
    
    # Create light palette
    light_palette = QPalette()
    
    # Base colors
    light_palette.setColor(QPalette.Window, QColor(240, 240, 240))
    light_palette.setColor(QPalette.WindowText, Qt.black)
    light_palette.setColor(QPalette.Base, QColor(255, 255, 255))
    light_palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    light_palette.setColor(QPalette.Text, Qt.black)
    light_palette.setColor(QPalette.Button, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ButtonText, Qt.black)
    
    # Link colors
    light_palette.setColor(QPalette.Link, QColor(0, 102, 204))
    light_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    light_palette.setColor(QPalette.HighlightedText, Qt.white)
    
    # Apply palette
    app.setPalette(light_palette)

def main():
    """Application entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GestureBind - Gesture-Based Shortcut Mapper")
    parser.add_argument("--config", help="Path to custom configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--minimize", action="store_true", help="Start minimized to tray")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.info("Debug mode enabled")
    
    # Determine config path
    if args.config:
        # Resolve the user-provided path
        config_path = resolve_path(args.config)
        if not os.path.exists(config_path):
            logger.error(f"Custom configuration file not found at: {config_path}")
            sys.exit(1)
    else:
        # Use default config path relative to the workspace root
        default_config_path = resolve_path("gesturebind/config/default_config.yaml")
        
        # Check if the default config exists, create if not
        if not os.path.exists(default_config_path):
            logger.warning(f"Default config not found at {default_config_path}. Creating example config.")
            # Ensure the directory exists before creating the file
            os.makedirs(os.path.dirname(default_config_path), exist_ok=True)
            from utils.config_manager import create_default_config
            if not create_default_config(default_config_path):
                 logger.error(f"Failed to create default configuration file at {default_config_path}")
                 # Set config_path to None or handle error appropriately
                 config_path = None 
            else:
                 config_path = default_config_path
        else:
             config_path = default_config_path
    
    # Check if config_path is valid before proceeding
    if not config_path:
        logger.critical("Configuration path could not be determined. Exiting.")
        sys.exit(1)

    logging.info(f"Using configuration from: {config_path}")
    
    # Create config manager
    # ConfigManager's __init__ now correctly receives an absolute path within the workspace
    config_manager = ConfigManager(config_path)
    config = config_manager.load_config()
    
    # Add a check for empty config after loading
    if not config:
        logger.error("Configuration could not be loaded. Please check the config file and logs.")
        # Provide a minimal default config to prevent downstream errors
        config = {"app": {}, "ui": {}, "actions": {}, "profiles": {}}
        # Optionally, exit if config is critical
        # sys.exit(1)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setOrganizationName("GestureBind")
    app.setApplicationName("GestureBind")
    app.setApplicationVersion(config.get("app", {}).get("version", "1.0.0"))
    
    # Apply theme if specified
    theme = config.get("ui", {}).get("theme", "system")
    if theme == "dark":
        logging.info("Applying dark theme")
        set_dark_theme(app)
    elif theme == "light":
        logging.info("Applying light theme")
        set_light_theme(app)
    else:
        # Use system theme
        logging.info("Using system theme")
    
    # Create main window
    main_window = MainWindow(config_manager)
    
    # Force show the window regardless of settings
    main_window.show()
    main_window.raise_()  # Bring window to front
    main_window.activateWindow()  # Activate window (important for some window managers)
    
    # Log that window is displayed
    logging.info("Main window displayed")
    
    # Run the application
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())