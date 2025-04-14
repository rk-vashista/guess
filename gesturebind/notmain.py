#!/usr/bin/env python3
"""
GestureBind - Gesture-Based Shortcut Mapper

A desktop application that allows users to bind hand gestures to system actions.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QStyleFactory
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

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
        config_path = args.config
    else:
        # Use default config
        default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         "config", "default_config.yaml")
        
        # Check if the default config exists
        if not os.path.exists(default_config_path):
            logging.warning(f"Default config not found at {default_config_path}. Creating example config.")
            from utils.config_manager import create_default_config
            create_default_config(default_config_path)
        
        config_path = default_config_path
    
    logging.info(f"Using configuration from: {config_path}")
    
    # Create config manager
    config_manager = ConfigManager(config_path)
    config = config_manager.load_config()
    
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