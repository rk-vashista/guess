"""
User interface module for GestureBind

Contains UI components for the main window, settings panel, and gesture trainer.
"""

import os
import sys

# Fix imports by adding the project root to sys.path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use direct imports instead of trying to import from the package itself
# Import after adding the project root to sys.path
from ui.main_window import MainWindow
from ui.settings_panel import SettingsPanel
from ui.gesture_trainer import GestureTrainer
from ui.action_overlay import ActionOverlay

__all__ = ['MainWindow', 'SettingsPanel', 'GestureTrainer', 'ActionOverlay']