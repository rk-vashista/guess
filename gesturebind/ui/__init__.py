"""
User interface module for GestureBind

Contains UI components for the main window, settings panel, and gesture trainer.
"""

from .main_window import MainWindow
from .settings_panel import SettingsPanel
from .gesture_trainer import GestureTrainer
from .action_overlay import ActionOverlay

__all__ = ['MainWindow', 'SettingsPanel', 'GestureTrainer', 'ActionOverlay']