"""
Action overlay UI module for GestureBind

This module provides a visual overlay to display detected gestures and triggered actions.
"""

import logging
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPalette

logger = logging.getLogger(__name__)

class ActionOverlay(QWidget):
    """
    Floating overlay window to show gesture detection feedback.
    """
    
    def __init__(self, parent=None):
        """Initialize the overlay window"""
        super().__init__(parent, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        
        # Default settings
        self.timeout_ms = 1500  # How long to show overlay
        self.opacity = 0.85
        self.corner_radius = 15
        
        # Set transparent background
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Initialize UI
        self._init_ui()
        
        # Setup timer for auto-hide
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide_overlay)
        
        # Animation for appearing/disappearing
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def _init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Gesture name label
        self.gesture_label = QLabel()
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #ffffff;"
        )
        layout.addWidget(self.gesture_label)
        
        # Action description label
        self.action_label = QLabel()
        self.action_label.setAlignment(Qt.AlignCenter)
        self.action_label.setStyleSheet(
            "font-size: 16px; color: #ffffff;"
        )
        layout.addWidget(self.action_label)
        
        # Overlay appearance
        self.setMinimumSize(300, 100)
        self.setMaximumSize(500, 150)
        
        # Position in the top-right corner by default
        self._position_overlay()
    
    def _position_overlay(self):
        """Position the overlay in the top-right corner of the screen"""
        desktop = QApplication.desktop()
        screen_rect = desktop.availableGeometry()
        
        width = 350
        height = 100
        padding = 20
        
        # Top-right position
        x = screen_rect.width() - width - padding
        y = padding + 50  # Add a bit more padding at the top to avoid system menus
        
        self.setGeometry(x, y, width, height)
    
    def set_timeout(self, timeout_ms):
        """Set how long the overlay stays visible"""
        self.timeout_ms = timeout_ms
    
    def show_action(self, gesture_name, action_description):
        """Show the overlay with gesture and action information"""
        # Set text content
        self.gesture_label.setText(gesture_name.replace('_', ' ').title())
        self.action_label.setText(action_description)
        
        # Stop any existing timer and animation
        self.hide_timer.stop()
        self.animation.stop()
        
        # Show the overlay
        if not self.isVisible():
            # Position the widget before showing
            self._position_overlay()
            
            # Set up animation to slide in from the right
            start_rect = self.geometry()
            start_rect.setX(start_rect.x() + 50)
            self.setGeometry(start_rect)
            
            # Target animation rectangle
            end_rect = self.geometry()
            end_rect.setX(end_rect.x() - 50)
            
            # Configure animation
            self.animation.setStartValue(start_rect)
            self.animation.setEndValue(end_rect)
            
            # Show and start animation
            self.show()
            self.animation.start()
        
        # Restart the hide timer
        self.hide_timer.start(self.timeout_ms)
        
        logger.debug(f"Showing action overlay for gesture: {gesture_name}")
    
    def hide_overlay(self):
        """Animate hiding the overlay"""
        # Animate sliding out to the right
        start_rect = self.geometry()
        end_rect = self.geometry()
        end_rect.setX(end_rect.x() + 50)
        
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        
        # Connect finished signal to hide the widget
        self.animation.finished.connect(self._hide_after_animation)
        
        # Start animation
        self.animation.start()
    
    def _hide_after_animation(self):
        """Hide the widget after the animation completes"""
        self.hide()
        # Disconnect to avoid repeated calls
        self.animation.finished.disconnect(self._hide_after_animation)
    
    def paintEvent(self, event):
        """Custom paint event to draw rounded rectangle background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create semi-transparent dark background
        painter.setBrush(QBrush(QColor(40, 40, 40, int(self.opacity * 255))))
        painter.setPen(QPen(QColor(60, 60, 60, int(self.opacity * 255)), 1))
        
        # Draw rounded rectangle
        painter.drawRoundedRect(self.rect(), self.corner_radius, self.corner_radius)