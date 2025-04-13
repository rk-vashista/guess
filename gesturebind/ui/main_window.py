"""
Main window UI module for GestureBind.

This module provides the primary application window and interface.
"""

import os
import sys
import logging
import cv2  # Added cv2 import
from PyQt5.QtWidgets import (QMainWindow, QAction, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QSystemTrayIcon, QMenu, QStatusBar, QMessageBox,
                            QGridLayout, QCheckBox, QComboBox, QApplication,
                            QFileDialog)  # Added QFileDialog import
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QIcon, QPixmap, QImage

from core.gesture_detector import GestureDetector
from core.camera_manager import CameraManager
from core.action_mapper import ActionMapper
from models.gesture_classifier import GestureClassifier
from ui.gesture_trainer import GestureTrainer
from ui.settings_panel import SettingsPanel
from ui.action_overlay import ActionOverlay  # Import the ActionOverlay component
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """
    Main application window for GestureBind.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the main window.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.load_config()
        
        self.detection_active = False
        self.camera_manager = None
        self.gesture_detector = None
        self.action_mapper = None
        self.gesture_classifier = None
        
        self.frame_update_timer = QTimer(self)
        self.frame_update_timer.timeout.connect(self.update_frame)
        
        self.last_detected_gesture = None
        self.last_gesture_confidence = 0.0
        
        # Initialize the action overlay for gesture feedback
        self.action_overlay = ActionOverlay()
        
        # Set overlay timeout from config
        overlay_timeout = self.config.get("ui", {}).get("overlay_timeout_ms", 1500)
        self.action_overlay.set_timeout(overlay_timeout)
        
        self._init_ui()
        self._init_components()
        self._setup_tray_icon()
        
        logger.info("GestureBind main window initialized")
    
    def _init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("GestureBind - Gesture-Based Shortcut Mapper")
        self.setMinimumSize(900, 700)
        
        # Create action overlay for gesture feedback
        self.action_overlay = ActionOverlay(self)
        overlay_timeout = self.config.get("ui", {}).get("overlay_timeout_ms", 1500)
        self.action_overlay.set_timeout(overlay_timeout)
        
        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Main tab - Gesture detection and control
        self.main_tab = QWidget()
        self.tab_widget.addTab(self.main_tab, "Gesture Control")
        
        # Training tab - Gesture training interface
        self.training_tab = QWidget()
        self.tab_widget.addTab(self.training_tab, "Train Gestures")
        
        # Settings tab - Application settings
        self.settings_tab = QWidget()
        self.tab_widget.addTab(self.settings_tab, "Settings")
        
        # Initialize the tab contents
        self._init_main_tab()
        self._init_training_tab()
        self._init_settings_tab()
        
        # Status bar for showing application status
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Create menu bar
        self._create_menu_bar()
    
    def _init_main_tab(self):
        """Initialize the main tab content"""
        layout = QGridLayout(self.main_tab)
        
        # Camera preview section
        preview_group = QWidget()
        preview_layout = QVBoxLayout(preview_group)
        
        preview_header = QHBoxLayout()
        
        self.camera_label = QLabel("Camera not started")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #000; color: #fff;")
        preview_layout.addWidget(self.camera_label)
        
        self.detection_toggle = QPushButton("Start Detection")
        self.detection_toggle.clicked.connect(self.toggle_detection)
        preview_header.addWidget(self.detection_toggle)
        
        self.camera_selector = QComboBox()
        self.camera_selector.addItem("Default Camera (0)")
        preview_header.addWidget(QLabel("Camera:"))
        preview_header.addWidget(self.camera_selector)
        
        preview_layout.insertLayout(0, preview_header)
        
        # Gesture recognition section
        recognition_group = QWidget()
        recognition_layout = QVBoxLayout(recognition_group)
        
        self.detection_status = QLabel("Detection Inactive")
        self.detection_status.setStyleSheet("font-weight: bold; color: #666;")
        recognition_layout.addWidget(self.detection_status)
        
        self.last_gesture_label = QLabel("No gesture detected")
        recognition_layout.addWidget(self.last_gesture_label)
        
        self.confidence_label = QLabel("Confidence: 0%")
        recognition_layout.addWidget(self.confidence_label)
        
        self.action_label = QLabel("Action: None")
        recognition_layout.addWidget(self.action_label)
        
        # Profile selection
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Active Profile:"))
        
        self.profile_selector = QComboBox()
        for profile in self.config.get("profiles", {}).keys():
            self.profile_selector.addItem(profile)
        
        # Set default profile
        default_profile = self.config.get("actions", {}).get("default_profile", "default")
        index = self.profile_selector.findText(default_profile)
        if index >= 0:
            self.profile_selector.setCurrentIndex(index)
        
        self.profile_selector.currentTextChanged.connect(self._change_profile)
        profile_layout.addWidget(self.profile_selector)
        
        recognition_layout.addLayout(profile_layout)
        
        # Arrange the panels
        layout.addWidget(preview_group, 0, 0, 2, 2)
        layout.addWidget(recognition_group, 0, 2, 1, 1)
        
        # Statistics section
        stats_group = QWidget()
        stats_layout = QVBoxLayout(stats_group)
        
        self.fps_label = QLabel("FPS: 0")
        stats_layout.addWidget(self.fps_label)
        
        self.resolution_label = QLabel("Resolution: -")
        stats_layout.addWidget(self.resolution_label)
        
        # Add statistics to the layout
        layout.addWidget(stats_group, 1, 2, 1, 1)
    
    def _init_training_tab(self):
        """Initialize the training tab content"""
        layout = QVBoxLayout(self.training_tab)
        
        # Create and add the gesture trainer component
        self.gesture_trainer = GestureTrainer(self.config_manager)
        layout.addWidget(self.gesture_trainer)
        
        # Connect the model trained signal
        self.gesture_trainer.model_trained.connect(self._on_model_trained)
    
    def _init_settings_tab(self):
        """Initialize the settings tab content"""
        layout = QVBoxLayout(self.settings_tab)
        
        # Create and add the settings panel component
        self.settings_panel = SettingsPanel(self.config_manager)
        layout.addWidget(self.settings_panel)
        
        # Connect settings changed signal
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
    
    def _create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        export_action = QAction("Export Configuration", self)
        export_action.triggered.connect(self._export_configuration)
        file_menu.addAction(export_action)
        
        import_action = QAction("Import Configuration", self)
        import_action.triggered.connect(self._import_configuration)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        fps_action = QAction("Show FPS", self)
        fps_action.setCheckable(True)
        fps_action.setChecked(True)
        fps_action.triggered.connect(lambda checked: self.fps_label.setVisible(checked))
        view_menu.addAction(fps_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About GestureBind", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self._open_documentation)
        help_menu.addAction(docs_action)
    
    def _init_components(self):
        """Initialize the application components"""
        try:
            # Initialize components only when needed to save resources
            if not self.detection_active:
                return
                
            # Initialize camera manager
            self.camera_manager = CameraManager(self.config)
            if not self.camera_manager.initialize():
                raise Exception("Failed to initialize camera")
            
            # Initialize gesture detector
            self.gesture_detector = GestureDetector(self.config)
            
            # Initialize action mapper
            self.action_mapper = ActionMapper(self.config)
            
            # Initialize gesture classifier
            self.gesture_classifier = GestureClassifier(self.config)
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize components: {str(e)}"
            )
            self.detection_active = False
    
    def _setup_tray_icon(self):
        """Set up system tray icon if supported"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("System tray not available")
            return
        
        self.tray_icon = QSystemTrayIcon(self)
        
        # Create a default icon (placeholder)
        # In a real implementation you'd use a proper app icon
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.blue)
        icon = QIcon(pixmap)
        self.tray_icon.setIcon(icon)
        
        # Create a context menu for the tray icon
        tray_menu = QMenu()
        
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)
        
        toggle_detection_action = QAction("Start Detection", self)
        toggle_detection_action.triggered.connect(self.toggle_detection)
        tray_menu.addAction(toggle_detection_action)
        
        tray_menu.addSeparator()
        
        quit_action = QAction("Exit", self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._tray_icon_activated)
        
        # Show the icon if configured
        if self.config.get("ui", {}).get("minimize_to_tray", True):
            self.tray_icon.show()
    
    def setup_tray_icon(self):
        """Set up system tray icon if supported - exposed as public method"""
        self._setup_tray_icon()
        if hasattr(self, 'tray_icon') and self.tray_icon:
            self.tray_icon.show()
            return True
        return False
    
    def _tray_icon_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
    
    @pyqtSlot()
    def toggle_detection(self):
        """Toggle gesture detection on/off"""
        if self.detection_active:
            # Stop detection
            self.detection_active = False
            self.detection_toggle.setText("Start Detection")
            self.detection_status.setText("Detection Inactive")
            self.detection_status.setStyleSheet("font-weight: bold; color: #666;")
            
            # Stop updating frames
            self.frame_update_timer.stop()
            
            # Release camera resources
            if self.camera_manager:
                self.camera_manager.stop_capture()
                
            self.camera_label.setText("Camera not started")
            self.statusBar.showMessage("Detection stopped")
            
        else:
            # Initialize components if needed
            if not self.camera_manager:
                self._init_components()
                if not self.camera_manager:
                    return
            
            # Start detection
            self.detection_active = True
            self.detection_toggle.setText("Stop Detection")
            self.detection_status.setText("Detection Active")
            self.detection_status.setStyleSheet("font-weight: bold; color: green;")
            
            # Start camera
            if self.camera_manager.start_capture():
                # Start frame update timer
                self.frame_update_timer.start(33)  # ~30 FPS UI update rate
                self.statusBar.showMessage("Detection started")
            else:
                QMessageBox.warning(
                    self,
                    "Camera Error",
                    "Failed to start camera. Please check your camera connection and settings."
                )
                self.toggle_detection()  # Turn off detection
    
    @pyqtSlot()
    def update_frame(self):
        """Update the camera preview frame and process gestures"""
        if not self.detection_active or not self.camera_manager:
            return
        
        # Get latest frame
        frame = self.camera_manager.get_frame()
        if frame is None:
            return
        
        # Process frame with gesture detector
        gesture_name, confidence, processed_frame = self.gesture_detector.detect_gesture(frame)
        
        # Update UI with the processed frame
        height, width, channels = processed_frame.shape
        bytes_per_line = channels * width
        
        # Convert BGR to RGB
        cvt_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        q_img = QImage(cvt_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img))
        
        # Update stats
        stats = self.camera_manager.get_stats()
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
        self.resolution_label.setText(f"Resolution: {stats['resolution']}")
        
        # If a gesture was detected
        if gesture_name:
            # Update UI
            self.last_detected_gesture = gesture_name
            self.last_gesture_confidence = confidence
            
            self.last_gesture_label.setText(f"Detected: {gesture_name}")
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
            
            # Execute mapped action
            if self.action_mapper:
                action_success = self.action_mapper.execute_action(gesture_name)
                if action_success:
                    self.action_label.setText(f"Action: Executed for {gesture_name}")
                else:
                    self.action_label.setText(f"Action: No mapping for {gesture_name}")
            
            # Show action overlay
            if self.config.get("ui", {}).get("overlay_feedback", True):
                action_text = "No action mapped"
                if self.action_mapper and gesture_name in self.action_mapper.current_profile:
                    action_text = self.action_mapper.current_profile[gesture_name]
                self.action_overlay.show_action(gesture_name, action_text)
    
    @pyqtSlot(str)
    def _change_profile(self, profile_name):
        """Change the active gesture profile"""
        if self.action_mapper:
            success = self.action_mapper.set_profile(profile_name)
            if success:
                self.statusBar.showMessage(f"Switched to profile: {profile_name}")
            else:
                self.statusBar.showMessage(f"Failed to switch profile")
        
        # Update config
        self.config["actions"]["default_profile"] = profile_name
        self.config_manager.save_config()
    
    @pyqtSlot(str)
    def _on_model_trained(self, profile_name):
        """Handle model trained signal from gesture trainer"""
        # Reload the gesture detector with the newly trained model
        if self.gesture_detector:
            # In a real implementation, we would reinitialize the detector or classifier
            # Here we'll just show a message
            self.statusBar.showMessage(f"Model trained for profile: {profile_name}")
    
    @pyqtSlot()
    def _on_settings_changed(self):
        """Handle settings changed signal"""
        # Reload configuration
        self.config = self.config_manager.load_config()
        
        # If detection is active, restart it to apply new settings
        if self.detection_active:
            was_active = True
            self.toggle_detection()  # Stop detection
        else:
            was_active = False
        
        # Reinitialize components with new settings
        self.camera_manager = None
        self.gesture_detector = None
        self.action_mapper = None
        self.gesture_classifier = None
        
        # Restart detection if it was active
        if was_active:
            self.toggle_detection()  # Start detection with new settings
        
        self.statusBar.showMessage("Settings applied")
    
    @pyqtSlot()
    def _export_configuration(self):
        """Export application configuration"""
        export_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Configuration",
            os.path.expanduser("~/gesturebind_config.yaml"),
            "YAML Files (*.yaml);;All Files (*)"
        )
        
        if export_path:
            success = self.config_manager.export_config(export_path)
            if success:
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Configuration exported to {export_path}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    "Failed to export configuration"
                )
    
    @pyqtSlot()
    def _import_configuration(self):
        """Import application configuration"""
        import_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Configuration",
            os.path.expanduser("~"),
            "YAML Files (*.yaml);;All Files (*)"
        )
        
        if import_path:
            success = self.config_manager.import_config(import_path)
            if success:
                # Reload configuration
                self.config = self.config_manager.load_config()
                
                # Update UI
                self._on_settings_changed()
                
                QMessageBox.information(
                    self,
                    "Import Successful",
                    "Configuration imported successfully"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Import Failed",
                    "Failed to import configuration"
                )
    
    @pyqtSlot()
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About GestureBind",
            """
            <h2>GestureBind v1.0</h2>
            <p>Gesture-based shortcut mapper application</p>
            <p>Copyright © 2025</p>
            <p>License: MIT</p>
            """
        )
    
    @pyqtSlot()
    def _open_documentation(self):
        """Open documentation"""
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation is available at:\nhttps://github.com/yourusername/gesturebind/docs"
        )
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.config.get("ui", {}).get("minimize_to_tray", True) and self.tray_icon and self.tray_icon.isVisible():
            # Minimize to tray instead of closing
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                "GestureBind",
                "GestureBind is still running in the background",
                QSystemTrayIcon.Information,
                2000
            )
        else:
            # Stop detection if active
            if self.detection_active:
                self.toggle_detection()
            
            # Close the application
            event.accept()