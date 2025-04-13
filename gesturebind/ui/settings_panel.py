"""
Settings panel UI module for GestureBind

This module provides the settings configuration interface.
"""

import os
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QComboBox, QSpinBox, QSlider,
                           QGroupBox, QFormLayout, QTabWidget, QCheckBox,
                           QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal

logger = logging.getLogger(__name__)

class SettingsPanel(QWidget):
    """
    Settings panel UI component for configuring GestureBind settings.
    """
    
    # Signal emitted when settings are changed
    settings_changed = pyqtSignal()
    
    def __init__(self, config_manager):
        """
        Initialize the settings panel.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.load_config()
        self.changes_made = False
        
        self._init_ui()
        self._populate_settings()
        
        logger.info("Settings panel initialized")
    
    def _init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for settings categories
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Camera settings tab
        self.camera_tab = QWidget()
        self._init_camera_tab()
        self.tab_widget.addTab(self.camera_tab, "Camera")
        
        # Detection settings tab
        self.detection_tab = QWidget()
        self._init_detection_tab()
        self.tab_widget.addTab(self.detection_tab, "Detection")
        
        # UI settings tab
        self.ui_tab = QWidget()
        self._init_ui_tab()
        self.tab_widget.addTab(self.ui_tab, "Interface")
        
        # Privacy settings tab
        self.privacy_tab = QWidget()
        self._init_privacy_tab()
        self.tab_widget.addTab(self.privacy_tab, "Privacy")
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_settings)
        buttons_layout.addWidget(self.reset_button)
        
        # Spacer
        buttons_layout.addStretch(1)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_changes)
        buttons_layout.addWidget(self.cancel_button)
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_settings)
        buttons_layout.addWidget(self.apply_button)
        
        main_layout.addLayout(buttons_layout)
    
    def _init_camera_tab(self):
        """Initialize the camera settings tab"""
        layout = QFormLayout(self.camera_tab)
        
        # Device selection
        self.camera_device_combo = QComboBox()
        self.camera_device_combo.addItem("Default Camera (0)", 0)
        self.camera_device_combo.addItem("Camera 1", 1)
        self.camera_device_combo.addItem("Camera 2", 2)
        self.camera_device_combo.currentIndexChanged.connect(self._setting_changed)
        layout.addRow("Camera Device:", self.camera_device_combo)
        
        # Resolution
        resolution_layout = QHBoxLayout()
        
        self.camera_width_spin = QSpinBox()
        self.camera_width_spin.setRange(320, 1920)
        self.camera_width_spin.setSingleStep(8)
        self.camera_width_spin.valueChanged.connect(self._setting_changed)
        resolution_layout.addWidget(self.camera_width_spin)
        
        resolution_layout.addWidget(QLabel("x"))
        
        self.camera_height_spin = QSpinBox()
        self.camera_height_spin.setRange(240, 1080)
        self.camera_height_spin.setSingleStep(8)
        self.camera_height_spin.valueChanged.connect(self._setting_changed)
        resolution_layout.addWidget(self.camera_height_spin)
        
        layout.addRow("Resolution:", resolution_layout)
        
        # FPS
        self.camera_fps_spin = QSpinBox()
        self.camera_fps_spin.setRange(1, 60)
        self.camera_fps_spin.valueChanged.connect(self._setting_changed)
        layout.addRow("Frames Per Second:", self.camera_fps_spin)
    
    def _init_detection_tab(self):
        """Initialize the detection settings tab"""
        layout = QVBoxLayout(self.detection_tab)
        
        # Detection engine
        engine_group = QGroupBox("Detection Engine")
        engine_layout = QFormLayout(engine_group)
        
        self.detection_engine_combo = QComboBox()
        self.detection_engine_combo.addItem("MediaPipe", "mediapipe")
        self.detection_engine_combo.addItem("YOLOv8", "yolov8")
        self.detection_engine_combo.currentIndexChanged.connect(self._setting_changed)
        engine_layout.addRow("Engine:", self.detection_engine_combo)
        
        layout.addWidget(engine_group)
        
        # Detection parameters
        params_group = QGroupBox("Detection Parameters")
        params_layout = QFormLayout(params_group)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self._setting_changed)
        
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self.confidence_slider)
        self.confidence_value = QLabel("50%")
        confidence_layout.addWidget(self.confidence_value)
        
        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence_value.setText(f"{value}%")
        )
        
        params_layout.addRow("Confidence Threshold:", confidence_layout)
        
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.1, 5.0)
        self.cooldown_spin.setSingleStep(0.1)
        self.cooldown_spin.setDecimals(1)
        self.cooldown_spin.setSuffix(" seconds")
        self.cooldown_spin.valueChanged.connect(self._setting_changed)
        params_layout.addRow("Cooldown Period:", self.cooldown_spin)
        
        layout.addWidget(params_group)
        
        # Detection area
        area_group = QGroupBox("Detection Area")
        area_layout = QFormLayout(area_group)
        
        self.left_margin_spin = QDoubleSpinBox()
        self.left_margin_spin.setRange(0.0, 0.5)
        self.left_margin_spin.setSingleStep(0.05)
        self.left_margin_spin.setDecimals(2)
        self.left_margin_spin.valueChanged.connect(self._setting_changed)
        area_layout.addRow("Left Margin:", self.left_margin_spin)
        
        self.right_margin_spin = QDoubleSpinBox()
        self.right_margin_spin.setRange(0.5, 1.0)
        self.right_margin_spin.setSingleStep(0.05)
        self.right_margin_spin.setDecimals(2)
        self.right_margin_spin.valueChanged.connect(self._setting_changed)
        area_layout.addRow("Right Margin:", self.right_margin_spin)
        
        self.top_margin_spin = QDoubleSpinBox()
        self.top_margin_spin.setRange(0.0, 0.5)
        self.top_margin_spin.setSingleStep(0.05)
        self.top_margin_spin.setDecimals(2)
        self.top_margin_spin.valueChanged.connect(self._setting_changed)
        area_layout.addRow("Top Margin:", self.top_margin_spin)
        
        self.bottom_margin_spin = QDoubleSpinBox()
        self.bottom_margin_spin.setRange(0.5, 1.0)
        self.bottom_margin_spin.setSingleStep(0.05)
        self.bottom_margin_spin.setDecimals(2)
        self.bottom_margin_spin.valueChanged.connect(self._setting_changed)
        area_layout.addRow("Bottom Margin:", self.bottom_margin_spin)
        
        layout.addWidget(area_group)
    
    def _init_ui_tab(self):
        """Initialize the UI settings tab"""
        layout = QFormLayout(self.ui_tab)
        
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("System", "system")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.currentIndexChanged.connect(self._setting_changed)
        layout.addRow("Theme:", self.theme_combo)
        
        # Show preview
        self.show_preview_check = QCheckBox()
        self.show_preview_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Show Camera Preview:", self.show_preview_check)
        
        # Show overlay feedback
        self.overlay_feedback_check = QCheckBox()
        self.overlay_feedback_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Show Action Overlay:", self.overlay_feedback_check)
        
        # Overlay timeout
        self.overlay_timeout_spin = QSpinBox()
        self.overlay_timeout_spin.setRange(500, 5000)
        self.overlay_timeout_spin.setSingleStep(100)
        self.overlay_timeout_spin.setSuffix(" ms")
        self.overlay_timeout_spin.valueChanged.connect(self._setting_changed)
        layout.addRow("Overlay Duration:", self.overlay_timeout_spin)
        
        # Minimize to tray
        self.minimize_to_tray_check = QCheckBox()
        self.minimize_to_tray_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Minimize to Tray:", self.minimize_to_tray_check)
        
        # Start minimized
        self.start_minimized_check = QCheckBox()
        self.start_minimized_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Start Minimized:", self.start_minimized_check)
        
        # Launch at startup
        self.launch_startup_check = QCheckBox()
        self.launch_startup_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Launch at Startup:", self.launch_startup_check)
    
    def _init_privacy_tab(self):
        """Initialize the privacy settings tab"""
        layout = QVBoxLayout(self.privacy_tab)
        
        # Privacy info
        info_label = QLabel(
            "GestureBind respects your privacy by processing all data locally on your device. "
            "No video or gesture data is transmitted over the internet."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Privacy options
        privacy_group = QGroupBox("Privacy Options")
        privacy_layout = QFormLayout(privacy_group)
        
        self.local_processing_check = QCheckBox()
        self.local_processing_check.setChecked(True)
        self.local_processing_check.setEnabled(False)  # This is always enforced
        privacy_layout.addRow("Process all data locally:", self.local_processing_check)
        
        # Logging level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItem("Debug", "debug")
        self.log_level_combo.addItem("Info", "info")
        self.log_level_combo.addItem("Warning", "warning")
        self.log_level_combo.addItem("Error", "error")
        self.log_level_combo.addItem("Critical", "critical")
        self.log_level_combo.currentIndexChanged.connect(self._setting_changed)
        privacy_layout.addRow("Logging Level:", self.log_level_combo)
        
        # Data storage options
        self.user_data_path = QPushButton("Change Data Directory")
        self.user_data_path.clicked.connect(self._change_data_path)
        self.user_data_path_label = QLabel()
        privacy_layout.addRow("Data Storage:", self.user_data_path)
        privacy_layout.addRow("", self.user_data_path_label)
        
        # Clear data button
        self.clear_data_button = QPushButton("Clear All User Data")
        self.clear_data_button.clicked.connect(self._clear_user_data)
        privacy_layout.addRow("", self.clear_data_button)
        
        layout.addWidget(privacy_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
    
    def _populate_settings(self):
        """Populate settings controls from current configuration"""
        try:
            # Camera settings
            camera_config = self.config.get("camera", {})
            
            device_id = camera_config.get("device_id", 0)
            index = self.camera_device_combo.findData(device_id)
            if index >= 0:
                self.camera_device_combo.setCurrentIndex(index)
            
            self.camera_width_spin.setValue(camera_config.get("width", 640))
            self.camera_height_spin.setValue(camera_config.get("height", 480))
            self.camera_fps_spin.setValue(camera_config.get("fps", 15))
            
            # Detection settings
            detection_config = self.config.get("detection", {})
            
            engine = detection_config.get("engine", "mediapipe")
            index = self.detection_engine_combo.findData(engine)
            if index >= 0:
                self.detection_engine_combo.setCurrentIndex(index)
            
            confidence = int(detection_config.get("confidence_threshold", 0.7) * 100)
            self.confidence_slider.setValue(confidence)
            self.confidence_value.setText(f"{confidence}%")
            
            self.cooldown_spin.setValue(detection_config.get("cooldown_seconds", 1.0))
            
            area = detection_config.get("detection_area", {})
            self.left_margin_spin.setValue(area.get("left", 0.1))
            self.right_margin_spin.setValue(area.get("right", 0.9))
            self.top_margin_spin.setValue(area.get("top", 0.1))
            self.bottom_margin_spin.setValue(area.get("bottom", 0.9))
            
            # UI settings
            ui_config = self.config.get("ui", {})
            
            theme = ui_config.get("theme", "system")
            index = self.theme_combo.findData(theme)
            if index >= 0:
                self.theme_combo.setCurrentIndex(index)
            
            self.show_preview_check.setChecked(ui_config.get("show_preview", True))
            self.overlay_feedback_check.setChecked(ui_config.get("overlay_feedback", True))
            self.overlay_timeout_spin.setValue(ui_config.get("overlay_timeout_ms", 1500))
            self.minimize_to_tray_check.setChecked(ui_config.get("minimize_to_tray", True))
            self.start_minimized_check.setChecked(ui_config.get("start_minimized", False))
            
            # App settings
            app_config = self.config.get("app", {})
            
            self.launch_startup_check.setChecked(app_config.get("launch_at_startup", False))
            
            # Log level
            log_level = app_config.get("log_level", "info")
            index = self.log_level_combo.findData(log_level)
            if index >= 0:
                self.log_level_combo.setCurrentIndex(index)
            
            # User data path
            user_data_path = app_config.get("save_user_data_path", "data/user_gestures")
            self.user_data_path_label.setText(user_data_path)
            
            # Reset change tracking
            self.changes_made = False
            self.apply_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            
        except Exception as e:
            logger.error(f"Error populating settings: {e}")
            QMessageBox.warning(
                self,
                "Settings Error",
                f"Failed to populate settings: {str(e)}"
            )
    
    @pyqtSlot()
    def _setting_changed(self):
        """Handle when a setting is changed"""
        self.changes_made = True
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
    
    @pyqtSlot()
    def _apply_settings(self):
        """Apply settings changes"""
        try:
            # Camera settings
            if "camera" not in self.config:
                self.config["camera"] = {}
                
            self.config["camera"]["device_id"] = self.camera_device_combo.currentData()
            self.config["camera"]["width"] = self.camera_width_spin.value()
            self.config["camera"]["height"] = self.camera_height_spin.value()
            self.config["camera"]["fps"] = self.camera_fps_spin.value()
            
            # Detection settings
            if "detection" not in self.config:
                self.config["detection"] = {}
                
            self.config["detection"]["engine"] = self.detection_engine_combo.currentData()
            self.config["detection"]["confidence_threshold"] = self.confidence_slider.value() / 100.0
            self.config["detection"]["cooldown_seconds"] = self.cooldown_spin.value()
            
            if "detection_area" not in self.config["detection"]:
                self.config["detection"]["detection_area"] = {}
                
            self.config["detection"]["detection_area"]["left"] = self.left_margin_spin.value()
            self.config["detection"]["detection_area"]["right"] = self.right_margin_spin.value()
            self.config["detection"]["detection_area"]["top"] = self.top_margin_spin.value()
            self.config["detection"]["detection_area"]["bottom"] = self.bottom_margin_spin.value()
            
            # UI settings
            if "ui" not in self.config:
                self.config["ui"] = {}
                
            self.config["ui"]["theme"] = self.theme_combo.currentData()
            self.config["ui"]["show_preview"] = self.show_preview_check.isChecked()
            self.config["ui"]["overlay_feedback"] = self.overlay_feedback_check.isChecked()
            self.config["ui"]["overlay_timeout_ms"] = self.overlay_timeout_spin.value()
            self.config["ui"]["minimize_to_tray"] = self.minimize_to_tray_check.isChecked()
            self.config["ui"]["start_minimized"] = self.start_minimized_check.isChecked()
            
            # App settings
            if "app" not in self.config:
                self.config["app"] = {}
                
            self.config["app"]["launch_at_startup"] = self.launch_startup_check.isChecked()
            self.config["app"]["log_level"] = self.log_level_combo.currentData()
            
            # Save the configuration
            success = self.config_manager.save_config()
            
            if success:
                self.changes_made = False
                self.apply_button.setEnabled(False)
                self.cancel_button.setEnabled(False)
                
                # Emit signal to notify that settings have changed
                self.settings_changed.emit()
                
                QMessageBox.information(
                    self,
                    "Settings Applied",
                    "Settings have been applied successfully."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Settings Error",
                    "Failed to save settings."
                )
                
        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            QMessageBox.critical(
                self,
                "Settings Error",
                f"Failed to apply settings: {str(e)}"
            )
    
    @pyqtSlot()
    def _cancel_changes(self):
        """Cancel settings changes"""
        if self.changes_made:
            # Reset to original settings
            self._populate_settings()
            
            self.changes_made = False
            self.apply_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
    
    @pyqtSlot()
    def _reset_settings(self):
        """Reset settings to defaults"""
        confirm = QMessageBox.question(
            self,
            "Confirm Reset",
            "Are you sure you want to reset all settings to their default values?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Reset configuration
                self.config = self.config_manager.reset_to_default()
                
                # Repopulate settings
                self._populate_settings()
                
                # Emit signal to notify that settings have changed
                self.settings_changed.emit()
                
                QMessageBox.information(
                    self,
                    "Settings Reset",
                    "Settings have been reset to defaults."
                )
                
            except Exception as e:
                logger.error(f"Error resetting settings: {e}")
                QMessageBox.critical(
                    self,
                    "Settings Error",
                    f"Failed to reset settings: {str(e)}"
                )
    
    @pyqtSlot()
    def _change_data_path(self):
        """Change the user data directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            confirm = QMessageBox.question(
                self,
                "Confirm Change",
                "Changing the data directory will not move existing data. "
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if confirm == QMessageBox.Yes:
                # Update config
                if "app" not in self.config:
                    self.config["app"] = {}
                
                self.config["app"]["save_user_data_path"] = directory
                
                # Update UI
                self.user_data_path_label.setText(directory)
                
                # Mark as changed
                self._setting_changed()
    
    @pyqtSlot()
    def _clear_user_data(self):
        """Clear all user data"""
        confirm = QMessageBox.question(
            self,
            "Confirm Data Deletion",
            "Are you sure you want to delete all user data? "
            "This will remove all your custom gestures and settings. "
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Get data directory
                data_path = self.config.get("app", {}).get("save_user_data_path", "data/user_gestures")
                data_path = os.path.expanduser(data_path)
                
                if os.path.exists(data_path):
                    # Delete all files in the directory
                    for file in os.listdir(data_path):
                        file_path = os.path.join(data_path, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                import shutil
                                shutil.rmtree(file_path)
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {e}")
                    
                    QMessageBox.information(
                        self,
                        "Data Cleared",
                        "All user data has been deleted."
                    )
                else:
                    QMessageBox.information(
                        self,
                        "No Data",
                        "No user data found to delete."
                    )
                
            except Exception as e:
                logger.error(f"Error clearing user data: {e}")
                QMessageBox.critical(
                    self,
                    "Data Error",
                    f"Failed to clear user data: {str(e)}"
                )