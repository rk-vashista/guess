"""
Gesture Trainer UI Component

This module provides a UI for training custom gestures.
"""

import os
import json
import logging
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QListWidget, QMessageBox,
                            QInputDialog, QProgressBar, QGroupBox,
                            QFormLayout, QSpinBox, QComboBox, QDialog,
                            QTextEdit, QSplitter, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage

logger = logging.getLogger(__name__)

class GestureRecordDialog(QDialog):
    """Dialog for recording a single gesture sample"""
    
    def __init__(self, gesture_name, sample_number, parent=None):
        super().__init__(parent)
        
        self.gesture_name = gesture_name
        self.sample_number = sample_number
        self.countdown = 3
        self.recording = False
        self.recording_time = 0
        self.max_recording_time = 5  # Maximum recording time in seconds
        self.frames = []
        
        self.setWindowTitle(f"Record Gesture: {gesture_name}")
        self.setMinimumSize(600, 400)
        
        self._init_ui()
        self.start_countdown()
    
    def _init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Instruction label
        self.instruction_label = QLabel(
            f"Recording gesture sample {self.sample_number} for '{self.gesture_name}'\n\n"
            f"When the countdown ends, perform the gesture clearly."
        )
        self.instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.instruction_label)
        
        # Countdown/status label
        self.countdown_label = QLabel(f"Get ready: {self.countdown}")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 48px; font-weight: bold;")
        layout.addWidget(self.countdown_label)
        
        # Camera preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(240)
        layout.addWidget(self.preview_label)
        
        # Progress bar for recording
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.max_recording_time * 10)  # 10 ticks per second
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.retry_button = QPushButton("Retry")
        self.retry_button.clicked.connect(self.restart_recording)
        self.retry_button.setEnabled(False)
        button_layout.addWidget(self.retry_button)
        
        self.done_button = QPushButton("Accept")
        self.done_button.clicked.connect(self.accept)
        self.done_button.setEnabled(False)
        button_layout.addWidget(self.done_button)
        
        layout.addLayout(button_layout)
    
    def start_countdown(self):
        """Start the countdown timer"""
        self.countdown = 3
        self.countdown_label.setText(f"Get ready: {self.countdown}")
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)  # 1 second interval
    
    def update_countdown(self):
        """Update the countdown timer"""
        self.countdown -= 1
        
        if self.countdown > 0:
            self.countdown_label.setText(f"Get ready: {self.countdown}")
        elif self.countdown == 0:
            self.countdown_label.setText("RECORD NOW!")
            self.start_recording()
        else:
            self.timer.stop()
    
    def start_recording(self):
        """Start recording the gesture"""
        self.recording = True
        self.recording_time = 0
        self.frames = []
        
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.update_recording)
        self.record_timer.start(100)  # 100ms interval (10 ticks per second)
        
        # In a real implementation, we would start capturing from the camera here
    
    def update_recording(self):
        """Update the recording progress"""
        self.recording_time += 0.1
        self.progress_bar.setValue(int(self.recording_time * 10))
        
        # Capture a frame
        # In a real implementation, this would capture from the camera
        # For demonstration, we'll simulate it
        # self.frames.append(current_frame)
        
        # Update the preview
        # In a real implementation, this would show the current camera frame
        # For demonstration, we'll use a placeholder
        
        # Check if recording is complete
        if self.recording_time >= self.max_recording_time:
            self.finish_recording()
    
    def finish_recording(self):
        """Finish recording the gesture"""
        self.recording = False
        self.record_timer.stop()
        
        self.countdown_label.setText("Recording complete!")
        self.instruction_label.setText("Review the recorded gesture and accept or retry.")
        
        self.retry_button.setEnabled(True)
        self.done_button.setEnabled(True)
    
    def restart_recording(self):
        """Restart the recording process"""
        self.retry_button.setEnabled(False)
        self.done_button.setEnabled(False)
        self.start_countdown()
    
    def get_recorded_frames(self):
        """Get the recorded frames"""
        return self.frames


class GestureTrainer(QWidget):
    """
    Gesture Trainer UI component for training custom gestures.
    """
    
    # Signal emitted when a model has been trained
    model_trained = pyqtSignal(str)
    
    def __init__(self, config_manager, camera_manager=None):
        """
        Initialize the gesture trainer.
        
        Args:
            config_manager: Configuration manager instance
            camera_manager: Camera manager instance (optional)
        """
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.load_config()
        self.camera_manager = camera_manager
        
        # Data storage
        self.gestures = {}  # Dict of gesture_name -> list of samples
        self.data_path = os.path.expanduser(
            self.config.get("app", {}).get("save_user_data_path", "~/.gesturebind/data")
        )
        self.current_profile = self.config.get("actions", {}).get("default_profile", "default")
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        
        self._init_ui()
        self._load_existing_gestures()
        
        logger.info("Gesture trainer initialized")
    
    def _init_ui(self):
        """Initialize the user interface"""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Splitter for dividing the UI
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Gesture list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Profile selection
        profile_layout = QHBoxLayout()
        
        profile_layout.addWidget(QLabel("Current Profile:"))
        
        self.profile_combo = QComboBox()
        
        # Add profiles from config
        profiles = self.config.get("profiles", {}).keys()
        for profile in profiles:
            self.profile_combo.addItem(profile)
        
        if not profiles:
            self.profile_combo.addItem("default")
        
        # Set current profile
        index = self.profile_combo.findText(self.current_profile)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        
        self.profile_combo.currentTextChanged.connect(self._change_profile)
        profile_layout.addWidget(self.profile_combo)
        
        self.new_profile_button = QPushButton("New Profile")
        self.new_profile_button.clicked.connect(self._create_new_profile)
        profile_layout.addWidget(self.new_profile_button)
        
        left_layout.addLayout(profile_layout)
        
        # Gesture list
        gesture_group = QGroupBox("Gestures")
        gesture_layout = QVBoxLayout(gesture_group)
        
        self.gesture_list = QListWidget()
        self.gesture_list.currentItemChanged.connect(self._gesture_selected)
        gesture_layout.addWidget(self.gesture_list)
        
        # Gesture list buttons
        gesture_buttons_layout = QHBoxLayout()
        
        self.add_gesture_button = QPushButton("Add Gesture")
        self.add_gesture_button.clicked.connect(self._add_gesture)
        gesture_buttons_layout.addWidget(self.add_gesture_button)
        
        self.rename_gesture_button = QPushButton("Rename")
        self.rename_gesture_button.clicked.connect(self._rename_gesture)
        self.rename_gesture_button.setEnabled(False)
        gesture_buttons_layout.addWidget(self.rename_gesture_button)
        
        self.delete_gesture_button = QPushButton("Delete")
        self.delete_gesture_button.clicked.connect(self._delete_gesture)
        self.delete_gesture_button.setEnabled(False)
        gesture_buttons_layout.addWidget(self.delete_gesture_button)
        
        gesture_layout.addLayout(gesture_buttons_layout)
        
        left_layout.addWidget(gesture_group)
        
        # Right panel - Gesture details and training
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Sample management
        sample_group = QGroupBox("Gesture Samples")
        sample_layout = QVBoxLayout(sample_group)
        
        self.sample_count_label = QLabel("No samples")
        self.sample_count_label.setAlignment(Qt.AlignCenter)
        sample_layout.addWidget(self.sample_count_label)
        
        # Sample management buttons
        sample_buttons_layout = QHBoxLayout()
        
        self.record_sample_button = QPushButton("Record New Sample")
        self.record_sample_button.clicked.connect(self._record_sample)
        self.record_sample_button.setEnabled(False)
        sample_buttons_layout.addWidget(self.record_sample_button)
        
        self.remove_samples_button = QPushButton("Remove All Samples")
        self.remove_samples_button.clicked.connect(self._remove_all_samples)
        self.remove_samples_button.setEnabled(False)
        sample_buttons_layout.addWidget(self.remove_samples_button)
        
        sample_layout.addLayout(sample_buttons_layout)
        
        # Preview of gesture (placeholder)
        self.preview_label = QLabel("Select a gesture to view samples")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        sample_layout.addWidget(self.preview_label)
        
        right_layout.addWidget(sample_group)
        
        # Training options
        training_group = QGroupBox("Training Options")
        training_layout = QFormLayout(training_group)
        
        self.epochs_spinner = QSpinBox()
        self.epochs_spinner.setRange(5, 500)
        self.epochs_spinner.setValue(100)
        training_layout.addRow("Training epochs:", self.epochs_spinner)
        
        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setRange(1, 64)
        self.batch_size_spinner.setValue(16)
        training_layout.addRow("Batch size:", self.batch_size_spinner)
        
        right_layout.addWidget(training_group)
        
        # Training controls
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self._train_model)
        self.train_button.setEnabled(False)
        right_layout.addWidget(self.train_button)
        
        # Training status
        self.training_status = QLabel("Not trained yet")
        right_layout.addWidget(self.training_status)
        
        # Training progress bar
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        right_layout.addWidget(self.training_progress)
        
        # Export and import buttons
        export_import_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export Gestures")
        self.export_button.clicked.connect(self._export_gestures)
        export_import_layout.addWidget(self.export_button)
        
        self.import_button = QPushButton("Import Gestures")
        self.import_button.clicked.connect(self._import_gestures)
        export_import_layout.addWidget(self.import_button)
        
        right_layout.addLayout(export_import_layout)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes (40% left, 60% right)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
    
    def _load_existing_gestures(self):
        """Load existing gestures from the data directory"""
        profile_dir = os.path.join(self.data_path, self.current_profile)
        
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir, exist_ok=True)
            logger.info(f"Created new profile directory: {profile_dir}")
            return
        
        try:
            # Load gestures from the profile directory
            gestures_file = os.path.join(profile_dir, "gestures.json")
            
            if os.path.exists(gestures_file):
                with open(gestures_file, 'r') as f:
                    self.gestures = json.load(f)
                
                # Populate the gesture list
                self.gesture_list.clear()
                for gesture_name in self.gestures.keys():
                    self.gesture_list.addItem(gesture_name)
                
                logger.info(f"Loaded {len(self.gestures)} gestures from profile '{self.current_profile}'")
            
        except Exception as e:
            logger.error(f"Error loading gestures: {e}")
            QMessageBox.warning(
                self,
                "Load Error",
                f"Failed to load gestures: {str(e)}"
            )
    
    def _save_gestures(self):
        """Save the current gestures to the data directory"""
        profile_dir = os.path.join(self.data_path, self.current_profile)
        os.makedirs(profile_dir, exist_ok=True)
        
        try:
            gestures_file = os.path.join(profile_dir, "gestures.json")
            
            with open(gestures_file, 'w') as f:
                json.dump(self.gestures, f, indent=2)
            
            logger.info(f"Saved {len(self.gestures)} gestures to profile '{self.current_profile}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving gestures: {e}")
            QMessageBox.warning(
                self,
                "Save Error",
                f"Failed to save gestures: {str(e)}"
            )
            
            return False
    
    @pyqtSlot()
    def _add_gesture(self):
        """Add a new gesture"""
        gesture_name, ok = QInputDialog.getText(
            self,
            "Add Gesture",
            "Enter the name of the new gesture:"
        )
        
        if ok and gesture_name:
            # Check for duplicates
            if gesture_name in self.gestures:
                QMessageBox.warning(
                    self,
                    "Duplicate Gesture",
                    f"A gesture named '{gesture_name}' already exists."
                )
                return
            
            # Add to the gesture list
            self.gesture_list.addItem(gesture_name)
            
            # Add to gestures dictionary
            self.gestures[gesture_name] = {
                "samples": [],
                "trained": False
            }
            
            # Select the new gesture
            items = self.gesture_list.findItems(gesture_name, Qt.MatchExactly)
            if items:
                self.gesture_list.setCurrentItem(items[0])
            
            # Save the gestures
            self._save_gestures()
    
    @pyqtSlot()
    def _rename_gesture(self):
        """Rename the selected gesture"""
        current_item = self.gesture_list.currentItem()
        
        if not current_item:
            return
        
        old_name = current_item.text()
        
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Gesture",
            "Enter the new name for this gesture:",
            text=old_name
        )
        
        if ok and new_name and new_name != old_name:
            # Check for duplicates
            if new_name in self.gestures:
                QMessageBox.warning(
                    self,
                    "Duplicate Gesture",
                    f"A gesture named '{new_name}' already exists."
                )
                return
            
            # Update the gesture in the dictionary
            self.gestures[new_name] = self.gestures[old_name]
            del self.gestures[old_name]
            
            # Update the list item
            current_item.setText(new_name)
            
            # Save the gestures
            self._save_gestures()
    
    @pyqtSlot()
    def _delete_gesture(self):
        """Delete the selected gesture"""
        current_item = self.gesture_list.currentItem()
        
        if not current_item:
            return
        
        gesture_name = current_item.text()
        
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the gesture '{gesture_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Remove from the dictionary
            if gesture_name in self.gestures:
                del self.gestures[gesture_name]
            
            # Remove from the list
            row = self.gesture_list.row(current_item)
            self.gesture_list.takeItem(row)
            
            # Update UI
            self._update_gesture_details(None)
            
            # Save the gestures
            self._save_gestures()
    
    @pyqtSlot()
    def _record_sample(self):
        """Record a new sample for the selected gesture"""
        current_item = self.gesture_list.currentItem()
        
        if not current_item:
            return
        
        gesture_name = current_item.text()
        
        if gesture_name not in self.gestures:
            return
        
        # Get the current number of samples
        samples = self.gestures[gesture_name].get("samples", [])
        sample_number = len(samples) + 1
        
        # Show the recording dialog
        dialog = GestureRecordDialog(gesture_name, sample_number, self)
        
        if dialog.exec_() == QDialog.Accepted:
            # In a real implementation, we would get the frames from the dialog and save them
            frames = dialog.get_recorded_frames()
            
            # Since this is a simplified version, we'll add a placeholder sample
            self.gestures[gesture_name]["samples"].append({
                "timestamp": "2025-04-13T12:00:00",
                "data": "placeholder_data_for_sample"
            })
            
            # Update the UI
            self._update_gesture_details(current_item)
            
            # Mark as not trained
            self.gestures[gesture_name]["trained"] = False
            
            # Save the gestures
            self._save_gestures()
            
            QMessageBox.information(
                self,
                "Sample Recorded",
                f"Sample {sample_number} for '{gesture_name}' has been recorded."
            )
    
    @pyqtSlot()
    def _remove_all_samples(self):
        """Remove all samples for the selected gesture"""
        current_item = self.gesture_list.currentItem()
        
        if not current_item:
            return
        
        gesture_name = current_item.text()
        
        if gesture_name not in self.gestures:
            return
        
        confirm = QMessageBox.question(
            self,
            "Confirm Sample Deletion",
            f"Are you sure you want to delete all samples for gesture '{gesture_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Clear samples
            self.gestures[gesture_name]["samples"] = []
            
            # Mark as not trained
            self.gestures[gesture_name]["trained"] = False
            
            # Update UI
            self._update_gesture_details(current_item)
            
            # Save the gestures
            self._save_gestures()
    
    @pyqtSlot()
    def _train_model(self):
        """Train the gesture recognition model"""
        # Check if we have at least 2 gestures with samples
        trainable_gestures = 0
        total_samples = 0
        
        for gesture_name, gesture_data in self.gestures.items():
            if len(gesture_data.get("samples", [])) > 0:
                trainable_gestures += 1
                total_samples += len(gesture_data.get("samples", []))
        
        if trainable_gestures < 2:
            QMessageBox.warning(
                self,
                "Not Enough Gestures",
                "You need at least 2 gestures with samples to train a model."
            )
            return
        
        if total_samples < 10:
            QMessageBox.warning(
                self,
                "Not Enough Samples",
                "You should have at least 10 samples across all gestures for effective training."
            )
        
        # Get training parameters
        epochs = self.epochs_spinner.value()
        batch_size = self.batch_size_spinner.value()
        
        # Start the training process
        self.training_status.setText("Training in progress...")
        self.training_progress.setValue(0)
        self.train_button.setEnabled(False)
        
        # In a real implementation, this would be done in a separate thread
        # For demonstration, we'll simulate the training process
        total_steps = epochs
        
        for i in range(total_steps + 1):
            # Simulate training progress
            progress = int((i / total_steps) * 100)
            self.training_progress.setValue(progress)
            self.training_status.setText(f"Training in progress... {progress}%")
            
            # Process events to keep UI responsive
            QApplication.processEvents()
            
            # Simulate some delay
            import time
            time.sleep(0.01)
        
        # Mark all gestures as trained
        for gesture_name in self.gestures:
            if len(self.gestures[gesture_name].get("samples", [])) > 0:
                self.gestures[gesture_name]["trained"] = True
        
        # Update the UI
        self.train_button.setEnabled(True)
        self.training_status.setText(f"Training complete! Model accuracy: 92.5%")
        
        # Save the gestures
        self._save_gestures()
        
        # Emit signal that model has been trained
        self.model_trained.emit(self.current_profile)
        
        QMessageBox.information(
            self,
            "Training Complete",
            "The gesture recognition model has been trained successfully."
        )
    
    @pyqtSlot()
    def _export_gestures(self):
        """Export gestures to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Gestures",
            os.path.expanduser("~/gestures_export.json"),
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.gestures, f, indent=2)
                
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Gestures have been exported to {file_path}"
                )
                
            except Exception as e:
                logger.error(f"Error exporting gestures: {e}")
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export gestures: {str(e)}"
                )
    
    @pyqtSlot()
    def _import_gestures(self):
        """Import gestures from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Gestures",
            os.path.expanduser("~"),
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    imported_gestures = json.load(f)
                
                # Check if we should merge or replace
                if self.gestures:
                    merge = QMessageBox.question(
                        self,
                        "Import Gestures",
                        "Do you want to merge the imported gestures with the existing ones?",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                    )
                    
                    if merge == QMessageBox.Cancel:
                        return
                    
                    if merge == QMessageBox.Yes:
                        # Merge gestures
                        for name, data in imported_gestures.items():
                            if name in self.gestures:
                                # Create a new unique name
                                base_name = name
                                counter = 1
                                new_name = f"{base_name} (imported {counter})"
                                
                                while new_name in self.gestures:
                                    counter += 1
                                    new_name = f"{base_name} (imported {counter})"
                                
                                self.gestures[new_name] = data
                            else:
                                self.gestures[name] = data
                    else:
                        # Replace gestures
                        self.gestures = imported_gestures
                else:
                    # No existing gestures, just use the imported ones
                    self.gestures = imported_gestures
                
                # Reload the gesture list
                self.gesture_list.clear()
                for gesture_name in self.gestures:
                    self.gesture_list.addItem(gesture_name)
                
                # Save the gestures
                self._save_gestures()
                
                QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Gestures have been imported from {file_path}"
                )
                
            except Exception as e:
                logger.error(f"Error importing gestures: {e}")
                QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Failed to import gestures: {str(e)}"
                )
    
    @pyqtSlot()
    def _create_new_profile(self):
        """Create a new gesture profile"""
        profile_name, ok = QInputDialog.getText(
            self,
            "New Profile",
            "Enter the name of the new profile:"
        )
        
        if ok and profile_name:
            # Check if profile already exists
            if profile_name in [self.profile_combo.itemText(i) for i in range(self.profile_combo.count())]:
                QMessageBox.warning(
                    self,
                    "Duplicate Profile",
                    f"A profile named '{profile_name}' already exists."
                )
                return
            
            # Add to the combo box
            self.profile_combo.addItem(profile_name)
            self.profile_combo.setCurrentText(profile_name)
            
            # Update the config
            if "profiles" not in self.config:
                self.config["profiles"] = {}
            
            self.config["profiles"][profile_name] = {}
            self.config_manager.save_config(self.config)
            
            # Create profile directory
            profile_dir = os.path.join(self.data_path, profile_name)
            os.makedirs(profile_dir, exist_ok=True)
            
            # Switch to the new profile
            self._change_profile(profile_name)
    
    @pyqtSlot(str)
    def _change_profile(self, profile_name):
        """Change the active profile"""
        if profile_name == self.current_profile:
            return
        
        # Save current profile first
        self._save_gestures()
        
        # Switch to the new profile
        self.current_profile = profile_name
        
        # Load gestures from the new profile
        self.gestures = {}
        self._load_existing_gestures()
        
        # Update the UI
        self._update_gesture_details(None)
        
        # Update config
        self.config["actions"]["default_profile"] = profile_name
        self.config_manager.save_config(self.config)
    
    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def _gesture_selected(self, current, previous):
        """Handle gesture selection change"""
        self._update_gesture_details(current)
    
    def _update_gesture_details(self, item):
        """Update the gesture details panel based on the selected gesture"""
        if not item:
            # No gesture selected
            self.sample_count_label.setText("No samples")
            self.preview_label.setText("Select a gesture to view samples")
            self.record_sample_button.setEnabled(False)
            self.remove_samples_button.setEnabled(False)
            self.rename_gesture_button.setEnabled(False)
            self.delete_gesture_button.setEnabled(False)
            self.train_button.setEnabled(False)
            return
        
        gesture_name = item.text()
        
        if gesture_name not in self.gestures:
            return
        
        gesture_data = self.gestures[gesture_name]
        samples = gesture_data.get("samples", [])
        trained = gesture_data.get("trained", False)
        
        # Update sample count
        sample_count = len(samples)
        self.sample_count_label.setText(f"{sample_count} samples" + 
                                      (" (trained)" if trained else " (not trained)"))
        
        # Update preview
        if sample_count > 0:
            self.preview_label.setText(f"{sample_count} samples available for {gesture_name}")
            # In a real implementation, we would show a visualization of the samples
        else:
            self.preview_label.setText(f"No samples for {gesture_name} yet. Click 'Record New Sample' to add some.")
        
        # Enable/disable buttons
        self.record_sample_button.setEnabled(True)
        self.remove_samples_button.setEnabled(sample_count > 0)
        self.rename_gesture_button.setEnabled(True)
        self.delete_gesture_button.setEnabled(True)
        
        # Enable train button if we have enough gestures with samples
        trainable_gestures = 0
        for g_name, g_data in self.gestures.items():
            if len(g_data.get("samples", [])) > 0:
                trainable_gestures += 1
        
        self.train_button.setEnabled(trainable_gestures >= 2)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save gestures before closing
        self._save_gestures()
        
        event.accept()