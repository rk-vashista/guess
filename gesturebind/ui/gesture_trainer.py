"""
Gesture Trainer UI Component

This module provides a UI for training custom gestures.
"""

import os
import json
import logging
import numpy as np
import cv2
import datetime
from pathlib import Path
import random

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QListWidget, QListWidgetItem, QMessageBox,
                            QInputDialog, QProgressBar, QGroupBox,
                            QFormLayout, QSpinBox, QComboBox, QDialog,
                            QSplitter, QFileDialog, QApplication)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage

# Import the gesture detector if available
try:
    from core.gesture_detector import GestureDetector
except ImportError:
    GestureDetector = None

# Import the gesture classifier if available
try:
    from models.gesture_classifier import GestureClassifier
except ImportError:
    GestureClassifier = None

logger = logging.getLogger(__name__)


class GestureAugmenter:
    """Class for augmenting gesture samples to improve training with few samples"""
    
    @staticmethod
    def _ensure_numeric_format(landmarks):
        """Convert landmarks from dictionary format to numeric array if needed"""
        if isinstance(landmarks, list) and landmarks and isinstance(landmarks[0], dict):
            # Convert from list of dicts to flat array
            numeric_landmarks = []
            for lm in landmarks:
                numeric_landmarks.extend([lm['x'], lm['y'], lm['z']])
            return np.array(numeric_landmarks)
        elif isinstance(landmarks, dict):
            # Single landmark dict
            return np.array([landmarks['x'], landmarks['y'], landmarks['z']])
        else:
            # Already in numeric format
            return np.array(landmarks)
    
    @staticmethod
    def center_landmarks(landmarks):
        """Centers landmarks to their centroid."""
        # Convert to numeric format first
        landmarks = GestureAugmenter._ensure_numeric_format(landmarks)
        landmarks = landmarks.reshape(-1, 3)
        center = np.mean(landmarks, axis=0)
        return (landmarks - center).flatten()

    @staticmethod
    def rotate_landmarks(landmarks, angle_deg):
        """Rotate around Z-axis (XY plane)"""
        # Convert to numeric format first
        landmarks = GestureAugmenter._ensure_numeric_format(landmarks)
        angle_rad = np.radians(angle_deg)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        landmarks = landmarks.reshape(-1, 3)
        xy_rotated = np.dot(landmarks[:, :2], rot_matrix)
        return np.hstack((xy_rotated, landmarks[:, 2:])).flatten()

    @staticmethod
    def mirror_landmarks(landmarks):
        """Flip along X-axis"""
        # Convert to numeric format first
        landmarks = GestureAugmenter._ensure_numeric_format(landmarks)
        landmarks = landmarks.reshape(-1, 3)
        landmarks[:, 0] = -landmarks[:, 0]
        return landmarks.flatten()

    @staticmethod
    def time_warp(landmarks):
        """Simulate frame lag by jittering a few random joints more than others"""
        # Convert to numeric format first
        landmarks = GestureAugmenter._ensure_numeric_format(landmarks)
        landmarks = landmarks.reshape(-1, 3)
        indices = random.sample(range(len(landmarks)), k=min(5, len(landmarks)))
        for i in indices:
            landmarks[i] += np.random.normal(0, 0.02, size=3)
        return landmarks.flatten()

    @staticmethod
    def augment_sample(sample):
        """Apply multiple augmentations to one sample"""
        augmented = []

        # Convert the sample to numeric format first
        sample = GestureAugmenter._ensure_numeric_format(sample)
        
        # Apply center normalization
        sample = GestureAugmenter.center_landmarks(sample)
        augmented.append(sample.tolist())

        for _ in range(19):
            aug = sample.copy()

            # Apply random augmentation combinations
            if random.random() < 0.5:
                aug = GestureAugmenter.rotate_landmarks(aug, angle_deg=random.uniform(-25, 25))
            if random.random() < 0.5:
                aug = GestureAugmenter.mirror_landmarks(aug)
            if random.random() < 0.5:
                aug = GestureAugmenter.time_warp(aug)

            # Add noise and scale
            jitter = np.random.normal(0, 0.01, size=len(aug))
            scale = np.random.uniform(0.9, 1.1)
            aug = (np.array(aug) + jitter) * scale

            augmented.append(aug.tolist())
        
        return augmented

    @staticmethod
    def generate_augmented_samples(samples, target_count=100):
        """Generate a specific number of samples from base samples"""
        all_augmented = []
        for sample in samples:
            all_augmented.extend(GestureAugmenter.augment_sample(sample))

        # Trim/pad to exactly the target count
        if len(all_augmented) > target_count:
            all_augmented = all_augmented[:target_count]
        elif len(all_augmented) < target_count:
            # Make sure we have at least one sample before trying to pad
            if all_augmented:
                all_augmented += random.choices(all_augmented, k=target_count - len(all_augmented))
            
        return all_augmented


class GestureRecordDialog(QDialog):
    """Dialog for recording a single gesture sample"""
    
    def __init__(self, gesture_name, sample_number, camera_manager=None, gesture_detector=None, parent=None):
        super().__init__(parent)
        
        self.gesture_name = gesture_name
        self.sample_number = sample_number
        self.countdown = 3
        self.recording = False
        self.recording_time = 0
        self.max_recording_time = 5  # Maximum recording time in seconds
        self.frames = []
        self.landmarks = []
        self.camera_manager = camera_manager
        self.gesture_detector = gesture_detector
        self.best_landmarks = None
        self.best_confidence = 0.0
        
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
        self.preview_label.setStyleSheet("background-color: #000000;")
        layout.addWidget(self.preview_label)
        
        # Progress bar for recording
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.max_recording_time * 10)  # 10 ticks per second
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Confidence indicator
        self.confidence_label = QLabel("Confidence: 0%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
        
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
        
        # Start video preview
        if self.camera_manager:
            self.camera_preview_timer = QTimer(self)
            self.camera_preview_timer.timeout.connect(self.update_preview)
            self.camera_preview_timer.start(33)  # ~30 FPS
        else:
            # Show placeholder if no camera manager
            self.show_placeholder_preview()
    
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
        self.landmarks = []
        self.best_landmarks = None
        self.best_confidence = 0.0
        
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.update_recording)
        self.record_timer.start(100)  # 100ms interval (10 ticks per second)
    
    def update_preview(self):
        """Update the camera preview"""
        if not self.camera_manager:
            return
        
        # Get current frame from camera manager
        ret, frame = self.camera_manager.get_frame()
        
        if not ret or frame is None:
            logger.error("Failed to get frame from camera")
            return
            
        # Process with gesture detector if available
        processed_frame = frame.copy()
        if self.gesture_detector and self.recording:
            try:
                # Process the frame to get landmarks
                if self.gesture_detector.detection_engine == "mediapipe":
                    # Use the direct processing method for display only
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.gesture_detector.hands.process(rgb_frame)
                    
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]  # Use first hand
                        
                        # Extract landmarks into a more usable format - without modifying MediaPipe objects
                        landmark_list = []
                        for lm in hand_landmarks.landmark:
                            landmark_list.append({
                                'x': lm.x,
                                'y': lm.y,
                                'z': lm.z
                            })
                        
                        # Add to landmarks collection
                        self.landmarks.append(landmark_list)
                        
                        # Calculate confidence based on visibility of landmarks
                        confidence = sum(1.0 for lm in landmark_list if lm['x'] > 0 and lm['y'] > 0) / len(landmark_list)
                        confidence = min(1.0, max(0.0, confidence))
                        
                        # Keep track of the best landmarks
                        if not self.best_landmarks or confidence > self.best_confidence:
                            self.best_confidence = confidence
                            self.best_landmarks = landmark_list
                            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
                        
                        # Draw landmarks on the frame (using MediaPipe's drawing utilities)
                        self.gesture_detector.mp_drawing.draw_landmarks(
                            processed_frame,
                            hand_landmarks,
                            self.gesture_detector.mp_hands.HAND_CONNECTIONS,
                            self.gesture_detector.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.gesture_detector.mp_drawing_styles.get_default_hand_connections_style()
                        )
                
            except Exception as e:
                logger.error(f"Error in gesture detection during recording: {e}")
        
        # Convert frame to QPixmap for display
        h, w, ch = processed_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.preview_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
    
    def show_placeholder_preview(self):
        """Show a placeholder preview when no camera is available"""
        # Create a placeholder image
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "No Camera Available",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Convert to QPixmap
        h, w, ch = placeholder.shape
        bytes_per_line = ch * w
        qt_image = QImage(placeholder.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.preview_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def update_recording(self):
        """Update the recording progress"""
        self.recording_time += 0.1
        self.progress_bar.setValue(int(self.recording_time * 10))
        
        # Get the current frame if camera is available
        if self.camera_manager:
            ret, frame = self.camera_manager.get_frame()
            if ret:
                self.frames.append(frame)
        
        # Check if recording is complete
        if self.recording_time >= self.max_recording_time:
            self.finish_recording()
    
    def finish_recording(self):
        """Finish recording the gesture"""
        self.recording = False
        if hasattr(self, 'record_timer'):
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
    
    def get_best_landmarks(self):
        """Get the best landmarks from the recording"""
        return self.best_landmarks
    
    def get_recorded_frames(self):
        """Get the recorded frames"""
        return self.frames
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop all timers
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
            
        if hasattr(self, 'record_timer') and self.record_timer.isActive():
            self.record_timer.stop()
            
        if hasattr(self, 'camera_preview_timer') and self.camera_preview_timer.isActive():
            self.camera_preview_timer.stop()
            
        event.accept()


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
        
        # Create gesture detector if possible
        self.gesture_detector = None
        if GestureDetector and camera_manager:
            try:
                self.gesture_detector = GestureDetector(self.config)
                logger.info("Created gesture detector for training interface")
            except Exception as e:
                logger.error(f"Failed to create gesture detector: {e}")
                
        # Create gesture classifier
        self.gesture_classifier = None
        if GestureClassifier:
            try:
                self.gesture_classifier = GestureClassifier(self.config)
                logger.info("Created gesture classifier for training interface")
            except Exception as e:
                logger.error(f"Failed to create gesture classifier: {e}")
        
        # Data storage
        self.gestures = {}  # Dict of gesture_name -> dict of info and samples
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
        
        self.quick_sample_button = QPushButton("Quick 5→100 Samples")
        self.quick_sample_button.setToolTip("Record just 5 samples and automatically generate 100 total variations")
        self.quick_sample_button.clicked.connect(self._quick_sample_with_augmentation)
        self.quick_sample_button.setEnabled(False)
        sample_buttons_layout.addWidget(self.quick_sample_button)
        
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
        
        # Model type selection
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("K-Nearest Neighbors", "knn")
        self.model_type_combo.addItem("Neural Network", "nn")
        training_layout.addRow("Model type:", self.model_type_combo)
        
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
        if not self.gesture_classifier:
            logger.warning("Cannot load gestures - no gesture classifier available")
            return
            
        try:
            # Get gestures from the classifier
            gesture_info = self.gesture_classifier.get_all_gestures()
            
            if gesture_info:
                # Convert from classifier format to our format
                for gesture_name, info in gesture_info.items():
                    self.gestures[gesture_name] = {
                        "samples": [],  # We don't load actual samples in the UI
                        "sample_count": info.get("sample_count", 0),
                        "trained": info.get("trained", False)
                    }
                
                # Populate the gesture list
                self.gesture_list.clear()
                for gesture_name in self.gestures.keys():
                    self.gesture_list.addItem(gesture_name)
                
                logger.info(f"Loaded {len(self.gestures)} gestures from classifier")
            else:
                logger.info("No existing gestures found")
            
        except Exception as e:
            logger.error(f"Error loading gestures: {e}")
            QMessageBox.warning(
                self,
                "Load Error",
                f"Failed to load gestures: {str(e)}"
            )
    
    def _save_gestures(self):
        """Save the current gestures configuration"""
        # We don't need to save anything here as the actual gesture data
        # is saved by the classifier when samples are added
        return True
    
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
                "sample_count": 0,
                "trained": False
            }
            
            # Select the new gesture
            items = self.gesture_list.findItems(gesture_name, Qt.MatchExactly)
            if items:
                self.gesture_list.setCurrentItem(items[0])
    
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
            
            # Not implemented yet - we'd need to rename in the classifier data
            QMessageBox.information(
                self,
                "Not Implemented",
                "Renaming gestures is not yet implemented."
            )
            
            # Update the list item just for UI purposes
            current_item.setText(new_name)
    
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
            # Delete from classifier
            if self.gesture_classifier:
                success = self.gesture_classifier.delete_gesture(gesture_name)
                if not success:
                    QMessageBox.warning(
                        self,
                        "Delete Failed",
                        f"Failed to delete gesture '{gesture_name}' from classifier data."
                    )
                    return
            
            # Remove from the dictionary
            if gesture_name in self.gestures:
                del self.gestures[gesture_name]
            
            # Remove from the list
            row = self.gesture_list.row(current_item)
            self.gesture_list.takeItem(row)
            
            # Update UI
            self._update_gesture_details(None)
    
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
        sample_count = self.gestures[gesture_name].get("sample_count", 0)
        sample_number = sample_count + 1
        
        # Show the recording dialog
        dialog = GestureRecordDialog(
            gesture_name, 
            sample_number, 
            self.camera_manager, 
            self.gesture_detector,
            self
        )
        
        if dialog.exec_() == QDialog.Accepted:
            # Get the best landmarks from the recording
            landmarks = dialog.get_best_landmarks()
            
            if not landmarks:
                QMessageBox.warning(
                    self,
                    "Record Failed",
                    "No valid hand landmarks were detected during recording."
                )
                return
            
            # Add the sample to the classifier
            if self.gesture_classifier:
                success = self.gesture_classifier.add_gesture_sample(gesture_name, landmarks)
                
                if success:
                    # Augment the sample
                    augmented_samples = GestureAugmenter.generate_augmented_samples([landmarks])
                    for aug_sample in augmented_samples:
                        self.gesture_classifier.add_gesture_sample(gesture_name, aug_sample)
                    
                    # Update local tracking
                    self.gestures[gesture_name]["sample_count"] = sample_count + len(augmented_samples)
                    self.gestures[gesture_name]["trained"] = False
                    
                    # Update the UI
                    self._update_gesture_details(current_item)
                    
                    QMessageBox.information(
                        self,
                        "Sample Recorded",
                        f"Sample {sample_number} for '{gesture_name}' has been recorded and augmented."
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Record Failed",
                        f"Failed to save sample for gesture '{gesture_name}'."
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Classifier Unavailable",
                    "Gesture classifier is not available, cannot save sample."
                )
    
    @pyqtSlot()
    def _quick_sample_with_augmentation(self):
        """Record 5 samples and automatically generate 100 total variations"""
        current_item = self.gesture_list.currentItem()
        
        if not current_item:
            return
        
        gesture_name = current_item.text()
        
        if gesture_name not in self.gestures:
            return
        
        # Get the current number of samples
        sample_count = self.gestures[gesture_name].get("sample_count", 0)
        
        # Show the recording dialog for 5 samples
        for sample_number in range(sample_count + 1, sample_count + 6):
            dialog = GestureRecordDialog(
                gesture_name, 
                sample_number, 
                self.camera_manager, 
                self.gesture_detector,
                self
            )
            
            if dialog.exec_() == QDialog.Accepted:
                # Get the best landmarks from the recording
                landmarks = dialog.get_best_landmarks()
                
                if not landmarks:
                    QMessageBox.warning(
                        self,
                        "Record Failed",
                        "No valid hand landmarks were detected during recording."
                    )
                    return
                
                # Add the sample to the classifier
                if self.gesture_classifier:
                    success = self.gesture_classifier.add_gesture_sample(gesture_name, landmarks)
                    
                    if success:
                        # Augment the sample
                        augmented_samples = GestureAugmenter.generate_augmented_samples([landmarks])
                        for aug_sample in augmented_samples:
                            self.gesture_classifier.add_gesture_sample(gesture_name, aug_sample)
                        
                        # Update local tracking
                        self.gestures[gesture_name]["sample_count"] = sample_count + len(augmented_samples)
                        self.gestures[gesture_name]["trained"] = False
                        
                        # Update the UI
                        self._update_gesture_details(current_item)
                        
                        QMessageBox.information(
                            self,
                            "Sample Recorded",
                            f"Sample {sample_number} for '{gesture_name}' has been recorded and augmented."
                        )
                    else:
                        QMessageBox.warning(
                            self,
                            "Record Failed",
                            f"Failed to save sample for gesture '{gesture_name}'."
                        )
                else:
                    QMessageBox.warning(
                        self,
                        "Classifier Unavailable",
                        "Gesture classifier is not available, cannot save sample."
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
            # Not implemented yet - we'd need to delete samples from classifier data
            QMessageBox.information(
                self,
                "Not Implemented",
                "Removing samples is not yet implemented."
            )
            
            # Mark as not trained for UI purposes
            self.gestures[gesture_name]["trained"] = False
            
            # Update UI
            self._update_gesture_details(current_item)
    
    @pyqtSlot()
    def _train_model(self):
        """Train the gesture recognition model"""
        # Check if we have at least 2 gestures with samples
        trainable_gestures = 0
        total_samples = 0
        
        for gesture_name, gesture_data in self.gestures.items():
            if gesture_data.get("sample_count", 0) > 0:
                trainable_gestures += 1
                total_samples += gesture_data.get("sample_count", 0)
        
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
        model_type_idx = self.model_type_combo.currentIndex()
        model_type = self.model_type_combo.itemData(model_type_idx)
        
        # Set model type in classifier if available
        if self.gesture_classifier:
            self.gesture_classifier.model_type = model_type
        
        # Start the training process
        self.training_status.setText("Training in progress...")
        self.training_progress.setValue(0)
        self.train_button.setEnabled(False)
        
        # Train the model
        success = False
        if self.gesture_classifier:
            try:
                # Process events to keep UI responsive
                QApplication.processEvents()
                
                # Perform the actual training
                success = self.gesture_classifier.train_model()
                
                # Show final progress
                self.training_progress.setValue(100)
                
                if success:
                    # Mark all gestures with samples as trained
                    for gesture_name in self.gestures:
                        if self.gestures[gesture_name].get("sample_count", 0) > 0:
                            self.gestures[gesture_name]["trained"] = True
                    
                    # Update the UI
                    self.train_button.setEnabled(True)
                    self.training_status.setText(f"Training complete! Model type: {model_type}")
                    
                    # Emit signal that model has been trained
                    self.model_trained.emit(self.current_profile)
                    
                    QMessageBox.information(
                        self,
                        "Training Complete",
                        "The gesture recognition model has been trained successfully."
                    )
                else:
                    self.training_status.setText("Training failed!")
                    QMessageBox.warning(
                        self,
                        "Training Failed",
                        "Failed to train the gesture recognition model."
                    )
                    self.train_button.setEnabled(True)
                    
            except Exception as e:
                logger.error(f"Error training model: {e}")
                self.training_status.setText(f"Training error: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Training Error",
                    f"An error occurred during training: {str(e)}"
                )
                self.train_button.setEnabled(True)
        else:
            QMessageBox.warning(
                self,
                "Classifier Unavailable",
                "Gesture classifier is not available, cannot train model."
            )
            self.training_status.setText("Classifier not available")
            self.train_button.setEnabled(True)
    
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
                # Load the actual gesture data from the classifier
                if self.gesture_classifier:
                    gesture_data = self.gesture_classifier._load_gesture_data()
                    
                    if not gesture_data:
                        QMessageBox.warning(
                            self,
                            "Export Failed",
                            "No gesture data available for export."
                        )
                        return
                    
                    # Export the data
                    with open(file_path, 'w') as f:
                        json.dump(gesture_data, f, indent=2)
                    
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Gestures have been exported to {file_path}\n\nTotal gestures exported: {len(gesture_data)}"
                    )
                    
                    # Log what was exported
                    gesture_names = list(gesture_data.keys())
                    logger.info(f"Exported gestures: {gesture_names}")
                else:
                    QMessageBox.warning(
                        self,
                        "Classifier Unavailable",
                        "Gesture classifier is not available, cannot export gestures."
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
                # Load the gesture data from the file
                with open(file_path, 'r') as f:
                    imported_gestures = json.load(f)
                
                if not imported_gestures:
                    QMessageBox.warning(
                        self,
                        "Import Failed",
                        "No valid gesture data found in the file."
                    )
                    return
                
                if not self.gesture_classifier:
                    QMessageBox.warning(
                        self,
                        "Classifier Unavailable",
                        "Gesture classifier is not available, cannot import gestures."
                    )
                    return
                
                # Show import details
                details = f"Found {len(imported_gestures)} gestures in the file:\n"
                for name, data in imported_gestures.items():
                    sample_count = len(data.get("samples", []))
                    details += f"- {name}: {sample_count} samples\n"
                
                QMessageBox.information(
                    self,
                    "Import Details",
                    details
                )
                
                # Load existing gesture data
                existing_data = self.gesture_classifier._load_gesture_data()
                
                # Check if we should merge or replace
                if existing_data:
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
                            if name in existing_data:
                                # Create a new unique name
                                base_name = name
                                counter = 1
                                new_name = f"{base_name} (imported {counter})"
                                
                                while new_name in existing_data:
                                    counter += 1
                                    new_name = f"{base_name} (imported {counter})"
                                
                                existing_data[new_name] = data
                                logger.info(f"Renamed imported gesture '{name}' to '{new_name}'")
                            else:
                                existing_data[name] = data
                        
                        logger.info(f"Merged {len(imported_gestures)} imported gestures with existing data")
                    else:
                        # Replace gestures
                        existing_data = imported_gestures
                        logger.info(f"Replaced existing gestures with {len(imported_gestures)} imported gestures")
                else:
                    # No existing gestures, just use the imported ones
                    existing_data = imported_gestures
                    logger.info(f"Imported {len(imported_gestures)} gestures to empty dataset")
                
                # Save the merged/replaced gesture data
                self.gesture_classifier._save_gesture_data(existing_data)
                
                # Reload the gesture list
                self._load_existing_gestures()
                
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
        
        # Switch to the new profile
        self.current_profile = profile_name
        
        # Load gestures for the new profile
        self._load_existing_gestures()
        
        # Update the UI
        self._update_gesture_details(None)
        
        # Update config safely
        if "actions" in self.config:
            if isinstance(self.config["actions"], dict):
                self.config["actions"]["default_profile"] = profile_name
                self.config_manager.save_config() # Save the updated config
            else:
                logger.warning("Config structure issue: 'actions' is not a dictionary. Cannot update default profile.")
        else:
            logger.warning("Config structure issue: 'actions' key missing. Cannot update default profile.")
    
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
            self.quick_sample_button.setEnabled(False)
            self.remove_samples_button.setEnabled(False)
            self.rename_gesture_button.setEnabled(False)
            self.delete_gesture_button.setEnabled(False)
            self.train_button.setEnabled(False)
            return
        
        gesture_name = item.text()
        
        if gesture_name not in self.gestures:
            return
        
        gesture_data = self.gestures[gesture_name]
        trained = gesture_data.get("trained", False)
        sample_count = gesture_data.get("sample_count", 0)
        
        # Update sample count
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
        self.quick_sample_button.setEnabled(True)
        self.remove_samples_button.setEnabled(sample_count > 0)
        self.rename_gesture_button.setEnabled(True)
        self.delete_gesture_button.setEnabled(True)
        
        # Enable train button if we have enough gestures with samples
        trainable_gestures = 0
        for g_name, g_data in self.gestures.items():
            if g_data.get("sample_count", 0) > 0:
                trainable_gestures += 1
        
        self.train_button.setEnabled(trainable_gestures >= 2)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Nothing to save specifically here since we save immediately when changes are made
        event.accept()