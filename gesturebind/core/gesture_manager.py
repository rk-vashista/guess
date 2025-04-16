"""
GestureManager module for GestureBind.

This module provides the central integration point for gesture detection and action mapping.
It coordinates between camera input, gesture detection, and action execution.
"""

import logging
import threading
import time
from typing import List, Tuple, Dict, Any, Optional, Callable

# Fix imports to use direct imports rather than namespace package imports
from core.camera_manager import CameraManager
from core.gesture_detector import GestureDetector
from core.action_mapper import ActionMapper
from models.gesture_classifier import GestureClassifier
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GestureManager:
    """
    Manages the integration between camera input, gesture detection, and action execution.
    Acts as a central coordinator for the gesture recognition system.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the gesture manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.config = config_manager.load_config()
        
        self.camera_manager = None
        self.gesture_detector = None
        self.action_mapper = None
        self.gesture_classifier = None
        
        self.processing_thread = None
        self.is_processing = False
        self.detection_paused = False
        
        self.latest_frame = None
        self.latest_processed_frame = None
        self.latest_gesture = (None, 0.0, None)  # (name, confidence, handedness)
        
        self.observers = []
        
        self._cooldown_timer = 0
        self._cooldown_period = self.config.get("detection", {}).get("cooldown_ms", 1000) / 1000.0
        self._confidence_threshold = self.config.get("detection", {}).get("confidence_threshold", 0.7)
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all required components"""
        try:
            # Initialize camera manager
            self.camera_manager = CameraManager(self.config)
            
            # Initialize gesture classifier
            self.gesture_classifier = GestureClassifier(self.config)
            
            # Initialize gesture detector with classifier
            self.gesture_detector = GestureDetector(self.config)
            self.gesture_detector.set_classifier(self.gesture_classifier)
            
            # Initialize action mapper
            self.action_mapper = ActionMapper(self.config)
            
            # Load default profile
            default_profile = self.config.get("actions", {}).get("default_profile", "default")
            self.action_mapper.set_profile(default_profile)
            
            logger.info("GestureManager components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing GestureManager components: {e}")
            raise
    
    def get_camera_status(self):
        """
        Get the current status of the camera.
        
        Returns:
            dict: Camera status information
        """
        if not hasattr(self, 'camera_manager') or self.camera_manager is None:
            return {
                "running": False,
                "has_error": True,
                "error_message": "Camera manager not initialized"
            }
        
        return self.camera_manager.get_status()
    
    def register_observer(self, observer):
        """
        Register an observer to receive gesture events.
        
        Args:
            observer: Object with on_frame_update and on_action_executed methods
        """
        self.observers.append(observer)
    
    def unregister_observer(self, observer):
        """
        Remove an observer.
        
        Args:
            observer: Observer to remove
        """
        if observer in self.observers:
            self.observers.remove(observer)
    
    def start_processing(self) -> bool:
        """
        Start gesture processing.
        
        Returns:
            bool: True if processing started successfully
        """
        if self.is_processing:
            logger.warning("Gesture processing already started")
            return True
        
        try:
            # Check camera availability before starting
            camera_available, error_msg = self.camera_manager.check_camera_availability()
            if not camera_available:
                logger.error(f"Cannot start processing: {error_msg}")
                return False
                
            # Start camera capture directly
            if not self.camera_manager.start_capture():
                camera_status = self.camera_manager.get_status()
                logger.error(f"Failed to start camera capture: {camera_status.get('error_message', 'Unknown error')}")
                return False
            
            # Start processing thread
            self.is_processing = True
            self.detection_paused = False
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info("Gesture processing started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting gesture processing: {e}")
            self.is_processing = False
            return False
    
    def stop_processing(self):
        """Stop gesture processing"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Stop camera capture
        if self.camera_manager:
            self.camera_manager.stop_capture()
        
        logger.info("Gesture processing stopped")
    
    def pause_detection(self):
        """Pause gesture detection without stopping camera"""
        self.detection_paused = True
        logger.info("Gesture detection paused")
    
    def resume_detection(self):
        """Resume gesture detection"""
        self.detection_paused = False
        logger.info("Gesture detection resumed")
    
    def set_profile(self, profile_name: str) -> bool:
        """
        Set the active gesture profile.
        
        Args:
            profile_name: Name of the profile to set
            
        Returns:
            bool: True if profile was set successfully
        """
        if self.action_mapper:
            success = self.action_mapper.set_profile(profile_name)
            if success:
                logger.info(f"Changed to profile: {profile_name}")
            else:
                logger.warning(f"Failed to change to profile: {profile_name}")
            return success
        return False
    
    def get_latest_processed_frame(self):
        """
        Get the latest processed frame with gesture visualizations.
        
        Returns:
            numpy.ndarray: Latest processed frame or None
        """
        return self.latest_processed_frame
    
    def get_latest_gesture_info(self) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Get the latest detected gesture information.
        
        Returns:
            Tuple: (gesture_name, confidence, handedness) 
        """
        return self.latest_gesture
    
    def _processing_loop(self):
        """Main processing loop for gesture detection and action execution"""
        fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        last_fps_update = time.time()
        frames_count = 0
        current_fps = 0
        
        while self.is_processing:
            # Get frame from camera
            ret, frame = self.camera_manager.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # Store latest frame
            self.latest_frame = frame.copy()
            
            # Skip gesture processing if detection is paused
            if self.detection_paused:
                # Still draw landmarks but don't process gestures
                self.latest_processed_frame = self.gesture_detector.draw_landmarks(frame)
                self._notify_frame_update()
                continue
            
            # Process frame for gesture detection
            gesture_name, confidence, processed_frame = self.gesture_detector.detect_gesture(frame)
            
            # Store processed frame
            self.latest_processed_frame = processed_frame
            
            # Check if we detected a gesture with sufficient confidence
            if gesture_name and confidence >= self._confidence_threshold:
                # Get handedness from gesture detector if available
                handedness = self.gesture_detector.get_handedness() if hasattr(self.gesture_detector, 'get_handedness') else None
                
                # Check cooldown
                current_time = time.time()
                if current_time - self._cooldown_timer >= self._cooldown_period:
                    # Update latest gesture info
                    self.latest_gesture = (gesture_name, confidence, handedness)
                    
                    # Execute action if mapped
                    if self.action_mapper:
                        action_type, action_data, action_description = self.action_mapper.get_action_for_gesture(gesture_name)
                        if action_type:
                            # Execute the action
                            success = self.action_mapper.execute_action(action_type, action_data)
                            
                            if success:
                                # Reset cooldown timer
                                self._cooldown_timer = current_time
                                
                                # Notify observers about the action execution
                                self._notify_action_executed(gesture_name, f"{action_type}:{action_data}", action_description)
            
            # Calculate FPS
            frames_count += 1
            current_time = time.time()
            if current_time - last_fps_update >= fps_update_interval:
                current_fps = frames_count / (current_time - last_fps_update)
                last_fps_update = current_time
                frames_count = 0
                
                # Log FPS (instead of updating stats)
                logger.debug(f"Processing at {current_fps:.1f} FPS")
            
            # Notify frame update
            self._notify_frame_update()
            
            # Limit processing rate if needed
            target_fps = self.config.get("detection", {}).get("target_fps", 30)
            if target_fps > 0:
                sleep_time = 1.0 / target_fps - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    def _notify_frame_update(self):
        """Notify observers about frame update"""
        for observer in self.observers:
            if hasattr(observer, 'on_frame_update'):
                observer.on_frame_update()
    
    def _notify_action_executed(self, gesture: str, action: Any, description: str):
        """
        Notify observers about action execution.
        
        Args:
            gesture: Name of the detected gesture
            action: Action object/data
            description: Human-readable description of the action
        """
        for observer in self.observers:
            if hasattr(observer, 'on_action_executed'):
                observer.on_action_executed(gesture, action, description)