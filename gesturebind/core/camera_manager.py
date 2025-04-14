"""
Camera manager module for GestureBind

This module handles video capture and frame processing.
"""

import cv2
import threading
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Manages camera capture and provides frames for gesture detection.
    """
    
    def __init__(self, config):
        """
        Initialize the camera manager with the provided configuration.
        
        Args:
            config (dict): Application configuration
        """
        self.config = config
        camera_config = config.get("camera", {})
        
        # Set camera properties
        self.device_id = camera_config.get("device_id", 0)
        self.width = camera_config.get("width", 640)
        self.height = camera_config.get("height", 480)
        self.fps = camera_config.get("fps", 30)
        self.flip_horizontal = camera_config.get("flip_horizontal", False)
        self.flip_vertical = camera_config.get("flip_vertical", False)
        
        # Initialize capture variables
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.current_frame = None
        
        logger.info(f"Camera manager initialized for device {self.device_id}")
    
    def start_capture(self):
        """
        Start the camera capture process.
        
        Returns:
            bool: Success or failure
        """
        if self.running:
            logger.warning("Camera capture already running")
            return True
        
        try:
            # Try to find a working camera if the default one doesn't work
            camera_indices_to_try = [self.device_id]
            
            # Add some common indices to try if the default fails
            if self.device_id != 0:
                camera_indices_to_try.append(0)  # Try default camera
            
            # Also try other common indices
            for idx in [1, 2, -1]:  # -1 is often used for default camera on some systems
                if idx not in camera_indices_to_try:
                    camera_indices_to_try.append(idx)
            
            # Try each camera index until one works
            success = False
            for cam_idx in camera_indices_to_try:
                logger.info(f"Trying to open camera with index {cam_idx}")
                
                # Initialize the capture
                self.cap = cv2.VideoCapture(cam_idx)
                
                # Check if camera opened successfully
                if not self.cap.isOpened():
                    logger.warning(f"Failed to open camera device {cam_idx}")
                    continue
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Read a test frame to verify capture works
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {cam_idx}")
                    self.cap.release()
                    continue
                
                # This camera works, update the device ID and break the loop
                self.device_id = cam_idx
                success = True
                break
            
            if not success:
                logger.error("Could not find any working camera")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False
                
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera {self.device_id} started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Start the capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_thread)
            self.thread.daemon = True
            self.thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera capture: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def stop_capture(self):
        """
        Stop the camera capture process.
        
        Returns:
            bool: Success or failure
        """
        if not self.running:
            logger.warning("Camera capture already stopped")
            return True
        
        try:
            # Stop the thread
            self.running = False
            if self.thread is not None:
                self.thread.join(timeout=1.0)
            
            # Release the camera
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            logger.info("Camera capture stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping camera capture: {e}")
            return False
    
    def _capture_thread(self):
        """Thread function for continuous frame capture"""
        try:
            while self.running:
                # Capture frame from camera
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    # Try to recover by reopening camera after short delay
                    time.sleep(0.1)
                    continue
                
                # Apply transformations if needed
                if self.flip_horizontal or self.flip_vertical:
                    flip_code = 1 if self.flip_horizontal and not self.flip_vertical else \
                              0 if not self.flip_horizontal and self.flip_vertical else -1
                    frame = cv2.flip(frame, flip_code)
                
                # Update the current frame
                with self.lock:
                    self.current_frame = frame
                
                # Sleep to maintain the requested frame rate
                time.sleep(1.0 / self.fps)
                
        except Exception as e:
            logger.error(f"Error in capture thread: {e}")
            self.running = False
    
    def get_frame(self):
        """
        Get the most recent camera frame.
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the camera frame
        """
        if not self.running or self.cap is None:
            return False, None
        
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
        
        return False, None
    
    def get_device_list(self):
        """
        Get a list of available camera devices.
        
        Returns:
            list: List of available camera device IDs
        """
        available_devices = []
        
        # Check first 10 camera indices to find working cameras
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_devices.append(i)
                cap.release()
        
        return available_devices
    
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        self.stop_capture()