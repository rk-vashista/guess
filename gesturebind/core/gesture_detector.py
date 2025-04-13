"""
Gesture detector module for GestureBind

This module handles detection and classification of gestures from camera frames.
"""

import os
import time
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class GestureDetector:
    """
    Detects and classifies hand gestures from video frames.
    """
    
    def __init__(self, config):
        """
        Initialize the gesture detector.
        
        Args:
            config (dict): Application configuration dictionary
        """
        self.config = config
        
        # Detection settings
        self.engine = config.get("detection", {}).get("engine", "mediapipe")
        self.confidence_threshold = config.get("detection", {}).get("confidence_threshold", 0.7)
        self.cooldown_ms = config.get("detection", {}).get("cooldown_ms", 500)
        self.min_detection_duration_ms = config.get("detection", {}).get("min_detection_duration_ms", 200)
        self.use_landmark_smoothing = config.get("advanced", {}).get("landmark_smoothing", True)
        
        # State variables
        self.last_detection_time = 0
        self.current_gesture = None
        self.current_gesture_start_time = 0
        self.previous_landmarks = None
        
        # Initialize detection engine
        self._initialize_engine()
        
        logger.info(f"Gesture detector initialized using {self.engine} engine")
    
    def _initialize_engine(self):
        """Initialize the selected detection engine"""
        if self.engine == "mediapipe":
            self._initialize_mediapipe()
        elif self.engine == "yolov8":
            self._initialize_yolov8()
        else:
            logger.warning(f"Unsupported engine: {self.engine}, falling back to mock engine")
            self._initialize_mock_engine()
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe Hands for gesture detection"""
        try:
            import mediapipe as mp
            
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Initialize hands with custom parameters
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self.confidence_threshold - 0.2,  # Lower threshold for detection
                min_tracking_confidence=self.confidence_threshold - 0.1    # Slightly higher for tracking
            )
            
            logger.info("MediaPipe Hands initialized successfully")
            self.engine_initialized = True
            
        except ImportError:
            logger.error("MediaPipe import failed, falling back to mock engine")
            self._initialize_mock_engine()
        except Exception as e:
            logger.error(f"Error initializing MediaPipe: {e}")
            self._initialize_mock_engine()
    
    def _initialize_yolov8(self):
        """Initialize YOLOv8 for gesture detection"""
        try:
            # In a real implementation, this would load the YOLOv8 model
            # and set up the inference pipeline
            
            # For now, we'll just log that it's not fully implemented
            logger.warning("YOLOv8 engine not fully implemented, falling back to mock engine")
            self._initialize_mock_engine()
            
        except Exception as e:
            logger.error(f"Error initializing YOLOv8: {e}")
            self._initialize_mock_engine()
    
    def _initialize_mock_engine(self):
        """Initialize a mock engine for testing without ML dependencies"""
        logger.info("Using mock gesture detection engine")
        self.engine = "mock"
        self.engine_initialized = True
    
    def update_config(self, config):
        """
        Update detector configuration.
        
        Args:
            config (dict): New configuration dictionary
        """
        # Update configuration
        self.config = config
        
        # Update detection settings
        self.confidence_threshold = config.get("detection", {}).get("confidence_threshold", 0.7)
        self.cooldown_ms = config.get("detection", {}).get("cooldown_ms", 500)
        self.min_detection_duration_ms = config.get("detection", {}).get("min_detection_duration_ms", 200)
        self.use_landmark_smoothing = config.get("advanced", {}).get("landmark_smoothing", True)
        
        # Check if engine changed, if so reinitialize
        new_engine = config.get("detection", {}).get("engine", "mediapipe")
        if new_engine != self.engine:
            self.engine = new_engine
            self._initialize_engine()
        
        logger.info(f"Gesture detector configuration updated")
    
    def _smooth_landmarks(self, landmarks, alpha=0.5):
        """
        Apply smoothing to landmarks to reduce jitter.
        
        Args:
            landmarks: Detected landmarks
            alpha: Smoothing factor (0-1, lower = more smoothing)
            
        Returns:
            Smoothed landmarks
        """
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return landmarks
        
        # Apply exponential smoothing
        smoothed = []
        for i, landmark in enumerate(landmarks):
            if i < len(self.previous_landmarks):
                prev = self.previous_landmarks[i]
                # Interpolate between previous and current
                smoothed_landmark = {
                    'x': alpha * landmark['x'] + (1 - alpha) * prev['x'],
                    'y': alpha * landmark['y'] + (1 - alpha) * prev['y'],
                    'z': alpha * landmark['z'] + (1 - alpha) * prev['z']
                }
                smoothed.append(smoothed_landmark)
            else:
                smoothed.append(landmark)
        
        self.previous_landmarks = smoothed
        return smoothed
    
    def _process_with_mediapipe(self, frame):
        """
        Process frame using MediaPipe Hands.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (gesture_name, confidence, processed_frame)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Copy frame for visualization
        processed_frame = frame.copy()
        
        gesture_name = None
        confidence = 0.0
        
        # Check if hands detected
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get handedness (left/right)
                handedness = hand_info.classification[0].label
                
                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                
                # Apply smoothing if enabled
                if self.use_landmark_smoothing:
                    landmarks = self._smooth_landmarks(landmarks)
                
                # Classify gesture based on landmarks
                gesture, conf = self._classify_gesture(landmarks, handedness)
                
                # Keep the highest confidence gesture
                if conf > confidence and conf >= self.confidence_threshold:
                    gesture_name = gesture
                    confidence = conf
                
                # Draw landmarks on the processed frame
                self.mp_drawing.draw_landmarks(
                    processed_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Add label with gesture name and confidence
                if gesture:
                    h, w, _ = processed_frame.shape
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    wrist_y = int(hand_landmarks.landmark[0].y * h)
                    
                    # Draw label background
                    label_text = f"{gesture} ({conf:.1%})"
                    (text_width, text_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    cv2.rectangle(
                        processed_frame,
                        (wrist_x, wrist_y - text_height - 10),
                        (wrist_x + text_width + 10, wrist_y),
                        (0, 0, 0), -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        processed_frame, 
                        label_text, 
                        (wrist_x + 5, wrist_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 1
                    )
        
        return gesture_name, confidence, processed_frame
    
    def _process_with_yolov8(self, frame):
        """
        Process frame using YOLOv8.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (gesture_name, confidence, processed_frame)
        """
        # In a real implementation, this would run the YOLOv8 model
        # For now, we'll just return a mock result
        processed_frame = frame.copy()
        
        # Add "not implemented" text
        h, w, _ = processed_frame.shape
        text = "YOLOv8 not fully implemented"
        cv2.putText(
            processed_frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
        
        return None, 0.0, processed_frame
    
    def _process_with_mock(self, frame):
        """
        Process frame using mock detection for testing.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (gesture_name, confidence, processed_frame)
        """
        processed_frame = frame.copy()
        
        # Add mock detection text
        cv2.putText(
            processed_frame,
            "Mock Detection Active",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Simple mock detection based on frame brightness in the center
        h, w, _ = frame.shape
        center_roi = frame[h//3:2*h//3, w//3:2*w//3]
        
        # Calculate average brightness in the center
        gray_center = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = gray_center.mean()
        
        # Simulated gesture detection based on brightness
        gesture_name = None
        confidence = 0.0
        
        if avg_brightness > 150:
            gesture_name = "open_palm"
            confidence = 0.8
            
            # Draw a rectangle in the center to indicate detection
            cv2.rectangle(
                processed_frame,
                (w//3, h//3),
                (2*w//3, 2*h//3),
                (0, 255, 0),
                2
            )
        elif avg_brightness > 100:
            gesture_name = "fist"
            confidence = 0.7
            
            # Draw a rectangle in the center to indicate detection
            cv2.rectangle(
                processed_frame,
                (w//3, h//3),
                (2*w//3, 2*h//3),
                (0, 200, 200),
                2
            )
        
        # Display current mock gesture if detected
        if gesture_name:
            cv2.putText(
                processed_frame,
                f"{gesture_name} ({confidence:.1%})",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
        
        return gesture_name, confidence, processed_frame
    
    def _classify_gesture(self, landmarks, handedness):
        """
        Classify the gesture based on hand landmarks.
        
        Args:
            landmarks: List of landmark coordinates
            handedness: "Left" or "Right"
            
        Returns:
            tuple: (gesture_name, confidence)
        """
        # In a real implementation, this would use a trained ML model
        # or a rule-based classifier to identify gestures
        
        # For this implementation, we'll use a simple rule-based approach
        # Basic gesture recognition using finger positions
        
        # Get fingertips and palm landmarks
        if len(landmarks) < 21:
            return None, 0.0
            
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Middle of palm for reference
        palm_center = {
            'x': landmarks[0]['x'],
            'y': landmarks[0]['y'],
            'z': landmarks[0]['z']
        }
        
        # Check if fingers are extended by comparing y coordinates
        # (this is a simplified approach and would need refinement in a real app)
        fingers_extended = [
            thumb_tip['x'] > wrist['x'] if handedness == "Right" else thumb_tip['x'] < wrist['x'],  # Thumb
            index_tip['y'] < palm_center['y'] - 0.05,  # Index
            middle_tip['y'] < palm_center['y'] - 0.05,  # Middle
            ring_tip['y'] < palm_center['y'] - 0.05,  # Ring
            pinky_tip['y'] < palm_center['y'] - 0.05   # Pinky
        ]
        
        # Classify gestures based on finger positions
        if sum(fingers_extended[1:]) >= 4:
            # All fingers except thumb are extended
            if not fingers_extended[0]:
                return "open_palm", 0.85
            else:
                return "open_hand", 0.8
        
        elif not any(fingers_extended):
            # No fingers extended
            return "fist", 0.9
        
        elif fingers_extended[1] and not any(fingers_extended[2:]):
            # Only index finger extended
            return "pointing", 0.85
        
        elif fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]):
            # Index and middle extended
            return "peace_sign", 0.8
        
        elif fingers_extended[0] and not any(fingers_extended[1:]):
            # Only thumb extended
            if thumb_tip['y'] < palm_center['y']:
                return "thumbs_up", 0.7
            else:
                return "thumbs_down", 0.7
        
        # If no clear gesture pattern is recognized
        return None, 0.3
    
    def detect_gesture(self, frame):
        """
        Process a frame and detect gestures.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (gesture_name, confidence, processed_frame)
                gesture_name: Name of detected gesture or None
                confidence: Detection confidence (0.0-1.0)
                processed_frame: Frame with visualization
        """
        # Check if we're in cooldown period
        current_time = time.time() * 1000  # Get current time in ms
        if (current_time - self.last_detection_time) < self.cooldown_ms:
            # In cooldown, return None for gesture
            return None, 0.0, frame.copy()
        
        # Process frame with appropriate engine
        if self.engine == "mediapipe":
            gesture_name, confidence, processed_frame = self._process_with_mediapipe(frame)
        elif self.engine == "yolov8":
            gesture_name, confidence, processed_frame = self._process_with_yolov8(frame)
        else:  # mock engine
            gesture_name, confidence, processed_frame = self._process_with_mock(frame)
        
        # Apply minimum detection duration logic
        if gesture_name:
            if self.current_gesture != gesture_name:
                # New gesture detected, start timer
                self.current_gesture = gesture_name
                self.current_gesture_start_time = current_time
                
                # Not yet held long enough
                return None, 0.0, processed_frame
            else:
                # Same gesture, check if held long enough
                if (current_time - self.current_gesture_start_time) >= self.min_detection_duration_ms:
                    # Gesture held long enough, trigger detection
                    self.last_detection_time = current_time
                    return gesture_name, confidence, processed_frame
                else:
                    # Not held long enough yet
                    return None, 0.0, processed_frame
        else:
            # No gesture detected
            self.current_gesture = None
            return None, 0.0, processed_frame
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        # Release MediaPipe resources if initialized
        if hasattr(self, 'hands'):
            self.hands.close()