"""
Gesture detector module for GestureBind

This module handles detection and classification of gestures from camera frames.
"""

import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# Attempt to import ML libraries, gracefully handle if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Import YOLOv8 if available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

class GestureDetector:
    """
    Detects and classifies hand gestures from video frames.
    """
    
    def __init__(self, config, camera_manager=None):
        """
        Initialize the gesture detector with configuration.
        
        Args:
            config (dict): Configuration dictionary
            camera_manager (CameraManager, optional): Camera manager instance
        """
        self.config = config
        self.camera_manager = camera_manager
        
        # Detection settings
        detection_config = config.get("detection", {})
        self.detection_engine = detection_config.get("engine", "mediapipe")
        self.confidence_threshold = detection_config.get("confidence_threshold", 0.7)
        self.cooldown_ms = detection_config.get("cooldown_ms", 500)
        self.min_detection_duration_ms = detection_config.get("min_detection_duration_ms", 200)
        
        # Preprocessing settings
        preprocessing_config = config.get("preprocessing", {})
        self.use_grayscale = preprocessing_config.get("use_grayscale", False)
        self.resize_frame = preprocessing_config.get("resize_frame", False)
        self.resize_width = preprocessing_config.get("resize_width", 320)
        self.resize_height = preprocessing_config.get("resize_height", 240)
        self.use_roi = preprocessing_config.get("use_roi", False)
        self.roi_x = preprocessing_config.get("roi_x", 100)
        self.roi_y = preprocessing_config.get("roi_y", 100)
        self.roi_width = preprocessing_config.get("roi_width", 440)
        self.roi_height = preprocessing_config.get("roi_height", 280)
        
        # Landmark smoothing
        smoothing_config = config.get("advanced", {})
        self.landmark_smoothing = smoothing_config.get("landmark_smoothing", True)
        self.smoothing_factor = smoothing_config.get("smoothing_factor", 0.5)
        self.smoothing_window = smoothing_config.get("smoothing_window", 3)
        
        # State tracking
        self.last_detection_time = 0
        self.current_gesture = None
        self.gesture_start_time = 0
        self.prev_landmarks = None
        self.landmark_history = []
        self.current_handedness = None
        
        # Initialize the detection engine
        self.engine = None
        self.engine_initialized = False
        self.classifier = None
        self._init_detection_engine()
    
    def _init_detection_engine(self):
        """Initialize the selected detection engine"""
        if self.detection_engine == "mediapipe":
            self._initialize_mediapipe()
        elif self.detection_engine == "yolov8":
            self._initialize_yolov8()
        else:
            self._initialize_mock_engine()
    
    def set_classifier(self, classifier):
        """
        Set the ML-based gesture classifier.
        
        Args:
            classifier: Trained gesture classifier instance
        """
        self.classifier = classifier
        logger.info("Gesture classifier set")
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe Hands for gesture detection"""
        try:
            if not MEDIAPIPE_AVAILABLE:
                logger.error("MediaPipe is not available. Please install it with 'pip install mediapipe'")
                self._initialize_mock_engine()
                return
            
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Configure MediaPipe Hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=self.confidence_threshold
            )
            
            self.engine = "mediapipe"
            self.engine_initialized = True
            logger.info("MediaPipe Hands initialized for gesture detection")
            
        except ImportError:
            logger.error("MediaPipe not available. Please install it with 'pip install mediapipe'")
            self._initialize_mock_engine()
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Hands: {e}")
            self._initialize_mock_engine()
    
    def _initialize_yolov8(self):
        """Initialize YOLOv8 for gesture detection"""
        try:
            if not YOLO_AVAILABLE:
                logger.error("YOLOv8 is not available. Please install it with 'pip install ultralytics'")
                self._initialize_mock_engine()
                return
            
            # Get model path from config
            model_path = self.config.get("detection", {}).get("yolo_model_path", "")
            if not model_path or not os.path.exists(model_path):
                logger.error(f"YOLOv8 model not found at: {model_path}")
                self._initialize_mock_engine()
                return
            
            # Load the YOLO model
            self.yolo_model = YOLO(model_path)
            self.engine = "yolov8"
            self.engine_initialized = True
            logger.info(f"YOLOv8 initialized with model: {model_path}")
            
        except ImportError:
            logger.error("YOLOv8 not available. Please install it with 'pip install ultralytics'")
            self._initialize_mock_engine()
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8: {e}")
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
        
        # Update preprocessing settings
        self.preprocess_config = config.get("preprocessing", {})
        self.use_grayscale = self.preprocess_config.get("use_grayscale", False)
        self.resize_frame = self.preprocess_config.get("resize_frame", False)
        self.resize_width = self.preprocess_config.get("resize_width", 640)
        self.resize_height = self.preprocess_config.get("resize_height", 480)
        self.use_roi = self.preprocess_config.get("use_roi", False)
        self.roi_x = self.preprocess_config.get("roi_x", 0)
        self.roi_y = self.preprocess_config.get("roi_y", 0)
        self.roi_width = self.preprocess_config.get("roi_width", 640)
        self.roi_height = self.preprocess_config.get("roi_height", 480)
        
        # Update smoothing parameters
        self.smoothing_factor = config.get("advanced", {}).get("smoothing_factor", 0.5)
        self.smoothing_window = config.get("advanced", {}).get("smoothing_window", 3)
        
        # Check if engine changed, if so reinitialize
        new_engine = config.get("detection", {}).get("engine", "mediapipe")
        if new_engine != self.engine:
            self.detection_engine = new_engine
            self._init_detection_engine()
        
        logger.info(f"Gesture detector configuration updated")

    def _preprocess_frame(self, frame):
        """
        Apply preprocessing steps to the input frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Preprocessed frame
        """
        # Make a copy to avoid modifying the original
        processed = frame.copy()
        
        # Apply ROI cropping if enabled
        if self.use_roi:
            try:
                h, w = processed.shape[:2]
                x = min(max(0, self.roi_x), w)
                y = min(max(0, self.roi_y), h)
                width = min(self.roi_width, w - x)
                height = min(self.roi_height, h - y)
                processed = processed[y:y+height, x:x+width]
            except Exception as e:
                logger.error(f"Error applying ROI: {e}")
        
        # Resize if enabled
        if self.resize_frame:
            try:
                processed = cv2.resize(
                    processed, 
                    (self.resize_width, self.resize_height), 
                    interpolation=cv2.INTER_AREA
                )
            except Exception as e:
                logger.error(f"Error resizing frame: {e}")
            
        # Convert to grayscale if enabled (but only for internal processing)
        if self.use_grayscale:
            try:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                # Convert back to BGR for compatibility with detection engines
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                logger.error(f"Error converting to grayscale: {e}")
        
        return processed
    
    def _smooth_landmarks(self, landmarks, alpha=None):
        """
        Apply smoothing to hand landmarks to reduce jitter
        
        Args:
            landmarks: MediaPipe hand landmarks
            alpha: Smoothing factor (0-1, lower = more smoothing)
            
        Returns:
            Smoothed landmarks
        """
        if not self.landmark_smoothing or landmarks is None:
            return landmarks
            
        if alpha is None:
            alpha = self.smoothing_factor
            
        # Add current landmarks to history
        self.landmark_history.append(landmarks)
        
        # Keep history limited to window size
        if len(self.landmark_history) > self.smoothing_window:
            self.landmark_history.pop(0)
            
        # Apply exponential moving average
        smoothed = landmarks
        
        if self.prev_landmarks is not None:
            # Create a copy to avoid modifying original landmarks
            smoothed = []
            for i, lm in enumerate(landmarks):
                if i < len(self.prev_landmarks):
                    # Smooth x, y, z coordinates
                    smoothed_x = alpha * lm.x + (1 - alpha) * self.prev_landmarks[i].x
                    smoothed_y = alpha * lm.y + (1 - alpha) * self.prev_landmarks[i].y
                    smoothed_z = alpha * lm.z + (1 - alpha) * self.prev_landmarks[i].z
                    
                    # Create new landmark with smoothed values
                    smoothed_lm = type(lm)()
                    smoothed_lm.x = smoothed_x
                    smoothed_lm.y = smoothed_y
                    smoothed_lm.z = smoothed_z
                    smoothed.append(smoothed_lm)
                else:
                    smoothed.append(lm)
                    
        # Update previous landmarks
        self.prev_landmarks = smoothed
            
        return smoothed
    
    def _process_with_mediapipe(self, frame):
        """
        Process a frame with MediaPipe Hands
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (gesture_name, confidence, processed_frame)
        """
        if not self.engine_initialized or self.engine != "mediapipe":
            return None, 0, frame
            
        # Process the frame
        try:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(frame_rgb)
            
            # Create a copy for visualization
            processed_frame = frame.copy()
            
            # Check if we have hand landmarks
            if results.multi_hand_landmarks:
                # Get the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get handedness (left/right)
                handedness = None
                if results.multi_handedness:
                    handedness_info = results.multi_handedness[0]
                    handedness = handedness_info.classification[0].label
                    self.current_handedness = handedness
                
                # Apply smoothing
                if self.landmark_smoothing:
                    hand_landmarks.landmark = self._smooth_landmarks(hand_landmarks.landmark)
                
                # Draw hand landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    processed_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Classify the gesture using the ML model
                if self.classifier:
                    gesture_name, confidence = self._classify_gesture(hand_landmarks, handedness)
                    
                    # Check for cooldown
                    current_time = time.time() * 1000  # convert to ms
                    cooldown_expired = (current_time - self.last_detection_time) > self.cooldown_ms
                    
                    if gesture_name and confidence >= self.confidence_threshold and cooldown_expired:
                        # Track gesture duration
                        if self.current_gesture != gesture_name:
                            self.current_gesture = gesture_name
                            self.gesture_start_time = current_time
                        
                        # Check if gesture has persisted long enough
                        gesture_duration = current_time - self.gesture_start_time
                        if gesture_duration >= self.min_detection_duration_ms:
                            # Update the detection time
                            self.last_detection_time = current_time
                            
                            # Display the gesture name
                            cv2.putText(
                                processed_frame,
                                f"{gesture_name} ({confidence:.1%})",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )
                            
                            return gesture_name, confidence, processed_frame
            
            # Reset current gesture if no hands detected
            if not results.multi_hand_landmarks:
                self.current_gesture = None
            
            return None, 0, processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame with MediaPipe: {e}")
            return None, 0, frame
    
    def _process_with_yolov8(self, frame):
        """
        Process a frame with YOLOv8
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (gesture_name, confidence, processed_frame)
        """
        if not self.engine_initialized or self.engine != "yolov8":
            return None, 0, frame
            
        # Process the frame
        try:
            # Create a copy for visualization
            processed_frame = frame.copy()
            
            # Run YOLOv8 detection
            results = self.yolo_model(frame, conf=self.confidence_threshold)
            
            # Process detection results
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the highest confidence detection
                result = results[0]
                
                # Extract boxes and class names
                boxes = result.boxes.cpu().numpy()
                class_names = result.names
                
                # Find the detection with highest confidence
                max_conf = 0
                max_conf_idx = -1
                
                for i, box in enumerate(boxes):
                    conf = box.conf[0]
                    if conf > max_conf:
                        max_conf = conf
                        max_conf_idx = i
                
                if max_conf_idx >= 0:
                    # Get the best detection
                    box = boxes[max_conf_idx]
                    class_id = int(box.cls[0])
                    conf = box.conf[0]
                    
                    # Get the gesture name
                    gesture_name = class_names.get(class_id, f"unknown-{class_id}")
                    
                    # Calculate cooldown
                    current_time = time.time() * 1000
                    cooldown_expired = (current_time - self.last_detection_time) > self.cooldown_ms
                    
                    if gesture_name and conf >= self.confidence_threshold and cooldown_expired:
                        # Track gesture duration
                        if self.current_gesture != gesture_name:
                            self.current_gesture = gesture_name
                            self.gesture_start_time = current_time
                        
                        # Check if gesture has persisted long enough
                        gesture_duration = current_time - self.gesture_start_time
                        if gesture_duration >= self.min_detection_duration_ms:
                            # Update the detection time
                            self.last_detection_time = current_time
                            
                            # Draw the detection box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Display the gesture name
                            cv2.putText(
                                processed_frame,
                                f"{gesture_name} ({conf:.1%})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )
                            
                            return gesture_name, conf, processed_frame
            
            # Reset current gesture if no detection
            if len(results) == 0 or len(results[0].boxes) == 0:
                self.current_gesture = None
                
            return None, 0, processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame with YOLOv8: {e}")
            return None, 0, frame
    
    def _process_with_mock(self, frame):
        """
        Process a frame with mock detector (for testing)
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (gesture_name, confidence, processed_frame)
        """
        # Create a copy for visualization
        processed_frame = frame.copy()
        
        # Show a message
        cv2.putText(
            processed_frame,
            "Mock detector active (no ML engine)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # Simulate random gesture detection periodically
        current_time = time.time() * 1000
        cooldown_expired = (current_time - self.last_detection_time) > self.cooldown_ms
        
        if cooldown_expired and np.random.random() < 0.05:  # 5% chance per frame
            # Pick a random gesture
            gestures = ["wave", "pinch", "fist", "point", "ok", "swipe"]
            gesture_name = gestures[np.random.randint(0, len(gestures))]
            confidence = np.random.uniform(0.7, 0.95)
            
            # Update the detection time
            self.last_detection_time = current_time
            
            # Display the gesture name
            cv2.putText(
                processed_frame,
                f"{gesture_name} ({confidence:.1%})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            return gesture_name, confidence, processed_frame
            
        return None, 0, processed_frame
    
    def _classify_gesture(self, landmarks, handedness):
        """
        Classify hand landmarks into a gesture using the ML classifier
        
        Args:
            landmarks: MediaPipe hand landmarks
            handedness: Hand classification (left/right)
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        if self.classifier is None:
            return None, 0.0
            
        try:
            # Extract landmark features
            landmark_features = []
            for landmark in landmarks.landmark:
                # Normalize coordinates (x, y, z)
                landmark_features.extend([landmark.x, landmark.y, landmark.z])
            
            # Add handedness as a feature (0 for left, 1 for right)
            hand_value = 0.5  # default if unknown
            if handedness:
                if handedness.lower() == "left":
                    hand_value = 0.0
                elif handedness.lower() == "right":
                    hand_value = 1.0
            
            # Add handedness to features
            landmark_features.append(hand_value)
            
            # Convert to numpy array
            features = np.array(landmark_features, dtype=np.float32)
            
            # Classify the landmarks
            gesture_name, confidence = self.classifier.classify(features)
            
            return gesture_name, confidence
            
        except Exception as e:
            logger.error(f"Error classifying gesture: {e}")
            return None, 0.0
    
    def detect_gesture(self, frame):
        """
        Detect and classify gestures in a frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (gesture_name, confidence, processed_frame)
        """
        if frame is None or not self.engine_initialized:
            return None, 0.0, frame
            
        try:
            # Preprocess the frame
            processed = self._preprocess_frame(frame)
            
            # Process with selected engine
            if self.engine == "mediapipe":
                return self._process_with_mediapipe(processed)
            elif self.engine == "yolov8":
                return self._process_with_yolov8(processed)
            else:  # Mock engine
                return self._process_with_mock(processed)
                
        except Exception as e:
            logger.error(f"Error detecting gesture: {e}")
            return None, 0.0, frame
    
    def process_frame(self, frame):
        """
        Process a frame and return gesture detection results
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (gesture_name, confidence, processed_frame, handedness)
        """
        gesture_name, confidence, processed_frame = self.detect_gesture(frame)
        return gesture_name, confidence, processed_frame, self.current_handedness
    
    def draw_landmarks(self, frame):
        """
        Draw hand landmarks on a frame without gesture detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with landmarks drawn
        """
        if not self.engine_initialized or frame is None:
            return frame
            
        # Create a copy for visualization
        processed_frame = frame.copy()
        
        try:
            # Process with MediaPipe to get landmarks
            if self.engine == "mediapipe":
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(frame_rgb)
                
                # Draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            processed_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
            elif self.engine == "yolov8":
                # Run YOLOv8 detection
                results = self.yolo_model(frame, conf=self.confidence_threshold)
                
                # Draw the detection boxes
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Draw the box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add class name and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = r.names[cls]
                        
                        cv2.putText(
                            processed_frame,
                            f"{name} ({conf:.1%})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            
        return processed_frame
    
    def get_handedness(self):
        """
        Get the handedness of the last detected hand
        
        Returns:
            String: "Left" or "Right" or None
        """
        return self.current_handedness
    
    def __del__(self):
        """Clean up resources"""
        if self.engine == "mediapipe" and hasattr(self, 'hands'):
            self.hands.close()