"""
Test cases for gesture detector module
"""

import unittest
import sys
import os
import numpy as np
import cv2

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.gesture_detector import GestureDetector

class MockConfig:
    """Mock configuration for testing"""
    
    def __init__(self):
        self.config = {
            "detection": {
                "engine": "mediapipe",
                "confidence_threshold": 0.7,
                "cooldown_ms": 100,
                "min_detection_duration_ms": 50
            },
            "advanced": {
                "landmark_smoothing": True
            }
        }

class MockFrame:
    """Mock frame utility for gesture detector tests"""
    
    @staticmethod
    def create_blank_frame(width=640, height=480):
        """Create a blank frame with specified dimensions"""
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    @staticmethod
    def create_hand_frame(gesture_type="open_palm"):
        """
        Create a test frame with a simulated hand
        
        This is a simplified version for testing - in reality we'd use
        real images or generate more sophisticated mock data
        """
        frame = MockFrame.create_blank_frame()
        
        # Draw a basic shape representing hand based on gesture type
        if gesture_type == "open_palm":
            # Draw simplified open palm
            cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)
            for i in range(5):
                cv2.rectangle(frame, (220 + i*40, 100), (240 + i*40, 150), (255, 255, 255), -1)
        elif gesture_type == "fist":
            # Draw simplified fist
            cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)
        elif gesture_type == "pointing":
            # Draw simplified pointing hand
            cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)
            cv2.rectangle(frame, (300, 50), (320, 150), (255, 255, 255), -1)
        
        return frame

class TestGestureDetector(unittest.TestCase):
    """Test cases for GestureDetector class"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_config = MockConfig().config
        
        # Skip tests if mediapipe is not available
        try:
            import mediapipe as mp
            self.skip_mediapipe = False
        except ImportError:
            self.skip_mediapipe = True
    
    def test_initialization(self):
        """Test if GestureDetector initializes correctly"""
        if self.skip_mediapipe:
            self.skipTest("MediaPipe not available")
        
        detector = GestureDetector(self.mock_config)
        
        # Check that detector was initialized
        self.assertIsNotNone(detector)
        self.assertEqual(detector.engine, "mediapipe")
        self.assertEqual(detector.confidence_threshold, 0.7)
        self.assertEqual(detector.cooldown_ms, 100)
    
    def test_empty_frame(self):
        """Test detection with an empty frame"""
        if self.skip_mediapipe:
            self.skipTest("MediaPipe not available")
        
        detector = GestureDetector(self.mock_config)
        frame = MockFrame.create_blank_frame()
        
        gesture_name, confidence, processed_frame = detector.detect_gesture(frame)
        
        # Should not detect any gesture in an empty frame
        self.assertIsNone(gesture_name)
        self.assertEqual(confidence, 0.0)
        
        # Processed frame should have same dimensions as input
        self.assertEqual(processed_frame.shape, frame.shape)
    
    def test_cooldown_period(self):
        """Test cooldown period between detections"""
        if self.skip_mediapipe:
            self.skipTest("MediaPipe not available")
        
        detector = GestureDetector(self.mock_config)
        
        # First detection
        frame = MockFrame.create_hand_frame("open_palm")
        detector.detect_gesture(frame)
        
        # Second detection immediately after (should be blocked by cooldown)
        gesture_name, confidence, _ = detector.detect_gesture(frame)
        
        # During cooldown, should return None even if gesture is present
        self.assertIsNone(gesture_name)
        self.assertEqual(confidence, 0.0)
        
        # Wait for cooldown to expire
        import time
        time.sleep(0.15)  # Wait longer than cooldown_ms
        
        # This won't be a full test of the real detection since our mock frame
        # isn't sophisticated enough to trigger actual detection in most cases.
        # In a real implementation with real image data, we'd test that
        # the detection now succeeds after cooldown.
    
    def test_changing_config(self):
        """Test changing detector configuration"""
        if self.skip_mediapipe:
            self.skipTest("MediaPipe not available")
        
        detector = GestureDetector(self.mock_config)
        original_threshold = detector.confidence_threshold
        
        # Change configuration
        new_config = self.mock_config.copy()
        new_config["detection"]["confidence_threshold"] = 0.9
        detector.update_config(new_config)
        
        # Check that configuration was updated
        self.assertEqual(detector.confidence_threshold, 0.9)
        self.assertNotEqual(detector.confidence_threshold, original_threshold)
    
    def test_mock_engine_fallback(self):
        """Test fallback to mock engine if requested engine not available"""
        # Set config to request a non-existent engine
        invalid_config = self.mock_config.copy()
        invalid_config["detection"]["engine"] = "nonexistent_engine"
        
        # This should fall back to a mock engine or default
        detector = GestureDetector(invalid_config)
        
        # It should still be usable
        frame = MockFrame.create_blank_frame()
        gesture_name, confidence, processed_frame = detector.detect_gesture(frame)
        
        # Should still return a valid result
        self.assertIsNotNone(processed_frame)
        # The other values might be None/0.0 depending on implementation

if __name__ == '__main__':
    unittest.main()