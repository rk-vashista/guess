"""
Test cases for action mapper module
"""

import unittest
import sys
import os
import platform

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.action_mapper import ActionMapper

class TestActionMapper(unittest.TestCase):
    """Test cases for ActionMapper class"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock config for testing
        self.config = {
            "actions": {
                "default_profile": "test_profile"
            },
            "profiles": {
                "test_profile": {
                    "open_palm": "keyboard:ctrl+c",
                    "fist": "keyboard:escape",
                    "pointing": "mouse:mouse_click",
                    "peace_sign": "media:play_pause"
                },
                "alternate_profile": {
                    "open_palm": "keyboard:ctrl+v",
                    "fist": "keyboard:enter"
                }
            }
        }
        
        self.action_mapper = ActionMapper(self.config)
    
    def test_initialization(self):
        """Test if ActionMapper initializes correctly"""
        # Check that the default profile was loaded
        self.assertEqual(self.action_mapper.current_profile_name, "test_profile")
        
        # Check that the profile has the correct mappings
        self.assertIn("open_palm", self.action_mapper.current_profile)
        self.assertEqual(self.action_mapper.current_profile["open_palm"], "keyboard:ctrl+c")
        self.assertEqual(self.action_mapper.current_profile["fist"], "keyboard:escape")
        
        # Check that the OS platform was detected
        self.assertIsNotNone(self.action_mapper.os_platform)
        
        # Check that action handlers were initialized
        self.assertIn("keyboard", self.action_mapper.action_handlers)
        self.assertIn("mouse", self.action_mapper.action_handlers)
        self.assertIn("media", self.action_mapper.action_handlers)
        self.assertIn("system", self.action_mapper.action_handlers)
        self.assertIn("app", self.action_mapper.action_handlers)
    
    def test_set_profile(self):
        """Test profile switching"""
        # Test switching to a valid profile
        success = self.action_mapper.set_profile("alternate_profile")
        self.assertTrue(success)
        self.assertEqual(self.action_mapper.current_profile_name, "alternate_profile")
        self.assertEqual(self.action_mapper.current_profile["open_palm"], "keyboard:ctrl+v")
        
        # Test switching to an invalid profile
        success = self.action_mapper.set_profile("nonexistent_profile")
        self.assertFalse(success)
        self.assertEqual(self.action_mapper.current_profile_name, "alternate_profile")
    
    def test_get_profiles(self):
        """Test getting available profiles"""
        profiles = self.action_mapper.get_profiles()
        self.assertIsInstance(profiles, list)
        self.assertEqual(len(profiles), 2)
        self.assertIn("test_profile", profiles)
        self.assertIn("alternate_profile", profiles)
    
    def test_add_gesture_mapping(self):
        """Test adding a gesture mapping"""
        # Add to an existing profile
        success = self.action_mapper.add_gesture_mapping(
            "test_profile",
            "thumbs_up",
            "media:volume_up"
        )
        self.assertTrue(success)
        self.assertIn("thumbs_up", self.action_mapper.profiles["test_profile"])
        self.assertEqual(
            self.action_mapper.profiles["test_profile"]["thumbs_up"],
            "media:volume_up"
        )
        
        # Add to a new profile
        success = self.action_mapper.add_gesture_mapping(
            "new_profile",
            "wave",
            "keyboard:alt+tab"
        )
        self.assertTrue(success)
        self.assertIn("new_profile", self.action_mapper.profiles)
        self.assertIn("wave", self.action_mapper.profiles["new_profile"])
        self.assertEqual(
            self.action_mapper.profiles["new_profile"]["wave"],
            "keyboard:alt+tab"
        )
    
    def test_remove_gesture_mapping(self):
        """Test removing a gesture mapping"""
        # Remove an existing mapping
        success = self.action_mapper.remove_gesture_mapping(
            "test_profile",
            "fist"
        )
        self.assertTrue(success)
        self.assertNotIn("fist", self.action_mapper.profiles["test_profile"])
        
        # Try to remove a non-existent mapping
        success = self.action_mapper.remove_gesture_mapping(
            "test_profile",
            "nonexistent_gesture"
        )
        self.assertFalse(success)
        
        # Try to remove from a non-existent profile
        success = self.action_mapper.remove_gesture_mapping(
            "nonexistent_profile",
            "open_palm"
        )
        self.assertFalse(success)
    
    def test_action_parsing(self):
        """Test parsing of action strings"""
        # Test with valid action string
        valid_action = "keyboard:ctrl+c"
        result = self.action_mapper._parse_action(valid_action)
        self.assertTrue(result[0])
        self.assertEqual(result[1], "keyboard")
        self.assertEqual(result[2], "ctrl+c")
        
        # Test with invalid action string (no colon)
        invalid_action = "keyboard_ctrl+c"
        result = self.action_mapper._parse_action(invalid_action)
        self.assertFalse(result[0])
        
        # Test with invalid action type
        invalid_type_action = "invalid_type:action"
        result = self.action_mapper._parse_action(invalid_type_action)
        self.assertFalse(result[0])
    
    def _parse_action(self, action):
        """Helper method to parse action strings (to be added to ActionMapper)"""
        if ":" not in action:
            return (False, None, None)
            
        action_type, action_command = action.split(":", 1)
        
        if action_type not in self.action_mapper.action_handlers:
            return (False, None, None)
            
        return (True, action_type, action_command)

if __name__ == '__main__':
    unittest.main()