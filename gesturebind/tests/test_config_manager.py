"""
Test cases for configuration manager module
"""

import unittest
import sys
import os
import tempfile
import yaml
import shutil
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test configuration
        self.test_dir = tempfile.mkdtemp()
        
        # Create a default config file for testing
        self.default_config_path = os.path.join(self.test_dir, "default_config.yaml")
        self.default_config = {
            "app": {
                "version": "1.0.0",
                "save_user_data_path": "~/gesturebind/data",
                "log_level": "info"
            },
            "camera": {
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "detection": {
                "engine": "mediapipe",
                "confidence_threshold": 0.7
            },
            "ui": {
                "theme": "system",
                "show_preview": True
            }
        }
        
        # Write default config to the file
        with open(self.default_config_path, 'w') as f:
            yaml.dump(self.default_config, f)
        
        # Create the config manager instance
        self.config_manager = ConfigManager(self.default_config_path)
        
        # Override the user config directory for testing
        self.config_manager.user_config_dir = Path(self.test_dir) / ".gesturebind"
        self.config_manager.user_config_path = self.config_manager.user_config_dir / "user_config.yaml"
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)
    
    def test_load_config(self):
        """Test loading configuration"""
        # Test loading default config
        config = self.config_manager.load_config()
        
        # Check that configuration was loaded correctly
        self.assertIsInstance(config, dict)
        self.assertIn("app", config)
        self.assertIn("camera", config)
        self.assertIn("detection", config)
        self.assertIn("ui", config)
        
        # Check specific values
        self.assertEqual(config["app"]["version"], "1.0.0")
        self.assertEqual(config["camera"]["width"], 640)
        self.assertEqual(config["detection"]["engine"], "mediapipe")
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        # Modify configuration
        self.config_manager.config = self.default_config.copy()
        self.config_manager.config["camera"]["width"] = 1280
        self.config_manager.config["camera"]["height"] = 720
        
        # Save configuration
        result = self.config_manager.save_config()
        self.assertTrue(result)
        
        # Check that the user config file was created
        self.assertTrue(self.config_manager.user_config_path.exists())
        
        # Create a new config manager instance to load the saved config
        new_config_manager = ConfigManager(self.default_config_path)
        new_config_manager.user_config_dir = Path(self.test_dir) / ".gesturebind"
        new_config_manager.user_config_path = new_config_manager.user_config_dir / "user_config.yaml"
        
        # Load the configuration
        config = new_config_manager.load_config()
        
        # Check that the modified values are present
        self.assertEqual(config["camera"]["width"], 1280)
        self.assertEqual(config["camera"]["height"], 720)
    
    def test_update_config(self):
        """Test updating specific configuration values"""
        # Update a specific value
        result = self.config_manager.update_config("ui.theme", "dark")
        self.assertTrue(result)
        self.assertEqual(self.config_manager.config["ui"]["theme"], "dark")
        
        # Update a value with nested path that doesn't exist yet
        result = self.config_manager.update_config("new_section.nested.value", 42)
        self.assertTrue(result)
        self.assertEqual(self.config_manager.config["new_section"]["nested"]["value"], 42)
    
    def test_get_value(self):
        """Test getting configuration values"""
        # Get existing value
        value = self.config_manager.get_value("camera.width")
        self.assertEqual(value, 640)
        
        # Get non-existent value with default
        value = self.config_manager.get_value("nonexistent.path", "default_value")
        self.assertEqual(value, "default_value")
        
        # Get non-existent value without default
        value = self.config_manager.get_value("nonexistent.path")
        self.assertIsNone(value)
    
    def test_reset_to_default(self):
        """Test resetting configuration to defaults"""
        # Modify the configuration
        self.config_manager.config["camera"]["width"] = 1280
        self.config_manager.config["new_section"] = {"test": True}
        
        # Reset to default
        config = self.config_manager.reset_to_default()
        
        # Check that the modifications are gone
        self.assertEqual(config["camera"]["width"], 640)
        self.assertNotIn("new_section", config)
    
    def test_export_import_config(self):
        """Test exporting and importing configuration"""
        # Modify the configuration
        self.config_manager.config["camera"]["width"] = 1280
        self.config_manager.config["test_export"] = True
        
        # Export configuration
        export_path = os.path.join(self.test_dir, "exported_config.yaml")
        result = self.config_manager.export_config(export_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(export_path))
        
        # Create a new config manager
        new_config_manager = ConfigManager(self.default_config_path)
        
        # Import the exported configuration
        result = new_config_manager.import_config(export_path)
        self.assertTrue(result)
        
        # Check that the imported configuration has the modifications
        self.assertEqual(new_config_manager.config["camera"]["width"], 1280)
        self.assertTrue(new_config_manager.config.get("test_export"))

if __name__ == '__main__':
    unittest.main()