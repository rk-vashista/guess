#!/usr/bin/env python3
"""
Camera Diagnostics Tool for GestureBind

This script helps diagnose camera-related issues by checking camera availability
and testing basic camera functionality.
"""

import sys
import os
import cv2
import logging
import time
import yaml
import subprocess
import platform

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesturebind.core.camera_manager import CameraManager

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("camera_diagnostics")

def load_config():
    """Load the application configuration"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "gesturebind/data/config/user_config.yaml")
    
    # If user config doesn't exist, try default config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "gesturebind/config/default_config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def check_camera_permissions():
    """Check if the user has permissions to access the camera"""
    print("\n=== Checking Camera Permissions ===")
    
    if platform.system() == "Linux":
        # Check if user is in video group
        try:
            result = subprocess.run(["groups"], capture_output=True, text=True)
            groups = result.stdout.strip().split()
            
            if "video" in groups:
                print("✅ User is in the 'video' group")
            else:
                print("❌ User is NOT in the 'video' group. You may need to run:")
                print("   sudo usermod -a -G video $USER")
                print("   (Then log out and log back in)")
            
            # Check permissions on video devices
            if os.path.exists("/dev/video0"):
                result = subprocess.run(["ls", "-l", "/dev/video0"], capture_output=True, text=True)
                print(f"Video device permissions: {result.stdout.strip()}")
                
                # Suggest fixing permissions if needed
                if "crw-rw----" in result.stdout and "video" in result.stdout:
                    print("✅ Video device has correct permissions")
                else:
                    print("⚠️ Video device may have incorrect permissions")
            else:
                print("❌ No video device found at /dev/video0")
        except Exception as e:
            print(f"❌ Error checking permissions: {e}")
    elif platform.system() == "Windows":
        print("ℹ️ On Windows, camera permissions are managed through Settings > Privacy > Camera")
    else:
        print(f"ℹ️ Permission checking not implemented for {platform.system()}")

def check_camera_in_use():
    """Check if camera might be in use by another application"""
    print("\n=== Checking if Camera is in Use ===")
    
    if platform.system() == "Linux":
        try:
            result = subprocess.run(["fuser", "/dev/video0"], capture_output=True, text=True)
            if result.stdout.strip():
                print("❌ Camera appears to be in use by another process")
                print(f"   Process IDs: {result.stdout.strip()}")
                return True
            else:
                print("✅ Camera does not appear to be in use by another process")
                return False
        except FileNotFoundError:
            print("ℹ️ Cannot check if camera is in use (fuser command not found)")
            print("   Install with: sudo apt-get install psmisc")
        except Exception as e:
            print(f"ℹ️ Error checking if camera is in use: {e}")
    else:
        print(f"ℹ️ Camera use checking not implemented for {platform.system()}")
    
    return False

def detect_cameras():
    """Detect available cameras in the system"""
    available_cameras = []
    
    print("\n=== Checking Available Cameras ===")
    
    # Try multiple camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available_cameras.append({
                    "index": i,
                    "resolution": f"{width}x{height}",
                    "fps": fps
                })
                print(f"✅ Camera {i}: Available ({width}x{height} @ {fps}fps)")
            else:
                print(f"⚠️ Camera {i}: Device found but cannot read frames")
            cap.release()
        else:
            print(f"❌ Camera {i}: Not available")
    
    return available_cameras

def test_camera_capture(device_id=0, num_frames=5):
    """Test direct camera capture without using CameraManager"""
    print(f"\n=== Testing Direct Camera Capture (device {device_id}) ===")
    
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"❌ Failed to open camera {device_id}")
        return False
    
    print(f"✅ Successfully opened camera {device_id}")
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Try to read multiple frames (sometimes the first frame fails)
    success_count = 0
    for i in range(num_frames):
        print(f"  Attempting to read frame {i+1}/{num_frames}...")
        ret, frame = cap.read()
        if ret:
            success_count += 1
            print(f"  ✅ Successfully read frame {i+1} with size {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"  ❌ Failed to read frame {i+1}")
        time.sleep(0.1)  # Small delay between frame captures
    
    cap.release()
    print(f"Successfully read {success_count}/{num_frames} frames")
    return success_count > 0

def test_camera_manager():
    """Test the CameraManager implementation"""
    config = load_config()
    
    if not config:
        print("❌ Failed to load configuration")
        return False
    
    print("\n=== Testing CameraManager ===")
    
    # Extract the camera device ID from config
    device_id = config.get("camera", {}).get("device_id", 0)
    print(f"📝 Config specifies camera device_id: {device_id}")
    
    # Create camera manager
    camera_manager = CameraManager(config)
    
    # Check availability
    print("\n- Checking camera availability...")
    is_available, msg = camera_manager.check_camera_availability()
    if is_available:
        print(f"✅ Camera availability check passed: {msg}")
    else:
        print(f"❌ Camera availability check failed: {msg}")
    
    # Test starting the camera
    print("\n- Testing camera start...")
    result = camera_manager.start_capture()
    if result:
        print("✅ Camera started successfully")
        
        # Add a delay to let the camera properly initialize
        print(f"  Waiting 1 second for camera to initialize...")
        time.sleep(1.0)
        
        # Test getting a frame
        print("\n- Testing frame capture...")
        success = False
        
        # Try multiple times to get a frame
        for i in range(3):
            print(f"  Attempt {i+1}/3 to capture frame...")
            success, frame = camera_manager.get_frame()
            if success and frame is not None:
                print(f"✅ Successfully captured frame of size {frame.shape[1]}x{frame.shape[0]}")
                break
            else:
                print(f"❌ Failed to capture frame on attempt {i+1}")
                time.sleep(0.5)  # Wait before trying again
        
        # Stop the camera
        print("\n- Testing camera stop...")
        result = camera_manager.stop_capture()
        if result:
            print("✅ Camera stopped successfully")
        else:
            print("❌ Failed to stop camera")
    else:
        camera_status = camera_manager.get_status()
        print(f"❌ Failed to start camera: {camera_status.get('error_message', 'Unknown error')}")
    
    return result

def main():
    """Main function"""
    print("=== GestureBind Camera Diagnostic Tool ===")
    print("This tool will help diagnose camera-related issues.")
    
    # Check permissions first
    check_camera_permissions()
    
    # Check if camera is in use
    camera_in_use = check_camera_in_use()
    
    # Step 1: Check system cameras
    available_cameras = detect_cameras()
    
    if not available_cameras:
        print("\n❌ No cameras were detected on your system. Please check if:")
        print("  - Your camera is properly connected")
        print("  - Your camera drivers are installed")
        print("  - Your camera is not being used by another application")
        print("  - You have proper permissions to access the camera")
        return
    
    # Test direct camera capture
    if available_cameras:
        test_camera_capture(available_cameras[0]['index'])
    
    # Step 2: Test the camera manager
    test_camera_manager()
    
    print("\n=== Recommendations ===")
    if len(available_cameras) > 0:
        print(f"- Available camera indices: {', '.join(str(cam['index']) for cam in available_cameras)}")
        print("- Set the correct camera device_id in your configuration file:")
        print("  Edit the file: gesturebind/data/config/user_config.yaml")
        print("  Change the 'device_id' value under 'camera' section")
        
        if camera_in_use:
            print("\n⚠️ Your camera appears to be in use by another application.")
            print("  Close any applications that might be using the camera (browsers, video conferencing apps, etc.)")
        
        print("\nIf problems persist, try these additional steps:")
        print("1. Restart your computer to free up camera resources")
        print("2. Check for camera driver updates")
        print("3. Try a different USB port if using an external camera")
        print("4. Try increasing the capture_delay in the camera_manager.py file")
    else:
        print("- No cameras detected. Please connect a camera to your system.")

if __name__ == "__main__":
    main()
