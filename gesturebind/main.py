#!/usr/bin/env python3
"""
GestureBind Launcher with Module Reload Approach

This script completely avoids Qt plugin conflicts by:
1. Ensuring PyQt5 is loaded before OpenCV
2. Using a subprocess to launch the application with clean environment
"""

import os
import sys
import subprocess
import importlib

def prepare_environment():
    """Prepare environment variables for the child process"""
    env = os.environ.copy()
    
    # Find PyQt5 plugins directory
    try:
        import PyQt5
        pyqt_dir = os.path.dirname(PyQt5.__file__)
        qt_plugin_path = os.path.join(pyqt_dir, "Qt5", "plugins")
        
        if os.path.exists(qt_plugin_path):
            # Set Qt plugin paths
            env["QT_PLUGIN_PATH"] = qt_plugin_path
            env["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(qt_plugin_path, "platforms")
            
            # Use xcb platform for Linux - more compatible than minimal
            if sys.platform.startswith('linux'):
                env["QT_QPA_PLATFORM"] = "xcb"
            else:
                env["QT_QPA_PLATFORM"] = "minimal"
            
            # Add Wayland-specific settings for Linux
            if 'XDG_SESSION_TYPE' in env and env['XDG_SESSION_TYPE'] == 'wayland':
                print("Detected Wayland session, setting appropriate QT environment variables")
                # Set QT_QPA_PLATFORM to wayland explicitly
                env["QT_QPA_PLATFORM"] = "wayland"
            
            # Tell Qt not to use the system's plugin path
            env["QT_NO_SYSTEM_PLUGIN_PATH"] = "1"
            
            # Ensure Python knows we're in a non-interactive environment
            env["PYTHONUNBUFFERED"] = "1"
            
            # Disable auto Qt platform detection
            env["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
            
            print(f"Set Qt plugin path to: {qt_plugin_path}")
            print(f"Platform plugins available: {os.listdir(os.path.join(qt_plugin_path, 'platforms'))}")
            print(f"Using QT_QPA_PLATFORM={env.get('QT_QPA_PLATFORM', 'not set')}")
            
    except Exception as e:
        print(f"Error setting up environment: {e}")
    
    return env

def preload_qt_modules():
    """Preload Qt modules to ensure they're loaded before OpenCV"""
    try:
        # Import PyQt5 modules in the correct order
        import PyQt5
        from PyQt5 import QtCore
        from PyQt5 import QtGui
        from PyQt5 import QtWidgets
        
        # Force initialization of Qt
        app = QtWidgets.QApplication.instance()
        if not app:
            # Create a dummy application to initialize Qt properly
            app = QtWidgets.QApplication([])
        
        print("Successfully preloaded PyQt5 modules")
        return True
    except Exception as e:
        print(f"Error preloading Qt modules: {e}")
        return False

if __name__ == "__main__":
    # Try to fix the import order issue
    if preload_qt_modules():
        # Launch the application in a separate process with the right environment
        env = prepare_environment()
        
        # Get the path to the notmain.py script (renamed from main to avoid recursive imports)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_script = os.path.join(script_dir, "notmain.py")
        
        if os.path.exists(app_script):
            print(f"Launching {app_script} in a clean environment...")
            
            # Use a subprocess to run with the prepared environment
            result = subprocess.call([sys.executable, app_script], env=env)
            sys.exit(result)
        else:
            print(f"Error: Could not find application script at {app_script}")
            sys.exit(1)
    else:
        print("Failed to preload Qt modules")
        sys.exit(1)