"""
Action mapper module for GestureBind

This module handles the mapping between detected gestures and system actions.
"""

import os
import logging
import subprocess
import platform

# Import platform-specific keyboard/mouse control libraries based on OS
system = platform.system()
if system == "Windows":
    import pythoncom  # For COM objects
    try:
        import pyautogui
        import keyboard
        from pynput import mouse
    except ImportError:
        logging.error("Required libraries missing. Install pyautogui, keyboard, and pynput.")
elif system == "Linux":
    try:
        import pyautogui
        from pynput import keyboard, mouse
    except ImportError:
        logging.error("Required libraries missing. Install pyautogui and pynput.")
elif system == "Darwin":  # macOS
    try:
        import pyautogui
        from pynput import keyboard, mouse
    except ImportError:
        logging.error("Required libraries missing. Install pyautogui and pynput.")
else:
    logging.error(f"Unsupported platform: {system}")

logger = logging.getLogger(__name__)

class ActionMapper:
    """
    Maps detected gestures to system actions based on configured profiles.
    """
    
    def __init__(self, config):
        """
        Initialize the action mapper.
        
        Args:
            config (dict): Application configuration
        """
        self.config = config
        
        # Current active profile
        self.active_profile_name = self.config.get("actions", {}).get("default_profile", "default")
        self.active_profile = self._get_profile(self.active_profile_name)
        
        # Initialize system-specific keyboard/mouse controllers
        self._init_controllers()
        
        logger.info(f"ActionMapper initialized with profile: {self.active_profile_name}")
    
    def _init_controllers(self):
        """Initialize platform-specific input controllers"""
        if system == "Windows":
            # Initialize COM for Windows
            try:
                pythoncom.CoInitialize()
            except Exception as e:
                logger.error(f"Failed to initialize COM: {e}")
        
        # Create keyboard controller
        try:
            if system == "Windows":
                # Windows uses the keyboard module directly
                self.keyboard_controller = None  # Will use keyboard module directly
            else:
                # Linux and macOS use pynput
                self.keyboard_controller = keyboard.Controller()
        except Exception as e:
            logger.error(f"Failed to initialize keyboard controller: {e}")
            self.keyboard_controller = None
        
        # Create mouse controller
        try:
            self.mouse_controller = mouse.Controller()
        except Exception as e:
            logger.error(f"Failed to initialize mouse controller: {e}")
            self.mouse_controller = None
    
    def _get_profile(self, profile_name):
        """
        Get the specified profile configuration.
        
        Args:
            profile_name (str): Name of the profile
            
        Returns:
            dict: Profile configuration or empty dict if not found
        """
        profiles = self.config.get("profiles", {})
        return profiles.get(profile_name, {})
    
    def set_active_profile(self, profile_name):
        """
        Set the active gesture mapping profile.
        
        Args:
            profile_name (str): Name of the profile to activate
            
        Returns:
            bool: True if profile was found and set, False otherwise
        """
        profile = self._get_profile(profile_name)
        if profile:
            self.active_profile_name = profile_name
            self.active_profile = profile
            logger.info(f"Active profile set to: {profile_name}")
            return True
        else:
            logger.warning(f"Profile '{profile_name}' not found, keeping current profile")
            return False
    
    def set_profile(self, profile_name):
        """
        Set the active gesture mapping profile (alias for set_active_profile).
        
        Args:
            profile_name (str): Name of the profile to activate
            
        Returns:
            bool: True if profile was found and set, False otherwise
        """
        return self.set_active_profile(profile_name)
    
    def get_action_for_gesture(self, gesture_name):
        """
        Get the configured action for a detected gesture.
        
        Args:
            gesture_name (str): Name of the detected gesture
            
        Returns:
            tuple: (action_type, action_data, action_description) or (None, None, None) if not mapped
        """
        # Check if gesture has a mapping in active profile
        gesture_mappings = self.active_profile.get("gesture_mappings", {})
        
        # First try the new format with gesture_mappings dict
        if gesture_name in gesture_mappings:
            action_config = gesture_mappings[gesture_name]
            action_type = action_config.get("type")
            action_data = action_config.get("data")
            description = action_config.get("description", f"Action: {action_type}")
            return action_type, action_data, description
        
        # Fall back to the old format (direct mapping in profile)
        elif gesture_name in self.active_profile:
            action_string = self.active_profile[gesture_name]
            # Parse the action string (format: "type:data")
            if ":" in action_string:
                action_type, action_data = action_string.split(":", 1)
                return action_type, action_data, f"Action: {action_string}"
            
        # No mapping found
        logger.debug(f"No mapping found for gesture: {gesture_name}")
        return None, None, "No action mapped"
    
    def execute_action(self, action_type, action_data):
        """
        Execute the specified action.
        
        Args:
            action_type (str): Type of action to perform
            action_data: Data needed for the action
            
        Returns:
            bool: True if action executed successfully, False otherwise
        """
        if action_type is None or action_data is None:
            return False
        
        try:
            # Execute different action types
            if action_type == "keypress":
                return self._execute_keypress(action_data)
                
            elif action_type == "hotkey":
                return self._execute_hotkey(action_data)
                
            elif action_type == "launch_app":
                return self._execute_launch_app(action_data)
                
            elif action_type == "mouse_move":
                return self._execute_mouse_move(action_data)
                
            elif action_type == "mouse_click":
                return self._execute_mouse_click(action_data)
                
            elif action_type == "system_command":
                return self._execute_system_command(action_data)
                
            elif action_type == "shell_command":
                return self._execute_shell_command(action_data)
            
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            return False
    
    def _execute_keypress(self, key_name):
        """Execute a single key press"""
        logger.debug(f"Executing keypress: {key_name}")
        
        try:
            if system == "Windows":
                keyboard.press_and_release(key_name)
            else:
                # Handle special keys for Linux/macOS
                if key_name.lower() in ['ctrl', 'control']:
                    key = keyboard.Key.ctrl
                elif key_name.lower() in ['alt']:
                    key = keyboard.Key.alt
                elif key_name.lower() in ['shift']:
                    key = keyboard.Key.shift
                elif key_name.lower() in ['cmd', 'command', 'super', 'win', 'meta']:
                    key = keyboard.Key.cmd
                elif hasattr(keyboard.Key, key_name.lower()):
                    # For other special keys like space, escape, etc.
                    key = getattr(keyboard.Key, key_name.lower())
                else:
                    # For regular character keys
                    key = key_name
                
                # Press and release the key
                self.keyboard_controller.press(key)
                self.keyboard_controller.release(key)
                
            return True
        except Exception as e:
            logger.error(f"Failed to execute keypress: {e}")
            return False
    
    def _execute_hotkey(self, hotkey_combo):
        """Execute a key combination/hotkey"""
        logger.debug(f"Executing hotkey: {hotkey_combo}")
        
        try:
            if system == "Windows":
                keyboard.press_and_release(hotkey_combo)
            else:
                # Parse hotkey string for pynput
                keys = hotkey_combo.split('+')
                pressed_keys = []
                
                # Map special keys if needed
                for k in keys:
                    # Handle special keys
                    if k.lower() in ['ctrl', 'control']:
                        pressed_keys.append(keyboard.Key.ctrl)
                    elif k.lower() in ['alt']:
                        pressed_keys.append(keyboard.Key.alt)
                    elif k.lower() in ['shift']:
                        pressed_keys.append(keyboard.Key.shift)
                    elif k.lower() in ['cmd', 'command', 'super', 'win', 'meta']:
                        pressed_keys.append(keyboard.Key.cmd)
                    elif hasattr(keyboard.Key, k.lower()):
                        # For other special keys like space, escape, etc.
                        pressed_keys.append(getattr(keyboard.Key, k.lower()))
                    else:
                        # For regular character keys
                        pressed_keys.append(k)
                
                # Press all keys
                for k in pressed_keys:
                    self.keyboard_controller.press(k)
                
                # Release all keys in reverse order
                for k in reversed(pressed_keys):
                    self.keyboard_controller.release(k)
            
            return True
        except Exception as e:
            logger.error(f"Failed to execute hotkey: {e}")
            return False
    
    def _execute_launch_app(self, app_info):
        """Launch an application"""
        app_path = app_info.get("path", "")
        app_args = app_info.get("args", [])
        
        logger.debug(f"Launching app: {app_path} with args: {app_args}")
        
        if not app_path:
            logger.error("No application path provided")
            return False
        
        try:
            if system == "Windows":
                # Windows-specific launch
                cmd = f'start "" "{app_path}"'
                if app_args:
                    cmd += ' ' + ' '.join(app_args)
                subprocess.Popen(cmd, shell=True)
            else:
                # Linux and macOS
                subprocess.Popen([app_path] + app_args)
            
            return True
        except Exception as e:
            logger.error(f"Failed to launch application: {e}")
            return False
    
    def _execute_mouse_move(self, move_data):
        """Move the mouse cursor"""
        move_type = move_data.get("type", "absolute")
        x = move_data.get("x", 0)
        y = move_data.get("y", 0)
        
        logger.debug(f"Moving mouse: {move_type} x={x}, y={y}")
        
        try:
            if move_type == "relative":
                # Relative movement
                self.mouse_controller.move(x, y)
            else:
                # Absolute position
                screen_width, screen_height = pyautogui.size()
                
                # Convert percentage to pixels if needed
                if isinstance(x, str) and x.endswith('%'):
                    x = int(screen_width * float(x.strip('%')) / 100)
                if isinstance(y, str) and y.endswith('%'):
                    y = int(screen_height * float(y.strip('%')) / 100)
                
                self.mouse_controller.position = (x, y)
            
            return True
        except Exception as e:
            logger.error(f"Failed to move mouse: {e}")
            return False
    
    def _execute_mouse_click(self, click_data):
        """Perform a mouse click"""
        button = click_data.get("button", "left")
        double = click_data.get("double", False)
        
        logger.debug(f"Clicking mouse: button={button}, double={double}")
        
        try:
            # Map button string to pynput button object
            if button == "left":
                btn = mouse.Button.left
            elif button == "right":
                btn = mouse.Button.right
            elif button == "middle":
                btn = mouse.Button.middle
            else:
                logger.error(f"Unknown mouse button: {button}")
                return False
            
            # Perform the click
            if double:
                self.mouse_controller.click(btn, 2)
            else:
                self.mouse_controller.click(btn)
            
            return True
        except Exception as e:
            logger.error(f"Failed to click mouse: {e}")
            return False
    
    def _execute_system_command(self, command_data):
        """Execute a system command"""
        command = command_data.get("command", "")
        shell = command_data.get("shell", True)
        
        logger.debug(f"Executing system command: {command}")
        
        if not command:
            logger.error("No command provided")
            return False
        
        try:
            subprocess.run(command, shell=shell, check=True)
            return True
        except Exception as e:
            logger.error(f"Failed to execute system command: {e}")
            return False
    
    def _execute_shell_command(self, action_data):
        """
        Execute a shell command with optional terminal window.
        
        Args:
            action_data (dict): Dictionary containing command and terminal option
                - command (str): Shell command to execute
                - terminal (bool): Whether to run in a terminal window
                
        Returns:
            bool: True if command executed successfully, False otherwise
        """
        # Handle both string and dictionary formats for backward compatibility
        if isinstance(action_data, str):
            command = action_data
            terminal = False
        else:
            command = action_data.get("command", "")
            terminal = action_data.get("terminal", False)
        
        logger.debug(f"Executing shell command: '{command}', terminal: {terminal}")
        
        if not command:
            logger.error("No command provided")
            return False
        
        try:
            # Execute differently based on platform and terminal option
            if terminal:
                # Run command in a visible terminal window
                if system == "Windows":
                    # Windows: use cmd.exe
                    subprocess.Popen(f'start cmd.exe /K "{command}"', shell=True)
                elif system == "Darwin":  # macOS
                    # macOS: use Terminal.app
                    apple_script = f'tell application "Terminal" to do script "{command}"'
                    subprocess.Popen(['osascript', '-e', apple_script])
                else:
                    # Linux: try to detect terminal emulator
                    terminal_cmd = None
                    # Check common terminal emulators
                    for term in ["gnome-terminal", "konsole", "xterm", "terminator"]:
                        if subprocess.call(['which', term], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                            terminal_cmd = term
                            break
                    
                    if terminal_cmd:
                        if terminal_cmd == "gnome-terminal":
                            subprocess.Popen([terminal_cmd, "--", "bash", "-c", f"{command}; exec bash"])
                        else:
                            subprocess.Popen([terminal_cmd, "-e", f"bash -c '{command}; exec bash'"])
                    else:
                        # Fallback: run without terminal
                        logger.warning("No terminal emulator found, running command in background")
                        subprocess.Popen(command, shell=True)
            else:
                # Run command silently in background
                if system == "Windows":
                    # Hide console window on Windows
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    subprocess.Popen(command, shell=True, startupinfo=startupinfo)
                else:
                    # Standard execution for Unix-like systems
                    subprocess.Popen(command, shell=True, 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute shell command: {e}")
            return False
    
    def __del__(self):
        """Clean up resources"""
        if system == "Windows":
            # Uninitialize COM
            try:
                pythoncom.CoUninitialize()
            except:
                pass