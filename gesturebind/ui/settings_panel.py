"""
Settings panel UI module for GestureBind

This module provides the settings configuration interface.
"""

import os
import sys
import logging
import platform
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
                           QGroupBox, QFormLayout, QTabWidget, QCheckBox,
                           QMessageBox, QFileDialog, QTableWidgetItem, QTableWidget,
                           QHeaderView, QDialog, QLineEdit, QDialogButtonBox, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QKeySequence

# Determine the operating system for platform-specific UI labels
PLATFORM = platform.system()
# Set appropriate name for the "Super" key based on platform
if PLATFORM == "Windows":
    SUPER_KEY_NAME = "Win"
    SUPER_KEY_INTERNAL = "win"
elif PLATFORM == "Darwin":  # macOS
    SUPER_KEY_NAME = "Command"
    SUPER_KEY_INTERNAL = "cmd"
else:  # Linux and others
    SUPER_KEY_NAME = "Super"
    SUPER_KEY_INTERNAL = "super"

# Fix imports by adding the project root to sys.path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use direct imports instead of relative imports
try:
    from utils.path_manager import get_workspace_root, resolve_path, get_data_dir, ensure_workspace_paths
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported path_manager using direct import")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Error importing path_manager: {e}")
    
    # Define fallback minimal implementations if needed
    def get_workspace_root():
        """Get the absolute path to the workspace root directory"""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    def resolve_path(path, create_if_missing=False):
        """Basic path resolver"""
        return path
    
    def get_data_dir(subdir=None, create=True):
        """Get data directory"""
        root = get_workspace_root()
        if subdir:
            return os.path.join(root, "data", subdir)
        return os.path.join(root, "data")
    
    def ensure_workspace_paths():
        """Ensure all workspace paths exist"""
        get_data_dir(create=True)
        get_data_dir("gestures", create=True)
        get_data_dir("models", create=True)
        get_data_dir("default", create=True)

logger = logging.getLogger(__name__)

class SettingsPanel(QWidget):
    """
    Settings panel UI component for configuring GestureBind settings.
    """
    
    # Signal emitted when settings are changed
    settings_changed = pyqtSignal()
    
    def __init__(self, config_manager):
        """
        Initialize the settings panel.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.load_config()
        self.changes_made = False
        
        # Ensure all workspace directories exist
        ensure_workspace_paths()
        
        # Override any absolute paths in the config with workspace-relative paths
        self._fix_config_paths()
        
        # Rest of initialization
        self._init_ui()
        self._populate_settings()
        
        logger.info("Settings panel initialized")

    def _fix_config_paths(self):
        """Fix any absolute paths in the configuration to be workspace-relative"""
        # Fix app paths
        if "app" in self.config:
            if "save_user_data_path" in self.config["app"]:
                self.config["app"]["save_user_data_path"] = "./data"
            if "trained_gestures_path" in self.config["app"]:
                self.config["app"]["trained_gestures_path"] = "./data/gestures"
        
        # Fix advanced paths
        if "advanced" in self.config:
            if "gesture_data_folder" in self.config["advanced"]:
                self.config["advanced"]["gesture_data_folder"] = "./data/gestures"
            if "model_folder" in self.config["advanced"]:
                self.config["advanced"]["model_folder"] = "./data/models"
        
        # Fix other paths
        if "paths" in self.config:
            if "model_folder" in self.config["paths"]:
                self.config["paths"]["model_folder"] = "./data/models"
            if "gesture_data_folder" in self.config["paths"]:
                self.config["paths"]["gesture_data_folder"] = "./data/gestures"
    
    def _init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for settings categories
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Camera settings tab
        self.camera_tab = QWidget()
        self._init_camera_tab()
        self.tab_widget.addTab(self.camera_tab, "Camera")
        
        # Detection settings tab
        self.detection_tab = QWidget()
        self._init_detection_tab()
        self.tab_widget.addTab(self.detection_tab, "Detection")
        
        # Gesture mapping tab
        self.mapping_tab = QWidget()
        self._init_mapping_tab()
        self.tab_widget.addTab(self.mapping_tab, "Gesture Mapping")
        
        # UI settings tab
        self.ui_tab = QWidget()
        self._init_ui_tab()
        self.tab_widget.addTab(self.ui_tab, "Interface")
        
        # Privacy settings tab
        self.privacy_tab = QWidget()
        self._init_privacy_tab()
        self.tab_widget.addTab(self.privacy_tab, "Privacy")
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_settings)
        buttons_layout.addWidget(self.reset_button)
        
        # Spacer
        buttons_layout.addStretch(1)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_changes)
        buttons_layout.addWidget(self.cancel_button)
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_settings)
        buttons_layout.addWidget(self.apply_button)
        
        main_layout.addLayout(buttons_layout)
    
    def _init_camera_tab(self):
        """Initialize the camera settings tab"""
        layout = QFormLayout(self.camera_tab)
        
        # Device selection
        self.camera_device_combo = QComboBox()
        self.camera_device_combo.addItem("Default Camera (0)", 0)
        self.camera_device_combo.addItem("Camera 1", 1)
        self.camera_device_combo.addItem("Camera 2", 2)
        self.camera_device_combo.currentIndexChanged.connect(self._setting_changed)
        layout.addRow("Camera Device:", self.camera_device_combo)
        
        # Resolution
        resolution_layout = QHBoxLayout()
        
        self.camera_width_spin = QSpinBox()
        self.camera_width_spin.setRange(320, 1920)
        self.camera_width_spin.setSingleStep(8)
        self.camera_width_spin.valueChanged.connect(self._setting_changed)
        resolution_layout.addWidget(self.camera_width_spin)
        
        resolution_layout.addWidget(QLabel("x"))
        
        self.camera_height_spin = QSpinBox()
        self.camera_height_spin.setRange(240, 1080)
        self.camera_height_spin.setSingleStep(8)
        self.camera_height_spin.valueChanged.connect(self._setting_changed)
        resolution_layout.addWidget(self.camera_height_spin)
        
        layout.addRow("Resolution:", resolution_layout)
        
        # FPS
        self.camera_fps_spin = QSpinBox()
        self.camera_fps_spin.setRange(1, 60)
        self.camera_fps_spin.valueChanged.connect(self._setting_changed)
        layout.addRow("Frames Per Second:", self.camera_fps_spin)
    
    def _init_detection_tab(self):
        """Initialize the detection settings tab"""
        layout = QVBoxLayout(self.detection_tab)
        
        # Detection engine
        engine_group = QGroupBox("Detection Engine")
        engine_layout = QFormLayout(engine_group)
        
        self.detection_engine_combo = QComboBox()
        self.detection_engine_combo.addItem("MediaPipe", "mediapipe")
        self.detection_engine_combo.addItem("YOLOv8", "yolov8")
        self.detection_engine_combo.currentIndexChanged.connect(self._setting_changed)
        engine_layout.addRow("Engine:", self.detection_engine_combo)
        
        layout.addWidget(engine_group)
        
        # Detection parameters
        params_group = QGroupBox("Detection Parameters")
        params_layout = QFormLayout(params_group)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self._setting_changed)
        
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self.confidence_slider)
        self.confidence_value = QLabel("50%")
        confidence_layout.addWidget(self.confidence_value)
        
        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence_value.setText(f"{value}%")
        )
        
        params_layout.addRow("Confidence Threshold:", confidence_layout)
        
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.1, 5.0)
        self.cooldown_spin.setSingleStep(0.1)
        self.cooldown_spin.setDecimals(1)
        self.cooldown_spin.setSuffix(" seconds")
        self.cooldown_spin.valueChanged.connect(self._setting_changed)
        params_layout.addRow("Cooldown Period:", self.cooldown_spin)
        
        layout.addWidget(params_group)
        
        # Detection area
        area_group = QGroupBox("Detection Area")
        area_layout = QFormLayout(area_group)
        
        self.left_margin_spin = QDoubleSpinBox()
        self.left_margin_spin.setRange(0.0, 0.5)
        self.left_margin_spin.setSingleStep(0.05)
        self.left_margin_spin.setDecimals(2)
        self.left_margin_spin.valueChanged.connect(self._setting_changed)
        area_layout.addRow("Left Margin:", self.left_margin_spin)
        
        self.right_margin_spin = QDoubleSpinBox()
        self.right_margin_spin.setRange(0.5, 1.0)
        self.right_margin_spin.setSingleStep(0.05)
        self.right_margin_spin.setDecimals(2)
        self.right_margin_spin.valueChanged.connect(self._setting_changed)
        area_layout.addRow("Right Margin:", self.right_margin_spin)
        
        self.top_margin_spin = QDoubleSpinBox()
        self.top_margin_spin.setRange(0.0, 0.5)
        self.top_margin_spin.setSingleStep(0.05)
        self.top_margin_spin.setDecimals(2)
        self.top_margin_spin.valueChanged.connect(self._setting_changed)
        area_layout.addRow("Top Margin:", self.top_margin_spin)
        
        self.bottom_margin_spin = QDoubleSpinBox()
        self.bottom_margin_spin.setRange(0.5, 1.0)
        self.bottom_margin_spin.setSingleStep(0.05)
        self.bottom_margin_spin.setDecimals(2)
        self.bottom_margin_spin.valueChanged.connect(self._setting_changed)
        area_layout.addRow("Bottom Margin:", self.bottom_margin_spin)
        
        layout.addWidget(area_group)
    
    def _init_mapping_tab(self):
        """Initialize the gesture mapping tab"""
        from PyQt5.QtWidgets import (QListWidget, QTableWidget, QTableWidgetItem, 
                                   QHeaderView, QDialog, QLineEdit, QDialogButtonBox)
        
        layout = QVBoxLayout(self.mapping_tab)
        
        # Profile selection
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Profile:"))
        
        self.profile_combo = QComboBox()
        # Will be populated from config in _populate_settings
        profile_layout.addWidget(self.profile_combo)
        
        self.add_profile_btn = QPushButton("New Profile")
        self.add_profile_btn.clicked.connect(self._add_new_profile)
        profile_layout.addWidget(self.add_profile_btn)
        
        profile_layout.addStretch(1)
        layout.addLayout(profile_layout)
        
        # Gesture mappings table
        self.mappings_table = QTableWidget()
        self.mappings_table.setColumnCount(3)
        self.mappings_table.setHorizontalHeaderLabels(["Gesture", "Action", "Description"])
        self.mappings_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mappings_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.mappings_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mappings_table.itemDoubleClicked.connect(self._edit_mapping)
        layout.addWidget(self.mappings_table)
        
        # Buttons for adding/editing/removing mappings
        button_layout = QHBoxLayout()
        
        self.add_mapping_btn = QPushButton("Add Mapping")
        self.add_mapping_btn.clicked.connect(self._add_new_mapping)
        button_layout.addWidget(self.add_mapping_btn)
        
        self.edit_mapping_btn = QPushButton("Edit")
        self.edit_mapping_btn.clicked.connect(self._edit_selected_mapping)
        button_layout.addWidget(self.edit_mapping_btn)
        
        self.remove_mapping_btn = QPushButton("Remove")
        self.remove_mapping_btn.clicked.connect(self._remove_selected_mapping)
        button_layout.addWidget(self.remove_mapping_btn)
        
        layout.addLayout(button_layout)
        
        # Explanation
        help_text = """
        <b>How to map gestures:</b>
        <ol>
            <li>First train your custom gestures in the Trainer tab</li>
            <li>Then come back here to map each gesture to an action</li>
            <li>Click "Add Mapping" to create a new gesture-to-action mapping</li>
            <li>Select the gesture and the action you want to trigger when it's detected</li>
        </ol>
        <p>You can create different profiles for different scenarios (e.g., presentations, media control).</p>
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

    @pyqtSlot()
    def _add_new_mapping(self):
        """Add a new gesture-to-action mapping"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QFormLayout, QFileDialog
        
        profile_name = self.profile_combo.currentText()
        if not profile_name:
            QMessageBox.warning(
                self,
                "No Profile Selected",
                "Please select or create a profile first."
            )
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Gesture Mapping")
        dialog.setMinimumWidth(400)
        
        # Gesture list - get available gestures from classifier
        gesture_names = self._get_available_gestures()
        
        # Dialog layout
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        
        # Gesture selection
        gesture_combo = QComboBox()
        for gesture in gesture_names:
            gesture_combo.addItem(gesture)
            
        form.addRow("Gesture:", gesture_combo)
        
        # Action type selection
        action_type_combo = QComboBox()
        action_type_combo.addItem("Keypress", "keypress")
        action_type_combo.addItem("Hotkey Combination", "hotkey")
        action_type_combo.addItem("Mouse Click", "mouse_click")
        action_type_combo.addItem("Launch Application", "launch_app")
        action_type_combo.addItem("Shell Command", "shell_command")  # Add shell command option
        form.addRow("Action Type:", action_type_combo)
        
        # Create a stacked layout for different action data inputs
        from PyQt5.QtWidgets import QStackedLayout, QWidget
        
        # Create stack for different input types
        input_stack = QStackedLayout()
        
        # 1. Key capture widget for keypress and hotkey
        key_capture_widget = QWidget()
        key_capture_layout, key_capture_edit = self._create_key_capture_widget()
        key_capture_widget.setLayout(key_capture_layout)
        input_stack.addWidget(key_capture_widget)
        
        # 2. Mouse button selection
        mouse_widget = QWidget()
        mouse_layout = QHBoxLayout(mouse_widget)
        mouse_combo = QComboBox()
        mouse_combo.addItem("Left Button", "left")
        mouse_combo.addItem("Right Button", "right")
        mouse_combo.addItem("Middle Button", "middle")
        mouse_layout.addWidget(mouse_combo)
        double_click = QCheckBox("Double Click")
        mouse_layout.addWidget(double_click)
        input_stack.addWidget(mouse_widget)
        
        # 3. App launcher with file picker
        app_widget = QWidget()
        app_layout = QHBoxLayout(app_widget)
        app_path_edit = QLineEdit()
        app_layout.addWidget(app_path_edit, 1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(lambda: self._browse_for_application(app_path_edit))
        app_layout.addWidget(browse_button)
        input_stack.addWidget(app_widget)
        
        # 4. Shell command input
        shell_widget = QWidget()
        shell_layout = QVBoxLayout(shell_widget)
        
        # Use QTextEdit instead of QLineEdit for better multi-line support
        from PyQt5.QtWidgets import QTextEdit
        shell_command_edit = QTextEdit()
        shell_command_edit.setPlaceholderText("Enter shell command...")
        shell_command_edit.setMinimumHeight(100)  # Give more space for commands
        shell_command_edit.setAcceptRichText(False)  # Plain text only
        
        # Set up for better copy/paste
        shell_command_edit.setContextMenuPolicy(Qt.DefaultContextMenu)
        shell_layout.addWidget(shell_command_edit)
        
        # Instructions
        paste_help = QLabel(f"Press Ctrl+V (or {SUPER_KEY_NAME}+V on Mac) to paste. Multi-line commands supported.")
        paste_help.setStyleSheet("color: gray; font-style: italic; font-size: 9pt;")
        shell_layout.addWidget(paste_help)
        
        # Add buttons for clipboard operations
        clipboard_layout = QHBoxLayout()
        
        paste_button = QPushButton("Paste from Clipboard")
        paste_button.clicked.connect(lambda: self._paste_to_text_edit(shell_command_edit))
        clipboard_layout.addWidget(paste_button)
        
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: shell_command_edit.clear())
        clipboard_layout.addWidget(clear_button)
        
        shell_layout.addLayout(clipboard_layout)
        
        shell_hint = QLabel("Example: ls -la or python script.py")
        shell_hint.setStyleSheet("color: gray; font-style: italic;")
        shell_layout.addWidget(shell_hint)
        
        run_in_terminal_check = QCheckBox("Run in terminal window")
        shell_layout.addWidget(run_in_terminal_check)
        
        input_stack.addWidget(shell_widget)
        
        # Create wrapper widget for the stack
        stack_widget = QWidget()
        stack_widget.setLayout(input_stack)
        form.addRow("Action Data:", stack_widget)
        
        # Action description
        description_edit = QLineEdit()
        form.addRow("Description:", description_edit)
        
        layout.addLayout(form)
        
        # Function to update input stack based on action type
        def update_action_input(index):
            action_type = action_type_combo.currentData()
            if action_type in ("keypress", "hotkey"):
                input_stack.setCurrentIndex(0)  # Key capture
                # Update placeholder for clarity
                if action_type == "keypress":
                    key_capture_edit.setPlaceholderText("Press single key...")
                else:
                    key_capture_edit.setPlaceholderText("Press key combination...")
            elif action_type == "mouse_click":
                input_stack.setCurrentIndex(1)  # Mouse selection
            elif action_type == "launch_app":
                input_stack.setCurrentIndex(2)  # App launcher
            elif action_type == "shell_command":
                input_stack.setCurrentIndex(3)  # Shell command input
        
        # Connect the action type change to stack update
        action_type_combo.currentIndexChanged.connect(update_action_input)
        update_action_input(0)  # Initialize with default selection
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            gesture = gesture_combo.currentText()
            action_type = action_type_combo.currentData()
            
            # Get the appropriate action data based on type
            if action_type in ("keypress", "hotkey"):
                action_data = key_capture_edit.text()
            elif action_type == "mouse_click":
                button = mouse_combo.currentData()
                action_data = {
                    "button": button,
                    "double": double_click.isChecked()
                }
            elif action_type == "launch_app":
                action_data = app_path_edit.text()
            elif action_type == "shell_command":
                action_data = {
                    "command": shell_command_edit.toPlainText(),
                    "terminal": run_in_terminal_check.isChecked()
                }
            else:
                action_data = ""
                
            # Get or generate description
            description = description_edit.text()
            if not description:
                if action_type in ("keypress", "hotkey"):
                    description = f"{action_type.title()}: {action_data}"
                elif action_type == "mouse_click":
                    click_type = "Double-click" if double_click.isChecked() else "Click"
                    description = f"{click_type} {mouse_combo.currentText()}"
                elif action_type == "launch_app":
                    import os
                    app_name = os.path.basename(action_data)
                    description = f"Launch: {app_name}"
                elif action_type == "shell_command":
                    command_text = shell_command_edit.toPlainText()
                    # Handle multiline commands in description
                    if "\n" in command_text:
                        command_lines = command_text.split("\n")
                        first_line = command_lines[0]
                        description = f"Command: {first_line[:20]}... (+{len(command_lines)-1} more lines)"
                    else:
                        # Truncate long commands for display
                        if len(command_text) > 30:
                            command_text = command_text[:27] + "..."
                        description = f"Command: {command_text}"
            
            # Add mapping to configuration
            if "profiles" not in self.config:
                self.config["profiles"] = {}
            
            if profile_name not in self.config["profiles"]:
                self.config["profiles"][profile_name] = {}
            
            if "gesture_mappings" not in self.config["profiles"][profile_name]:
                self.config["profiles"][profile_name]["gesture_mappings"] = {}
            
            # Prepare and store the action configuration
            action_config = {
                "type": action_type,
                "data": action_data,
                "description": description
            }
            
            # Add to config
            self.config["profiles"][profile_name]["gesture_mappings"][gesture] = action_config
            
            # Update UI
            self._load_profile_mappings()
            self._setting_changed()

    def _populate_gesture_mapping_ui(self):
        """Populate the gesture mapping interface from current configuration"""
        # Clear existing items
        self.profile_combo.clear()
        self.mappings_table.setRowCount(0)
        
        # Get profiles from config
        profiles = self.config.get("profiles", {})
        
        # Add profiles to combo box
        for profile_name in profiles.keys():
            self.profile_combo.addItem(profile_name)
        
        # Select active profile
        active_profile = self.config.get("actions", {}).get("default_profile", "default")
        index = self.profile_combo.findText(active_profile)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        
        # Connect change signal (after populating to avoid triggering during setup)
        self.profile_combo.currentIndexChanged.connect(self._profile_selected)
        
        # Load mappings for selected profile
        self._load_profile_mappings()

    def _load_profile_mappings(self):
        """Load mappings for the currently selected profile"""
        profile_name = self.profile_combo.currentText()
        if not profile_name:
            return
        
        # Get mappings for this profile
        profiles = self.config.get("profiles", {})
        profile = profiles.get(profile_name, {})
        mappings = profile.get("gesture_mappings", {})
        
        # Clear and populate table
        self.mappings_table.setRowCount(0)
        
        row = 0
        for gesture_name, action_config in mappings.items():
            self.mappings_table.insertRow(row)
            
            # Gesture name
            self.mappings_table.setItem(row, 0, QTableWidgetItem(gesture_name))
            
            # Action type and data
            action_type = action_config.get("type", "")
            action_data = action_config.get("data", "")
            action_text = f"{action_type}: {action_data}"
            self.mappings_table.setItem(row, 1, QTableWidgetItem(action_text))
            
            # Description
            description = action_config.get("description", "")
            self.mappings_table.setItem(row, 2, QTableWidgetItem(description))
            
            row += 1
        
        # Mark settings as unchanged since this is just initial load
        self.changes_made = False

    @pyqtSlot(int)
    def _profile_selected(self, index):
        """Handle when a different profile is selected"""
        self._load_profile_mappings()
        self._setting_changed()

    @pyqtSlot()
    def _add_new_profile(self):
        """Add a new gesture mapping profile"""
        profile_name, ok = QInputDialog.getText(
            self, 
            "New Profile", 
            "Enter name for new profile:"
        )
        
        if ok and profile_name:
            # Check if profile already exists
            if profile_name in self.config.get("profiles", {}):
                QMessageBox.warning(
                    self,
                    "Duplicate Profile",
                    f"Profile '{profile_name}' already exists."
                )
                return
            
            # Add new profile to config
            if "profiles" not in self.config:
                self.config["profiles"] = {}
            
            self.config["profiles"][profile_name] = {
                "gesture_mappings": {}
            }
            
            # Update UI
            self.profile_combo.addItem(profile_name)
            self.profile_combo.setCurrentText(profile_name)
            
            # Mark settings as changed
            self._setting_changed()

    @pyqtSlot(QTableWidgetItem)
    def _edit_mapping(self, item):
        """Edit a gesture mapping when double-clicked"""
        if not item:
            return
            
        row = item.row()
        gesture_name = self.mappings_table.item(row, 0).text()
        profile_name = self.profile_combo.currentText()
        
        # Get current mapping data
        try:
            mapping_data = self.config["profiles"][profile_name]["gesture_mappings"][gesture_name]
        except KeyError:
            QMessageBox.warning(
                self,
                "Mapping Not Found",
                f"Mapping for '{gesture_name}' not found in profile '{profile_name}'."
            )
            return
            
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QFormLayout, QStackedLayout, QWidget
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Mapping for '{gesture_name}'")
        dialog.setMinimumWidth(400)
        
        # Dialog layout
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        
        # Display gesture name (not editable)
        gesture_label = QLabel(gesture_name)
        form.addRow("Gesture:", gesture_label)
        
        # Action type selection
        action_type_combo = QComboBox()
        action_type_combo.addItem("Keypress", "keypress")
        action_type_combo.addItem("Hotkey Combination", "hotkey")
        action_type_combo.addItem("Mouse Click", "mouse_click")
        action_type_combo.addItem("Launch Application", "launch_app")
        action_type_combo.addItem("Shell Command", "shell_command")  # Add shell command option
        
        # Set current action type
        current_action_type = mapping_data.get("type", "keypress")
        index = action_type_combo.findData(current_action_type)
        if index >= 0:
            action_type_combo.setCurrentIndex(index)
            
        form.addRow("Action Type:", action_type_combo)
        
        # Create a stacked layout for different action data inputs
        input_stack = QStackedLayout()
        
        # 1. Key capture widget for keypress and hotkey
        key_capture_widget = QWidget()
        # Get current value for key binding
        current_key = ""
        if current_action_type in ("keypress", "hotkey"):
            current_key = str(mapping_data.get("data", ""))
        key_capture_layout, key_capture_edit = self._create_key_capture_widget(current_key)
        key_capture_widget.setLayout(key_capture_layout)
        input_stack.addWidget(key_capture_widget)
        
        # 2. Mouse button selection
        mouse_widget = QWidget()
        mouse_layout = QHBoxLayout(mouse_widget)
        mouse_combo = QComboBox()
        mouse_combo.addItem("Left Button", "left")
        mouse_combo.addItem("Right Button", "right")
        mouse_combo.addItem("Middle Button", "middle")
        
        # Set current mouse button if applicable
        if current_action_type == "mouse_click" and isinstance(mapping_data.get("data"), dict):
            current_button = mapping_data["data"].get("button", "left")
            current_double = mapping_data["data"].get("double", False)
            
            index = mouse_combo.findData(current_button)
            if index >= 0:
                mouse_combo.setCurrentIndex(index)
                
            double_click = QCheckBox("Double Click")
            double_click.setChecked(current_double)
        else:
            double_click = QCheckBox("Double Click")
            
        mouse_layout.addWidget(mouse_combo)
        mouse_layout.addWidget(double_click)
        input_stack.addWidget(mouse_widget)
        
        # 3. App launcher with file picker
        app_widget = QWidget()
        app_layout = QHBoxLayout(app_widget)
        app_path_edit = QLineEdit()
        
        # Set current app path if applicable
        if current_action_type == "launch_app":
            app_path_edit.setText(str(mapping_data.get("data", "")))
            
        app_layout.addWidget(app_path_edit, 1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(lambda: self._browse_for_application(app_path_edit))
        app_layout.addWidget(browse_button)
        input_stack.addWidget(app_widget)
        
        # 4. Shell command input
        shell_widget = QWidget()
        shell_layout = QVBoxLayout(shell_widget)
        
        # Use QTextEdit instead of QLineEdit for better multi-line support
        from PyQt5.QtWidgets import QTextEdit
        shell_command_edit = QTextEdit()
        shell_command_edit.setPlaceholderText("Enter shell command...")
        shell_command_edit.setMinimumHeight(100)  # Give more space for commands
        shell_command_edit.setAcceptRichText(False)  # Plain text only
        
        # Set up for better copy/paste
        shell_command_edit.setContextMenuPolicy(Qt.DefaultContextMenu)
        shell_layout.addWidget(shell_command_edit)
        
        # Instructions
        paste_help = QLabel(f"Press Ctrl+V (or {SUPER_KEY_NAME}+V on Mac) to paste. Multi-line commands supported.")
        paste_help.setStyleSheet("color: gray; font-style: italic; font-size: 9pt;")
        shell_layout.addWidget(paste_help)
        
        # Add buttons for clipboard operations
        clipboard_layout = QHBoxLayout()
        
        paste_button = QPushButton("Paste from Clipboard")
        paste_button.clicked.connect(lambda: self._paste_to_text_edit(shell_command_edit))
        clipboard_layout.addWidget(paste_button)
        
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: shell_command_edit.clear())
        clipboard_layout.addWidget(clear_button)
        
        shell_layout.addLayout(clipboard_layout)
        
        shell_hint = QLabel("Example: ls -la or python script.py")
        shell_hint.setStyleSheet("color: gray; font-style: italic;")
        shell_layout.addWidget(shell_hint)
        
        run_in_terminal_check = QCheckBox("Run in terminal window")
        shell_layout.addWidget(run_in_terminal_check)
        
        input_stack.addWidget(shell_widget)
        
        # Set current shell command if applicable
        if current_action_type == "shell_command" and isinstance(mapping_data.get("data"), dict):
            shell_command_edit.setText(mapping_data["data"].get("command", ""))
            run_in_terminal_check.setChecked(mapping_data["data"].get("terminal", False))
        
        # Create wrapper widget for the stack
        stack_widget = QWidget()
        stack_widget.setLayout(input_stack)
        form.addRow("Action Data:", stack_widget)
        
        # Action description
        description_edit = QLineEdit(mapping_data.get("description", ""))
        form.addRow("Description:", description_edit)
        
        layout.addLayout(form)
        
        # Function to update input stack based on action type
        def update_action_input(index):
            action_type = action_type_combo.currentData()
            if action_type in ("keypress", "hotkey"):
                input_stack.setCurrentIndex(0)  # Key capture
                # Update placeholder for clarity
                if action_type == "keypress":
                    key_capture_edit.setPlaceholderText("Press single key...")
                else:
                    key_capture_edit.setPlaceholderText("Press key combination...")
            elif action_type == "mouse_click":
                input_stack.setCurrentIndex(1)  # Mouse selection
            elif action_type == "launch_app":
                input_stack.setCurrentIndex(2)  # App launcher
            elif action_type == "shell_command":
                input_stack.setCurrentIndex(3)  # Shell command input
        
        # Connect the action type change to stack update
        action_type_combo.currentIndexChanged.connect(update_action_input)
        update_action_input(action_type_combo.currentIndex())  # Initialize with current selection
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            action_type = action_type_combo.currentData()
            
            # Get the appropriate action data based on type
            if action_type in ("keypress", "hotkey"):
                action_data = key_capture_edit.text()
            elif action_type == "mouse_click":
                action_data = {
                    "button": mouse_combo.currentData(),
                    "double": double_click.isChecked()
                }
            elif action_type == "launch_app":
                action_data = app_path_edit.text()
            elif action_type == "shell_command":
                action_data = {
                    "command": shell_command_edit.toPlainText(),  # Use toPlainText() for QTextEdit
                    "terminal": run_in_terminal_check.isChecked()
                }
            else:
                action_data = ""
            
            # Get or generate description
            description = description_edit.text()
            if not description:
                if action_type in ("keypress", "hotkey"):
                    description = f"{action_type.title()}: {action_data}"
                elif action_type == "mouse_click":
                    click_type = "Double-click" if double_click.isChecked() else "Click"
                    description = f"{click_type} {mouse_combo.currentText()}"
                elif action_type == "launch_app":
                    import os
                    app_name = os.path.basename(action_data)
                    description = f"Launch: {app_name}"
                elif action_type == "shell_command":
                    command_text = shell_command_edit.toPlainText()
                    # Handle multiline commands in description
                    if "\n" in command_text:
                        command_lines = command_text.split("\n")
                        first_line = command_lines[0]
                        description = f"Command: {first_line[:20]}... (+{len(command_lines)-1} more lines)"
                    else:
                        # Truncate long commands for display
                        if len(command_text) > 30:
                            command_text = command_text[:27] + "..."
                        description = f"Command: {command_text}"
            
            # Prepare and store the updated action configuration
            action_config = {
                "type": action_type,
                "data": action_data,
                "description": description
            }
            
            # Update the config
            self.config["profiles"][profile_name]["gesture_mappings"][gesture_name] = action_config
            
            # Update UI
            self._load_profile_mappings()
            self._setting_changed()

    @pyqtSlot()
    def _edit_selected_mapping(self):
        """Edit the selected gesture mapping"""
        selected_items = self.mappings_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a mapping to edit."
            )
            return
            
        # Get row of first selected item
        row = selected_items[0].row()
        self._edit_mapping(self.mappings_table.item(row, 0))

    @pyqtSlot()
    def _remove_selected_mapping(self):
        """Remove the selected gesture mapping"""
        selected_items = self.mappings_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a mapping to remove."
            )
            return
            
        # Get row of first selected item
        row = selected_items[0].row()
        gesture_name = self.mappings_table.item(row, 0).text()
        profile_name = self.profile_combo.currentText()
        
        # Confirm removal
        confirm = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove the mapping for '{gesture_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Remove from config
                del self.config["profiles"][profile_name]["gesture_mappings"][gesture_name]
                
                # Update UI
                self.mappings_table.removeRow(row)
                self._setting_changed()
            except KeyError:
                QMessageBox.warning(
                    self,
                    "Mapping Not Found",
                    f"Mapping for '{gesture_name}' not found in profile '{profile_name}'."
                )

    def _get_available_gestures(self):
        """Get a list of available trained gestures from the workspace data directory"""
        available_gestures = set() # Use a set to avoid duplicates
        
        # Define the primary paths to search within the workspace data directory
        # Use path_manager functions to ensure consistency
        gesture_paths_to_check = [
            get_data_dir("gestures"),  # ./data/gestures
            get_data_dir("default")   # ./data/default
        ]
        
        # Add paths from configuration, resolved to workspace
        if "app" in self.config and "trained_gestures_path" in self.config["app"]:
            path = resolve_path(self.config["app"]["trained_gestures_path"], create_if_missing=True)
            if path not in gesture_paths_to_check:
                gesture_paths_to_check.append(path)
        
        if "paths" in self.config and "gesture_data_folder" in self.config["paths"]:
            path = resolve_path(self.config["paths"]["gesture_data_folder"], create_if_missing=True)
            if path not in gesture_paths_to_check:
                gesture_paths_to_check.append(path)

        logger.info(f"Searching for gestures in: {gesture_paths_to_check}")
        
        # Search for gestures in all specified locations
        for gesture_path in gesture_paths_to_check:
            if os.path.exists(gesture_path) and os.path.isdir(gesture_path):
                logger.info(f"Scanning directory: {gesture_path}")
                try:
                    items = os.listdir(gesture_path)
                    logger.debug(f"Contents of {gesture_path}: {items}")

                    # Check for the main gesture data file first
                    json_path = os.path.join(gesture_path, "gesture_data.json")
                    if os.path.exists(json_path) and os.path.isfile(json_path):
                        logger.debug(f"Found gesture_data.json in {gesture_path}")
                        import json
                        try:
                            with open(json_path, 'r') as f:
                                gesture_data = json.load(f)
                            # Expecting format like {"gesture_name": {"samples": [...], "trained": bool}}
                            if isinstance(gesture_data, dict):
                                # Check if it's the old format {"gestures": {...}}
                                if "gestures" in gesture_data and isinstance(gesture_data["gestures"], dict):
                                     gesture_dict = gesture_data["gestures"]
                                else:
                                     gesture_dict = gesture_data # Assume new format

                                for gesture_name in gesture_dict.keys():
                                    if isinstance(gesture_name, str) and gesture_name: # Basic validation
                                        available_gestures.add(gesture_name)
                                        logger.info(f"Found gesture '{gesture_name}' in {json_path}")
                                    else:
                                        logger.warning(f"Skipping invalid gesture key '{gesture_name}' in {json_path}")
                            else:
                                logger.warning(f"Unexpected format in {json_path}, expected a dictionary.")
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON from {json_path}")
                        except Exception as e:
                            logger.error(f"Error reading gesture data from {json_path}: {e}")

                    # Also check for individual files/directories as gestures
                    valid_extensions = ['.gesture', '.data', '.json', '.yaml', '.yml', '.pkl', '.model']
                    for item in items:
                        item_path = os.path.join(gesture_path, item)
                        
                        # Directories might represent gesture names
                        if os.path.isdir(item_path):
                            # Check if directory name looks like a valid gesture name (basic check)
                            if item and not item.startswith('.'):
                                available_gestures.add(item)
                                logger.info(f"Found potential gesture directory: {item}")
                        
                        # Files with specific extensions might be gesture data
                        elif os.path.isfile(item_path):
                            name, ext = os.path.splitext(item)
                            # Ensure name is valid and extension is recognized, ignore gesture_data.json itself
                            if name and not name.startswith('.') and name != "gesture_data" and ext.lower() in valid_extensions:
                                available_gestures.add(name)
                                logger.info(f"Found potential gesture file: {name}{ext}")
                            elif name == "gesture_data" and ext.lower() == ".json":
                                pass # Already handled above
                            else:
                                logger.debug(f"Skipping file (invalid name/extension): {item}")
                
                except Exception as e:
                    logger.error(f"Error scanning directory {gesture_path}: {e}")
            else:
                 logger.warning(f"Gesture path does not exist or is not a directory: {gesture_path}")
        
        # Convert set to sorted list for consistent UI order
        final_gestures = sorted(list(available_gestures))
        
        # Log final results
        if final_gestures:
            logger.info(f"Final list of available gestures: {final_gestures}")
        else:
            logger.warning("No trained gestures found after scanning all paths.")
            # Optionally, provide more context if possible
            logger.info(f"Checked paths: {gesture_paths_to_check}")
            # You might add a check here to see if the gesture_data.json file exists but is empty/invalid

        return final_gestures

    def _ensure_gesture_storage_directory(self):
        """Ensure the directory for storing trained gestures exists"""
        # Use the path manager to ensure all directories exist
        ensure_workspace_paths()

    def _populate_settings(self):
        """Populate settings controls from current configuration"""
        try:
            # Camera settings
            camera_config = self.config.get("camera", {})
            
            device_id = camera_config.get("device_id", 0)
            index = self.camera_device_combo.findData(device_id)
            if index >= 0:
                self.camera_device_combo.setCurrentIndex(index)
            
            self.camera_width_spin.setValue(camera_config.get("width", 640))
            self.camera_height_spin.setValue(camera_config.get("height", 480))
            self.camera_fps_spin.setValue(camera_config.get("fps", 15))
            
            # Detection settings
            detection_config = self.config.get("detection", {})
            
            engine = detection_config.get("engine", "mediapipe")
            index = self.detection_engine_combo.findData(engine)
            if index >= 0:
                self.detection_engine_combo.setCurrentIndex(index)
            
            confidence = int(detection_config.get("confidence_threshold", 0.7) * 100)
            self.confidence_slider.setValue(confidence)
            self.confidence_value.setText(f"{confidence}%")
            
            self.cooldown_spin.setValue(detection_config.get("cooldown_seconds", 1.0))
            
            area = detection_config.get("detection_area", {})
            self.left_margin_spin.setValue(area.get("left", 0.1))
            self.right_margin_spin.setValue(area.get("right", 0.9))
            self.top_margin_spin.setValue(area.get("top", 0.1))
            self.bottom_margin_spin.setValue(area.get("bottom", 0.9))
            
            # Gesture mapping settings
            self._populate_gesture_mapping_ui()
            
            # UI settings
            ui_config = self.config.get("ui", {})
            
            theme = ui_config.get("theme", "system")
            index = self.theme_combo.findData(theme)
            if index >= 0:
                self.theme_combo.setCurrentIndex(index)
            
            self.show_preview_check.setChecked(ui_config.get("show_preview", True))
            self.overlay_feedback_check.setChecked(ui_config.get("overlay_feedback", True))
            self.overlay_timeout_spin.setValue(ui_config.get("overlay_timeout_ms", 1500))
            self.minimize_to_tray_check.setChecked(ui_config.get("minimize_to_tray", True))
            self.start_minimized_check.setChecked(ui_config.get("start_minimized", False))
            
            # App settings
            app_config = self.config.get("app", {})
            
            self.launch_startup_check.setChecked(app_config.get("launch_at_startup", False))
            
            # Log level
            log_level = app_config.get("log_level", "info")
            index = self.log_level_combo.findData(log_level)
            if index >= 0:
                self.log_level_combo.setCurrentIndex(index)
            
            # User data path
            user_data_path = app_config.get("save_user_data_path", "data/user_gestures")
            self.user_data_path_label.setText(user_data_path)
            
            # Reset change tracking
            self.changes_made = False
            self.apply_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            
        except Exception as e:
            logger.error(f"Error populating settings: {e}")
            QMessageBox.warning(
                self,
                "Settings Error",
                f"Failed to populate settings: {str(e)}"
            )
    
    @pyqtSlot()
    def _setting_changed(self):
        """Handle when a setting is changed"""
        self.changes_made = True
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
    
    @pyqtSlot()
    def _apply_settings(self):
        """Apply settings changes"""
        try:
            # Camera settings
            if "camera" not in self.config:
                self.config["camera"] = {}
                
            self.config["camera"]["device_id"] = self.camera_device_combo.currentData()
            self.config["camera"]["width"] = self.camera_width_spin.value()
            self.config["camera"]["height"] = self.camera_height_spin.value()
            self.config["camera"]["fps"] = self.camera_fps_spin.value()
            
            # Detection settings
            if "detection" not in self.config:
                self.config["detection"] = {}
                
            self.config["detection"]["engine"] = self.detection_engine_combo.currentData()
            self.config["detection"]["confidence_threshold"] = self.confidence_slider.value() / 100.0
            self.config["detection"]["cooldown_seconds"] = self.cooldown_spin.value()
            
            if "detection_area" not in self.config["detection"]:
                self.config["detection"]["detection_area"] = {}
                
            self.config["detection"]["detection_area"]["left"] = self.left_margin_spin.value()
            self.config["detection"]["detection_area"]["right"] = self.right_margin_spin.value()
            self.config["detection"]["detection_area"]["top"] = self.top_margin_spin.value()
            self.config["detection"]["detection_area"]["bottom"] = self.bottom_margin_spin.value()
            
            # UI settings
            if "ui" not in self.config:
                self.config["ui"] = {}
                
            self.config["ui"]["theme"] = self.theme_combo.currentData()
            self.config["ui"]["show_preview"] = self.show_preview_check.isChecked()
            self.config["ui"]["overlay_feedback"] = self.overlay_feedback_check.isChecked()
            self.config["ui"]["overlay_timeout_ms"] = self.overlay_timeout_spin.value()
            self.config["ui"]["minimize_to_tray"] = self.minimize_to_tray_check.isChecked()
            self.config["ui"]["start_minimized"] = self.start_minimized_check.isChecked()
            
            # App settings
            if "app" not in self.config:
                self.config["app"] = {}
                
            self.config["app"]["launch_at_startup"] = self.launch_startup_check.isChecked()
            self.config["app"]["log_level"] = self.log_level_combo.currentData()
            
            # Save the configuration
            success = self.config_manager.save_config()
            
            if success:
                self.changes_made = False
                self.apply_button.setEnabled(False)
                self.cancel_button.setEnabled(False)
                
                # Emit signal to notify that settings have changed
                self.settings_changed.emit()
                
                QMessageBox.information(
                    self,
                    "Settings Applied",
                    "Settings have been applied successfully."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Settings Error",
                    "Failed to save settings."
                )
                
        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            QMessageBox.critical(
                self,
                "Settings Error",
                f"Failed to apply settings: {str(e)}"
            )
    
    @pyqtSlot()
    def _cancel_changes(self):
        """Cancel settings changes"""
        if self.changes_made:
            # Reset to original settings
            self._populate_settings()
            
            self.changes_made = False
            self.apply_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
    
    @pyqtSlot()
    def _reset_settings(self):
        """Reset settings to defaults"""
        confirm = QMessageBox.question(
            self,
            "Confirm Reset",
            "Are you sure you want to reset all settings to their default values? This will also delete all custom gestures.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # First delete all gesture data
                gesture_data_file = get_data_dir("default") + "/gesture_data.json"
                model_files_to_delete = []
                
                # Find model files for all profiles
                models_dir = get_data_dir("models")
                if os.path.exists(models_dir):
                    for filename in os.listdir(models_dir):
                        if filename.startswith("gesture_"):
                            model_files_to_delete.append(os.path.join(models_dir, filename))
                
                # Delete gesture data file
                if os.path.exists(gesture_data_file):
                    try:
                        os.remove(gesture_data_file)
                        logger.info(f"Deleted gesture data file: {gesture_data_file}")
                    except Exception as e:
                        logger.error(f"Failed to delete gesture data file: {e}")
                
                # Delete model files
                for file_path in model_files_to_delete:
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted model file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete model file: {e}")
                
                # Reset configuration
                self.config = self.config_manager.reset_to_default()
                
                # Repopulate settings
                self._populate_settings()
                
                # Emit signal to notify that settings have changed
                self.settings_changed.emit()
                
                QMessageBox.information(
                    self,
                    "Settings Reset",
                    "Settings have been reset to defaults and all gesture data has been cleared."
                )
                
            except Exception as e:
                logger.error(f"Error resetting settings: {e}")
                QMessageBox.critical(
                    self,
                    "Settings Error",
                    f"Failed to reset settings: {str(e)}"
                )

    @pyqtSlot()
    def _change_data_path(self):
        """Change the user data directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            confirm = QMessageBox.question(
                self,
                "Confirm Change",
                "Changing the data directory will not move existing data. "
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if confirm == QMessageBox.Yes:
                # Update config
                if "app" not in self.config:
                    self.config["app"] = {}
                
                self.config["app"]["save_user_data_path"] = directory
                
                # Update UI
                self.user_data_path_label.setText(directory)
                
                # Mark as changed
                self._setting_changed()
    
    @pyqtSlot()
    def _clear_user_data(self):
        """Clear all user data"""
        confirm = QMessageBox.question(
            self,
            "Confirm Data Deletion",
            "Are you sure you want to delete all user data? "
            "This will remove all your custom gestures and settings. "
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Get data directory
                data_path = self.config.get("app", {}).get("save_user_data_path", "data/user_gestures")
                data_path = os.path.expanduser(data_path)
                
                if os.path.exists(data_path):
                    # Delete all files in the directory
                    for file in os.listdir(data_path):
                        file_path = os.path.join(data_path, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                import shutil
                                shutil.rmtree(file_path)
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {e}")
                    
                    QMessageBox.information(
                        self,
                        "Data Cleared",
                        "All user data has been deleted."
                    )
                else:
                    QMessageBox.information(
                        self,
                        "No Data",
                        "No user data found to delete."
                    )
                
            except Exception as e:
                logger.error(f"Error clearing user data: {e}")
                QMessageBox.critical(
                    self,
                    "Data Error",
                    f"Failed to clear user data: {str(e)}"
                )

    def _init_ui_tab(self):
        """Initialize the UI settings tab"""
        layout = QFormLayout(self.ui_tab)
        
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("System", "system")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.currentIndexChanged.connect(self._setting_changed)
        layout.addRow("Theme:", self.theme_combo)
        
        # Show preview
        self.show_preview_check = QCheckBox()
        self.show_preview_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Show Camera Preview:", self.show_preview_check)
        
        # Show overlay feedback
        self.overlay_feedback_check = QCheckBox()
        self.overlay_feedback_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Show Action Overlay:", self.overlay_feedback_check)
        
        # Overlay timeout
        self.overlay_timeout_spin = QSpinBox()
        self.overlay_timeout_spin.setRange(500, 5000)
        self.overlay_timeout_spin.setSingleStep(100)
        self.overlay_timeout_spin.setSuffix(" ms")
        self.overlay_timeout_spin.valueChanged.connect(self._setting_changed)
        layout.addRow("Overlay Duration:", self.overlay_timeout_spin)
        
        # Minimize to tray
        self.minimize_to_tray_check = QCheckBox()
        self.minimize_to_tray_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Minimize to Tray:", self.minimize_to_tray_check)
        
        # Start minimized
        self.start_minimized_check = QCheckBox()
        self.start_minimized_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Start Minimized:", self.start_minimized_check)
        
        # Launch at startup
        self.launch_startup_check = QCheckBox()
        self.launch_startup_check.stateChanged.connect(self._setting_changed)
        layout.addRow("Launch at Startup:", self.launch_startup_check)

    def _init_privacy_tab(self):
        """Initialize the privacy settings tab"""
        layout = QVBoxLayout(self.privacy_tab)
        
        # Privacy info
        info_label = QLabel(
            "GestureBind respects your privacy by processing all data locally on your device. "
            "No video or gesture data is transmitted over the internet."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Privacy options
        privacy_group = QGroupBox("Privacy Options")
        privacy_layout = QFormLayout(privacy_group)
        
        self.local_processing_check = QCheckBox()
        self.local_processing_check.setChecked(True)
        self.local_processing_check.setEnabled(False)  # This is always enforced
        privacy_layout.addRow("Process all data locally:", self.local_processing_check)
        
        # Logging level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItem("Debug", "debug")
        self.log_level_combo.addItem("Info", "info")
        self.log_level_combo.addItem("Warning", "warning")
        self.log_level_combo.addItem("Error", "error")
        self.log_level_combo.addItem("Critical", "critical")
        self.log_level_combo.currentIndexChanged.connect(self._setting_changed)
        privacy_layout.addRow("Logging Level:", self.log_level_combo)
        
        # Data storage options
        self.user_data_path = QPushButton("Change Data Directory")
        self.user_data_path.clicked.connect(self._change_data_path)
        self.user_data_path_label = QLabel()
        privacy_layout.addRow("Data Storage:", self.user_data_path)
        privacy_layout.addRow("", self.user_data_path_label)
        
        # Clear data button
        self.clear_data_button = QPushButton("Clear All User Data")
        self.clear_data_button.clicked.connect(self._clear_user_data)
        privacy_layout.addRow("", self.clear_data_button)
        
        layout.addWidget(privacy_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)

    def _create_key_capture_widget(self, initial_value=""):
        """
        Create a widget that can capture keyboard input for sequential key combinations
        
        Args:
            initial_value: Initial text to display
            
        Returns:
            Tuple of (layout, line_edit) containing the widget layout and the line edit where the key is stored
        """
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QFrame, QGridLayout, QStackedWidget
        
        # Create main layout for the key capture widget
        main_layout = QVBoxLayout()
        
        # Create the container for the key sequence (key slots)
        key_sequence_layout = QHBoxLayout()
        
        # Create fields for up to 3 keys in the sequence
        key_slots = []
        key_combos = []
        
        # Tracking the active key slot and whether we're waiting for a keypress
        self.active_key_slot = 0
        self.waiting_for_keypress = False
        
        # Create a readable representation of key sequence for storage
        key_sequence_edit = QLineEdit(initial_value)
        key_sequence_edit.setReadOnly(True)
        key_sequence_edit.setVisible(False)  # Hidden storage field
        
        # Parse initial value into key slots if provided
        initial_keys = []
        if initial_value:
            initial_keys = initial_value.split("+")
        
        # Label to show instructions
        instructions_label = QLabel("Select or press the keys in sequence")
        instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setStyleSheet("font-style: italic;")
        main_layout.addWidget(instructions_label)
        
        # Function to update the full key sequence from individual slots
        def update_key_sequence():
            keys = []
            for slot in key_slots:
                if slot.text() and slot.text() != "Press Key...":
                    keys.append(slot.text())
            key_sequence_edit.setText("+".join(keys))
        
        # Function to select next key slot
        def select_next_slot(current_index):
            if current_index < len(key_slots) - 1:
                activate_slot(current_index + 1)
                return True
            return False
        
        # Function to activate a specific slot
        def activate_slot(index):
            nonlocal self
            if index < 0 or index >= len(key_slots):
                return
                
            self.active_key_slot = index
            self.waiting_for_keypress = True
            
            # Update styling for all slots
            for i, slot in enumerate(key_slots):
                if i == index:
                    slot.setStyleSheet("background-color: #e0f0ff; border: 2px solid #3399ff; padding: 4px;")
                    slot.setText("Press Key..." if not slot.text() or slot.text() == "Press Key..." else slot.text())
                else:
                    if slot.text() and slot.text() != "Press Key...":
                        slot.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc; padding: 4px;")
                    else:
                        slot.setStyleSheet("background-color: #f8f8f8; border: 1px dashed #cccccc; padding: 4px;")
            
            # Set focus to the active slot to capture keypresses
            key_slots[index].setFocus()
            
            # Update the instruction label
            if index == 0:
                instructions_label.setText("Select first key (required)")
            elif index == 1:
                instructions_label.setText("Select second key (required)")
            else:
                instructions_label.setText("Select third key (optional)")
        
        # Create the key slots
        for i in range(3):  # 3 key slots
            key_frame = QFrame()
            key_frame.setFrameShape(QFrame.StyledPanel)
            key_frame.setMinimumWidth(120)
            key_frame.setMinimumHeight(30)
            
            # Layout for this key slot
            slot_layout = QVBoxLayout(key_frame)
            slot_layout.setContentsMargins(0, 0, 0, 0)
            
            # Label for this key
            key_label = QLabel(f"Key {i+1}")
            key_label.setAlignment(Qt.AlignCenter)
            
            # Text field for the key
            key_edit = QLineEdit()
            key_edit.setReadOnly(True)
            key_edit.setAlignment(Qt.AlignCenter)
            
            # Set initial value if available
            if i < len(initial_keys):
                key_edit.setText(initial_keys[i])
            else:
                key_edit.setText("Press Key..." if i <= 1 else "")  # First two keys required
            
            # Create special key dropdown for this slot
            special_keys_combo = QComboBox()
            special_keys_combo.addItem("Select special key...", None)
            special_keys_combo.addItem("Ctrl", "ctrl")
            special_keys_combo.addItem("Shift", "shift")
            special_keys_combo.addItem("Alt", "alt")
            special_keys_combo.addItem(SUPER_KEY_NAME, SUPER_KEY_INTERNAL)  # Platform-specific Super key name
            special_keys_combo.addItem("F1", "f1")
            special_keys_combo.addItem("F2", "f2")
            special_keys_combo.addItem("F3", "f3")
            special_keys_combo.addItem("F4", "f4")
            special_keys_combo.addItem("F5", "f5")
            special_keys_combo.addItem("F6", "f6")
            special_keys_combo.addItem("F7", "f7")
            special_keys_combo.addItem("F8", "f8")
            special_keys_combo.addItem("F9", "f9")
            special_keys_combo.addItem("F10", "f10")
            special_keys_combo.addItem("F11", "f11")
            special_keys_combo.addItem("F12", "f12")
            special_keys_combo.addItem("Esc", "esc")
            special_keys_combo.addItem("Tab", "tab")
            special_keys_combo.addItem("Backspace", "backspace")
            special_keys_combo.addItem("Enter", "enter")
            special_keys_combo.addItem("Space", "space")
            special_keys_combo.addItem("Delete", "delete")
            special_keys_combo.addItem("Home", "home")
            special_keys_combo.addItem("End", "end")
            special_keys_combo.addItem("PageUp", "pageup")
            special_keys_combo.addItem("PageDown", "pagedown")
            special_keys_combo.addItem("Left", "left")
            special_keys_combo.addItem("Right", "right")
            special_keys_combo.addItem("Up", "up")
            special_keys_combo.addItem("Down", "down")
            
            # Function to handle dropdown selection for this slot
            def create_special_key_handler(slot_index):
                def handle_special_key(index):
                    if index == 0:  # "Select special key..." option
                        return
                        
                    special_key = key_combos[slot_index].currentData()
                    key_slots[slot_index].setText(special_key)
                    key_combos[slot_index].setCurrentIndex(0)  # Reset dropdown
                    update_key_sequence()
                    
                    # Move to next slot
                    if not select_next_slot(slot_index):
                        # If last slot, set focus back to the dialog
                        key_slots[slot_index].clearFocus()
                        
                return handle_special_key
            
            special_keys_combo.currentIndexChanged.connect(create_special_key_handler(i))
            key_combos.append(special_keys_combo)
            
            # Add key label and edit to the slot layout
            slot_layout.addWidget(key_label)
            slot_layout.addWidget(key_edit)
            slot_layout.addWidget(special_keys_combo)
            
            # Store the key edit for later access
            key_slots.append(key_edit)
            
            # Add frame to the sequence layout
            key_sequence_layout.addWidget(key_frame)
            
            # Add connecting "+" label between key slots (except after the last one)
            if i < 2:  # Only add plus sign between keys, not after the last one
                plus_label = QLabel("+")
                plus_label.setAlignment(Qt.AlignCenter)
                plus_label.setStyleSheet("font-size: 16px; font-weight: bold;")
                key_sequence_layout.addWidget(plus_label)
        
        # Add key sequence layout to main layout
        main_layout.addLayout(key_sequence_layout)
        
        # Controls layout (for clear button)
        controls_layout = QHBoxLayout()
        controls_layout.addStretch(1)
        
        # Clear button to reset all key slots
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(lambda: self._clear_key_slots(key_slots, key_sequence_edit))
        controls_layout.addWidget(clear_button)
        
        main_layout.addLayout(controls_layout)
        
        # Hidden final key sequence storage
        main_layout.addWidget(key_sequence_edit)
        
        # Custom event filter to capture key presses for the active slot
        class KeySequenceFilter(QObject):
            def __init__(self, parent, key_slots, key_sequence_edit):
                super().__init__(parent)
                self.key_slots = key_slots
                self.key_sequence_edit = key_sequence_edit
                
                # Dictionary mapping key constants to readable names - with platform-specific handling
                self.key_map = {
                    Qt.Key_Super_L: SUPER_KEY_INTERNAL,
                    Qt.Key_Super_R: SUPER_KEY_INTERNAL,
                    Qt.Key_Meta: SUPER_KEY_INTERNAL,
                    Qt.Key_Alt: "alt",
                    Qt.Key_AltGr: "alt",
                    Qt.Key_Control: "ctrl",
                    Qt.Key_Shift: "shift",
                    Qt.Key_Escape: "esc",
                    Qt.Key_Return: "enter",
                    Qt.Key_Enter: "enter",
                    Qt.Key_Tab: "tab",
                    Qt.Key_Space: "space",
                    Qt.Key_Backspace: "backspace",
                    Qt.Key_Delete: "delete",
                    Qt.Key_Home: "home",
                    Qt.Key_End: "end",
                    Qt.Key_PageUp: "pageup",
                    Qt.Key_PageDown: "pagedown",
                    Qt.Key_Left: "left",
                    Qt.Key_Right: "right",
                    Qt.Key_Up: "up",
                    Qt.Key_Down: "down",
                    Qt.Key_F1: "f1",
                    Qt.Key_F2: "f2",
                    Qt.Key_F3: "f3",
                    Qt.Key_F4: "f4",
                    Qt.Key_F5: "f5",
                    Qt.Key_F6: "f6",
                    Qt.Key_F7: "f7",
                    Qt.Key_F8: "f8",
                    Qt.Key_F9: "f9",
                    Qt.Key_F10: "f10",
                    Qt.Key_F11: "f11",
                    Qt.Key_F12: "f12",
                }
                
                # Add platform-specific mappings - important for macOS which has unique keys
                if PLATFORM == "Darwin":  # macOS
                    self.key_map[Qt.Key_Meta] = "cmd"
                    self.key_map[Qt.Key_Control] = "ctrl"
                    self.key_map[Qt.Key_Option] = "alt"
                    self.key_map[Qt.Key_Command] = "cmd"
            
            def eventFilter(self, obj, event):
                if not self.parent().waiting_for_keypress:
                    return False
                    
                if event.type() == QEvent.KeyPress:
                    key = event.key()
                    modifiers = event.modifiers()
                    
                    # Get the active slot index
                    active_index = self.parent().active_key_slot
                    
                    # Skip if modifier key alone
                    if key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta, 
                              Qt.Key_Super_L, Qt.Key_Super_R, Qt.Key_AltGr):
                        return True
                    
                    # Special handling for standard shortcut keys (Ctrl+V, Ctrl+C, etc.)
                    # Allow application shortcut processing to proceed if standard shortcut keys are pressed
                    if modifiers & (Qt.ControlModifier | Qt.MetaModifier) and key in (
                        Qt.Key_C, Qt.Key_V, Qt.Key_X, Qt.Key_A, Qt.Key_Z):
                        # Let the standard shortcuts work normally
                        return False
                    
                    # Convert key to readable text
                    if key in self.key_map:
                        key_text = self.key_map[key]
                    else:
                        # For regular keys (letters, numbers, symbols)
                        key_text = QKeySequence(key).toString().lower()
                        
                        # For some reason, sometimes QKeySequence returns empty string
                        # Try to handle common cases manually
                        if not key_text and key >= Qt.Key_A and key <= Qt.Key_Z:
                            key_text = chr(key - Qt.Key_A + ord('a'))
                        elif not key_text and key >= Qt.Key_0 and key <= Qt.Key_9:
                            key_text = chr(key - Qt.Key_0 + ord('0'))
                        
                    # Log for debugging
                    logger.debug(f"Key pressed: {key}, mapped to: {key_text}, modifiers: {modifiers}")
                    
                    # Update the active slot with the pressed key
                    if key_text:  # Only update if we got a valid key text
                        self.key_slots[active_index].setText(key_text)
                        
                        # Update the full key sequence
                        update_key_sequence()
                        
                        # Move to the next slot if possible
                        if not select_next_slot(active_index):
                            # If we're at the last slot, clear focus
                            self.key_slots[active_index].clearFocus()
                    
                    return True
                
                return False
        
        # Create and install the event filter on each key slot
        key_filter = KeySequenceFilter(self, key_slots, key_sequence_edit)
        for slot in key_slots:
            slot.installEventFilter(key_filter)
        
        # Initialize the first slot as active
        activate_slot(0)
        
        # Return the main layout and the hidden key sequence edit
        return main_layout, key_sequence_edit
        
    @pyqtSlot()
    def _clear_key_slots(self, key_slots, key_sequence_edit):
        """Clear all key slots and reset the sequence"""
        for i, slot in enumerate(key_slots):
            if i <= 1:  # First two slots required
                slot.setText("Press Key...")
            else:
                slot.setText("")  # Third slot optional
        
        key_sequence_edit.setText("")
        
        # Reactivate the first slot
        self.active_key_slot = 0
        self.waiting_for_keypress = True
        
        # Update styling
        for i, slot in enumerate(key_slots):
            if i == 0:
                slot.setStyleSheet("background-color: #e0f0ff; border: 2px solid #3399ff; padding: 4px;")
            else:
                slot.setStyleSheet("background-color: #f8f8f8; border: 1px dashed #cccccc; padding: 4px;")
        
        # Set focus to the first slot
        key_slots[0].setFocus()

    def _paste_from_clipboard(self, text_edit):
        """Paste text from clipboard into the given text edit widget"""
        from PyQt5.QtWidgets import QApplication
        
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        
        if text:
            # Insert text at cursor position
            current_text = text_edit.text()
            cursor_pos = text_edit.cursorPosition()
            
            # Combine text
            new_text = current_text[:cursor_pos] + text + current_text[cursor_pos:]
            
            # Set text and move cursor to end of pasted text
            text_edit.setText(new_text)
            text_edit.setCursorPosition(cursor_pos + len(text))
            
            # Ensure text edit has focus
            text_edit.setFocus()
        else:
            logger.debug("Clipboard is empty, nothing to paste")

    def _paste_to_text_edit(self, text_edit):
        """Paste text from clipboard into the given text edit widget (works with QTextEdit)"""
        from PyQt5.QtWidgets import QApplication
        
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        
        if text:
            # For QTextEdit, use insertPlainText for best compatibility
            text_edit.insertPlainText(text)
            # Ensure text edit has focus
            text_edit.setFocus()
        else:
            logger.debug("Clipboard is empty, nothing to paste")

    def focusInEvent(self, event):
        """Handle when the application regains focus"""
        super().focusInEvent(event)
        # This helps with ensuring keyboard shortcuts work after switching applications
        QApplication.processEvents()
            
    def eventFilter(self, obj, event):
        """Custom event filter to better handle focus and key events"""
        if isinstance(obj, QLineEdit) and event.type() == QEvent.FocusIn:
            # Reset state when a line edit receives focus
            obj.deselect()  # Avoid full text selection which can lead to accidental deletion
            
        # Allow standard event processing
        return super().eventFilter(event)

# Register event filters for the entire application at startup
from PyQt5.QtWidgets import QApplication
app = QApplication.instance()
if app:
    # Create a global event filter for the application
    class GlobalEventFilter(QObject):
        def eventFilter(self, obj, event):
            # Handle application focus changes
            if event.type() == QEvent.ApplicationActivate:
                # App regained focus, process events to ensure widgets update correctly
                QApplication.processEvents()
            return False
    
    # Install the global event filter
    global_filter = GlobalEventFilter()
    app.installEventFilter(global_filter)