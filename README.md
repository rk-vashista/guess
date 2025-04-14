# GestureBind - Gesture-Based Shortcut Mapper

<div align="center">
  
![GestureBind Logo](https://via.placeholder.com/150x150.png?text=GestureBind)

**Control your computer with hand gestures**

</div>

## 📖 Overview

GestureBind is a versatile application that allows users to map hand and body gestures to custom system actions and shortcuts, enabling touchless computer control. By leveraging computer vision and machine learning technologies, GestureBind recognizes gestures captured through your webcam and triggers corresponding actions like keyboard shortcuts, mouse movements, or launching applications.

## ✨ Features

- **Real-time Gesture Detection**: Continuous webcam-based detection using MediaPipe and OpenCV
- **Predefined Gesture Recognition**: Out-of-the-box support for common gestures (peace sign, thumb up, etc.)
- **Custom Gesture Training**: (Coming soon) Train the system with your own personalized gestures
- **Action Mapping System**:
  - Keyboard shortcuts and hotkeys
  - Mouse clicks and movements
  - Application launching
  - System command execution
- **Profile Management**: Create and switch between gesture mapping profiles for different applications
- **Visual Feedback**: On-screen overlay confirmation when gestures are detected
- **Adjustable Detection Settings**: Fine-tune sensitivity, cooldown periods, and confidence thresholds
- **Background Operation**: Runs in system tray for continuous operation
- **Cross-Platform Support**: Windows, Linux, and macOS compatibility
- **Privacy-Focused**: All processing happens locally on your device

## 🛠️ Requirements

- Python 3.8 or later
- Webcam/camera device
- The following Python packages (installed automatically):
  - OpenCV
  - MediaPipe
  - PyQt5
  - PyAutoGUI
  - TensorFlow (for ML-based gesture recognition)
  - PyYAML
  - NumPy
  - pynput

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gesturebind.git
cd gesturebind

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

For Windows users, you can also use the included installer script:
```batch
scripts\install.bat
```

For Linux/macOS users:
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

## 🎮 Usage

### Getting Started

1. Launch the application by running `python main.py`
2. Allow camera access when prompted
3. Navigate to the "Gesture Control" tab
4. Click "Start Detection" to begin recognizing gestures
5. Perform gestures in front of your camera to trigger actions

### Mapping Gestures to Actions

1. Go to the "Settings" tab
2. Select or create a profile
3. Click "Add Mapping" to associate a gesture with an action:
   - Choose a gesture from the dropdown list
   - Select an action type (hotkey, mouse action, application launch)
   - Configure the specific action parameters
   - Save your mapping

### Using Profiles

Profiles allow you to create different sets of gesture mappings for various applications or contexts:

- **Default Profile**: Always available system-wide
- **Media Profile**: Optimized for media playback controls
- **Presentation Profile**: Useful for slideshow navigation

Switch between profiles from the main screen dropdown menu.

## ⚙️ Configuration

GestureBind uses YAML configuration files stored in:

- Default configuration: `config/default_config.yaml`
- User configuration: `data/config/user_config.yaml`

### Configuration Options

```yaml
# Example configuration
detection:
  engine: mediapipe  # Options: mediapipe, yolov8
  confidence_threshold: 0.7
  min_detection_duration_ms: 200

ui:
  theme: system  # Options: system, dark, light
  show_preview: true
  camera_preview_fps: 15
  overlay_feedback: true
  
profiles:
  default:
    gesture_mappings:
      peace:
        type: hotkey
        data: ctrl+l
        description: Hotkey: ctrl+l
```

## 📋 Development Status

GestureBind is currently in active development. The core functionality for gesture detection and action execution is implemented, but some features are still in progress:

- ✅ MediaPipe hands integration
- ✅ Basic gesture recognition
- ✅ Action mapping system
- ✅ Configuration management
- ✅ UI framework with PyQt5
- ⚠️ Custom gesture training (in progress)
- ⚠️ ML-based gesture classification (in progress)
- ⚠️ YOLOv8 integration (placeholder only)

## 🧪 Contributing

Contributions are welcome! If you'd like to help improve GestureBind:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

MIT

## 🔒 Privacy

GestureBind processes all camera feeds locally on your device. No video or gesture data is uploaded externally, ensuring your privacy is maintained.