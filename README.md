# 👋 GestureBind

<div align="center">
  

**Control your computer with hand gestures ✌️ 👆 👍**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?style=flat&logo=opencv&logoColor=white&color=5C3EE8)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0F9D58.svg)](https://developers.google.com/mediapipe)

</div>

---

## 🪄 Overview

**GestureBind** transforms how you interact with your computer by mapping hand gestures to system actions. Wave goodbye to keyboard shortcuts and mouse clicks—literally!

Using your webcam and advanced computer vision, GestureBind recognizes hand gestures and instantly triggers corresponding actions, from launching apps to executing keyboard shortcuts.



## ✨ Features

<table>
  <tr>
    <td width="50%">
      <h3>🎯 Real-time Detection</h3>
      <p>Continuous webcam monitoring using MediaPipe and OpenCV for instant response to your gestures.</p>
    </td>
    <td width="50%">
      <h3>🎮 Action Mapping</h3>
      <p>Map gestures to keyboard shortcuts, mouse movements, app launches, or system commands.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔄 Profile Management</h3>
      <p>Create and switch between gesture maps for different applications and contexts.</p>
    </td>
    <td width="50%">
      <h3>🔒 Privacy-Focused</h3>
      <p>All processing happens locally—your camera feed never leaves your device.</p>
    </td>
  </tr>
</table>

### Additional Features

- **Predefined Gestures**: Out-of-the-box support for common hand positions
- **Custom Gesture Training**: *(Coming soon)* Create your own personalized gestures
- **Visual Feedback**: On-screen confirmation when gestures are detected
- **Adjustable Settings**: Fine-tune sensitivity, cooldown periods, and detection thresholds
- **Background Operation**: Runs quietly in your system tray
- **Cross-Platform**: Works on Windows, Linux, and macOS

## 🛠️ Requirements

- Python 3.8 or later
- Webcam/camera device
- Required Python packages (installed automatically):
  - OpenCV
  - MediaPipe
  - PyQt5
  - PyAutoGUI
  - TensorFlow
  - PyYAML
  - NumPy
  - pynput

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gesturebind.git
cd gesturebind

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python gesturebind/main.py
```

### Quick Install Scripts

<table>
  <tr>
    <td>
      <b>Windows</b><br>
      <pre>scripts\install.bat</pre>
    </td>
    <td>
      <b>Linux/macOS</b><br>
      <pre>chmod +x scripts/install.sh<br>./scripts/install.sh</pre>
    </td>
  </tr>
</table>

## 🎮 Usage

### Getting Started in 3 Simple Steps

<ol>
  <li>
    <b>Launch GestureBind</b>
    <p>Run <code>python gesturebind/main.py</code> and allow camera access</p>
  </li>
  <li>
    <b>Start Detection</b>
    <p>Click the "Start Detection" button in the main window</p>
  </li>
  <li>
    <b>Control with Gestures</b>
    <p>Perform gestures in front of your camera to trigger actions</p>
  </li>
</ol>

### Mapping Gestures to Actions

1. Navigate to the Settings panel
2. Select or create a profile
3. Configure gesture mappings:
   - Choose from predefined gestures
   - Select an action type (hotkey, mouse action, app launch)
   - Configure the specific parameters
   - Save your mapping


## ⚙️ Configuration

GestureBind uses YAML configuration files:

- Default configuration: `gesturebind/config/default_config.yaml`
- User configuration: `gesturebind/data/config/user_config.yaml`

### Example Configuration

```yaml
# Core detection settings
detection:
  engine: mediapipe  # Options: mediapipe, yolov8 (placeholder)
  confidence_threshold: 0.7
  min_detection_duration_ms: 200

# UI preferences
ui:
  theme: system  # Options: system, dark, light
  show_preview: true
  camera_preview_fps: 15
  overlay_feedback: true
  minimize_to_tray: true
  start_minimized: false
  
# Gesture mappings by profile
profiles:
  default:
    gesture_mappings:
      peace:
        type: hotkey
        data: Ctrl+V
        description: 'Paste clipboard content'
      thumb:
        type: shell_command
        data:
          command: flatpak run com.spotify.Client
          terminal: false
        description: 'Launch Spotify'
```

## 📋 Development Status

<div align="center">
  <h3>GestureBind is in active development</h3>
</div>

### ✅ Already Implemented

- MediaPipe hands integration with landmark detection
- Landmark smoothing for more stable gesture recognition
- Rule-based gesture classification for basic gestures
- Action mapping system (keyboard, apps, shell commands)
- Configuration management with YAML
- PyQt5 UI framework with system tray integration
- Cross-platform support foundation

### 🚧 In Progress

- Visual feedback overlay for gestures
- UI organization and navigation improvements
- YOLOv8 integration (placeholder implementation)
- Profile management interface enhancements

### 🔮 Coming Soon

- Custom gesture training interface
- Machine learning-based gesture classification
- Gesture embedding storage and similarity matching
- Built-in application presets
- Import/Export for gesture profiles
- Drag-and-drop action mapping interface
- Enhanced action confirmation feedback

## 🔍 Next Steps & Priorities

### High Priority
1. Implement Gesture Training UI
2. Integrate ML-based gesture classification
3. Complete action configuration interface
4. Enhance visual feedback system

### Medium Priority
1. Complete YOLOv8 integration
2. Add built-in application presets
3. Implement import/export functionality
4. Improve test coverage

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest gesturebind/tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Check our implementation status and priorities for areas that need attention.

## 📝 License

GestureBind is available under the MIT License. See the LICENSE file for details.

## 🔒 Privacy

We take your privacy seriously. GestureBind processes all camera feeds locally on your device. No video or gesture data is uploaded externally, ensuring your privacy is maintained at all times.

---

<div align="center">
  <p><i>Have questions or feedback? Open an issue on GitHub or reach out to the maintainers.</i></p>
</div>