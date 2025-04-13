# GestureBind - Gesture-Based Shortcut Mapper

GestureBind is a versatile application that allows users to map hand and body gestures to custom system actions and shortcuts, enabling touchless computer control.

## Features

- Real-time gesture detection using webcam
- Predefined gesture recognition (fist, open palm, peace sign, etc.)
- Custom gesture training system
- Gesture-to-action mapping for keyboard shortcuts, mouse actions, and application launching
- Action feedback overlay
- Adjustable detection sensitivity 
- Background/tray mode operation
- Cross-platform support (Windows, Linux, macOS)
- Local processing for privacy and security
- Built-in profiles for popular applications
- Import/export configuration options

## Requirements

See `requirements.txt` for all dependencies.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gesturebind.git
cd gesturebind

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

1. Launch the application
2. Allow camera access when prompted
3. Use the UI to map gestures to actions
4. Enable gesture detection to start using your configured shortcuts

## Configuration

Configuration files are stored in `config/`. The default configuration is `default_config.yaml`.

## License

MIT

## Privacy

GestureBind processes all camera feeds locally on your device. No video or gesture data is uploaded externally.