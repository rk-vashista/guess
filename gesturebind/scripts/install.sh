#!/bin/bash
# GestureBind Installation Script

echo "===== GestureBind Installation ====="
echo "Installing dependencies for GestureBind"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Installing pip..."
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y python3-pip
    elif command -v dnf &> /dev/null; then
        # Fedora
        sudo dnf install -y python3-pip
    elif command -v brew &> /dev/null; then
        # macOS with Homebrew
        brew install python
    else
        echo "Could not install pip. Please install pip manually."
        exit 1
    fi
fi

# Check for Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 7 ]); then
    echo "Python 3.7 or higher is required. You have $PYTHON_VERSION"
    exit 1
fi

# Install required packages
echo "Installing Python dependencies..."
pip install -r ../requirements.txt

# Install system dependencies for OpenCV and MediaPipe
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    echo "Installing system dependencies for Ubuntu/Debian..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-opencv \
        libgtk-3-dev \
        libcanberra-gtk3-module \
        libatlas-base-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev
elif command -v dnf &> /dev/null; then
    # Fedora
    echo "Installing system dependencies for Fedora..."
    sudo dnf install -y \
        gtk3-devel \
        atlas-devel \
        libjpeg-turbo-devel \
        libpng-devel \
        libtiff-devel
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "macOS detected. Most dependencies will be installed via pip."
    # If using Homebrew, ensure required libraries are available
    if command -v brew &> /dev/null; then
        brew install gtk+3
    fi
fi

echo "Creating application shortcut..."
# Create desktop shortcut file
DESKTOP_FILE="$HOME/.local/share/applications/gesturebind.desktop"
mkdir -p "$(dirname "$DESKTOP_FILE")"

cat > "$DESKTOP_FILE" << EOL
[Desktop Entry]
Name=GestureBind
Comment=Gesture-Based Shortcut Mapper
Exec=python3 $(realpath ../main.py)
Icon=$(realpath ../resources/icon.png)
Terminal=false
Type=Application
Categories=Utility;
EOL

echo "Installation complete!"
echo "You can start GestureBind by running 'python main.py' from the main directory"
echo "or by using the application shortcut in your system menu."