# Installation Guide for Frame Extractor

This guide provides installation instructions for the Frame Extractor application on macOS.

## Prerequisites

- **macOS**: 10.15 (Catalina) or newer
- **Python**: 3.8 or newer
- **FFmpeg**: Required for video processing

## Option 1: Install from Source (Development)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/frame_extractor.git
cd frame_extractor
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

Using Homebrew (recommended):
```bash
brew install ffmpeg
```

Or using MacPorts:
```bash
sudo port install ffmpeg
```

### 5. Run the Application

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run the application
python run.py
```

## Option 2: Install Packaged Application (Release)

### 1. Download the Application

Download the latest release from the repository's Releases page.

### 2. Install FFmpeg (if not already installed)

Using Homebrew (recommended):
```bash
brew install ffmpeg
```

Or using MacPorts:
```bash
sudo port install ffmpeg
```

### 3. Move to Applications Folder

Drag the downloaded Frame Extractor.app to your Applications folder.

### 4. First Run

When running the application for the first time, you may need to right-click on the application and select "Open" to bypass macOS security restrictions for applications from unidentified developers.

## Building the Application from Source

To build the application as a standalone macOS app:

### 1. Install py2app

```bash
pip install py2app
```

### 2. Generate the Application Icon

```bash
cd resources
python generate_icon.py
cd ..
```

### 3. Build the Application

```bash
python package_macos.py py2app
```

The packaged application will be created in the `dist` directory.

## Troubleshooting

### FFmpeg Not Found

If you encounter an error indicating FFmpeg is not found:

1. Verify FFmpeg is installed:
   ```bash
   ffmpeg -version
   ```

2. If installed but not found by the application, ensure the FFmpeg binary is in your PATH:
   ```bash
   echo $PATH
   ```

3. You may need to create a symbolic link to the FFmpeg binary in a location that's in your PATH:
   ```bash
   sudo ln -s /path/to/ffmpeg /usr/local/bin/ffmpeg
   ```

### OpenCV Issues

If you encounter errors related to OpenCV on Apple Silicon:

1. Uninstall the current OpenCV package:
   ```bash
   pip uninstall opencv-python
   ```

2. Install the headless version:
   ```bash
   pip install opencv-python-headless
   ```

### Permission Issues

If you encounter permission issues when running the application:

1. Grant full disk access to Terminal or the application in System Preferences > Security & Privacy > Privacy > Full Disk Access.

2. For the packaged app, you may need to remove the quarantine attribute:
   ```bash
   xattr -d com.apple.quarantine /Applications/Frame\ Extractor.app
   ```

## Support

If you encounter any issues, please file a bug report on the repository's Issues page. 