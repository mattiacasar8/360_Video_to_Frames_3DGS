# Frame Extractor User Guide

This guide will help you get started with Frame Extractor, a macOS application for processing video files to extract high-quality frames for 3D Gaussian Splatting and photogrammetry.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Main Interface](#main-interface)
3. [Workflow](#workflow)
4. [Advanced Settings](#advanced-settings)
5. [Processing 360° Videos](#processing-360-videos)
6. [Keyboard Shortcuts](#keyboard-shortcuts)
7. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

After installing Frame Extractor (see [INSTALL.md](INSTALL.md)), activate the virtual environment and launch the application:

```bash
# Navigate to the project directory
cd frame_extractor

# Activate the virtual environment
source venv/bin/activate

# Run the application
python run.py
```

You'll be presented with the main window after the application starts.

## Main Interface

The application interface is divided into several sections:

### Video List Panel

- Located on the left side
- Displays videos added for processing
- Shows video information (resolution, duration, etc.)
- Indicates 360° videos with a special icon
- Shows processing status for each video

### Parameter Panel

- Located on the right side
- Contains tabs for different parameter categories:
  - **General**: Basic extraction settings
  - **Blur Detection**: Settings for blur detection algorithms
  - **Similarity Detection**: Settings for similar frame detection
  - **360° Video**: Settings specific to 360° video processing

### Progress Panel

- Located at the bottom
- Shows overall progress during processing
- Displays detailed statistics (processed frames, extracted frames, etc.)
- Shows estimated time remaining

### Toolbar

- Located at the top
- Provides quick access to common functions:
  - Add Video
  - Remove Video
  - Start Processing
  - Stop Processing
  - Select Output Directory

## Workflow

### 1. Add Videos

Add videos to the processing queue using one of these methods:
- Drag and drop video files onto the application window
- Click the "Add Video" button in the toolbar
- Use the "File > Add Video" menu option

### 2. Configure Settings

Adjust processing parameters based on your needs:

#### General Settings

- **Output Directory**: Where extracted frames will be saved
- **Maximum Frames**: Limit the number of frames extracted per video
- **Frame Interval**: Select frames at regular intervals (e.g., every 5 frames)

#### Blur Detection

- **Laplacian Threshold**: Higher values mean stricter blur detection
- **FFT Threshold**: Higher values mean stricter frequency-based blur detection
- **Enable/Disable**: Toggle which blur detection methods to use

#### Similarity Detection

- **SSIM Threshold**: Higher values mean less similar frames will be kept
- **Histogram Threshold**: Higher values mean frames with more similar color distributions will be excluded
- **Enable/Disable**: Toggle which similarity detection methods to use

#### 360° Video Settings

- **Projection Type**: Choose between equirectangular and cubemap projections
- **Output Format**: Select output image format (JPEG, PNG, etc.)
- **Quality**: For JPEG output, control compression quality

### 3. Start Processing

Click the "Start Processing" button to begin extracting frames.

The application will:
1. Process each video in sequence
2. Apply blur detection to filter out blurry frames
3. Apply similarity detection to filter out redundant frames
4. Save qualifying frames to the output directory
5. Update progress information in real-time

### 4. Review Results

After processing completes:
- The status of each video will update
- You can access the extracted frames in the output directory
- A summary of results will be displayed

## Advanced Settings

### Hardware Acceleration

The application automatically uses hardware acceleration when available:

- **Apple Silicon**: Uses Metal/VideoToolbox acceleration
- **Intel Macs**: Uses VideoToolbox acceleration
- **OpenCL**: Used for OpenCV operations when available

You can toggle hardware acceleration in "Preferences > Performance".

### Custom Frame Naming

You can customize the naming pattern for extracted frames in "Preferences > Output Format":

- `{video}`: Original video filename
- `{frame}`: Frame number
- `{timestamp}`: Timestamp in the video
- `{date}`: Current date
- `{quality}`: Blur score

## Processing 360° Videos

The application automatically detects 360° videos and provides specialized processing:

1. Videos are detected as 360° based on metadata
2. You can manually toggle 360° mode for videos without proper metadata
3. Choose the projection type in the "360° Video" tab:
   - **Equirectangular**: Default format for 360° panoramas
   - **Cubemap**: Six-faced cube representation

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Add Video | Cmd+O |
| Remove Selected Video | Delete |
| Start Processing | Cmd+R |
| Stop Processing | Cmd+. |
| Select Output Directory | Cmd+D |
| Quit Application | Cmd+Q |
| Show Preferences | Cmd+, |

## Tips and Best Practices

### For Best Quality Frames:

1. **Use High-Resolution Source Videos**
   - Higher resolution source material produces better frames for 3D reconstruction

2. **Adjust Blur Detection Thresholds Carefully**
   - Start with default values and adjust gradually
   - Preview mode can help you calibrate settings for your specific footage

3. **Balance Similarity Threshold**
   - Too high: May miss important differences between frames
   - Too low: May extract too many redundant frames

4. **For 360° Videos**
   - Equirectangular projection is generally better for photogrammetry
   - Make sure your 360° camera has proper stabilization

5. **Use Batch Processing for Consistency**
   - Process related videos in a single batch with identical settings
   - This ensures consistent frame selection across the entire dataset

6. **Output Format Recommendations**
   - PNG: Best quality, larger file size
   - JPEG (95% quality): Good balance of quality and size for most purposes 