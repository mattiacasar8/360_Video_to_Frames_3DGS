# Frame Extractor for 3D Gaussian Splatting

A macOS application for processing video files (including 360° videos) to extract high-quality frames for 3D Gaussian Splatting and photogrammetry.

## Features

- Batch process multiple videos sequentially with hardware acceleration
- Support for 360° video conversion to equirectangular frames
- Advanced blur detection using combined Laplacian variance and FFT analysis
- Multi-method similar frame detection using both SSIM and histogram comparison
- Organized output structure for extracted frames
- Progress tracking and statistics during processing
- Native macOS interface with dark mode support
- Optimized for Apple Silicon with Metal hardware acceleration

## Requirements

- macOS (including Apple Silicon support)
- Python 3.8 or higher
- FFmpeg (for video processing)
- OpenCV (for image analysis)

## Installation

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   For optimal performance on Apple Silicon:
   ```bash
   pip install opencv-python-headless
   ```

4. Run the application:
   ```bash
   # Make sure your virtual environment is activated
   source venv/bin/activate
   
   # Run the application
   python run.py
   ```

## Usage

1. Add videos to the processing queue using drag-and-drop
2. Configure processing parameters:
   - Blur detection thresholds (Laplacian and FFT)
   - Frame similarity thresholds (SSIM and histogram)
   - Frame extraction interval
   - Maximum frames per video
3. Select output destination
4. Start processing
5. Review extracted frames

## Advanced Features

### Blur Detection

The application uses a combined approach for blur detection:
- Laplacian variance method: Analyzes image edges and transitions
- FFT (Fast Fourier Transform): Analyzes image frequency components

This combined approach provides more robust blur detection across different types of videos.

### Similarity Detection

To avoid redundant frames, the application uses multiple methods:
- SSIM (Structural Similarity Index): Analyzes structural patterns in images
- Histogram comparison: Analyzes color and intensity distributions

### Hardware Acceleration

The application automatically detects and utilizes hardware acceleration:
- Apple Silicon Macs: Uses Metal acceleration via VideoToolbox
- Intel Macs: Uses VideoToolbox acceleration
- OpenCL: Used for OpenCV operations when available

## Development

See the [TODO.md](TODO.md) file for the current development status and roadmap. 