# Implementation TODO List

## Phase 1: Core Functionality
- [x] Set up project structure
  - Created basic directory structure and initialized files
  - Added documentation and type hints throughout the codebase
- [x] Implement FFmpeg wrapper for frame extraction
  - Implemented in VideoProcessor class using OpenCV and FFmpeg
  - Added support for metadata extraction and frame processing
  - Added hardware acceleration for Apple Silicon using Metal/videotoolbox
- [x] Implement blur detection with OpenCV
  - Created BlurDetector class using Laplacian variance method
  - Enhanced with FFT (frequency domain analysis) for more robust blur detection
  - Added configurable threshold and analysis functions
- [x] Implement frame similarity detection
  - Created SimilarityDetector class using SSIM from scikit-image
  - Enhanced with histogram comparison for more robust detection
  - Added optimization by downscaling images for faster comparison
- [x] Create 360° video conversion functionality
  - Implemented SphericalConverter class for 360° video handling
  - Added detection of 360° video based on metadata
  - Added hardware-accelerated extraction using FFmpeg
- [x] Design basic command-line interface for testing
  - Implemented CLI in main.py with argument parsing
  - Added progress reporting for CLI operation

## Phase 2: User Interface
- [x] Create main application window
  - Implemented MainWindow class with proper layout
  - Added menu and toolbar with essential functions
- [x] Implement drag-and-drop video selection
  - Created VideoListWidget for managing video queue
  - Added drag-and-drop support for video files
- [x] Add parameter adjustment controls
  - Implemented ParameterPanel with sliders and input fields
  - Added help text for parameter descriptions
  - Added controls for new blur and similarity detection parameters
- [x] Implement preview functionality
  - Added basic video list with 360° detection
  - Status display for processing state
- [x] Create progress tracking display
  - Implemented ProgressWidget with detailed statistics
  - Added overall and per-video progress tracking
- [x] Design output configuration panel
  - Added output directory selection
  - Created data flow between parameters and processor

## Phase 3: Integration and Testing
- [x] Connect UI to core processing functions
  - Integrated improved BlurDetector with combined Laplacian/FFT approach
  - Integrated enhanced SimilarityDetector with SSIM/histogram comparison
  - Connected hardware-accelerated frame extraction for better performance
- [x] Implement batch processing logic
  - Added hardware-accelerated batch processing with FFmpeg
  - Improved overall processing workflow
- [x] Add error handling and logging
  - Added comprehensive logging
  - Added try/except blocks for robust error handling
- [x] Perform performance optimization
  - Added hardware acceleration detection for Apple Silicon
  - Added OpenCL/Metal acceleration for OpenCV
  - Implemented more efficient batch frame extraction
- [x] Test with various video formats and sizes
  - Created tests for MP4, AVI, MOV and MKV formats
  - Added automated tests using synthetic video generation
- [x] Test specifically with 360° videos
  - Created tests for 360° video handling and extraction
  - Implemented tests for both metadata detection and frame extraction

## Phase 4: Packaging and Distribution
- [x] Package application for macOS
  - Created py2app packaging script for macOS distribution
  - Added application icon generation
- [x] Ensure proper handling of dependencies
  - Added dependency management in package scripts
  - Optimized package size by excluding unnecessary modules
- [x] Create installation instructions
  - Added comprehensive INSTALL.md with instructions for different install methods
  - Included troubleshooting guide for common issues
- [x] Write user documentation
  - Updated README.md with feature documentation
  - Added usage instructions in INSTALL.md
- [ ] Perform final testing on different Mac hardware 