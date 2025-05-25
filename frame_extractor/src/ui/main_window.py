"""Main window for the Frame Extractor application."""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import cv2
import tempfile

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QMessageBox, QApplication,
    QCheckBox, QGroupBox, QStatusBar, QToolBar,
    QSlider, QSpinBox, QDoubleSpinBox, QProgressBar, QToolButton,
    QTabWidget, QComboBox, QLineEdit, QDialog, QFormLayout
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot, QUrl
from PyQt6.QtGui import QIcon, QDragEnterEvent, QDropEvent, QDesktopServices, QAction

from ..core.video_processor import VideoProcessor
from .video_list_widget import VideoListWidget
from .parameter_panel import ParameterPanel
from .progress_widget import ProgressWidget


class ProcessingThread(QThread):
    """Thread for running video processing without blocking the UI."""
    
    # Signal for progress updates
    progress_update = pyqtSignal(dict)
    # Signal for process completion
    process_finished = pyqtSignal(dict)
    # Signal for frame-level progress
    frame_progress = pyqtSignal(str, int, int)
    
    def __init__(self, processor: VideoProcessor, video_paths: List[str], is_360_list: List[bool]):
        """Initialize the processing thread.
        
        Args:
            processor: The VideoProcessor instance
            video_paths: List of video file paths to process
            is_360_list: List indicating which videos are 360°
        """
        super().__init__()
        self.processor = processor
        self.video_paths = video_paths
        self.is_360_list = is_360_list
        
        # Add a logger to monitor frame processing
        self.logger = logging.getLogger("ProcessingThread")
        
        # Hook up to processor's logger to monitor progress
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure logging to catch processor messages."""
        # Create a custom handler to catch VideoProcessor log messages
        class ProcessorLogHandler(logging.Handler):
            def __init__(self, thread):
                super().__init__()
                self.thread = thread
                
            def emit(self, record):
                # Catch frame processing progress messages
                if "Processing frame" in record.getMessage():
                    try:
                        msg = record.getMessage()
                        # Extract current and total frame numbers
                        parts = msg.split()
                        current = int(parts[2])
                        total = int(parts[3].split('/')[1])
                        video_name = self.thread.current_video.stem if hasattr(self.thread, "current_video") else ""
                        self.thread.frame_progress.emit(video_name, current, total)
                    except Exception as e:
                        pass  # Failed to parse message
                        
                # Check for "Saved frames" messages
                if "Saved" in record.getMessage() and "frames" in record.getMessage():
                    try:
                        msg = record.getMessage()
                        if "so far" in msg:
                            # Extract the saved frames count
                            parts = msg.split()
                            saved_count = int(parts[1])
                            # Update stats with just the saved count
                            update_stats = {
                                'saved_frames_update': True,
                                'processor_stats': {
                                    'saved_frames': saved_count
                                }
                            }
                            self.thread.progress_update.emit(update_stats)
                    except Exception:
                        # If parsing fails, just emit a generic update
                        self.thread.progress_update.emit({"saved_frames_update": True})
        
        # Add custom handler to the VideoProcessor logger
        processor_logger = logging.getLogger('VideoProcessor')
        processor_logger.addHandler(ProcessorLogHandler(self))
    
    def run(self):
        """Run the video processing."""
        # Process each video individually to provide better progress updates
        self.stats = {
            'total_frames': 0,
            'blurry_frames': 0,
            'similar_frames': 0,
            'saved_frames': 0,
            'videos_processed': 0,
            'videos_failed': 0,
            'per_video': {}
        }
        
        for i, (video_path, is_360) in enumerate(zip(self.video_paths, self.is_360_list)):
            try:
                # Update current video
                self.current_video = Path(video_path)
                
                # Emit progress update for UI
                self.progress_update.emit({
                    'current_video': i + 1,
                    'total_videos': len(self.video_paths),
                    'video_path': video_path,
                    'videos_processed': i  # Add this for compatibility
                })
                
                # Process the video
                video_stats = self.processor.process_video(video_path, is_360)
                
                # Update overall statistics
                self.stats['total_frames'] += video_stats.get('total_frames', 0)
                self.stats['blurry_frames'] += video_stats.get('blurry_frames', 0)
                self.stats['similar_frames'] += video_stats.get('similar_frames', 0)
                self.stats['saved_frames'] += video_stats.get('saved_frames', 0)
                self.stats['videos_processed'] += 1
                self.stats['per_video'][video_path] = video_stats
                
                # Emit progress update for UI
                self.progress_update.emit(self.stats)
                
            except Exception as e:
                self.logger.error(f"Error processing video {video_path}: {e}")
                self.stats['videos_failed'] += 1
        
        # Emit completion signal
        self.process_finished.emit(self.stats)


class DirectorySelectionDialog(QDialog):
    """Dialog for selecting input and output directories."""
    
    def __init__(self, parent=None):
        """Initialize the dialog."""
        super().__init__(parent)
        self.setWindowTitle("Select Directories")
        self.setModal(True)
        
        # Create layout
        layout = QFormLayout(self)
        
        # Input directory
        self.input_dir = QLineEdit()
        self.input_dir.setReadOnly(True)
        self.input_btn = QPushButton("Browse...")
        self.input_btn.clicked.connect(self._select_input_dir)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_dir)
        input_layout.addWidget(self.input_btn)
        layout.addRow("Cubemap Directory:", input_layout)
        
        # Output directory
        self.output_dir = QLineEdit()
        self.output_dir.setReadOnly(True)
        self.output_btn = QPushButton("Browse...")
        self.output_btn.clicked.connect(self._select_output_dir)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(self.output_btn)
        layout.addRow("Faces Directory:", output_layout)
        
        # Buttons
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addRow("", button_layout)
    
    def _select_input_dir(self):
        """Select input directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Cubemap Directory",
            str(Path.home())
        )
        if dir_path:
            self.input_dir.setText(dir_path)
    
    def _select_output_dir(self):
        """Select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Faces Directory",
            str(Path.home())
        )
        if dir_path:
            self.output_dir.setText(dir_path)
    
    def get_directories(self) -> tuple[str, str]:
        """Get selected directories.
        
        Returns:
            Tuple of (input_dir, output_dir)
        """
        return self.input_dir.text(), self.output_dir.text()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        self.setWindowTitle("Frame Extractor for 3D Gaussian Splatting")
        self.setMinimumSize(1000, 700)
        
        # Set up logger
        self.logger = logging.getLogger("MainWindow")
        
        # Set up processor
        self.processor = None
        self.processing_thread = None
        
        # Initialize frame analysis cache
        # Dictionary structure: {video_path: {metadata: {fps, frame_count, duration}, frame_selector: FrameSelector instance}}
        self.frame_cache = {}
        
        # Set up UI
        self._setup_ui()
        
        # Connect signals and slots
        self._connect_signals()
        
        # Initialize video processor
        self._initialize_processor()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Create central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Set up splitter for main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Video list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Video list widget
        self.video_list = VideoListWidget()
        self.video_list.setAcceptDrops(True)
        left_layout.addWidget(self.video_list)
        
        # Buttons for adding/removing videos
        buttons_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Videos")
        self.remove_button = QPushButton("Remove Selected")
        self.clear_button = QPushButton("Clear All")
        
        buttons_layout.addWidget(self.add_button)
        buttons_layout.addWidget(self.remove_button)
        buttons_layout.addWidget(self.clear_button)
        
        left_layout.addLayout(buttons_layout)
        
        # Right panel - Parameters and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Parameter panel
        self.parameter_panel = ParameterPanel()
        right_layout.addWidget(self.parameter_panel)
        
        # Output directory selection
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout(output_group)
        
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        self.output_path.setPlaceholderText("Select output directory...")
        
        self.browse_button = QPushButton("Browse...")
        
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.browse_button)
        
        right_layout.addWidget(output_group)
        
        # Analysis results
        analysis_group = QGroupBox("Analysis Results")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Add cache status label at the top of analysis results
        self.cache_status_label = QLabel("Cache Status: No videos analyzed yet")
        self.cache_status_label.setStyleSheet("color: #666; font-weight: bold;")
        analysis_layout.addWidget(self.cache_status_label)
        
        self.analysis_text = QLabel("No analysis performed yet")
        self.analysis_text.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_text)
        
        # Workflow instructions label
        workflow_label = QLabel("Workflow: 1. Add Videos → 2. Analyze Videos → 3. Adjust Parameters → 4. Process")
        workflow_label.setStyleSheet("color: #666; font-style: italic;")
        analysis_layout.addWidget(workflow_label)
        
        right_layout.addWidget(analysis_group)
        
        # Progress widget
        self.progress_widget = ProgressWidget()
        right_layout.addWidget(self.progress_widget)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("Analyze Videos")
        self.analyze_button.setToolTip("Analyze videos to extract preview frames (required before processing)")
        self.analyze_button.setEnabled(False)
        
        self.process_button = QPushButton("Extract Frames")
        self.process_button.setToolTip("Process videos with current parameter settings")
        self.process_button.setEnabled(False)
        self.process_button.setMinimumHeight(50)
        
        buttons_layout.addWidget(self.analyze_button)
        buttons_layout.addWidget(self.process_button)
        
        right_layout.addLayout(buttons_layout)
        
        # Add cubemap to faces button
        self.cubemap_btn = QPushButton("Convert Cubemap to Faces")
        self.cubemap_btn.setToolTip("Convert existing cubemap images to individual faces")
        self.cubemap_btn.clicked.connect(self._convert_cubemap_to_faces)
        right_layout.addWidget(self.cubemap_btn)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        right_layout.addWidget(self.progress_bar)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([400, 600])
        
        # Set up status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set up menu bar
        self._setup_menu()
        
        # Set up toolbar
        self._setup_toolbar()
    
    def _setup_menu(self):
        """Set up the application menu."""
        # Create menu bar
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        # Add videos action
        add_action = QAction("&Add Videos...", self)
        add_action.setShortcut("Ctrl+O")
        add_action.triggered.connect(self._add_videos)
        file_menu.addAction(add_action)
        
        # Select output directory action
        output_action = QAction("Select &Output Directory...", self)
        output_action.triggered.connect(self._select_output_directory)
        file_menu.addAction(output_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        
        # Start processing action
        process_action = QAction("&Start Processing", self)
        process_action.triggered.connect(self._start_processing)
        tools_menu.addAction(process_action)
        
        # Reset parameters action
        reset_action = QAction("&Reset Parameters", self)
        reset_action.triggered.connect(self.parameter_panel.reset_parameters)
        tools_menu.addAction(reset_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbar(self):
        """Set up the application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add videos action
        add_action = QAction("Add Videos", self)
        # add_action.setIcon(QIcon.fromTheme("list-add"))
        add_action.triggered.connect(self._add_videos)
        toolbar.addAction(add_action)
        
        # Select output action
        output_action = QAction("Output Directory", self)
        # output_action.setIcon(QIcon.fromTheme("folder"))
        output_action.triggered.connect(self._select_output_directory)
        toolbar.addAction(output_action)
        
        toolbar.addSeparator()
        
        # Start processing action
        process_action = QAction("Start Processing", self)
        # process_action.setIcon(QIcon.fromTheme("media-playback-start"))
        process_action.triggered.connect(self._start_processing)
        toolbar.addAction(process_action)
        
        # Stop processing action
        self.stop_action = QAction("Stop Processing", self)
        # self.stop_action.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_action.triggered.connect(self._stop_processing)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)
        
        toolbar.addSeparator()
        
        # Open output folder action
        open_output_action = QAction("Open Output Folder", self)
        # open_output_action.setIcon(QIcon.fromTheme("folder-open"))
        open_output_action.triggered.connect(self._open_output_folder)
        toolbar.addAction(open_output_action)
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # Add/remove video buttons
        self.add_button.clicked.connect(self._add_videos)
        self.remove_button.clicked.connect(self.video_list.remove_selected)
        self.clear_button.clicked.connect(self.video_list.clear)
        
        # Output directory button
        self.browse_button.clicked.connect(self._select_output_directory)
        
        # Analysis and process buttons
        self.analyze_button.clicked.connect(self._analyze_videos)
        self.process_button.clicked.connect(self._start_processing)
        
        # Video list signals
        self.video_list.video_list_changed.connect(self._update_ui_state)
        
        # Parameter changes
        self.parameter_panel.parameters_changed.connect(self._update_analysis_from_cache)
    
    def _initialize_processor(self):
        """Initialize the video processor."""
        # Use user's desktop as the default output directory
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        
        # Get current parameters
        params = self.parameter_panel.get_parameters()
        
        # Create processor with default parameters
        self.processor = VideoProcessor(
            output_dir=desktop_path,
            enable_frame_skip=params['enable_frame_skip'],
            fps=params['fps'],
            enable_blur_detection=params['enable_blur_detection'],
            blur_threshold=params['blur_threshold'],
            enable_frame_selection=params['enable_frame_selection'],
            min_frame_gap=params['min_frame_gap']
        )
    
    def _update_ui_state(self):
        """Update UI state based on current application state."""
        has_videos = self.video_list.count() > 0
        has_output = bool(self.output_path.text())
        
        # Enable/disable buttons
        self.analyze_button.setEnabled(has_videos)
        self.process_button.setEnabled(has_videos and has_output)
        
        # Enable/disable remove and clear buttons
        self.remove_button.setEnabled(has_videos)
        self.clear_button.setEnabled(has_videos)
        
        # Update status bar
        if has_videos:
            self.status_bar.showMessage(f"{self.video_list.count()} videos in queue")
        else:
            self.status_bar.showMessage("Ready")
    
    def _update_analysis_from_cache(self):
        """Update analysis results from cache when parameters change."""
        if not self.frame_cache:
            # No cache available, nothing to do
            self.cache_status_label.setText("Cache Status: No videos analyzed yet")
            return
            
        # Get current parameters
        params = self.parameter_panel.get_parameters()
        
        # Calculate estimated frames for each video
        analysis_text = []
        total_frames = 0
        
        # Check if we have videos in the list
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if not item:
                continue
                
            video_path = item.data(Qt.ItemDataRole.UserRole)
            if not video_path or video_path not in self.frame_cache:
                continue
                
            # Get cached data
            cached_data = self.frame_cache[video_path]
            metadata = cached_data['metadata']
            frame_selector = cached_data['frame_selector']
            
            # Calculate estimated frames based on parameters
            if params['enable_frame_skip']:
                estimated_frames = int(metadata['duration'] * params['fps'])
            else:
                estimated_frames = metadata['frame_count']
                
            # Add to analysis
            video_name = Path(video_path).name
            analysis_text.append(f"{video_name}:")
            analysis_text.append(f"  Duration: {metadata['duration']:.1f} seconds")
            analysis_text.append(f"  Total frames: {metadata['frame_count']}")
            analysis_text.append(f"  Estimated frames to extract: {estimated_frames}")
            
            # Update frame selector parameters
            frame_selector.set_parameters(
                blur_threshold=params['blur_threshold'],
                min_frame_gap=params['min_frame_gap']
            )
            
            # Get preview statistics using cached frame selector
            preview = frame_selector.preview_filter_effects(
                blur_threshold=params['blur_threshold'] if params['enable_blur_detection'] else None,
                min_frame_gap=params['min_frame_gap'] if params['enable_frame_selection'] else None
            )
            
            # Update UI with preview statistics
            analysis_text.append(f"  Preview statistics:")
            analysis_text.append(f"    Frames passing blur threshold: {preview['frames_passing_blur']}")
            analysis_text.append(f"    Frames selected with minimum gap: {preview['frames_passing_gap']}")
            analysis_text.append(f"    Estimated final frames: {preview['estimated_final_frames']}")
            analysis_text.append("")
            
            # Update parameter panel previews
            self.parameter_panel.update_previews(frame_selector)
            
            # Add to total frames
            total_frames += estimated_frames
        
        if not analysis_text:
            analysis_text.append("No valid videos to analyze")
            self.cache_status_label.setText("Cache Status: No valid videos in cache")
        else:
            # Add total
            analysis_text.append(f"Total estimated frames: {total_frames}")
            cached_videos = len(self.frame_cache)
            total_videos = self.video_list.count()
            self.cache_status_label.setText(f"Cache Status: {cached_videos}/{total_videos} videos analyzed - Parameters updated instantly")
        
        # Update analysis text
        self.analysis_text.setText("\n".join(analysis_text))
    
    def _analyze_videos(self):
        """Analyze videos and show estimated frame counts."""
        if self.video_list.count() == 0:
            return
            
        # Get current parameters
        params = self.parameter_panel.get_parameters()
        
        # Clear the cache
        self.frame_cache = {}
        self.cache_status_label.setText("Cache Status: Analyzing videos...")
        
        # Calculate estimated frames for each video
        analysis_text = []
        total_frames = 0
        videos_analyzed = 0
        total_videos = self.video_list.count()
        
        # Setup progress tracking
        self.progress_bar.setRange(0, total_videos)
        self.progress_bar.setValue(0)
        
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if not item:
                continue
                
            video_path = item.data(Qt.ItemDataRole.UserRole)
            if not video_path or not os.path.exists(video_path):
                continue
                
            try:
                # Update progress bar
                self.progress_bar.setValue(videos_analyzed)
                video_name = Path(video_path).name
                self.status_bar.showMessage(f"Analyzing video {videos_analyzed+1}/{total_videos}: {video_name}")
                QApplication.processEvents()  # Ensure UI updates
                
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    analysis_text.append(f"Error: Could not open video {Path(video_path).name}")
                    continue
                    
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                # Save metadata to cache
                self.frame_cache[video_path] = {
                    'metadata': {
                        'fps': fps,
                        'frame_count': frame_count,
                        'duration': duration
                    }
                }
                
                # Calculate estimated frames based on parameters
                if params['enable_frame_skip']:
                    estimated_frames = int(duration * params['fps'])
                else:
                    estimated_frames = frame_count
                    
                # Add to analysis
                video_name = Path(video_path).name
                analysis_text.append(f"{video_name}:")
                analysis_text.append(f"  Duration: {duration:.1f} seconds")
                analysis_text.append(f"  Total frames: {frame_count}")
                analysis_text.append(f"  Estimated frames to extract: {estimated_frames}")
                
                # Extract frames for preview
                if estimated_frames > 0:
                    # Create temporary directory for frames
                    with tempfile.TemporaryDirectory() as temp_dir:
                        self.status_bar.showMessage(f"Analyzing video {videos_analyzed+1}/{total_videos}: {video_name} - Extracting sample frames...")
                        QApplication.processEvents()  # Ensure UI updates
                        
                        # Extract frames
                        frame_paths = self.processor.spherical_converter.extract_frames(
                            str(video_path),
                            temp_dir,
                            fps=params['fps'] if params['enable_frame_skip'] else fps,
                            is_360=item.data(Qt.ItemDataRole.UserRole + 1),  # is_360 flag
                            max_frames=min(50, estimated_frames)  # Limit to 50 frames for faster analysis
                        )
                        
                        if frame_paths:
                            self.status_bar.showMessage(f"Analyzing video {videos_analyzed+1}/{total_videos}: {video_name} - Scoring frames...")
                            QApplication.processEvents()  # Ensure UI updates
                            
                            # Create a copy of the frame selector for caching
                            frame_selector = type(self.processor.frame_selector)(
                                blur_threshold=params['blur_threshold'],
                                min_frame_gap=params['min_frame_gap']
                            )
                            
                            # Score frames with the cached frame selector
                            frame_selector.score_frames(frame_paths)
                            
                            # Store frame selector in cache
                            self.frame_cache[video_path]['frame_selector'] = frame_selector
                            
                            # Update previews
                            self.parameter_panel.update_previews(frame_selector)
                            
                            # Add preview statistics to analysis
                            preview = frame_selector.preview_filter_effects(
                                blur_threshold=params['blur_threshold'] if params['enable_blur_detection'] else None,
                                min_frame_gap=params['min_frame_gap'] if params['enable_frame_selection'] else None
                            )
                            
                            analysis_text.append(f"  Preview statistics:")
                            analysis_text.append(f"    Frames passing blur threshold: {preview['frames_passing_blur']}")
                            analysis_text.append(f"    Frames selected with minimum gap: {preview['frames_passing_gap']}")
                            analysis_text.append(f"    Estimated final frames: {preview['estimated_final_frames']}")
                
                analysis_text.append("")
                
                total_frames += estimated_frames
                videos_analyzed += 1
                
                cap.release()
                
                # Update progress bar after each video
                self.progress_bar.setValue(videos_analyzed)
                
            except Exception as e:
                analysis_text.append(f"Error analyzing {Path(video_path).name}: {str(e)}")
                continue
        
        if not analysis_text:
            analysis_text.append("No valid videos to analyze")
            self.cache_status_label.setText("Cache Status: No valid videos found")
        else:
            # Add total
            analysis_text.append(f"Total estimated frames: {total_frames}")
            self.cache_status_label.setText(f"Cache Status: {videos_analyzed}/{total_videos} videos analyzed - Try adjusting parameters")
        
        # Update analysis text
        self.analysis_text.setText("\n".join(analysis_text))
        self.status_bar.showMessage("Analysis complete - Adjust parameters and see results update instantly")
        
        # Reset progress bar
        self.progress_bar.setValue(0)
    
    def _start_processing(self):
        """Start processing videos."""
        # Check if we have videos
        if self.video_list.count() == 0:
            QMessageBox.warning(self, "No Videos", "Please add videos to process")
            return
        
        # Check if we have an output directory
        output_dir = self.output_path.text()
        if not output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current parameters
        params = self.parameter_panel.get_parameters()
        
        # Update processor parameters
        self.processor.update_parameters(
            enable_frame_skip=params['enable_frame_skip'],
            fps=params['fps'],
            enable_blur_detection=params['enable_blur_detection'],
            blur_threshold=params['blur_threshold'],
            enable_frame_selection=params['enable_frame_selection'],
            min_frame_gap=params['min_frame_gap']
        )
        
        # Get video paths and 360° status
        video_paths = self.video_list.get_video_paths()
        is_360_list = self.video_list.get_is_360_list()
        
        # Disable UI elements during processing
        self._set_processing_ui_state(True)
        
        # Initialize progress
        self.progress_widget.start_progress(len(video_paths))
        
        # Create and start processing thread
        self.processing_thread = ProcessingThread(self.processor, video_paths, is_360_list)
        self.processing_thread.progress_update.connect(self._update_progress)
        self.processing_thread.frame_progress.connect(self.progress_widget.update_frame_progress)
        self.processing_thread.process_finished.connect(self._processing_finished)
        self.processing_thread.start()
    
    def _stop_processing(self):
        """Stop the current processing operation."""
        if self.processing_thread and self.processing_thread.isRunning():
            # Terminate the thread
            self.processing_thread.terminate()
            self.processing_thread.wait()
            
            # Update UI
            self._set_processing_ui_state(False)
            self.progress_widget.reset_progress()
            self.status_bar.showMessage("Processing stopped by user")
    
    def _update_progress(self, stats: Dict[str, Any]):
        """Update the progress display.
        
        Args:
            stats: Current processing statistics
        """
        # Update progress widget
        self.progress_widget.update_stats(stats)
        
        # Update status bar - handle both update formats
        if 'current_video' in stats and 'total_videos' in stats:
            # New detailed progress format
            self.status_bar.showMessage(f"Processing video {stats['current_video']} of {stats['total_videos']}")
        elif 'videos_processed' in stats:
            # Original format
            self.status_bar.showMessage(f"Processing video {stats['videos_processed'] + 1} of {self.video_list.count()}")
        elif 'saved_frames_update' in stats:
            # Just a frames update
            saved_frames = self.progress_widget.saved_frames_label.text()
            self.status_bar.showMessage(f"Processing frames... {saved_frames} frames saved so far")
    
    def _processing_finished(self, stats: Dict[str, Any]):
        """Handle processing completion.
        
        Args:
            stats: Final processing statistics
        """
        # Update UI
        self._set_processing_ui_state(False)
        
        # Update progress widget with final stats
        self.progress_widget.complete_progress(stats)
        
        # Update status bar
        if stats['saved_frames'] > 0:
            self.status_bar.showMessage(f"Processing complete: {stats['saved_frames']} frames saved")
        else:
            self.status_bar.showMessage("Processing complete: No frames were saved. Check logs for errors.")
        
        # Show completion message with more detailed information
        completion_message = f"Processing complete!\n\n"
        
        if stats['saved_frames'] == 0:
            # Add warning about no frames being saved
            completion_message += "⚠️ WARNING: No frames were saved! This could be due to:\n"
            completion_message += "- Very strict blur detection settings\n"
            completion_message += "- Large minimum frame gap causing all frames to be filtered out\n"
            completion_message += "- Issues with 360° video conversion\n\n"
            completion_message += "Try adjusting your parameters and processing again.\n\n"
        
        completion_message += f"Videos processed: {stats['videos_processed']}\n"
        completion_message += f"Total frames extracted: {stats['total_frames']}\n"
        completion_message += f"Blurry frames rejected: {stats['blurry_frames']}\n"
        completion_message += f"Similar frames rejected: {stats['similar_frames']}\n"
        
        # Add information about repaired frames if any
        if 'repaired_frames' in stats and stats['repaired_frames'] > 0:
            completion_message += f"Missing faces repaired: {stats['repaired_frames']}\n"
            
        completion_message += f"Frames saved: {stats['saved_frames']}"
        
        # Use the correct message box API
        if stats['saved_frames'] == 0:
            QMessageBox.warning(
                self,
                "Processing Complete",
                completion_message
            )
        else:
            QMessageBox.information(
                self,
                "Processing Complete",
                completion_message
            )
        
        # If frames were saved, ask if user wants to open output directory
        if stats['saved_frames'] > 0:
            reply = QMessageBox.question(
                self,
                "Open Output Directory",
                "Would you like to open the output directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._open_output_folder()
    
    def _set_processing_ui_state(self, is_processing: bool):
        """Set UI state for processing/not processing.
        
        Args:
            is_processing: Whether processing is active
        """
        # Disable/enable buttons during processing
        self.add_button.setEnabled(not is_processing)
        self.remove_button.setEnabled(not is_processing)
        self.clear_button.setEnabled(not is_processing)
        self.browse_button.setEnabled(not is_processing)
        self.process_button.setEnabled(not is_processing)
        
        # Disable/enable parameter panel
        self.parameter_panel.setEnabled(not is_processing)
        
        # Disable/enable video list
        self.video_list.setEnabled(not is_processing)
        
        # Enable/disable stop action
        self.stop_action.setEnabled(is_processing)
    
    def _show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Frame Extractor",
            f"<h3>Frame Extractor for 3D Gaussian Splatting</h3>"
            f"<p>Version 0.1.0</p>"
            f"<p>A tool for extracting high-quality frames from videos for 3D reconstruction.</p>"
            f"<p>Copyright © {datetime.datetime.now().year}</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop processing if active
        if self.processing_thread and self.processing_thread.isRunning():
            # Ask for confirmation
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is still active. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._stop_processing()
            else:
                event.ignore()
                return
        
        event.accept()

    def _convert_cubemap_to_faces(self):
        """Convert cubemap images to individual faces."""
        # Show directory selection dialog
        dialog = DirectorySelectionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            input_dir, output_dir = dialog.get_directories()
            
            if not input_dir or not output_dir:
                QMessageBox.warning(
                    self,
                    "Invalid Directories",
                    "Please select both input and output directories."
                )
                return
            
            try:
                # Create converter and process directory
                from ..utils.cubemap_to_faces import CubemapToFaces
                converter = CubemapToFaces()
                processed = converter.process_directory(input_dir, output_dir)
                
                # Show success message
                QMessageBox.information(
                    self,
                    "Conversion Complete",
                    f"Successfully processed {processed} cubemap images."
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error converting cubemap images: {str(e)}"
                )

    def _add_videos(self):
        """Add videos to the processing queue."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Video files (*.mp4 *.avi *.mov *.mkv)")
        
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            self.video_list.add_videos(filenames)
    
    def _select_output_directory(self):
        """Select the output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        
        if directory:
            self.output_path.setText(directory)
            
            # Update processor
            self.processor.output_dir = Path(directory)
            
            # Update UI state
            self._update_ui_state()
    
    def _open_output_folder(self):
        """Open the output folder in the file explorer."""
        output_dir = self.output_path.text()
        
        if output_dir and os.path.exists(output_dir):
            # Convert the path to a QUrl object
            url = QUrl.fromLocalFile(output_dir)
            QDesktopServices.openUrl(url)
        else:
            QMessageBox.warning(self, "Error", "Output directory does not exist")

    def _processing_complete(self, stats):
        """Called when video processing is complete."""
        self.ui.progress_bar.hide()
        self.ui.statusbar.showMessage("Processing complete")
        
        # Update the UI based on the processing results
        total_frames = stats.get('total_frames', 0)
        blurry_frames = stats.get('blurry_frames', 0)
        saved_frames = stats.get('saved_frames', 0)
        
        # Show a message box with the results
        video_count = len(self.video_list.get_items())
        
        success_msg = f"Processed {video_count} videos.\n\n"
        success_msg += f"Total frames extracted: {total_frames}\n"
        
        if 'videos_processed' in stats and 'videos_failed' in stats:
            success_msg += f"Videos processed successfully: {stats['videos_processed']}\n"
            if stats['videos_failed'] > 0:
                success_msg += f"Videos failed: {stats['videos_failed']}\n"
                
        if 'repaired_frames' in stats and stats['repaired_frames'] > 0:
            success_msg += f"\nRepaired {stats['repaired_frames']} frames with missing cubemap faces\n"
            
        # Show details about the filters
        if self.ui.cb_blur_detection.isChecked():
            blurry_percent = (blurry_frames / total_frames * 100) if total_frames > 0 else 0
            success_msg += f"\nBlurry frames rejected: {blurry_frames} ({blurry_percent:.1f}%)\n"
            
        success_msg += f"\nFinal frames saved: {saved_frames}"
        
        # Show the output folder location
        output_dir = self.video_processor.output_dir
        success_msg += f"\n\nOutput saved to:\n{output_dir}"
        
        QMessageBox.information(self, "Processing Complete", success_msg)
        
        # Re-enable the UI
        self._set_ui_enabled(True) 