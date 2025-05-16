"""Main window for the Frame Extractor application."""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QMessageBox, QApplication,
    QCheckBox, QGroupBox, QStatusBar, QToolBar,
    QSlider, QSpinBox, QDoubleSpinBox, QProgressBar, QToolButton,
    QTabWidget, QComboBox, QLineEdit
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
        
        # Progress widget
        self.progress_widget = ProgressWidget()
        right_layout.addWidget(self.progress_widget)
        
        # Process button
        self.process_button = QPushButton("Start Processing")
        self.process_button.setEnabled(False)
        self.process_button.setMinimumHeight(50)
        right_layout.addWidget(self.process_button)
        
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
        
        # Process button
        self.process_button.clicked.connect(self._start_processing)
        
        # Video list signals
        self.video_list.video_list_changed.connect(self._update_ui_state)
    
    def _initialize_processor(self):
        """Initialize the video processor."""
        # Use user's desktop as the default output directory
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        
        # Create processor with default parameters
        self.processor = VideoProcessor(
            output_dir=desktop_path,
            blur_threshold=self.parameter_panel.get_blur_threshold(),
            max_frames=self.parameter_panel.get_max_frames(),
            min_frame_gap=self.parameter_panel.get_min_frame_gap(),
            frame_interval=self.parameter_panel.get_frame_interval()
        )
    
    def _update_ui_state(self):
        """Update UI state based on current application state."""
        has_videos = self.video_list.count() > 0
        has_output = bool(self.output_path.text())
        
        # Enable/disable process button
        self.process_button.setEnabled(has_videos and has_output)
        
        # Enable/disable remove and clear buttons
        self.remove_button.setEnabled(has_videos)
        self.clear_button.setEnabled(has_videos)
        
        # Update status bar
        if has_videos:
            self.status_bar.showMessage(f"{self.video_list.count()} videos in queue")
        else:
            self.status_bar.showMessage("Ready")
    
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
        
        # Update processor parameters before starting
        self._update_parameters()
        
        # Log the current settings
        blur_threshold = self.parameter_panel.get_blur_threshold()
        max_frames = self.parameter_panel.get_max_frames()
        min_frame_gap = self.parameter_panel.get_min_frame_gap()
        frame_interval = self.parameter_panel.get_frame_interval()
        
        self.logger.info(f"Starting processing with: Blur threshold = {blur_threshold}, Max frames = {max_frames}, Min gap = {min_frame_gap}, Frame interval = {frame_interval}")
        
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
        self.status_bar.showMessage(f"Processing complete: {stats['saved_frames']} frames saved")
        
        # Show completion message
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Processing complete!\n\n"
            f"Videos processed: {stats['videos_processed']}\n"
            f"Total frames extracted: {stats['total_frames']}\n"
            f"Blurry frames rejected: {stats['blurry_frames']}\n"
            f"Similar frames rejected: {stats['similar_frames']}\n"
            f"Frames saved: {stats['saved_frames']}"
        )
    
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

    def _update_parameters(self):
        """Update processing parameters based on UI settings."""
        # Get parameters from UI
        blur_threshold = self.parameter_panel.get_blur_threshold()
        frame_interval = self.parameter_panel.get_frame_interval()
        max_frames = self.parameter_panel.get_max_frames()
        min_frame_gap = self.parameter_panel.get_min_frame_gap()
        
        # Update the processor
        self.processor.update_parameters(
            blur_threshold=blur_threshold,
            frame_interval=frame_interval,
            max_frames=max_frames,
            min_frame_gap=min_frame_gap
        ) 