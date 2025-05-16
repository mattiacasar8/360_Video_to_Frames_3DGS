"""Progress widget for displaying processing status and statistics."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt
from typing import Dict, Any


class ProgressWidget(QWidget):
    """Widget for displaying processing progress and statistics."""
    
    def __init__(self, parent=None):
        """Initialize the ProgressWidget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up the UI
        self._setup_ui()
        
        # Initialize values
        self.reset_progress()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video progress group
        video_group = QGroupBox("Video Progress")
        video_layout = QVBoxLayout(video_group)
        
        # Current video info
        self.video_label = QLabel("No videos processing")
        video_layout.addWidget(self.video_label)
        
        # Video progress bar
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        video_layout.addWidget(self.video_progress)
        
        # Frame progress group
        frame_group = QGroupBox("Frame Progress")
        frame_layout = QVBoxLayout(frame_group)
        
        # Current frame info
        self.frame_label = QLabel("No frames processing")
        frame_layout.addWidget(self.frame_label)
        
        # Frame progress bar
        self.frame_progress = QProgressBar()
        self.frame_progress.setRange(0, 100)
        self.frame_progress.setValue(0)
        frame_layout.addWidget(self.frame_progress)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        
        # Statistics labels
        self.total_frames_label = QLabel("0")
        self.blurry_frames_label = QLabel("0")
        self.similar_frames_label = QLabel("0")
        self.saved_frames_label = QLabel("0")
        
        stats_layout.addRow("Total frames:", self.total_frames_label)
        stats_layout.addRow("Blurry frames:", self.blurry_frames_label)
        stats_layout.addRow("Similar frames:", self.similar_frames_label)
        stats_layout.addRow("Saved frames:", self.saved_frames_label)
        
        # Add groups to main layout
        main_layout.addWidget(video_group)
        main_layout.addWidget(frame_group)
        main_layout.addWidget(stats_group)
    
    def start_progress(self, total_videos: int):
        """Start tracking progress for a new processing operation.
        
        Args:
            total_videos: Total number of videos to process
        """
        self.total_videos = total_videos
        self.current_video = 0
        self.current_frame = 0
        self.total_frames = 0
        
        # Reset labels
        self.video_label.setText(f"Processing video 0/{total_videos}")
        self.frame_label.setText("Preparing...")
        
        # Reset progress bars
        self.video_progress.setRange(0, total_videos)
        self.video_progress.setValue(0)
        self.frame_progress.setRange(0, 100)  # Will be updated per video
        self.frame_progress.setValue(0)
        
        # Reset statistics
        self.total_frames_label.setText("0")
        self.blurry_frames_label.setText("0")
        self.similar_frames_label.setText("0")
        self.saved_frames_label.setText("0")
    
    def update_frame_progress(self, video_name: str, current_frame: int, total_frames: int):
        """Update the frame processing progress.
        
        Args:
            video_name: Name of the video being processed
            current_frame: Current frame being processed
            total_frames: Total frames to process
        """
        # Update frame information
        self.current_frame = current_frame
        self.total_frames = total_frames
        
        # Update frame label
        self.frame_label.setText(f"Processing '{video_name}' frame {current_frame}/{total_frames}")
        
        # Update frame progress bar
        self.frame_progress.setRange(0, total_frames)
        self.frame_progress.setValue(current_frame)
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update processing statistics.
        
        Args:
            stats: Current processing statistics
        """
        # Check for video progress info
        if 'current_video' in stats and 'total_videos' in stats:
            self.current_video = stats['current_video']
            self.total_videos = stats['total_videos']
            video_path = stats.get('video_path', '')
            
            # Update video progress info
            video_name = video_path.split('/')[-1] if '/' in video_path else video_path
            self.video_label.setText(f"Processing video {self.current_video}/{self.total_videos}: {video_name}")
            self.video_progress.setValue(self.current_video)
        
        # Update statistics
        if 'total_frames' in stats:
            self.total_frames_label.setText(str(stats['total_frames']))
        if 'blurry_frames' in stats:
            self.blurry_frames_label.setText(str(stats['blurry_frames']))
        if 'similar_frames' in stats:
            self.similar_frames_label.setText(str(stats['similar_frames']))
        if 'saved_frames' in stats:
            self.saved_frames_label.setText(str(stats['saved_frames']))
        
        # Handle saved frames update without statistics
        if stats.get('saved_frames_update', False):
            # Try to update just the saved frames count
            try:
                # Check if we can get this from the processor
                if 'processor_stats' in stats and 'saved_frames' in stats['processor_stats']:
                    self.saved_frames_label.setText(str(stats['processor_stats']['saved_frames']))
            except Exception:
                pass  # Ignore errors
    
    def complete_progress(self, stats: Dict[str, Any]):
        """Mark processing as complete and show final statistics.
        
        Args:
            stats: Final processing statistics
        """
        # Update video progress
        self.video_label.setText(f"Completed {stats.get('videos_processed', 0)}/{self.total_videos} videos")
        self.video_progress.setValue(self.total_videos)
        
        # Update frame progress
        self.frame_label.setText("Processing complete")
        self.frame_progress.setValue(self.frame_progress.maximum())
        
        # Update statistics
        self.update_stats(stats)
    
    def reset_progress(self):
        """Reset all progress indicators."""
        self.total_videos = 0
        self.current_video = 0
        
        # Reset labels
        self.video_label.setText("No videos processing")
        self.frame_label.setText("No frames processing")
        
        # Reset progress bars
        self.video_progress.setValue(0)
        self.frame_progress.setValue(0)
        
        # Reset statistics
        self.total_frames_label.setText("0")
        self.blurry_frames_label.setText("0")
        self.similar_frames_label.setText("0")
        self.saved_frames_label.setText("0") 