"""Video list widget for the Frame Extractor application."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QListWidget, QListWidgetItem, QWidget, QCheckBox, QHBoxLayout,
    QLabel, QMenu, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from ..core.spherical_converter import SphericalConverter


class VideoListItem(QWidget):
    """Custom widget for displaying a video in the list."""
    
    def __init__(self, video_path: str, parent=None):
        """Initialize the VideoListItem.
        
        Args:
            video_path: Path to the video file
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Store video path
        self.video_path = video_path
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Video name label
        self.video_name_label = QLabel(Path(video_path).name)
        layout.addWidget(self.video_name_label, 1)
        
        # 360° checkbox
        self.is_360_checkbox = QCheckBox("360°")
        layout.addWidget(self.is_360_checkbox)
        
        # Try to detect if this is a 360° video
        self._detect_360()
    
    def _detect_360(self):
        """Attempt to detect if the video is in 360° format."""
        try:
            # Use the SphericalConverter to detect 360° videos
            converter = SphericalConverter()
            is_360 = converter.is_360_video(self.video_path)
            
            # Set checkbox state
            self.is_360_checkbox.setChecked(is_360)
            
        except Exception as e:
            # If detection fails, leave unchecked
            self.is_360_checkbox.setChecked(False)
    
    def is_360(self) -> bool:
        """Check if the video is marked as 360°.
        
        Returns:
            True if the video is marked as 360°, False otherwise
        """
        return self.is_360_checkbox.isChecked()
    
    def set_is_360(self, is_360: bool):
        """Set the 360° status of the video.
        
        Args:
            is_360: Whether the video is in 360° format
        """
        self.is_360_checkbox.setChecked(is_360)


class VideoListWidget(QListWidget):
    """Widget for displaying and managing the list of videos to process."""
    
    video_list_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the VideoListWidget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Configure list widget
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        
        # Enable drag and drop
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        
        # Connect signals
        self.itemChanged.connect(self._on_item_changed)
        
        # Set up context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
    
    def add_videos(self, video_paths: List[str]):
        """Add videos to the list.
        
        Args:
            video_paths: List of paths to video files
        """
        # Ensure we have valid video files
        valid_paths = []
        for path in video_paths:
            if os.path.exists(path) and os.path.isfile(path):
                # Check file extension
                if Path(path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    valid_paths.append(path)
        
        # Add valid videos to the list
        for path in valid_paths:
            # Check if video is already in the list
            if not self._video_exists(path):
                # Create item
                item = QListWidgetItem(self)
                
                # Set video path in UserRole data for analysis
                item.setData(Qt.ItemDataRole.UserRole, path)
                
                # Create widget
                video_widget = VideoListItem(path)
                
                # Set item widget and size
                self.addItem(item)
                item.setSizeHint(video_widget.sizeHint())
                self.setItemWidget(item, video_widget)
        
        # Emit signal if we added any videos
        if valid_paths:
            self.video_list_changed.emit()
    
    def _video_exists(self, video_path: str) -> bool:
        """Check if a video is already in the list.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if the video is already in the list, False otherwise
        """
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            
            if isinstance(widget, VideoListItem) and widget.video_path == video_path:
                return True
        
        return False
    
    def remove_selected(self):
        """Remove selected videos from the list."""
        # Get selected items
        selected_items = self.selectedItems()
        
        # Remove items
        for item in selected_items:
            self.takeItem(self.row(item))
        
        # Emit signal if we removed any items
        if selected_items:
            self.video_list_changed.emit()
    
    def clear(self):
        """Clear all videos from the list."""
        if self.count() > 0:
            super().clear()
            self.video_list_changed.emit()
    
    def get_video_paths(self) -> List[str]:
        """Get the list of video paths.
        
        Returns:
            List of paths to video files
        """
        paths = []
        
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            
            if isinstance(widget, VideoListItem):
                paths.append(widget.video_path)
        
        return paths
    
    def get_is_360_list(self) -> List[bool]:
        """Get the list of 360° statuses.
        
        Returns:
            List indicating which videos are 360°
        """
        is_360_list = []
        
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            
            if isinstance(widget, VideoListItem):
                is_360_list.append(widget.is_360())
        
        return is_360_list
    
    def _on_item_changed(self, item):
        """Handle item change events."""
        self.video_list_changed.emit()
    
    def _show_context_menu(self, position):
        """Show context menu for the list item at the given position."""
        # Create menu
        menu = QMenu()
        
        # Add actions
        remove_action = menu.addAction("Remove Selected")
        mark_360_action = menu.addAction("Mark as 360°")
        unmark_360_action = menu.addAction("Unmark as 360°")
        
        # Show menu and handle result
        action = menu.exec(self.mapToGlobal(position))
        
        if action == remove_action:
            self.remove_selected()
        
        elif action == mark_360_action:
            # Mark selected items as 360°
            for item in self.selectedItems():
                widget = self.itemWidget(item)
                if isinstance(widget, VideoListItem):
                    widget.set_is_360(True)
        
        elif action == unmark_360_action:
            # Unmark selected items as 360°
            for item in self.selectedItems():
                widget = self.itemWidget(item)
                if isinstance(widget, VideoListItem):
                    widget.set_is_360(False)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events.
        
        Args:
            event: Drag enter event
        """
        # Accept file drops
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events.
        
        Args:
            event: Drop event
        """
        # Get dropped files
        file_paths = []
        
        for url in event.mimeData().urls():
            # Get local file path
            file_path = url.toLocalFile()
            
            # Add to list if it exists
            if os.path.exists(file_path) and os.path.isfile(file_path):
                file_paths.append(file_path)
        
        # Add videos
        if file_paths:
            self.add_videos(file_paths)
            event.acceptProposedAction() 