"""Parameter panel for the Frame Extractor application."""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox,
    QDoubleSpinBox, QGroupBox, QPushButton, QCheckBox, QFormLayout,
    QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal


class ParameterPanel(QWidget):
    """Widget for configuring processing parameters."""
    
    parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the ParameterPanel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up the UI
        self._setup_ui()
        
        # Connect signals
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for organizing parameter groups
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.blur_tab = QWidget()
        self.similarity_tab = QWidget()
        self.extraction_tab = QWidget()
        
        # Set up tab layouts
        self._setup_blur_tab()
        self._setup_similarity_tab()
        self._setup_extraction_tab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.blur_tab, "Blur Detection")
        self.tab_widget.addTab(self.similarity_tab, "Frame Selection")
        self.tab_widget.addTab(self.extraction_tab, "Frame Extraction")
        
        # Reset button
        self.reset_button = QPushButton("Reset to Defaults")
        
        # Add widgets to main layout
        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(self.reset_button)
        main_layout.addStretch()
    
    def _setup_blur_tab(self):
        """Set up the blur detection tab."""
        layout = QVBoxLayout(self.blur_tab)
        
        # Laplacian variance parameters
        laplacian_group = QGroupBox("Blur Detection")
        laplacian_layout = QFormLayout(laplacian_group)
        
        # Blur threshold slider and spin box
        blur_slider_layout = QHBoxLayout()
        
        self.blur_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_threshold_slider.setRange(10, 500)
        self.blur_threshold_slider.setValue(100)  # Default to 100.0
        self.blur_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.blur_threshold_slider.setTickInterval(50)
        
        self.blur_threshold_spin = QDoubleSpinBox()
        self.blur_threshold_spin.setRange(10, 500)
        self.blur_threshold_spin.setValue(100)  # Default to 100.0
        self.blur_threshold_spin.setSingleStep(10)
        self.blur_threshold_spin.setDecimals(1)
        
        blur_slider_layout.addWidget(self.blur_threshold_slider, 7)
        blur_slider_layout.addWidget(self.blur_threshold_spin, 3)
        
        laplacian_layout.addRow("Laplacian Threshold:", blur_slider_layout)
        
        # Help text for blur threshold
        blur_help = QLabel("Lower values keep only very sharp frames, higher values keep more blurry frames.")
        blur_help.setWordWrap(True)
        blur_help.setStyleSheet("color: #666; font-size: 10px;")
        laplacian_layout.addRow("", blur_help)
        
        # Add groups to tab layout
        layout.addWidget(laplacian_group)
        layout.addStretch()
    
    def _setup_similarity_tab(self):
        """Set up the frame selection tab."""
        layout = QVBoxLayout(self.similarity_tab)
        
        # Frame selection parameters
        selection_group = QGroupBox("Frame Selection")
        selection_layout = QFormLayout(selection_group)
        
        # Maximum frames slider and spin box
        max_frames_layout = QHBoxLayout()
        
        self.max_frames_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_frames_slider.setRange(5, 100)
        self.max_frames_slider.setValue(20)  # Default to 20
        self.max_frames_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.max_frames_slider.setTickInterval(5)
        
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(5, 100)
        self.max_frames_spin.setValue(20)  # Default to 20
        self.max_frames_spin.setSingleStep(1)
        
        max_frames_layout.addWidget(self.max_frames_slider, 7)
        max_frames_layout.addWidget(self.max_frames_spin, 3)
        
        selection_layout.addRow("Max Frames to Select:", max_frames_layout)
        
        # Help text for max frames
        max_frames_help = QLabel("Maximum number of frames to select per video.")
        max_frames_help.setWordWrap(True)
        max_frames_help.setStyleSheet("color: #666; font-size: 10px;")
        selection_layout.addRow("", max_frames_help)
        
        # Minimum frame gap
        min_gap_layout = QHBoxLayout()
        
        self.min_gap_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_gap_slider.setRange(1, 30)
        self.min_gap_slider.setValue(5)  # Default to 5
        self.min_gap_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.min_gap_slider.setTickInterval(1)
        
        self.min_gap_spin = QSpinBox()
        self.min_gap_spin.setRange(1, 30)
        self.min_gap_spin.setValue(5)  # Default to 5
        self.min_gap_spin.setSingleStep(1)
        
        min_gap_layout.addWidget(self.min_gap_slider, 7)
        min_gap_layout.addWidget(self.min_gap_spin, 3)
        
        selection_layout.addRow("Minimum Frame Gap:", min_gap_layout)
        
        # Help text for min gap
        min_gap_help = QLabel("Minimum gap between selected frames to ensure even distribution.")
        min_gap_help.setWordWrap(True)
        min_gap_help.setStyleSheet("color: #666; font-size: 10px;")
        selection_layout.addRow("", min_gap_help)
        
        # Add groups to tab layout
        layout.addWidget(selection_group)
        layout.addStretch()
    
    def _setup_extraction_tab(self):
        """Set up the frame extraction tab."""
        layout = QFormLayout(self.extraction_tab)
        
        # Frame interval
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 100)
        self.frame_interval_spin.setValue(15)
        self.frame_interval_spin.setSingleStep(1)
        
        layout.addRow("Frame Interval:", self.frame_interval_spin)
        
        # Help text for frame interval
        interval_help = QLabel("Number of frames to skip between extractions. Higher values extract fewer frames.")
        interval_help.setWordWrap(True)
        interval_help.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow("", interval_help)
        
        # Add spacer
        layout.addItem(QVBoxLayout())
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # Blur detection parameters
        self.blur_threshold_slider.valueChanged.connect(self._blur_slider_changed)
        self.blur_threshold_spin.valueChanged.connect(self._blur_spin_changed)
        
        # Frame selection parameters
        self.max_frames_slider.valueChanged.connect(self._max_frames_slider_changed)
        self.max_frames_spin.valueChanged.connect(self._max_frames_spin_changed)
        self.min_gap_slider.valueChanged.connect(self._min_gap_slider_changed)
        self.min_gap_spin.valueChanged.connect(self._min_gap_spin_changed)
        
        # Other parameters
        self.frame_interval_spin.valueChanged.connect(self.parameters_changed.emit)
        
        # Reset button
        self.reset_button.clicked.connect(self.reset_parameters)
    
    def _blur_slider_changed(self, value: int):
        """Handle blur threshold slider change.
        
        Args:
            value: New slider value
        """
        self.blur_threshold_spin.setValue(value)
        self.parameters_changed.emit()
    
    def _blur_spin_changed(self, value: float):
        """Handle blur threshold spin box change.
        
        Args:
            value: New spin box value
        """
        self.blur_threshold_slider.setValue(int(value))
        self.parameters_changed.emit()
    
    def _max_frames_slider_changed(self, value: int):
        """Handle max frames slider change.
        
        Args:
            value: New slider value
        """
        self.max_frames_spin.setValue(value)
        self.parameters_changed.emit()
    
    def _max_frames_spin_changed(self, value: int):
        """Handle max frames spin box change.
        
        Args:
            value: New spin box value
        """
        self.max_frames_slider.setValue(value)
        self.parameters_changed.emit()
    
    def _min_gap_slider_changed(self, value: int):
        """Handle min gap slider change.
        
        Args:
            value: New slider value
        """
        self.min_gap_spin.setValue(value)
        self.parameters_changed.emit()
    
    def _min_gap_spin_changed(self, value: int):
        """Handle min gap spin box change.
        
        Args:
            value: New spin box value
        """
        self.min_gap_slider.setValue(value)
        self.parameters_changed.emit()
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        # Blur threshold
        self.blur_threshold_slider.setValue(100)
        self.blur_threshold_spin.setValue(100)
        
        # Frame selection
        self.max_frames_slider.setValue(20)
        self.max_frames_spin.setValue(20)
        self.min_gap_slider.setValue(5)
        self.min_gap_spin.setValue(5)
        
        # Frame interval
        self.frame_interval_spin.setValue(15)
        
        # Emit signal that parameters have changed
        self.parameters_changed.emit()
    
    def get_blur_threshold(self) -> float:
        """Get the current blur threshold.
        
        Returns:
            Blur threshold value
        """
        return self.blur_threshold_spin.value()
    
    def get_max_frames(self) -> int:
        """Get the maximum number of frames to select.
        
        Returns:
            Maximum frames value
        """
        return self.max_frames_spin.value()
    
    def get_min_frame_gap(self) -> int:
        """Get the minimum gap between selected frames.
        
        Returns:
            Minimum frame gap value
        """
        return self.min_gap_spin.value()
    
    def get_frame_interval(self) -> int:
        """Get the current frame interval.
        
        Returns:
            Frame interval value
        """
        return self.frame_interval_spin.value() 