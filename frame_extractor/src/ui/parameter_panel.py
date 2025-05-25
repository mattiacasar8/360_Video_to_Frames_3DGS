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
        
        # Live update notice
        self.live_update_label = QLabel("Parameters update results in real-time")
        self.live_update_label.setStyleSheet("color: #060; background-color: #efe; padding: 5px; border-radius: 3px; font-weight: bold;")
        self.live_update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create tab widget for organizing parameter groups
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.extraction_tab = QWidget()
        self.blur_tab = QWidget()
        self.similarity_tab = QWidget()
        
        # Set up tab layouts
        self._setup_extraction_tab()
        self._setup_blur_tab()
        self._setup_similarity_tab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.extraction_tab, "Frame Extraction")
        self.tab_widget.addTab(self.blur_tab, "Blur Detection")
        self.tab_widget.addTab(self.similarity_tab, "Frame Selection")
        
        # Reset button
        reset_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.setToolTip("Reset all parameters to default values")
        reset_layout.addWidget(self.reset_button)
        
        # Add widgets to main layout
        main_layout.addWidget(self.live_update_label)
        main_layout.addWidget(self.tab_widget)
        main_layout.addLayout(reset_layout)
        main_layout.addStretch()
    
    def _setup_extraction_tab(self):
        """Set up the frame extraction tab."""
        layout = QVBoxLayout(self.extraction_tab)
        
        # Frame extraction parameters
        extraction_group = QGroupBox("Frame Extraction")
        extraction_layout = QFormLayout(extraction_group)
        
        # Enable frame skipping toggle
        self.enable_frame_skip = QCheckBox("Enable Frame Skipping")
        self.enable_frame_skip.setChecked(False)
        extraction_layout.addRow("", self.enable_frame_skip)
        
        # FPS selection
        fps_layout = QHBoxLayout()
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(2)  # Default to 2 FPS
        self.fps_spin.setSingleStep(1)
        self.fps_spin.setEnabled(False)  # Initially disabled
        
        fps_layout.addWidget(self.fps_spin)
        fps_layout.addWidget(QLabel("FPS"))
        
        extraction_layout.addRow("Extraction Rate:", fps_layout)
        
        # Help text for FPS
        fps_help = QLabel("Number of frames to extract per second. Lower values mean fewer frames.")
        fps_help.setWordWrap(True)
        fps_help.setStyleSheet("color: #666; font-size: 10px;")
        extraction_layout.addRow("", fps_help)
        
        # Add groups to tab layout
        layout.addWidget(extraction_group)
        layout.addStretch()
    
    def _setup_blur_tab(self):
        """Set up the blur detection tab."""
        layout = QVBoxLayout(self.blur_tab)
        
        # Blur detection parameters
        blur_group = QGroupBox("Blur Detection")
        blur_layout = QFormLayout(blur_group)
        
        # Enable blur detection toggle
        self.enable_blur_detection = QCheckBox("Enable Blur Detection")
        self.enable_blur_detection.setChecked(False)
        blur_layout.addRow("", self.enable_blur_detection)
        
        # Blur threshold slider and spin box
        blur_slider_layout = QHBoxLayout()
        
        self.blur_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_threshold_slider.setRange(10, 500)
        self.blur_threshold_slider.setValue(100)  # Default to 100.0
        self.blur_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.blur_threshold_slider.setTickInterval(50)
        self.blur_threshold_slider.setEnabled(False)  # Initially disabled
        
        self.blur_threshold_spin = QDoubleSpinBox()
        self.blur_threshold_spin.setRange(10, 500)
        self.blur_threshold_spin.setValue(100)  # Default to 100.0
        self.blur_threshold_spin.setSingleStep(10)
        self.blur_threshold_spin.setDecimals(1)
        self.blur_threshold_spin.setEnabled(False)  # Initially disabled
        
        blur_slider_layout.addWidget(self.blur_threshold_slider, 7)
        blur_slider_layout.addWidget(self.blur_threshold_spin, 3)
        
        blur_layout.addRow("Laplacian Threshold:", blur_slider_layout)
        
        # Preview label for blur detection
        self.blur_preview = QLabel("Preview: No frames scored yet")
        self.blur_preview.setWordWrap(True)
        blur_layout.addRow("", self.blur_preview)
        
        # Help text for blur threshold
        blur_help = QLabel("Higher values include more frames (lenient). Lower values keep only very sharp frames (strict).\nTypical values: <50 (very strict), 50-150 (moderate), >150 (lenient)")
        blur_help.setWordWrap(True)
        blur_help.setStyleSheet("color: #666; font-size: 10px;")
        blur_layout.addRow("", blur_help)
        
        # Add groups to tab layout
        layout.addWidget(blur_group)
        layout.addStretch()
    
    def _setup_similarity_tab(self):
        """Set up the frame selection tab."""
        layout = QVBoxLayout(self.similarity_tab)
        
        # Frame selection parameters
        selection_group = QGroupBox("Frame Selection")
        selection_layout = QFormLayout(selection_group)
        
        # Enable frame selection toggle
        self.enable_frame_selection = QCheckBox("Enable Frame Selection")
        self.enable_frame_selection.setChecked(False)
        selection_layout.addRow("", self.enable_frame_selection)
        
        # Minimum frame gap
        min_gap_layout = QHBoxLayout()
        
        self.min_gap_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_gap_slider.setRange(1, 30)
        self.min_gap_slider.setValue(5)  # Default to 5
        self.min_gap_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.min_gap_slider.setTickInterval(1)
        self.min_gap_slider.setEnabled(False)  # Initially disabled
        
        self.min_gap_spin = QSpinBox()
        self.min_gap_spin.setRange(1, 30)
        self.min_gap_spin.setValue(5)  # Default to 5
        self.min_gap_spin.setSingleStep(1)
        self.min_gap_spin.setEnabled(False)  # Initially disabled
        
        min_gap_layout.addWidget(self.min_gap_slider, 7)
        min_gap_layout.addWidget(self.min_gap_spin, 3)
        
        selection_layout.addRow("Minimum Frame Gap:", min_gap_layout)
        
        # Preview label for frame selection
        self.selection_preview = QLabel("Preview: No frames scored yet")
        self.selection_preview.setWordWrap(True)
        selection_layout.addRow("", self.selection_preview)
        
        # Help text for min gap
        min_gap_help = QLabel("Minimum number of frames that must separate selected frames.\nSmaller values (1-3) keep more frames, larger values (>10) keep fewer frames but more evenly distributed.\nNote: Large values may cause zero frames to be selected if too restrictive.")
        min_gap_help.setWordWrap(True)
        min_gap_help.setStyleSheet("color: #666; font-size: 10px;")
        selection_layout.addRow("", min_gap_help)
        
        # Add groups to tab layout
        layout.addWidget(selection_group)
        layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # Frame extraction signals
        self.enable_frame_skip.stateChanged.connect(self._toggle_frame_skip)
        self.fps_spin.valueChanged.connect(self._fps_changed)
        
        # Blur detection signals
        self.enable_blur_detection.stateChanged.connect(self._toggle_blur_detection)
        self.blur_threshold_slider.valueChanged.connect(self._blur_threshold_slider_changed)
        self.blur_threshold_spin.valueChanged.connect(self._blur_threshold_spin_changed)
        
        # Frame selection signals
        self.enable_frame_selection.stateChanged.connect(self._toggle_frame_selection)
        self.min_gap_slider.valueChanged.connect(self._min_gap_slider_changed)
        self.min_gap_spin.valueChanged.connect(self._min_gap_spin_changed)
        
        # Reset button
        self.reset_button.clicked.connect(self.reset_parameters)
    
    def _toggle_frame_skip(self, state: int):
        """Handle frame skip toggle.
        
        Args:
            state: New checkbox state
        """
        self.fps_spin.setEnabled(state == Qt.CheckState.Checked.value)
        self.parameters_changed.emit()
    
    def _toggle_blur_detection(self, state: int):
        """Handle blur detection toggle.
        
        Args:
            state: New checkbox state
        """
        enabled = state == Qt.CheckState.Checked.value
        self.blur_threshold_slider.setEnabled(enabled)
        self.blur_threshold_spin.setEnabled(enabled)
        self.parameters_changed.emit()
    
    def _toggle_frame_selection(self, state: int):
        """Handle frame selection toggle.
        
        Args:
            state: New checkbox state
        """
        enabled = state == Qt.CheckState.Checked.value
        self.min_gap_slider.setEnabled(enabled)
        self.min_gap_spin.setEnabled(enabled)
        self.parameters_changed.emit()
    
    def _fps_changed(self, value: int):
        """Handle FPS change.
        
        Args:
            value: New FPS value
        """
        self.parameters_changed.emit()
    
    def _blur_threshold_slider_changed(self, value: int):
        """Handle blur threshold slider change.
        
        Args:
            value: New slider value
        """
        self.blur_threshold_spin.setValue(value)
        self.parameters_changed.emit()
    
    def _blur_threshold_spin_changed(self, value: float):
        """Handle blur threshold spin box change.
        
        Args:
            value: New spin box value
        """
        self.blur_threshold_slider.setValue(int(value))
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
        # Frame extraction
        self.enable_frame_skip.setChecked(False)
        self.fps_spin.setValue(2)
        
        # Blur detection
        self.enable_blur_detection.setChecked(False)
        self.blur_threshold_slider.setValue(100)
        self.blur_threshold_spin.setValue(100)
        
        # Frame selection
        self.enable_frame_selection.setChecked(False)
        self.min_gap_slider.setValue(5)
        self.min_gap_spin.setValue(5)
        
        # Emit signal that parameters have changed
        self.parameters_changed.emit()
    
    def get_parameters(self) -> dict:
        """Get all current parameter values.
        
        Returns:
            Dictionary containing all parameter values
        """
        return {
            'enable_frame_skip': self.enable_frame_skip.isChecked(),
            'fps': self.fps_spin.value(),
            'enable_blur_detection': self.enable_blur_detection.isChecked(),
            'blur_threshold': self.blur_threshold_spin.value(),
            'enable_frame_selection': self.enable_frame_selection.isChecked(),
            'min_frame_gap': self.min_gap_spin.value()
        }

    def _setup_frame_skipping_tab(self):
        """Set up the frame skipping tab."""
        layout = QVBoxLayout()
        
        # Frame skipping toggle
        self.frame_skip_toggle = QCheckBox("Enable Frame Skipping")
        self.frame_skip_toggle.setChecked(False)
        self.frame_skip_toggle.stateChanged.connect(self._update_preview)
        layout.addWidget(self.frame_skip_toggle)
        
        # FPS control
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frames per second:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(1)
        self.fps_spin.setEnabled(False)
        self.fps_spin.valueChanged.connect(self._update_preview)
        fps_layout.addWidget(self.fps_spin)
        layout.addLayout(fps_layout)
        
        # Preview label
        self.frame_skip_preview = QLabel("Preview: No frame skipping enabled")
        self.frame_skip_preview.setWordWrap(True)
        layout.addWidget(self.frame_skip_preview)
        
        # Description
        description = QLabel(
            "Frame skipping allows you to extract frames at a specific rate.\n"
            "For example, setting 1 FPS will extract one frame per second.\n"
            "This is useful for reducing the number of frames when processing long videos."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        layout.addStretch()
        return layout
    
    def _setup_blur_detection_tab(self):
        """Set up the blur detection tab."""
        layout = QVBoxLayout()
        
        # Blur detection toggle
        self.blur_toggle = QCheckBox("Enable Blur Detection")
        self.blur_toggle.setChecked(False)
        self.blur_toggle.stateChanged.connect(self._update_preview)
        layout.addWidget(self.blur_toggle)
        
        # Blur threshold control
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Blur threshold:"))
        self.blur_threshold_spin = QDoubleSpinBox()
        self.blur_threshold_spin.setRange(0.1, 1000.0)
        self.blur_threshold_spin.setValue(100.0)
        self.blur_threshold_spin.setEnabled(False)
        self.blur_threshold_spin.valueChanged.connect(self._update_preview)
        threshold_layout.addWidget(self.blur_threshold_spin)
        layout.addLayout(threshold_layout)
        
        # Preview label
        self.blur_preview = QLabel("Preview: Blur detection disabled")
        self.blur_preview.setWordWrap(True)
        layout.addWidget(self.blur_preview)
        
        # Description
        description = QLabel(
            "Blur detection analyzes each frame and rejects blurry ones.\n"
            "Higher threshold values are more strict (reject more frames).\n"
            "Lower values are more lenient (keep more frames).\n"
            "Typical values range from 50 to 200."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        layout.addStretch()
        return layout
    
    def _setup_frame_selection_tab(self):
        """Set up the frame selection tab."""
        layout = QVBoxLayout()
        
        # Frame selection toggle
        self.frame_selection_toggle = QCheckBox("Enable Frame Selection")
        self.frame_selection_toggle.setChecked(False)
        self.frame_selection_toggle.stateChanged.connect(self._update_preview)
        layout.addWidget(self.frame_selection_toggle)
        
        # Minimum gap control
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Minimum gap between frames:"))
        self.min_gap_spin = QSpinBox()
        self.min_gap_spin.setRange(1, 1000)
        self.min_gap_spin.setValue(10)
        self.min_gap_spin.setEnabled(False)
        self.min_gap_spin.valueChanged.connect(self._update_preview)
        gap_layout.addWidget(self.min_gap_spin)
        layout.addLayout(gap_layout)
        
        # Preview label
        self.frame_selection_preview = QLabel("Preview: Frame selection disabled")
        self.frame_selection_preview.setWordWrap(True)
        layout.addWidget(self.frame_selection_preview)
        
        # Description
        description = QLabel(
            "Frame selection ensures a minimum gap between selected frames.\n"
            "This helps avoid selecting frames that are too similar.\n"
            "For example, a gap of 10 means at least 10 frames must separate each selected frame.\n"
            "This is useful for ensuring good coverage of the scene."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        layout.addStretch()
        return layout
    
    def _update_preview(self):
        """Update the preview labels based on current settings."""
        # Frame skipping preview
        if self.frame_skip_toggle.isChecked():
            fps = self.fps_spin.value()
            self.frame_skip_preview.setText(
                f"Preview: Will extract 1 frame every {1/fps:.2f} seconds\n"
                f"For a 1-minute video, this will extract approximately {60 * fps} frames"
            )
        else:
            self.frame_skip_preview.setText("Preview: No frame skipping enabled")
        
        # Blur detection preview
        if self.blur_toggle.isChecked():
            threshold = self.blur_threshold_spin.value()
            self.blur_preview.setText(
                f"Preview: Will reject frames with blur score below {threshold}\n"
                f"Higher threshold = more strict filtering"
            )
        else:
            self.blur_preview.setText("Preview: Blur detection disabled")
        
        # Frame selection preview
        if self.frame_selection_toggle.isChecked():
            gap = self.min_gap_spin.value()
            self.frame_selection_preview.setText(
                f"Preview: Will ensure at least {gap} frames between selected frames\n"
                f"This helps avoid selecting similar frames"
            )
        else:
            self.frame_selection_preview.setText("Preview: Frame selection disabled")

    def update_previews(self, frame_selector):
        """Update preview labels with current filter effects.
        
        Args:
            frame_selector: FrameSelector instance to get preview statistics
        """
        # Get preview statistics
        preview = frame_selector.preview_filter_effects(
            blur_threshold=self.blur_threshold_spin.value() if self.enable_blur_detection.isChecked() else None,
            min_frame_gap=self.min_gap_spin.value() if self.enable_frame_selection.isChecked() else None
        )
        
        # Update blur detection preview
        if preview['total_frames'] > 0:
            blur_text = (
                f"Preview: {preview['frames_passing_blur']} frames pass blur threshold\n"
                f"({preview['frames_passing_blur']/preview['total_frames']*100:.1f}% of total)"
            )
            # Add warning if no frames pass blur threshold
            if preview['frames_passing_blur'] == 0 and self.enable_blur_detection.isChecked():
                blur_text += "\n⚠️ Warning: No frames pass the current threshold. Try increasing the value."
        else:
            blur_text = "Preview: No frames scored yet"
        self.blur_preview.setText(blur_text)
        
        # Update frame selection preview
        if preview['total_frames'] > 0:
            selection_text = (
                f"Preview: {preview['frames_passing_gap']} frames selected with minimum gap\n"
                f"({preview['frames_passing_gap']/preview['total_frames']*100:.1f}% of total)"
            )
            # Add warning if no frames pass frame selection
            if preview['frames_passing_gap'] == 0 and self.enable_frame_selection.isChecked():
                selection_text += "\n⚠️ Warning: No frames selected with current gap. Try reducing the value."
        else:
            selection_text = "Preview: No frames scored yet"
        self.selection_preview.setText(selection_text) 