"""Frame selection functionality for the Frame Extractor application."""

import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import cv2
import numpy as np


class FrameSelector:
    """Selects the best frames from a video based on sharpness and distribution."""
    
    def __init__(self, blur_threshold: float = 100.0, max_frames: int = 20, min_frame_gap: int = 5):
        """Initialize the FrameSelector.
        
        Args:
            blur_threshold: Threshold for blur detection (lower means more strict)
            max_frames: Maximum number of frames to select
            min_frame_gap: Minimum gap between selected frames
        """
        self.blur_threshold = blur_threshold
        self.max_frames = max_frames
        self.min_frame_gap = min_frame_gap
        self.logger = logging.getLogger('FrameSelector')
        
        # Store frame scores and metadata
        self.frame_scores: Dict[str, Dict[str, float]] = {}
        self.frame_metadata: Dict[str, Dict[str, Any]] = {}
    
    def score_frames(self, frame_paths: List[str]) -> Dict[str, Dict[str, float]]:
        """Score all frames and store results for later filtering.
        
        Args:
            frame_paths: List of paths to frame files
            
        Returns:
            Dictionary mapping frame paths to their scores and metadata
        """
        self.frame_scores = {}
        self.frame_metadata = {}
        
        for i, path in enumerate(frame_paths):
            # Read frame
            frame = cv2.imread(path)
            if frame is None:
                self.logger.warning(f"Could not read frame at path {path}")
                continue
            
            # Calculate blur score
            _, sharpness_score = self.detect_blur(frame)
            
            # Store scores and metadata
            self.frame_scores[path] = {
                'sharpness': sharpness_score,
                'frame_number': i
            }
            
            # Log progress
            if (i + 1) % 10 == 0 or i == len(frame_paths) - 1:
                self.logger.info(f"Scored {i+1}/{len(frame_paths)} frames")
        
        return self.frame_scores
    
    def preview_filter_effects(self, 
                             blur_threshold: Optional[float] = None,
                             min_frame_gap: Optional[int] = None) -> Dict[str, Any]:
        """Preview how many frames would be affected by current filter settings.
        
        Args:
            blur_threshold: Optional new blur threshold to preview
            min_frame_gap: Optional new minimum frame gap to preview
            
        Returns:
            Dictionary containing preview statistics
        """
        if not self.frame_scores:
            return {
                'total_frames': 0,
                'frames_passing_blur': 0,
                'frames_passing_gap': 0,
                'estimated_final_frames': 0
            }
        
        # Use provided thresholds or current ones
        blur_threshold = blur_threshold if blur_threshold is not None else self.blur_threshold
        min_frame_gap = min_frame_gap if min_frame_gap is not None else self.min_frame_gap
        
        # Count frames passing blur threshold
        frames_passing_blur = [
            path for path, scores in self.frame_scores.items()
            if scores['sharpness'] >= blur_threshold
        ]
        
        # Simulate frame selection with minimum gap
        selected_frames = []
        selected_indices = []
        
        # Sort frames by sharpness score
        sorted_frames = sorted(
            frames_passing_blur,
            key=lambda x: self.frame_scores[x]['sharpness'],
            reverse=True
        )
        
        # First pass: select highest scoring frame
        if sorted_frames:
            best_frame = sorted_frames[0]
            selected_frames.append(best_frame)
            selected_indices.append(self.frame_scores[best_frame]['frame_number'])
        
        # Second pass: select remaining frames with min_gap constraint
        for frame_path in sorted_frames[1:]:
            current_index = self.frame_scores[frame_path]['frame_number']
            
            # Check if this frame is far enough from all previously selected frames
            if all(abs(current_index - selected_idx) >= min_frame_gap 
                  for selected_idx in selected_indices):
                selected_frames.append(frame_path)
                selected_indices.append(current_index)
                
                # Break if we've selected enough frames
                if len(selected_frames) >= self.max_frames:
                    break
        
        return {
            'total_frames': len(self.frame_scores),
            'frames_passing_blur': len(frames_passing_blur),
            'frames_passing_gap': len(selected_frames),
            'estimated_final_frames': len(selected_frames)
        }
    
    def apply_filters(self, 
                     blur_threshold: Optional[float] = None,
                     min_frame_gap: Optional[int] = None) -> List[str]:
        """Apply filters to get final frame selection.
        
        Args:
            blur_threshold: Optional new blur threshold to apply
            min_frame_gap: Optional new minimum frame gap to apply
            
        Returns:
            List of selected frame paths
        """
        if not self.frame_scores:
            return []
        
        # Get preview statistics
        preview = self.preview_filter_effects(blur_threshold, min_frame_gap)
        
        # Apply the same logic as preview but return the actual frames
        blur_threshold = blur_threshold if blur_threshold is not None else self.blur_threshold
        min_frame_gap = min_frame_gap if min_frame_gap is not None else self.min_frame_gap
        
        # Filter by blur threshold
        frames_passing_blur = [
            path for path, scores in self.frame_scores.items()
            if scores['sharpness'] >= blur_threshold
        ]
        
        # Apply minimum gap selection
        selected_frames = []
        selected_indices = []
        
        # Sort frames by sharpness score
        sorted_frames = sorted(
            frames_passing_blur,
            key=lambda x: self.frame_scores[x]['sharpness'],
            reverse=True
        )
        
        # First pass: select highest scoring frame
        if sorted_frames:
            best_frame = sorted_frames[0]
            selected_frames.append(best_frame)
            selected_indices.append(self.frame_scores[best_frame]['frame_number'])
        
        # Second pass: select remaining frames with min_gap constraint
        for frame_path in sorted_frames[1:]:
            current_index = self.frame_scores[frame_path]['frame_number']
            
            # Check if this frame is far enough from all previously selected frames
            if all(abs(current_index - selected_idx) >= min_frame_gap 
                  for selected_idx in selected_indices):
                selected_frames.append(frame_path)
                selected_indices.append(current_index)
                
                # Break if we've selected enough frames
                if len(selected_frames) >= self.max_frames:
                    break
        
        return selected_frames
    
    def detect_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """Detect if an image is blurry using Laplacian variance.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            Tuple of (is_blurry, sharpness_score)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()
        
        # Determine if image is blurry
        is_blurry = sharpness_score < self.blur_threshold
        
        return is_blurry, sharpness_score
    
    def set_parameters(self, blur_threshold: Optional[float] = None, 
                       max_frames: Optional[int] = None,
                       min_frame_gap: Optional[int] = None):
        """Update selector parameters.
        
        Args:
            blur_threshold: New threshold for blur detection
            max_frames: New maximum number of frames to select
            min_frame_gap: New minimum gap between selected frames
        """
        if blur_threshold is not None:
            self.blur_threshold = blur_threshold
            
        if max_frames is not None:
            self.max_frames = max_frames
            
        if min_frame_gap is not None:
            self.min_frame_gap = min_frame_gap 