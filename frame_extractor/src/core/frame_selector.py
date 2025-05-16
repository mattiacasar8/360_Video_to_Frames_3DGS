"""Frame selection functionality for the Frame Extractor application."""

import os
import re
import logging
from typing import Dict, List, Tuple, Optional
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
    
    def analyze_frames(self, frames: List[np.ndarray], frame_paths: List[str]) -> Dict[str, float]:
        """Analyze frames and compute sharpness scores.
        
        Args:
            frames: List of frame images
            frame_paths: List of frame paths corresponding to the frames
            
        Returns:
            Dictionary mapping frame paths to sharpness scores
        """
        frame_scores = {}
        
        for i, (frame, path) in enumerate(zip(frames, frame_paths)):
            # Calculate blur score
            _, sharpness_score = self.detect_blur(frame)
            frame_scores[path] = sharpness_score
            
            # Log every 10 frames for progress reporting
            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                self.logger.info(f"Analyzed {i+1}/{len(frames)} frames")
        
        return frame_scores
    
    def analyze_frame_files(self, frame_paths: List[str]) -> Dict[str, float]:
        """Analyze frame files and compute sharpness scores.
        
        Args:
            frame_paths: List of paths to frame files
            
        Returns:
            Dictionary mapping frame paths to sharpness scores
        """
        frame_scores = {}
        
        for i, path in enumerate(frame_paths):
            # Read frame
            frame = cv2.imread(path)
            if frame is None:
                self.logger.warning(f"Could not read frame at path {path}")
                continue
                
            # Calculate blur score
            _, sharpness_score = self.detect_blur(frame)
            frame_scores[path] = sharpness_score
            
            # Log every 10 frames for progress reporting
            if (i + 1) % 10 == 0 or i == len(frame_paths) - 1:
                self.logger.info(f"Analyzed {i+1}/{len(frame_paths)} frames")
        
        return frame_scores
    
    def select_best_frames(self, frame_scores: Dict[str, float]) -> List[str]:
        """Select N frames with highest sharpness scores that are evenly distributed.
        
        Args:
            frame_scores: Dictionary mapping frame paths to sharpness scores
            
        Returns:
            List of selected frame paths
        """
        self.logger.info(f"Selecting up to {self.max_frames} frames with minimum gap of {self.min_frame_gap}")
        
        # Sort frames by score (highest first)
        sorted_frames = sorted(frame_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_frames = []
        selected_indices = []
        
        # Create a mapping from frame path to its original index
        frame_indices = {}
        for path, _ in sorted_frames:
            # Extract frame number from filename
            filename = os.path.basename(path)
            match = re.search(r'(\d+)', filename)
            if match:
                frame_indices[path] = int(match.group())
            else:
                # If we can't extract a number, use the position in the list
                frame_indices[path] = list(frame_scores.keys()).index(path)
        
        # First pass: select the highest scoring frame
        if sorted_frames:
            best_frame, best_score = sorted_frames[0]
            selected_frames.append(best_frame)
            selected_indices.append(frame_indices[best_frame])
            self.logger.info(f"Selected highest scoring frame: {os.path.basename(best_frame)} (score: {best_score:.2f})")
        
        # Second pass: select remaining frames with min_gap constraint
        for frame_path, score in sorted_frames[1:]:
            if frame_path in selected_frames:
                continue
                
            current_index = frame_indices[frame_path]
            
            # Check if this frame is far enough from all previously selected frames
            if all(abs(current_index - selected_idx) >= self.min_frame_gap for selected_idx in selected_indices):
                selected_frames.append(frame_path)
                selected_indices.append(current_index)
                self.logger.debug(f"Selected frame: {os.path.basename(frame_path)} (score: {score:.2f}, index: {current_index})")
                
                # Break if we've selected enough frames
                if len(selected_frames) >= self.max_frames:
                    break
        
        # If we still need more frames and couldn't meet the min_gap requirement
        # for all frames, we'll relax it slightly
        if len(selected_frames) < self.max_frames:
            self.logger.info(f"Could only select {len(selected_frames)} frames with strict gap requirements. Relaxing gap requirements.")
            remaining_frames = [f for f, _ in sorted_frames if f not in selected_frames]
            
            # Sort remaining frames to prioritize those with best scores
            remaining_frames_scored = [(f, frame_scores[f]) for f in remaining_frames]
            remaining_frames_scored.sort(key=lambda x: x[1], reverse=True)
            
            # Add frames with relaxed gap requirement
            for frame_path, score in remaining_frames_scored:
                if frame_path in selected_frames:
                    continue
                    
                current_index = frame_indices[frame_path]
                
                # Gradually reduce the gap requirement
                adjusted_gap = self.min_frame_gap // 2
                
                while adjusted_gap > 0:
                    if all(abs(current_index - selected_idx) >= adjusted_gap for selected_idx in selected_indices):
                        selected_frames.append(frame_path)
                        selected_indices.append(current_index)
                        self.logger.debug(f"Selected frame with relaxed gap ({adjusted_gap}): {os.path.basename(frame_path)} (score: {score:.2f}, index: {current_index})")
                        break
                    
                    adjusted_gap -= 1
                    
                # Break if we've selected enough frames
                if len(selected_frames) >= self.max_frames:
                    break
        
        # Sort the selected frames by their original index for sequential processing
        selected_frames_with_indices = [(f, frame_indices[f]) for f in selected_frames]
        selected_frames_with_indices.sort(key=lambda x: x[1])
        
        final_selection = [f for f, _ in selected_frames_with_indices]
        
        self.logger.info(f"Selected {len(final_selection)} frames out of {len(frame_scores)} total frames")
        
        return final_selection
    
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