"""Similarity detection functionality for the Frame Extractor application."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim


class SimilarityDetector:
    """Detects similarity between images using SSIM and histogram comparison."""
    
    def __init__(self, ssim_threshold: float = 0.92, hist_threshold: float = 0.95, max_stored_frames: int = 10):
        """Initialize the SimilarityDetector.
        
        Args:
            ssim_threshold: SSIM threshold above which images are considered similar
            hist_threshold: Histogram comparison threshold above which images are considered similar
            max_stored_frames: Maximum number of frames to keep for comparison
        """
        self.ssim_threshold = ssim_threshold
        self.hist_threshold = hist_threshold
        self.max_stored_frames = max_stored_frames
        self.stored_frames = []
        self.stored_histograms = []
        self.downscale_size = (320, 240)  # Downscale images for faster comparison
    
    def reset(self):
        """Reset the stored frames and histograms."""
        self.stored_frames = []
        self.stored_histograms = []
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image for similarity comparison.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (processed_grayscale_image, histogram)
        """
        # For SSIM: Convert to grayscale and resize
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize for faster processing
        resized_gray = cv2.resize(gray, self.downscale_size)
        
        # For histogram comparison: Convert to HSV
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = cv2.resize(hsv, self.downscale_size)  # Resize for consistency
            
            # Calculate histogram for H and S channels (ignore V as it's affected by lighting)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            
            # Normalize
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        else:
            # If input is already grayscale, create a simple grayscale histogram
            hist = cv2.calcHist([resized_gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return resized_gray, hist
    
    def compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float, float]:
        """Compute similarity between two images using multiple methods.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Tuple of (combined_score, ssim_score, hist_score)
        """
        # Preprocess images
        gray1, hist1 = self.preprocess_image(img1)
        gray2, hist2 = self.preprocess_image(img2)
        
        # 1. Compute SSIM
        ssim_score = ssim(gray1, gray2)
        
        # 2. Compare histograms
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Combine scores (weighted)
        combined_score = 0.6 * ssim_score + 0.4 * hist_score
        
        return combined_score, ssim_score, hist_score
    
    def is_similar(self, image: np.ndarray) -> Tuple[bool, float]:
        """Determine if image is similar to any stored frames.
        
        Args:
            image: Input image to check
            
        Returns:
            Tuple of (is_similar, max_similarity_score)
        """
        if not self.stored_frames:
            return False, 0.0
        
        # Preprocess input image
        proc_gray, proc_hist = self.preprocess_image(image)
        
        # Compare with stored frames
        max_combined_score = 0.0
        for i, (stored_gray, stored_hist) in enumerate(zip(self.stored_frames, self.stored_histograms)):
            # 1. SSIM comparison
            ssim_score = ssim(proc_gray, stored_gray)
            
            # 2. Histogram comparison
            hist_score = cv2.compareHist(proc_hist, stored_hist, cv2.HISTCMP_CORREL)
            
            # Combine scores
            combined_score = 0.6 * ssim_score + 0.4 * hist_score
            
            # Keep track of maximum score
            if combined_score > max_combined_score:
                max_combined_score = combined_score
        
        # Check if image is similar to any stored frame
        is_similar = (max_combined_score > self.get_combined_threshold())
        
        return is_similar, max_combined_score
    
    def get_combined_threshold(self) -> float:
        """Get the combined threshold for similarity detection.
        
        Returns:
            Combined threshold
        """
        # Weighted combination of SSIM and histogram thresholds
        return 0.6 * self.ssim_threshold + 0.4 * self.hist_threshold
    
    def add_frame(self, image: np.ndarray):
        """Add frame to stored frames for future similarity checks.
        
        Args:
            image: Frame to add
        """
        # Preprocess image
        proc_gray, proc_hist = self.preprocess_image(image)
        
        # Add to stored frames, keeping only max_stored_frames
        self.stored_frames.append(proc_gray)
        self.stored_histograms.append(proc_hist)
        
        # Ensure we don't exceed max_stored_frames
        if len(self.stored_frames) > self.max_stored_frames:
            self.stored_frames.pop(0)  # Remove oldest frame
            self.stored_histograms.pop(0)  # Remove oldest histogram
    
    def set_threshold(self, ssim_threshold: float = None, hist_threshold: float = None):
        """Update the similarity thresholds.
        
        Args:
            ssim_threshold: New SSIM threshold value
            hist_threshold: New histogram threshold value
        """
        if ssim_threshold is not None:
            self.ssim_threshold = ssim_threshold
            
        if hist_threshold is not None:
            self.hist_threshold = hist_threshold
    
    def get_threshold(self) -> float:
        """Get the SSIM threshold (for backward compatibility).
        
        Returns:
            SSIM threshold value
        """
        return self.ssim_threshold
    
    def set_max_stored_frames(self, max_frames: int):
        """Set the maximum number of stored frames.
        
        Args:
            max_frames: Maximum number of frames to store
        """
        self.max_stored_frames = max_frames
        
        # Trim stored frames if needed
        if len(self.stored_frames) > self.max_stored_frames:
            self.stored_frames = self.stored_frames[-self.max_stored_frames:]
            self.stored_histograms = self.stored_histograms[-self.max_stored_frames:] 