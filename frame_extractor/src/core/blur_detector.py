"""Blur detection functionality for the Frame Extractor application."""

import cv2
import numpy as np
from typing import Tuple


class BlurDetector:
    """Detects blurry images using combined Laplacian variance and FFT methods."""
    
    def __init__(self, laplacian_threshold: float = 100.0, fft_threshold: float = 10.0, fft_size: int = 60):
        """Initialize the BlurDetector.
        
        Args:
            laplacian_threshold: Laplacian variance threshold below which an image is considered blurry
            fft_threshold: FFT threshold for high frequency content
            fft_size: Size of the high frequency region in FFT
        """
        self.laplacian_threshold = laplacian_threshold
        self.fft_threshold = fft_threshold
        self.fft_size = fft_size
    
    def compute_blur_score(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Compute the blur score for an image using combined methods.
        
        Uses both Laplacian variance and FFT methods for more robust blur detection.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (combined_score, laplacian_score, fft_score)
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 1. Laplacian variance method
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_score = laplacian.var()
        
        # 2. FFT method (frequency domain analysis)
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        
        # Compute FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # Create mask for high frequency region
        mask = np.zeros((h, w), np.uint8)
        mask[cy-self.fft_size:cy+self.fft_size, cx-self.fft_size:cx+self.fft_size] = 1
        
        # Compute magnitude
        magnitude = 20 * np.log(np.abs(fshift) + 1e-10)  # Add small constant to avoid log(0)
        
        # Mean value in high frequency region
        fft_score = np.mean(magnitude * mask)
        
        # Combine scores (weighted average)
        # Normalize each score relative to its threshold
        norm_lap_score = lap_score / self.laplacian_threshold
        norm_fft_score = fft_score / self.fft_threshold
        
        # Combined score (weighted)
        combined_score = 0.7 * norm_lap_score + 0.3 * norm_fft_score
        
        return combined_score, lap_score, fft_score
    
    def is_blurry(self, image: np.ndarray) -> Tuple[bool, float]:
        """Determine if an image is blurry using combined methods.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (is_blurry, blur_score)
        """
        combined_score, lap_score, fft_score = self.compute_blur_score(image)
        
        # Image is blurry if either method says it's blurry
        laplacian_blurry = lap_score < self.laplacian_threshold
        fft_blurry = fft_score < self.fft_threshold
        is_blurry = laplacian_blurry or fft_blurry
        
        return is_blurry, combined_score
    
    def set_threshold(self, laplacian_threshold: float = None, fft_threshold: float = None, fft_size: int = None):
        """Update the blur detection thresholds.
        
        Args:
            laplacian_threshold: New Laplacian threshold value
            fft_threshold: New FFT threshold value
            fft_size: New FFT region size
        """
        if laplacian_threshold is not None:
            self.laplacian_threshold = laplacian_threshold
            
        if fft_threshold is not None:
            self.fft_threshold = fft_threshold
            
        if fft_size is not None:
            self.fft_size = fft_size
        
    def get_threshold(self) -> float:
        """Get the current blur threshold (Laplacian threshold for backward compatibility).
        
        Returns:
            Current Laplacian threshold value
        """
        return self.laplacian_threshold
    
    def analyze_image_set(self, images: list) -> dict:
        """Analyze a set of images and provide statistics about blur scores.
        
        Args:
            images: List of images to analyze
            
        Returns:
            Dictionary with blur statistics: min, max, mean, median
        """
        if not images:
            return {
                'min_score': 0,
                'max_score': 0,
                'mean_score': 0,
                'median_score': 0,
                'blurry_percentage': 0,
                'sharp_percentage': 0,
                'laplacian_stats': {},
                'fft_stats': {}
            }
            
        # Compute scores for all images
        combined_scores = []
        lap_scores = []
        fft_scores = []
        
        for img in images:
            combined, lap, fft = self.compute_blur_score(img)
            combined_scores.append(combined)
            lap_scores.append(lap)
            fft_scores.append(fft)
        
        # Count blurry images
        blurry_count = sum(1 for i, img in enumerate(images) 
                          if lap_scores[i] < self.laplacian_threshold or fft_scores[i] < self.fft_threshold)
        
        return {
            'min_score': min(combined_scores),
            'max_score': max(combined_scores),
            'mean_score': sum(combined_scores) / len(combined_scores),
            'median_score': sorted(combined_scores)[len(combined_scores) // 2],
            'blurry_percentage': (blurry_count / len(images)) * 100,
            'sharp_percentage': ((len(images) - blurry_count) / len(images)) * 100,
            'laplacian_stats': {
                'min': min(lap_scores),
                'max': max(lap_scores),
                'mean': sum(lap_scores) / len(lap_scores)
            },
            'fft_stats': {
                'min': min(fft_scores),
                'max': max(fft_scores),
                'mean': sum(fft_scores) / len(fft_scores)
            }
        } 