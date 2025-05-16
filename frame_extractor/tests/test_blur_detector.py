"""Test the BlurDetector class."""

import unittest
import numpy as np
import cv2

from src.core.blur_detector import BlurDetector


class TestBlurDetector(unittest.TestCase):
    """Test the BlurDetector class."""
    
    def setUp(self):
        """Set up the test."""
        # Use higher thresholds for the test to ensure clear differentiation
        self.blur_detector = BlurDetector(laplacian_threshold=50.0, fft_threshold=5.0)
        
        # Create a sharp test image with high frequency components
        self.sharp_image = np.zeros((200, 200), dtype=np.uint8)
        # Create a checkerboard pattern (high frequency)
        for i in range(0, 200, 10):
            for j in range(0, 200, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    self.sharp_image[i:i+10, j:j+10] = 255
        
        # Create a blurry version of the test image
        self.blurry_image = cv2.GaussianBlur(self.sharp_image, (31, 31), 10)
        
        # Create a very blurry image - almost uniform
        self.very_blurry_image = cv2.GaussianBlur(self.sharp_image, (51, 51), 20)
    
    def test_compute_blur_score(self):
        """Test the compute_blur_score method."""
        # Sharp image should have a higher score
        combined_score, lap_score, fft_score = self.blur_detector.compute_blur_score(self.sharp_image)
        combined_blurry, lap_blurry, fft_blurry = self.blur_detector.compute_blur_score(self.blurry_image)
        
        # Both Laplacian and FFT scores should be higher for sharp images
        self.assertGreater(lap_score, lap_blurry)
        self.assertGreater(fft_score, fft_blurry)
    
    def test_is_blurry(self):
        """Test the is_blurry method."""
        # Test with very contrasting images to ensure clear results
        # Temporarily set very distinct thresholds
        self.blur_detector.set_threshold(laplacian_threshold=100.0, fft_threshold=10.0)
        
        # Sharp image should not be detected as blurry
        is_blurry_sharp, _ = self.blur_detector.is_blurry(self.sharp_image)
        
        # Very blurry image should be detected as blurry
        is_blurry_very_blurry, _ = self.blur_detector.is_blurry(self.very_blurry_image)
        
        self.assertFalse(is_blurry_sharp)
        self.assertTrue(is_blurry_very_blurry)
    
    def test_threshold_setting(self):
        """Test the threshold setting methods."""
        # Test setting thresholds
        new_laplacian = 200.0
        new_fft = 20.0
        
        self.blur_detector.set_threshold(laplacian_threshold=new_laplacian, fft_threshold=new_fft)
        self.assertEqual(self.blur_detector.laplacian_threshold, new_laplacian)
        self.assertEqual(self.blur_detector.fft_threshold, new_fft)
        
        # Test if changing thresholds affects classification
        # First set high thresholds (more lenient)
        self.blur_detector.set_threshold(laplacian_threshold=1.0, fft_threshold=1.0)
        is_blurry_low_threshold, _ = self.blur_detector.is_blurry(self.blurry_image)
        
        # Then set very high thresholds (more strict)
        self.blur_detector.set_threshold(laplacian_threshold=500.0, fft_threshold=50.0)
        is_blurry_high_threshold, _ = self.blur_detector.is_blurry(self.blurry_image)
        
        # With higher thresholds, more images should be classified as blurry
        self.assertNotEqual(is_blurry_low_threshold, is_blurry_high_threshold)


if __name__ == "__main__":
    unittest.main() 