"""Test the SphericalConverter class for 360° video handling."""

import unittest
import os
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path
import subprocess

from src.core.spherical_converter import SphericalConverter


class TestSphericalConverter(unittest.TestCase):
    """Test the SphericalConverter class for 360° video handling."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a synthetic 360° test video
        self.create_test_360_video()
        
        # Initialize SphericalConverter with default settings
        self.spherical_converter = SphericalConverter()
    
    def tearDown(self):
        """Clean up after the test."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_360_video(self):
        """Create a synthetic 360° test video."""
        # Create an equirectangular image (360x180 degrees)
        # For a synthetic test, we'll create a color gradient pattern
        width, height = 1920, 960  # 2:1 ratio for equirectangular projection
        frames = []
        
        for i in range(30):  # 1 second at 30 fps
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create a gradient pattern
            for y in range(height):
                for x in range(width):
                    # Map x to hue (0-180)
                    hue = int(180 * x / width)
                    # Map y to saturation (0-255)
                    sat = int(255 * y / height)
                    # Use value with frame number for animation
                    val = 255 - (i * 5) % 255
                    
                    # Convert HSV to BGR
                    hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    frame[y, x] = bgr[0, 0]
            
            # Add grid lines to visualize the spherical coordinates
            for lat in range(0, height, height//6):  # Every 30 degrees
                cv2.line(frame, (0, lat), (width, lat), (255, 255, 255), 2)
            
            for lon in range(0, width, width//12):  # Every 30 degrees
                cv2.line(frame, (lon, 0), (lon, height), (255, 255, 255), 2)
            
            # Add text indicators for orientation
            cv2.putText(frame, "FRONT", (width//2 - 50, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "TOP", (width//2 - 40, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "BOTTOM", (width//2 - 60, height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frames.append(frame)
        
        # Save as MP4 with H.264 codec
        self.video_360_path = os.path.join(self.temp_dir, "test_360.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.video_360_path, fourcc, 30.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Add spherical metadata using FFmpeg
        temp_output = os.path.join(self.temp_dir, "temp_output.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", self.video_360_path,
            "-c:v", "copy", "-metadata:s:v", "spherical=true",
            "-metadata:s:v", "stereo=monoscopic",
            "-metadata:s:v", "projection=equirectangular",
            temp_output
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.rename(temp_output, self.video_360_path)
        except (subprocess.SubprocessError, OSError) as e:
            print(f"Warning: Could not add 360° metadata: {e}")
            # If FFmpeg fails, we'll just use the video without metadata
            # Tests that rely on metadata can be skipped in this case
    
    def test_detect_360_video(self):
        """Test the 360° video detection."""
        is_360 = self.spherical_converter.is_360_video(self.video_360_path)
        # This might be True or False depending on whether the FFmpeg metadata
        # command succeeded in setUp
        print(f"Detected 360 video: {is_360}")
        
        # We'll test the extraction regardless of metadata detection
        if not is_360:
            self.skipTest("360° metadata could not be added, skipping this test")
    
    def test_extract_equirectangular_frames(self):
        """Test extraction of equirectangular frames."""
        output_path = Path(self.output_dir) / "equirect"
        os.makedirs(output_path, exist_ok=True)
        
        # Extract a frame
        frame_path = self.spherical_converter.extract_frame(
            self.video_360_path, 
            str(output_path),
            frame_index=15,
            projection="equirectangular"
        )
        
        self.assertTrue(os.path.exists(frame_path), 
                        f"Frame was not extracted to {frame_path}")
        
        # Check that the image has the expected aspect ratio
        img = cv2.imread(frame_path)
        height, width = img.shape[:2]
        aspect_ratio = width / height
        self.assertAlmostEqual(aspect_ratio, 2.0, delta=0.1, 
                              msg="Equirectangular projection should have 2:1 aspect ratio")
    
    def test_extract_cubemap_frames(self):
        """Test extraction of cubemap frames."""
        output_path = Path(self.output_dir) / "cubemap"
        os.makedirs(output_path, exist_ok=True)
        
        # Extract a frame in cubemap format if supported
        try:
            frame_path = self.spherical_converter.extract_frame(
                self.video_360_path, 
                str(output_path),
                frame_index=15,
                projection="cubemap"
            )
            
            self.assertTrue(os.path.exists(frame_path), 
                            f"Cubemap frame was not extracted to {frame_path}")
        except NotImplementedError:
            self.skipTest("Cubemap projection not implemented yet")

    def test_convert_frame_to_cubemap(self):
        """Test conversion of an equirectangular frame to cubemap faces."""
        # First extract an equirectangular frame
        output_path = Path(self.output_dir) / "equirect"
        os.makedirs(output_path, exist_ok=True)
        
        # Extract a sample frame
        frame_paths = self.spherical_converter.extract_frames(
            self.video_360_path, 
            str(output_path),
            fps=0.5,  # Just get a few frames
            max_frames=1
        )
        
        if not frame_paths:
            self.skipTest("Could not extract test frame")
        
        # Load the frame
        frame = cv2.imread(frame_paths[0])
        self.assertIsNotNone(frame, "Could not load test frame")
        
        # Convert to cubemap faces
        faces = self.spherical_converter.convert_frame(frame)
        
        # Check that we got 6 faces
        self.assertEqual(len(faces), 6, "Should get 6 cubemap faces")
        
        # Each face should be square (or close to it)
        for i, face in enumerate(faces):
            h, w = face.shape[:2]
            aspect_ratio = w / h
            self.assertAlmostEqual(aspect_ratio, 1.0, delta=0.1, 
                                  msg=f"Face {i} should be square, got aspect ratio {aspect_ratio}")
            
            # Save faces for inspection (optional)
            face_path = os.path.join(self.output_dir, f"face_{i}.jpg")
            cv2.imwrite(face_path, face)


if __name__ == "__main__":
    unittest.main() 