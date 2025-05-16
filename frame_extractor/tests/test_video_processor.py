"""Test the VideoProcessor class with various video formats."""

import unittest
import os
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path

from src.core.video_processor import VideoProcessor


class TestVideoProcessor(unittest.TestCase):
    """Test the VideoProcessor class with various video formats."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a simple test video in different formats
        self.create_test_videos()
        
        # Initialize VideoProcessor with default settings
        self.video_processor = VideoProcessor(
            blur_threshold=100.0,
            similarity_threshold=0.9,
            extract_interval=1,
            max_frames=100
        )
    
    def tearDown(self):
        """Clean up after the test."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_videos(self):
        """Create test videos in different formats."""
        # Create a simple test frame sequence
        frames = []
        for i in range(30):  # 1 second at 30 fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some simple graphics
            cv2.rectangle(frame, (50, 50), (590, 430), (0, 255, 0), 2)
            cv2.putText(frame, f"Frame {i}", (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)
        
        # Save as MP4 (H.264)
        self.mp4_path = os.path.join(self.temp_dir, "test_h264.mp4")
        self.write_video(self.mp4_path, frames, "mp4v", (640, 480))
        
        # Save as AVI (MJPEG)
        self.avi_path = os.path.join(self.temp_dir, "test_mjpeg.avi")
        self.write_video(self.avi_path, frames, "MJPG", (640, 480))
    
    def write_video(self, path, frames, fourcc_str, size):
        """Write frames to a video file."""
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(path, fourcc, 30.0, size)
        for frame in frames:
            out.write(frame)
        out.release()
    
    def test_extract_frames_mp4(self):
        """Test frame extraction from MP4."""
        # Process the video
        output_subfolder = self.video_processor.process_video(
            self.mp4_path, self.output_dir, progress_callback=None
        )
        
        # Check if frames were extracted
        output_path = Path(self.output_dir) / output_subfolder
        extracted_frames = list(output_path.glob("*.jpg"))
        
        self.assertTrue(len(extracted_frames) > 0, 
                        f"No frames were extracted from MP4. Path: {output_path}")
    
    def test_extract_frames_avi(self):
        """Test frame extraction from AVI."""
        # Process the video
        output_subfolder = self.video_processor.process_video(
            self.avi_path, self.output_dir, progress_callback=None
        )
        
        # Check if frames were extracted
        output_path = Path(self.output_dir) / output_subfolder
        extracted_frames = list(output_path.glob("*.jpg"))
        
        self.assertTrue(len(extracted_frames) > 0, 
                        f"No frames were extracted from AVI. Path: {output_path}")
    
    def test_metadata_extraction(self):
        """Test metadata extraction."""
        # Get metadata from MP4
        metadata = self.video_processor.get_video_metadata(self.mp4_path)
        
        # Check basic metadata
        self.assertEqual(metadata["width"], 640)
        self.assertEqual(metadata["height"], 480)
        self.assertEqual(metadata["fps"], 30)
        self.assertGreaterEqual(metadata["frame_count"], 30)
        self.assertFalse(metadata["is_360"])


if __name__ == "__main__":
    unittest.main() 