"""Video processing functionality for the Frame Extractor application."""

import os
import logging
import tempfile
import shutil
import glob
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import ffmpeg
import platform

from .frame_selector import FrameSelector
from .spherical_converter import SphericalConverter


class VideoProcessor:
    """Main processing engine for video files."""

    def __init__(self, 
                 output_dir: str = "output",
                 blur_threshold: float = 100.0,
                 max_frames: int = 20,
                 min_frame_gap: int = 5,
                 frame_interval: int = 15):
        """Initialize the VideoProcessor.
        
        Args:
            output_dir: Directory to save extracted frames
            blur_threshold: Threshold for blur detection (Laplacian)
            max_frames: Maximum number of frames to extract per video
            min_frame_gap: Minimum gap between selected frames
            frame_interval: Number of frames to skip between extractions
        """
        self.output_dir = Path(output_dir)
        self.blur_threshold = blur_threshold
        self.max_frames = max_frames
        self.min_frame_gap = min_frame_gap
        self.frame_interval = frame_interval
        
        # Initialize selectors with parameters
        self.frame_selector = FrameSelector(
            blur_threshold=blur_threshold,
            max_frames=max_frames,
            min_frame_gap=min_frame_gap
        )
        
        self.spherical_converter = SphericalConverter()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('VideoProcessor')
        
        # Check for OpenCV GPU acceleration
        self._setup_gpu_acceleration()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'blurry_frames': 0,
            'selected_frames': 0,
            'saved_frames': 0
        }
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_gpu_acceleration(self):
        """Set up GPU acceleration for OpenCV if available."""
        # Check if OpenCL is available (Metal for Apple Silicon)
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            self.logger.info("OpenCL acceleration is available")
            # For macOS with Apple Silicon, set Metal device
            if platform.system() == 'Darwin':
                if platform.processor() == 'arm' or platform.machine() == 'arm64':
                    # Apple Silicon specific optimizations
                    self.logger.info("Apple Silicon detected, optimizing for Metal")
    
    def process_video(self, video_path: str, is_360: bool = False) -> Dict[str, Any]:
        """Process a single video file with improved 360° handling.
        
        Args:
            video_path: Path to the video file
            is_360: Whether the video is 360° format
            
        Returns:
            Dictionary containing processing statistics
        """
        video_path = Path(video_path)
        self.logger.info(f"Processing video: {video_path}")
        
        # Reset per-video statistics
        video_stats = {
            'total_frames': 0,
            'blurry_frames': 0,
            'selected_frames': 0,
            'saved_frames': 0
        }
        
        # Create output directories for this video
        video_output_dir = self.output_dir / video_path.stem
        
        equirect_dir = video_output_dir / "equirect"
        cubemap_dir = video_output_dir / "cubemap"
        faces_dir = video_output_dir / "faces" 
        selected_dir = video_output_dir / "selected"
        
        for dir_path in [video_output_dir, equirect_dir, cubemap_dir, faces_dir, selected_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get video metadata first for smarter extraction
            self.logger.info("Fetching video metadata...")
            metadata = self.spherical_converter.get_video_metadata(str(video_path))
            
            # Auto-detect 360 if not specified
            if not is_360 and metadata.get('is_360', False):
                is_360 = True
                self.logger.info(f"Auto-detected 360° video: {video_path}")
                
            # Step 1: Extract all frames initially
            self.logger.info(f"Extracting frames from video at 1/{self.frame_interval} FPS...")
            
            # Extract frames to equirect directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Calculate frames per second to extract
                source_fps = metadata.get('fps', 30.0)
                extract_fps = source_fps / self.frame_interval
                
                # Extract frames
                frame_paths = self.spherical_converter.extract_frames(
                    str(video_path),
                    str(equirect_dir),
                    fps=extract_fps, 
                    is_360=is_360,
                    max_frames=None  # Extract all frames first
                )
                
                video_stats['total_frames'] = len(frame_paths)
                self.logger.info(f"Extracted {len(frame_paths)} frames")
                
                if len(frame_paths) == 0:
                    self.logger.warning("No frames were extracted from the video")
                    return video_stats
                
                # Step 2: Analyze all frames for blur and select the best ones
                self.logger.info("Analyzing frames for sharpness...")
                frame_scores = self.frame_selector.analyze_frame_files(frame_paths)
                
                # Step 3: Select the best frames with distribution constraints
                selected_frames = self.frame_selector.select_best_frames(frame_scores)
                video_stats['selected_frames'] = len(selected_frames)
                self.logger.info(f"Selected {len(selected_frames)} frames out of {len(frame_paths)}")
                
                # Step 4: Process only the selected frames
                self.logger.info("Processing selected frames...")
                for i, frame_path in enumerate(selected_frames):
                    self.logger.info(f"Processing selected frame {i+1}/{len(selected_frames)}: {os.path.basename(frame_path)}")
                    
                    # Read the original frame
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        self.logger.warning(f"Could not read frame at path {frame_path}")
                        continue
                    
                    # Copy to selected directory
                    frame_filename = Path(frame_path).name
                    frame_base = os.path.splitext(frame_filename)[0]
                    selected_path = selected_dir / frame_filename
                    cv2.imwrite(str(selected_path), frame)
                    
                    # Process 360° video if needed
                    if is_360:
                        self.logger.info(f"Converting 360° frame to cubemap faces...")
                        # First save as cubemap
                        cubemap_path = cubemap_dir / f"{frame_base}_cube.jpg"
                        
                        # Create cubemap using FFmpeg's v360 filter
                        cmd = [
                            self.spherical_converter.ffmpeg_path,
                            '-i', str(frame_path),
                            '-vf', 'v360=e:c6x1',
                            '-q:v', '1',
                            str(cubemap_path)
                        ]
                        
                        try:
                            import subprocess
                            subprocess.run(cmd, check=True, capture_output=True)
                            
                            # Read the cubemap image
                            cubemap = cv2.imread(str(cubemap_path))
                            if cubemap is None:
                                self.logger.warning(f"Failed to read cubemap image: {cubemap_path}")
                                continue
                            
                            # Split the cubemap into 6 individual faces
                            height = cubemap.shape[0]
                            face_width = cubemap.shape[1] // 6
                            
                            # Extract and save individual faces
                            # Fix face naming based on actual positions in the cubemap
                            face_names = ["lateral_1", "lateral_2", "top", "bottom", "lateral_3", "lateral_4"]
                            for j, face_name in enumerate(face_names):
                                face = cubemap[:, j*face_width:(j+1)*face_width]
                                face_path = faces_dir / f"{frame_base}_{face_name}.jpg"
                                cv2.imwrite(str(face_path), face)
                                video_stats['saved_frames'] += 1
                                
                        except Exception as e:
                            self.logger.error(f"Error processing cubemap for frame {frame_path}: {e}")
                    else:
                        # For regular videos, just save the frame
                        video_stats['saved_frames'] += 1
                
                # Create a report file
                with open(str(video_output_dir / "report.txt"), "w") as f:
                    f.write(f"Video: {video_path}\n")
                    f.write(f"Total frames analyzed: {video_stats['total_frames']}\n")
                    f.write(f"Blurry frames: {video_stats['blurry_frames']}\n")
                    f.write(f"Frames selected: {video_stats['selected_frames']}\n")
                    f.write(f"Total perspective views saved: {video_stats['saved_frames']}\n\n")
                    
                    f.write("Selected frames:\n")
                    for i, frame in enumerate(selected_frames):
                        frame_name = os.path.basename(frame)
                        f.write(f"{i+1}. {frame_name} - Sharpness score: {frame_scores[frame]:.2f}\n")
            
            # Update global statistics
            self.stats['total_frames'] += video_stats['total_frames']
            self.stats['blurry_frames'] += video_stats['blurry_frames']
            self.stats['selected_frames'] += video_stats['selected_frames']
            self.stats['saved_frames'] += video_stats['saved_frames']
            
            self.logger.info(f"Video processing complete: {video_stats['saved_frames']} views saved from {video_stats['selected_frames']} selected frames")
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
        
        return video_stats
    
    def process_videos(self, video_paths: List[str], is_360_list: Optional[List[bool]] = None) -> Dict[str, Any]:
        """Process multiple videos.
        
        Args:
            video_paths: List of paths to video files
            is_360_list: List indicating whether each video is 360° (if None, all are assumed not 360°)
            
        Returns:
            Dictionary containing overall processing statistics
        """
        # Reset overall statistics
        self.stats = {
            'total_frames': 0,
            'blurry_frames': 0,
            'selected_frames': 0,
            'saved_frames': 0,
            'videos_processed': 0,
            'videos_failed': 0,
            'per_video': {}
        }
        
        if is_360_list is None:
            is_360_list = [False] * len(video_paths)
        
        # Process each video
        for i, (video_path, is_360) in enumerate(zip(video_paths, is_360_list)):
            self.logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            try:
                video_stats = self.process_video(video_path, is_360)
                self.stats['per_video'][video_path] = video_stats
                self.stats['videos_processed'] += 1
            except Exception as e:
                self.logger.error(f"Failed to process video {video_path}: {e}")
                self.stats['videos_failed'] += 1
        
        return self.stats

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.stats
        
    def update_parameters(self, 
                         blur_threshold: Optional[float] = None,
                         max_frames: Optional[int] = None,
                         min_frame_gap: Optional[int] = None,
                         frame_interval: Optional[int] = None):
        """Update processing parameters.
        
        Args:
            blur_threshold: New blur threshold value
            max_frames: New maximum frames value
            min_frame_gap: New minimum frame gap value
            frame_interval: New frame interval value
        """
        if blur_threshold is not None:
            self.blur_threshold = blur_threshold
            self.frame_selector.set_parameters(blur_threshold=blur_threshold)
            
        if max_frames is not None:
            self.max_frames = max_frames
            self.frame_selector.set_parameters(max_frames=max_frames)
            
        if min_frame_gap is not None:
            self.min_frame_gap = min_frame_gap
            self.frame_selector.set_parameters(min_frame_gap=min_frame_gap)
            
        if frame_interval is not None:
            self.frame_interval = frame_interval 