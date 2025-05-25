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
import subprocess

from .frame_selector import FrameSelector
from .spherical_converter import SphericalConverter
from ..utils.cubemap_to_faces import CubemapToFaces


class VideoProcessor:
    """Main processing engine for video files."""

    def __init__(self, 
                 output_dir: str = "output",
                 enable_frame_skip: bool = False,
                 fps: int = 2,
                 enable_blur_detection: bool = False,
                 blur_threshold: float = 100.0,
                 enable_frame_selection: bool = False,
                 min_frame_gap: int = 5):
        """Initialize the VideoProcessor.
        
        Args:
            output_dir: Directory to save extracted frames
            enable_frame_skip: Whether to enable frame skipping
            fps: Frames per second to extract when frame skipping is enabled
            enable_blur_detection: Whether to enable blur detection
            blur_threshold: Threshold for blur detection (Laplacian)
            enable_frame_selection: Whether to enable frame selection
            min_frame_gap: Minimum gap between selected frames
        """
        self.output_dir = Path(output_dir)
        self.enable_frame_skip = enable_frame_skip
        self.fps = fps
        self.enable_blur_detection = enable_blur_detection
        self.blur_threshold = blur_threshold
        self.enable_frame_selection = enable_frame_selection
        self.min_frame_gap = min_frame_gap
        
        # Initialize selectors with parameters
        self.frame_selector = FrameSelector(
            blur_threshold=blur_threshold,
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
        
        # Ensure output directory exists with proper permissions
        self._ensure_output_directory()
    
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
    
    def _ensure_output_directory(self):
        """Ensure output directory exists with proper permissions."""
        try:
            # Create main output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set directory permissions to ensure application can read/write
            import stat
            current_permissions = self.output_dir.stat().st_mode
            desired_permissions = current_permissions | stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH
            
            # Apply permissions (equivalent to chmod 775)
            if current_permissions != desired_permissions:
                os.chmod(self.output_dir, desired_permissions)
                self.logger.info(f"Updated permissions for output directory: {self.output_dir}")
            
            # Verify we can write to the directory
            test_file = self.output_dir / ".write_test"
            try:
                with open(test_file, 'w') as f:
                    f.write("Test write access")
                test_file.unlink()  # Remove test file
                self.logger.info(f"Verified write access to output directory: {self.output_dir}")
            except Exception as e:
                self.logger.warning(f"Unable to write to output directory: {e}")
                
        except Exception as e:
            self.logger.error(f"Error ensuring output directory: {e}")
    
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
        
        # Create directories with proper permissions
        for dir_path in [video_output_dir, equirect_dir, cubemap_dir, faces_dir, selected_dir]:
            self._create_directory_with_permissions(dir_path)
        
        try:
            # Get video metadata first for smarter extraction
            self.logger.info("Fetching video metadata...")
            metadata = self.spherical_converter.get_video_metadata(str(video_path))
            
            # Auto-detect 360 if not specified
            if not is_360 and metadata.get('is_360', False):
                is_360 = True
                self.logger.info(f"Auto-detected 360° video: {video_path}")
                
            # Step 1: Extract frames
            self.logger.info("Extracting frames from video...")
            
            # Extract frames to equirect directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Calculate frames per second to extract
                source_fps = metadata.get('fps', 30.0)
                extract_fps = self.fps if self.enable_frame_skip else source_fps
                
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
                
                # Step 2: Score all frames
                self.logger.info("Scoring frames...")
                self.frame_selector.score_frames(frame_paths)
                
                # Apply filters based on enabled settings
                selected_frames = None
                if self.enable_blur_detection or self.enable_frame_selection:
                    # Determine which filters to apply
                    blur_threshold = self.blur_threshold if self.enable_blur_detection else None
                    min_frame_gap = self.min_frame_gap if self.enable_frame_selection else None
                    
                    self.logger.info(f"Applying filters: blur_threshold={blur_threshold}, min_frame_gap={min_frame_gap}")
                    
                    # Apply filters and get filtered frame paths
                    selected_frames = self.frame_selector.apply_filters(
                        blur_threshold=blur_threshold,
                        min_frame_gap=min_frame_gap
                    )
                    
                    # Log filtering results
                    self.logger.info(f"After filtering: {len(selected_frames)}/{len(frame_paths)} frames selected")
                    
                    # Track statistics
                    rejected_frames = len(frame_paths) - len(selected_frames)
                    if self.enable_blur_detection:
                        video_stats['blurry_frames'] = rejected_frames
                
                # If no filters applied, process all frames
                frame_paths_to_process = selected_frames if selected_frames is not None else frame_paths
                self.logger.info(f"Processing {len(frame_paths_to_process)} frames...")
                
                # Step 3: Process each selected frame
                processed_count = 0
                for i, frame_path in enumerate(frame_paths_to_process):
                    frame_base = Path(frame_path).stem
                    
                    # Log progress periodically
                    if (i + 1) % 10 == 0 or i == 0 or i == len(frame_paths_to_process) - 1:
                        self.logger.info(f"Processing frame {i+1}/{len(frame_paths_to_process)}")
                    
                    # Process 360° video if needed
                    if is_360:
                        self.logger.info(f"Converting 360° frame to cubemap faces: {frame_base}")
                        # First save as cubemap
                        cubemap_path = cubemap_dir / f"{frame_base}_cube.jpg"
                        
                        try:
                            # First check if the frame exists and can be read
                            frame = cv2.imread(str(frame_path))
                            if frame is None:
                                self.logger.error(f"Could not read frame at path: {frame_path}")
                                continue
                            
                            # Create cubemap using FFmpeg's v360 filter
                            cmd = [
                                self.spherical_converter.ffmpeg_path,
                                '-i', str(frame_path),
                                '-vf', 'v360=e:c6x1',
                                '-q:v', '1',
                                str(cubemap_path)
                            ]
                            
                            # Run FFmpeg command with detailed error capture
                            self.logger.info(f"Running FFmpeg to convert frame to cubemap: {' '.join(cmd)}")
                            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                            
                            # Verify cubemap was created
                            if not cubemap_path.exists():
                                self.logger.error(f"FFmpeg did not create cubemap file at: {cubemap_path}")
                                
                                # Try alternate approach - read frame and write directly as cubemap
                                self.logger.info(f"Trying alternate approach to create cubemap")
                                cubemap = self.spherical_converter.convert_frame(frame)[0]  # Get combined cubemap
                                cv2.imwrite(str(cubemap_path), cubemap)
                                
                                if not cubemap_path.exists():
                                    raise RuntimeError(f"Failed to create cubemap file after multiple attempts")
                            
                            # Verify the cubemap can be read
                            cubemap = cv2.imread(str(cubemap_path))
                            if cubemap is None:
                                self.logger.error(f"Created cubemap file but cannot read it: {cubemap_path}")
                                raise RuntimeError(f"Created cubemap file but cannot read it")
                            
                            # Get cubemap dimensions
                            height, width, _ = cubemap.shape
                            if width % 6 != 0:
                                self.logger.warning(f"Cubemap width {width} is not divisible by 6. This may cause issues in face extraction.")
                            
                            # Use CubemapToFaces utility to extract faces
                            cubemap_converter = CubemapToFaces()
                            
                            # Process single file instead of directory to avoid processing other files
                            self.logger.info(f"Extracting faces from cubemap: {cubemap_path}")
                            
                            # Process the single cubemap file - output directly to faces directory
                            faces_saved = cubemap_converter.process_file(
                                str(cubemap_path),
                                str(faces_dir)
                            )
                            
                            if faces_saved == 0:
                                self.logger.error(f"No faces were extracted from cubemap: {cubemap_path}")
                                
                                # Fallback: extract faces manually
                                self.logger.info(f"Trying manual face extraction as fallback")
                                faces_saved = self._manual_extract_faces(cubemap, frame_base, faces_dir)
                                
                                if faces_saved == 0:
                                    raise RuntimeError("No faces were extracted from the cubemap after multiple attempts")
                            
                            self.logger.info(f"Successfully extracted {faces_saved} faces from cubemap")
                            video_stats['saved_frames'] += faces_saved
                            processed_count += 1
                            
                        except subprocess.CalledProcessError as e:
                            self.logger.error(f"FFmpeg error processing cubemap: {e.stderr if hasattr(e, 'stderr') else str(e)}")
                            continue
                        except Exception as e:
                            self.logger.error(f"Error processing cubemap: {str(e)}")
                            continue
                    else:
                        # For non-360° videos, just copy the frame to selected directory
                        selected_path = selected_dir / f"{frame_base}.jpg"
                        shutil.copy2(frame_path, selected_path)
                        video_stats['saved_frames'] += 1
                        processed_count += 1
            
            # Update saved frames count in case of issues during processing
            self.logger.info(f"Final frame count check: {video_stats['saved_frames']} frames saved")
            if video_stats['saved_frames'] == 0 and len(frame_paths_to_process) > 0:
                self.logger.warning("No frames were saved despite having frames to process. Check filter settings and output paths.")
            
            self.logger.info(f"Video processing complete: {video_stats['saved_frames']} views saved")
            
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
        
        # Post-processing verification for 360° videos
        self._verify_and_repair_360_output()
        
        return self.stats

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.stats
        
    def update_parameters(self, 
                         enable_frame_skip: Optional[bool] = None,
                         fps: Optional[int] = None,
                         enable_blur_detection: Optional[bool] = None,
                         blur_threshold: Optional[float] = None,
                         enable_frame_selection: Optional[bool] = None,
                         min_frame_gap: Optional[int] = None):
        """Update processing parameters.
        
        Args:
            enable_frame_skip: Whether to enable frame skipping
            fps: Frames per second to extract
            enable_blur_detection: Whether to enable blur detection
            blur_threshold: New blur threshold value
            enable_frame_selection: Whether to enable frame selection
            min_frame_gap: New minimum frame gap value
        """
        if enable_frame_skip is not None:
            self.enable_frame_skip = enable_frame_skip
            
        if fps is not None:
            self.fps = fps
            
        if enable_blur_detection is not None:
            self.enable_blur_detection = enable_blur_detection
            
        if blur_threshold is not None:
            self.blur_threshold = blur_threshold
            self.frame_selector.set_parameters(blur_threshold=blur_threshold)
            
        if enable_frame_selection is not None:
            self.enable_frame_selection = enable_frame_selection
            
        if min_frame_gap is not None:
            self.min_frame_gap = min_frame_gap
            self.frame_selector.set_parameters(min_frame_gap=min_frame_gap) 

    def _manual_extract_faces(self, cubemap: np.ndarray, frame_base: str, output_dir: Path) -> int:
        """Manually extract faces from a cubemap when the utility fails.
        
        Args:
            cubemap: The cubemap image as a NumPy array
            frame_base: Base name for the output files
            output_dir: Directory to save the faces
            
        Returns:
            Number of faces successfully extracted
        """
        try:
            # Check cubemap dimensions
            height, width, _ = cubemap.shape
            if width % 6 != 0:
                self.logger.warning(f"Cubemap width {width} is not divisible by 6, attempting to fix")
                # Try to determine face width by dividing total width by 6
                face_width = width // 6
            else:
                face_width = width // 6
                
            self.logger.info(f"Manual extraction: Cubemap dimensions {width}x{height}, face width {face_width}")
            
            # Face names for consistent naming
            face_names = ["lateral_1", "lateral_2", "top", "bottom", "lateral_3", "lateral_4"]
            
            saved_faces = 0
            for i, face_name in enumerate(face_names):
                if i * face_width >= width:
                    self.logger.error(f"Manual extraction: Face index {i} exceeds cubemap width")
                    continue
                    
                # Extract face from cubemap
                x_start = i * face_width
                x_end = min((i + 1) * face_width, width)
                face = cubemap[:, x_start:x_end]
                
                # Skip empty or invalid faces
                if face.size == 0:
                    self.logger.warning(f"Manual extraction: Empty face at index {i}")
                    continue
                
                # Create output file path
                face_path = output_dir / f"{frame_base}_{face_name}.jpg"
                
                # Save face
                if cv2.imwrite(str(face_path), face):
                    saved_faces += 1
                    self.logger.debug(f"Manual extraction: Saved face {face_name} to {face_path}")
                else:
                    self.logger.warning(f"Manual extraction: Failed to save face {face_name} to {face_path}")
            
            return saved_faces
            
        except Exception as e:
            self.logger.error(f"Manual face extraction error: {str(e)}")
            return 0 

    def _create_directory_with_permissions(self, directory_path):
        """Create a directory with proper permissions.
        
        Args:
            directory_path: Path to the directory to create
        
        Returns:
            True if successful, False otherwise
        """
        try:
            dir_path = Path(directory_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Set proper permissions
            import stat
            permissions = stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH  # 0o775
            current_mode = dir_path.stat().st_mode
            
            # Only change if permissions are different
            if (current_mode & 0o777) != (permissions & 0o777):
                os.chmod(dir_path, permissions)
                self.logger.info(f"Set permissions for directory: {dir_path}")
            
            # Verify write access
            test_file = dir_path / ".write_test"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()
                self.logger.debug(f"Verified write access to: {dir_path}")
                return True
            except Exception as e:
                self.logger.warning(f"Cannot write to directory {dir_path}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating directory {directory_path}: {e}")
            return False 

    def _verify_and_repair_360_output(self):
        """Verify 360° output directories and repair any missing faces."""
        from src.utils.cubemap_to_faces import CubemapToFaces
        
        # Add repaired frames to statistics
        total_repaired_frames = 0
        
        # Check each video directory
        for video_path, stats in self.stats['per_video'].items():
            video_dir = self.output_dir / Path(video_path).stem
            
            # Skip if not a 360° video
            faces_dir = video_dir / "faces"
            cubemap_dir = video_dir / "cubemap"
            equirect_dir = video_dir / "equirect"
            
            # Check if this is a 360° video directory
            if not (cubemap_dir.exists() and equirect_dir.exists()):
                continue
                
            self.logger.info(f"Verifying 360° output for {video_dir.name}")
            
            # Find all equirectangular frames
            equirect_frames = sorted(list(equirect_dir.glob('*.jpg')))
            
            # Find existing faces
            existing_faces = set()
            if faces_dir.exists():
                for face_path in faces_dir.glob('*.jpg'):
                    # Extract the base frame name (without face suffix)
                    parts = face_path.stem.split('_')
                    if len(parts) >= 2 and parts[-1] in ["lateral_1", "lateral_2", "top", "bottom", "lateral_3", "lateral_4"]:
                        # Join all parts except the last one to get the frame name
                        frame_name = '_'.join(parts[:-1])
                        existing_faces.add(frame_name)
            
            # Count frames that need repair
            frames_needing_repair = []
            for frame_path in equirect_frames:
                frame_base = frame_path.stem
                if frame_base not in existing_faces:
                    frames_needing_repair.append(frame_path)
            
            if not frames_needing_repair:
                self.logger.info(f"All frames complete for {video_dir.name}")
                continue
                
            self.logger.info(f"Found {len(frames_needing_repair)} frames missing faces in {video_dir.name}")
            
            # Create faces directory if it doesn't exist
            faces_dir.mkdir(exist_ok=True)
            
            # Create converter
            cubemap_converter = CubemapToFaces()
            
            # Process each frame that doesn't have faces
            fixed_frames = 0
            for frame_path in frames_needing_repair:
                frame_base = frame_path.stem
                cubemap_path = cubemap_dir / f"{frame_base}_cube.jpg"
                
                # Skip if cubemap doesn't exist
                if not cubemap_path.exists():
                    self.logger.warning(f"Cubemap missing for {frame_base}, skipping")
                    continue
                
                # Extract faces from the cubemap
                self.logger.info(f"Extracting faces from cubemap: {cubemap_path.name}")
                try:
                    faces_saved = cubemap_converter.process_file(str(cubemap_path), str(faces_dir))
                    if faces_saved > 0:
                        self.logger.info(f"Extracted {faces_saved} faces from {cubemap_path.name}")
                        fixed_frames += 1
                        # Update saved frames in video stats
                        self.stats['per_video'][video_path]['saved_frames'] += faces_saved
                    else:
                        self.logger.warning(f"No faces were extracted from {cubemap_path.name}")
                except Exception as e:
                    self.logger.error(f"Error extracting faces from {cubemap_path.name}: {e}")
            
            if fixed_frames > 0:
                self.logger.info(f"Repaired {fixed_frames} frames in {video_dir.name}")
                total_repaired_frames += fixed_frames
            
        # Update global stats
        if total_repaired_frames > 0:
            self.stats['repaired_frames'] = total_repaired_frames
            # Also update the global saved_frames count
            self.stats['saved_frames'] = sum(
                stats.get('saved_frames', 0) 
                for stats in self.stats['per_video'].values()
            )
            
        # Update permissions on output directory
        self._ensure_output_directory() 