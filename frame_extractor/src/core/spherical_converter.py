"""Spherical video conversion functionality for the Frame Extractor application."""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import subprocess
import json
import os
import platform
import tempfile
from pathlib import Path
import logging


class SphericalConverter:
    """Handles conversion of 360° video frames to equirectangular format with hardware acceleration."""
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """Initialize the SphericalConverter.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable (if None, assumes in PATH)
        """
        self.ffmpeg_path = ffmpeg_path or 'ffmpeg'
        # Check for hardware acceleration support
        self.hw_accel = self._detect_hw_acceleration()
        
        # Set up logging
        self.logger = logging.getLogger('SphericalConverter')
    
    def _detect_hw_acceleration(self) -> str:
        """Detect available hardware acceleration for the current platform.
        
        Returns:
            String indicating the hardware acceleration option for FFmpeg
        """
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # Check for Apple Silicon
            processor = platform.processor()
            if processor == 'arm' or platform.machine() == 'arm64':
                return 'videotoolbox'  # Metal hardware acceleration
            else:
                return 'videotoolbox'  # Still use videotoolbox for Intel Macs
        elif system == 'Windows':
            return 'cuda'  # NVIDIA CUDA
        elif system == 'Linux':
            # Could be more complex on Linux, might need to detect NVIDIA/AMD/Intel
            return 'vaapi'  # Video Acceleration API (common on Linux)
        
        # Default to none if no specific acceleration detected
        return 'none'
    
    def is_360_video(self, video_path: str) -> bool:
        """Detect if a video is in 360° format by checking metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if video is 360°, False otherwise
        """
        try:
            # Use FFprobe to get video metadata in JSON format
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)
            
            # Check for 360 video indicators in metadata
            for stream in metadata.get('streams', []):
                if stream.get('codec_type') == 'video':
                    # Check for spherical video tags
                    tags = stream.get('tags', {})
                    if tags.get('spherical') == '1' or \
                       tags.get('stereo_mode') == '1' or \
                       tags.get('projection_type') == 'equirectangular':
                        return True
            
            return False
            
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError) as e:
            print(f"Error checking 360 video: {e}")
            return False
    
    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get metadata about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            # Use FFprobe to get video metadata in JSON format
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)
            
            # Extract relevant information
            video_info = {}
            
            for stream in metadata.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_info['width'] = stream.get('width')
                    video_info['height'] = stream.get('height')
                    video_info['codec'] = stream.get('codec_name')
                    video_info['is_360'] = self.is_360_video(video_path)
                    video_info['fps'] = self._parse_frame_rate(stream.get('r_frame_rate', '30/1'))
                    
                    # Get projection type if available
                    tags = stream.get('tags', {})
                    video_info['projection'] = tags.get('projection_type', 'unknown')
                    break
                    
            return video_info
                    
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError) as e:
            print(f"Error getting video metadata: {e}")
            return {}
    
    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """Parse frame rate string (often in fraction format).
        
        Args:
            frame_rate_str: Frame rate string, often in format like "30/1"
            
        Returns:
            Frame rate as float
        """
        try:
            if '/' in frame_rate_str:
                num, den = map(int, frame_rate_str.split('/'))
                return num / den
            else:
                return float(frame_rate_str)
        except (ValueError, ZeroDivisionError):
            return 30.0  # Default to 30fps in case of parsing error
    
    def extract_frames(self, video_path: str, output_dir: str, fps: float = 1.0, 
                      is_360: bool = False, max_frames: Optional[int] = None) -> List[str]:
        """Extract frames from video with hardware acceleration.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract
            is_360: Whether the video is 360°
            max_frames: Maximum number of frames to extract (None = no limit)
            
        Returns:
            List of paths to extracted frames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Base command with hardware acceleration if available
        cmd = [self.ffmpeg_path]
        
        # Add hardware acceleration if supported
        if self.hw_accel != 'none':
            cmd.extend(['-hwaccel', self.hw_accel])
        
        # Input file
        cmd.extend(['-i', video_path])
        
        # Apply frame rate filter
        filter_complex = f'fps={fps}'
        
        # If max frames is specified, add a trim filter
        if max_frames is not None:
            duration = max_frames / fps
            filter_complex = f'trim=duration={duration},{filter_complex}'
        
        # Set filters
        cmd.extend(['-vf', filter_complex])
        
        # High quality output
        cmd.extend(['-q:v', '1'])
        
        # Output pattern
        output_pattern = os.path.join(output_dir, 'frame_%04d.jpg')
        cmd.append(output_pattern)
        
        # Run FFmpeg
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Get list of extracted files
            frame_files = sorted([
                os.path.join(output_dir, f) for f in os.listdir(output_dir)
                if f.startswith('frame_') and f.endswith('.jpg')
            ])
            
            return frame_files
            
        except subprocess.SubprocessError as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def convert_frame(self, frame: np.ndarray, projection_type: str = 'equirectangular') -> List[np.ndarray]:
        """Convert a 360° video frame to multiple perspective views using cubemap projection.
        
        Converts an equirectangular frame to 6 individual cubemap faces for better 3D reconstruction.
        
        Args:
            frame: Input equirectangular frame (BGR format)
            projection_type: Type of projection to use
            
        Returns:
            List of converted frames (6 cubemap faces)
        """
        # Create a temporary directory to store the intermediate images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input frame as an image
            input_path = os.path.join(temp_dir, 'input_frame.jpg')
            cv2.imwrite(input_path, frame)
            
            # Ensure input image exists and is readable
            if not os.path.exists(input_path):
                self.logger.error("Failed to save input frame for conversion")
                return [frame]  # Return original frame if failed
                
            # Convert to cubemap using FFmpeg's v360 filter
            cubemap_path = os.path.join(temp_dir, 'cubemap.jpg')
            
            # First attempt with standard command
            success = self._try_convert_to_cubemap(input_path, cubemap_path)
            
            # If failed, try with force dimensions that are known to work well
            if not success:
                self.logger.warning("Standard conversion failed, trying with adjusted dimensions")
                # Resize the input image to dimensions that work well with FFmpeg's v360 filter
                try:
                    # Try a dimension that's divisible by common factors
                    resized_path = os.path.join(temp_dir, 'resized_input.jpg')
                    img = cv2.imread(input_path)
                    # Make sure height and width are even numbers
                    h, w = img.shape[:2]
                    new_w = w if w % 2 == 0 else w - 1
                    new_h = h if h % 2 == 0 else h - 1
                    if new_w != w or new_h != h:
                        img = cv2.resize(img, (new_w, new_h))
                        cv2.imwrite(resized_path, img)
                        self.logger.info(f"Resized input from {w}x{h} to {new_w}x{new_h}")
                        success = self._try_convert_to_cubemap(resized_path, cubemap_path)
                except Exception as e:
                    self.logger.error(f"Error during resize attempt: {e}")
                    success = False
            
            # If all FFmpeg approaches failed, try direct OpenCV conversion
            if not success:
                try:
                    self.logger.warning("FFmpeg conversion failed, trying OpenCV direct conversion")
                    # Try manual conversion using OpenCV
                    img = cv2.imread(input_path)
                    if img is None:
                        return [frame]
                    
                    # Get dimensions
                    h, w = img.shape[:2]
                    
                    # Create a 6-face cubemap (simple method - just divide the image)
                    # This is a basic fallback - not as accurate as FFmpeg's v360 filter
                    face_size = min(h // 3, w // 4)  # Make square faces
                    
                    # Create a new image to hold the cubemap (1 row, 6 columns)
                    cubemap = np.zeros((face_size, face_size * 6, 3), dtype=np.uint8)
                    
                    # Fill with samples from the source image (basic sampling)
                    for i in range(6):
                        x_start = (i * w) // 6
                        x_end = ((i + 1) * w) // 6
                        y_start = 0
                        y_end = h
                        
                        # Extract a region and resize to face size
                        region = img[y_start:y_end, x_start:x_end]
                        face = cv2.resize(region, (face_size, face_size))
                        
                        # Add to cubemap
                        cubemap[:, i*face_size:(i+1)*face_size] = face
                    
                    # Save the resulting cubemap
                    cv2.imwrite(cubemap_path, cubemap)
                    success = os.path.exists(cubemap_path)
                    
                    if success:
                        self.logger.info("Successfully created cubemap using OpenCV direct method")
                except Exception as e:
                    self.logger.error(f"OpenCV direct conversion failed: {e}")
                    return [frame]  # Return original frame if all methods failed
            
            # Read the cubemap image if it was successfully created
            if success and os.path.exists(cubemap_path):
                cubemap = cv2.imread(cubemap_path)
                if cubemap is None:
                    self.logger.error("Error: Failed to read cubemap image")
                    return [frame]  # Return original frame if failed
                
                # Ensure cubemap width is divisible by 6
                height, width, _ = cubemap.shape
                if width % 6 != 0:
                    self.logger.warning(f"Cubemap width {width} is not divisible by 6, adjusting")
                    # Adjust width to be divisible by 6
                    new_width = (width // 6) * 6
                    cubemap = cv2.resize(cubemap, (new_width, height))
                    # Re-save the adjusted cubemap
                    cv2.imwrite(cubemap_path, cubemap)
                    width = new_width
                
                # Split the cubemap into 6 individual faces
                face_width = width // 6
                
                faces = []
                for i in range(6):
                    face = cubemap[:, i*face_width:(i+1)*face_width]
                    faces.append(face)
                
                # If successful, return all 6 faces
                if len(faces) == 6:
                    return faces
            
            # If we got here, something went wrong
            self.logger.error("Cubemap conversion failed with all methods")
            return [frame]  # Return original frame as fallback
    
    def _try_convert_to_cubemap(self, input_path: str, output_path: str) -> bool:
        """Try to convert an equirectangular image to cubemap using FFmpeg.
        
        Args:
            input_path: Path to input image
            output_path: Path to save cubemap image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', input_path,
                '-vf', 'v360=e:c6x1',
                '-q:v', '1',
                output_path
            ]
            
            # Run FFmpeg
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Check if file was created
            if os.path.exists(output_path):
                # Verify the cubemap can be read
                img = cv2.imread(output_path)
                if img is not None:
                    return True
                else:
                    self.logger.error("Created cubemap file but cannot read it")
            else:
                self.logger.error("FFmpeg did not create cubemap file")
            
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error in cubemap conversion: {str(e)}")
            return False
    
    def process_360_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """Process a 360° frame and return multiple perspective views.
        
        This is a convenience method that calls convert_frame with appropriate settings.
        
        Args:
            frame: Input equirectangular frame (BGR format)
            
        Returns:
            List of processed frames
        """
        return self.convert_frame(frame, 'equirectangular')
    
    def convert_video(self, input_path: str, output_path: str, 
                    projection_type: str = 'equirectangular') -> bool:
        """Convert an entire 360° video to a different projection with hardware acceleration.
        
        This uses FFmpeg for the conversion.
        
        Args:
            input_path: Path to the input video file
            output_path: Path to save the converted video
            projection_type: Type of projection to convert to
            
        Returns:
            True if conversion succeeded, False otherwise
        """
        try:
            # Base command with hardware acceleration if available
            cmd = [self.ffmpeg_path]
            
            # Add hardware acceleration if supported
            if self.hw_accel != 'none':
                cmd.extend(['-hwaccel', self.hw_accel])
            
            # Input file
            cmd.extend(['-i', input_path])
            
            # Set video codec with hardware acceleration if available
            if self.hw_accel == 'videotoolbox':
                cmd.extend(['-c:v', 'h264_videotoolbox'])
            else:
                cmd.extend(['-c:v', 'libx264'])
            
            # Set quality and other parameters
            cmd.extend([
                '-preset', 'medium',
                '-crf', '23',
                '-metadata:s:v', 'spherical=true',
                '-metadata:s:v', f'projection={projection_type}'
            ])
            
            # Output file
            cmd.append(output_path)
            
            # Run FFmpeg
            subprocess.run(cmd, check=True)
            return os.path.exists(output_path)
            
        except subprocess.SubprocessError as e:
            print(f"Error converting video: {e}")
            return False
    
    def convert_frame_batch(self, frames: List[np.ndarray], 
                          projection_type: str = 'equirectangular') -> List[np.ndarray]:
        """Convert a batch of frames to the desired projection.
        
        Args:
            frames: List of input frames
            projection_type: Type of projection to convert to
            
        Returns:
            List of converted frames
        """
        # This is a convenience method for batch processing
        return [self.convert_frame(frame, projection_type) for frame in frames] 