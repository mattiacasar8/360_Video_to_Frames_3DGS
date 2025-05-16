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
            
            # Convert to cubemap using FFmpeg's v360 filter
            cubemap_path = os.path.join(temp_dir, 'cubemap.jpg')
            
            cmd = [
                self.ffmpeg_path,
                '-i', input_path,
                '-vf', 'v360=e:c6x1',
                '-q:v', '1',
                cubemap_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Read the cubemap image
                cubemap = cv2.imread(cubemap_path)
                if cubemap is None:
                    print("Error: Failed to read cubemap image")
                    return [frame]  # Return original frame if failed
                
                # Split the cubemap into 6 individual faces
                height = cubemap.shape[0]
                face_width = cubemap.shape[1] // 6
                
                faces = []
                for i in range(6):
                    face = cubemap[:, i*face_width:(i+1)*face_width]
                    faces.append(face)
                
                # If successful, return all 6 faces
                return faces
                
            except subprocess.SubprocessError as e:
                print(f"Error converting to cubemap: {e}")
                return [frame]  # Return original frame if failed
    
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