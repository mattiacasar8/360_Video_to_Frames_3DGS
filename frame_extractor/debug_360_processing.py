#!/usr/bin/env python3
"""Debug utility for 360° video processing in Frame Extractor."""

import os
import sys
import argparse
import logging
import subprocess
import cv2
import numpy as np
from pathlib import Path

# Add the parent directory to the path to allow importing from src
parent_dir = Path(__file__).parent
sys.path.append(str(parent_dir))

from src.core.spherical_converter import SphericalConverter
from src.utils.cubemap_to_faces import CubemapToFaces


def setup_logging(verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('debug_360')


def check_ffmpeg_installation():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.split('\n')[0]
        else:
            return False, "FFmpeg is installed but returned an error"
    except FileNotFoundError:
        return False, "FFmpeg not found - please install FFmpeg"
    except Exception as e:
        return False, f"Error checking FFmpeg: {str(e)}"


def check_video_metadata(video_path, logger):
    """Check video metadata including 360° status."""
    try:
        converter = SphericalConverter()
        metadata = converter.get_video_metadata(video_path)
        
        if not metadata:
            return False, "Failed to retrieve metadata"
        
        is_360 = metadata.get('is_360', False)
        
        logger.info(f"Video metadata: {metadata}")
        logger.info(f"Is 360° video: {is_360}")
        
        return True, metadata
    except Exception as e:
        return False, f"Error checking video metadata: {str(e)}"


def extract_test_frame(video_path, output_dir, logger):
    """Extract a test frame from the video."""
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create frame path
        frame_path = output_path / "test_frame.jpg"
        
        # Use FFmpeg to extract a frame
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '1',
            str(frame_path)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        if not frame_path.exists():
            return False, "Failed to extract test frame"
        
        logger.info(f"Extracted test frame to {frame_path}")
        return True, str(frame_path)
    except Exception as e:
        return False, f"Error extracting test frame: {str(e)}"


def convert_to_cubemap(frame_path, output_dir, logger):
    """Convert a frame to cubemap format."""
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create cubemap path
        cubemap_path = output_path / "test_cubemap.jpg"
        
        # Use FFmpeg to convert to cubemap
        cmd = [
            'ffmpeg',
            '-i', frame_path,
            '-vf', 'v360=e:c6x1',
            '-q:v', '1',
            str(cubemap_path)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False, f"FFmpeg error: {result.stderr}"
        
        if not cubemap_path.exists():
            return False, "Failed to create cubemap"
        
        logger.info(f"Created cubemap at {cubemap_path}")
        return True, str(cubemap_path)
    except Exception as e:
        return False, f"Error converting to cubemap: {str(e)}"


def extract_faces(cubemap_path, output_dir, logger):
    """Extract faces from a cubemap."""
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use CubemapToFaces to extract faces
        converter = CubemapToFaces(debug_mode=True)
        
        # Validate cubemap first
        is_valid, message = converter.validate_cubemap(cubemap_path)
        if not is_valid:
            logger.error(f"Invalid cubemap: {message}")
            
            # Try to fix the cubemap if possible
            logger.info("Attempting to fix cubemap...")
            frame = cv2.imread(cubemap_path)
            if frame is None:
                return False, "Could not read cubemap file"
            
            # Ensure width is divisible by 6
            height, width, _ = frame.shape
            if width % 6 != 0:
                new_width = (width // 6) * 6
                frame = cv2.resize(frame, (new_width, height))
                fixed_path = output_path / "fixed_cubemap.jpg"
                cv2.imwrite(str(fixed_path), frame)
                logger.info(f"Created fixed cubemap at {fixed_path}")
                cubemap_path = str(fixed_path)
        
        # Now extract faces
        faces_saved = converter.process_file(cubemap_path, str(output_path))
        
        if faces_saved == 0:
            return False, "Failed to extract faces"
        
        logger.info(f"Extracted {faces_saved} faces to {output_path}")
        return True, f"Extracted {faces_saved} faces"
    except Exception as e:
        return False, f"Error extracting faces: {str(e)}"


def check_directory_permissions(directory, logger):
    """Check directory permissions."""
    try:
        dir_path = Path(directory)
        
        # Check if directory exists
        if not dir_path.exists():
            return False, "Directory does not exist"
        
        # Check if it's a directory
        if not dir_path.is_dir():
            return False, "Not a directory"
        
        # Check read permission
        readable = os.access(dir_path, os.R_OK)
        
        # Check write permission
        writable = os.access(dir_path, os.W_OK)
        
        # Check execute permission
        executable = os.access(dir_path, os.X_OK)
        
        # Get permissions as string
        try:
            permissions = oct(os.stat(dir_path).st_mode)[-3:]
        except:
            permissions = "unknown"
        
        # Get owner
        try:
            owner = os.stat(dir_path).st_uid
        except:
            owner = "unknown"
        
        logger.info(f"Directory: {dir_path}")
        logger.info(f"Readable: {readable}")
        logger.info(f"Writable: {writable}")
        logger.info(f"Executable: {executable}")
        logger.info(f"Permissions: {permissions}")
        logger.info(f"Owner ID: {owner}")
        
        # Create test file
        test_file = dir_path / "test_perm.txt"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            logger.info("Successfully created test file")
            os.remove(test_file)
            logger.info("Successfully removed test file")
        except Exception as e:
            return False, f"Failed to write test file: {str(e)}"
        
        if readable and writable and executable:
            return True, "Directory has appropriate permissions"
        else:
            return False, f"Directory permissions issue: read={readable}, write={writable}, execute={executable}"
    except Exception as e:
        return False, f"Error checking directory permissions: {str(e)}"


def manual_test_extract(video_path, output_dir, logger):
    """Test the full extraction process manually."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Extract a frame
        success, frame_path = extract_test_frame(video_path, output_dir, logger)
        if not success:
            return False, frame_path  # frame_path contains the error message
        
        # 2. Convert to cubemap
        success, cubemap_path = convert_to_cubemap(frame_path, output_dir, logger)
        if not success:
            return False, cubemap_path  # cubemap_path contains the error message
        
        # 3. Extract faces
        success, message = extract_faces(cubemap_path, output_dir, logger)
        if not success:
            return False, message
        
        # 4. Check output directory permissions
        success, message = check_directory_permissions(output_dir, logger)
        if not success:
            return False, message
        
        return True, "Full test completed successfully"
    except Exception as e:
        return False, f"Error during manual test: {str(e)}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Debug 360° video processing')
    parser.add_argument('--video', help='Path to the 360° video file')
    parser.add_argument('--output-dir', help='Path to the output directory', default='debug_output')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Check if FFmpeg is installed
    logger.info("Checking FFmpeg installation...")
    success, message = check_ffmpeg_installation()
    if not success:
        logger.error(message)
        return 1
    logger.info(f"FFmpeg: {message}")
    
    # If video path is provided, check metadata
    if args.video:
        if not Path(args.video).exists():
            logger.error(f"Video file does not exist: {args.video}")
            return 1
        
        logger.info(f"Checking video metadata: {args.video}")
        success, metadata = check_video_metadata(args.video, logger)
        if not success:
            logger.error(f"Metadata error: {metadata}")
        
        # Run full manual test
        logger.info(f"Running full manual test on {args.video}")
        success, message = manual_test_extract(args.video, args.output_dir, logger)
        if not success:
            logger.error(f"Manual test failed: {message}")
            return 1
        logger.info(message)
    else:
        # If no video, just do a general environment check
        logger.info("No video specified, will only check environment")
        
        # Check output directory permissions
        logger.info(f"Checking output directory permissions: {args.output_dir}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        success, message = check_directory_permissions(args.output_dir, logger)
        if not success:
            logger.error(f"Directory permission issue: {message}")
            return 1
        logger.info(message)
    
    logger.info("Debug complete - check the log for details")
    return 0


if __name__ == "__main__":
    exit(main()) 