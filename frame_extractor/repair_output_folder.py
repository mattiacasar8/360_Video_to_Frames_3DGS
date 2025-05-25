#!/usr/bin/env python3
"""Repair tool for Frame Extractor output directories."""

import os
import sys
import cv2
import logging
import argparse
import numpy as np
import shutil
from pathlib import Path
import subprocess
import tempfile

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the utility classes
from src.utils.cubemap_to_faces import CubemapToFaces
from src.core.spherical_converter import SphericalConverter


def setup_logging(verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('repair_tool')


def check_output_structure(directory, logger):
    """Check the output directory structure.
    
    Args:
        directory: Path to the output directory
        logger: Logger instance
        
    Returns:
        Dict containing information about the directory
    """
    dir_path = Path(directory)
    
    # Information to collect
    info = {
        'valid': False,
        'exists': dir_path.exists(),
        'is_dir': dir_path.is_dir() if dir_path.exists() else False,
        'video_dirs': [],
        'has_360': False,
        'has_cubemaps': False,
        'has_equirect': False,
        'incomplete_videos': [],
        'repaired_videos': 0
    }
    
    if not info['exists'] or not info['is_dir']:
        logger.error(f"Directory {dir_path} does not exist or is not a directory")
        return info
    
    # Check for video-specific subdirectories
    for item in dir_path.iterdir():
        if item.is_dir():
            # Check if this is a video output directory
            faces_dir = item / "faces"
            cubemap_dir = item / "cubemap"
            equirect_dir = item / "equirect"
            
            if faces_dir.exists() or cubemap_dir.exists() or equirect_dir.exists():
                video_info = {
                    'name': item.name,
                    'path': str(item),
                    'has_faces': faces_dir.exists(),
                    'has_cubemap': cubemap_dir.exists(),
                    'has_equirect': equirect_dir.exists(),
                    'faces_count': len(list(faces_dir.glob('*.jpg'))) if faces_dir.exists() else 0,
                    'cubemap_count': len(list(cubemap_dir.glob('*.jpg'))) if cubemap_dir.exists() else 0,
                    'equirect_count': len(list(equirect_dir.glob('*.jpg'))) if equirect_dir.exists() else 0,
                    'incomplete': False
                }
                
                # Check if this is a 360 video (has cubemap or equirect dirs)
                if video_info['has_cubemap'] or video_info['has_equirect']:
                    info['has_360'] = True
                    
                    if video_info['has_cubemap']:
                        info['has_cubemaps'] = True
                    
                    if video_info['has_equirect']:
                        info['has_equirect'] = True
                    
                    # Check if processing was incomplete (has equirect or cubemap but no/few faces)
                    if video_info['has_equirect'] and video_info['equirect_count'] > 0:
                        if not video_info['has_faces'] or video_info['faces_count'] < video_info['equirect_count'] * 6:
                            video_info['incomplete'] = True
                            info['incomplete_videos'].append(video_info)
                
                info['video_dirs'].append(video_info)
    
    info['valid'] = len(info['video_dirs']) > 0
    logger.info(f"Found {len(info['video_dirs'])} video directories, {len(info['incomplete_videos'])} incomplete")
    
    return info


def repair_video_directory(video_info, logger):
    """Repair a video output directory.
    
    Args:
        video_info: Dict containing video directory information
        logger: Logger instance
        
    Returns:
        Number of fixed frames
    """
    path = Path(video_info['path'])
    fixed_frames = 0
    
    # Check if this is an incomplete 360 video
    if not video_info['incomplete']:
        logger.info(f"Video {path.name} is already complete, skipping")
        return 0
    
    logger.info(f"Repairing video directory: {path}")
    
    # Setup directories
    equirect_dir = path / "equirect"
    cubemap_dir = path / "cubemap"
    faces_dir = path / "faces"
    
    # Create faces directory if it doesn't exist
    faces_dir.mkdir(exist_ok=True)
    
    # Find all equirectangular frames
    equirect_frames = sorted(list(equirect_dir.glob('*.jpg')))
    logger.info(f"Found {len(equirect_frames)} equirectangular frames")
    
    # Find existing faces
    existing_faces = set(f.stem.split('_')[0] for f in faces_dir.glob('*.jpg'))
    logger.info(f"Found {len(existing_faces)} frames that already have faces")
    
    # Create converters
    converter = SphericalConverter()
    cubemap_converter = CubemapToFaces(debug_mode=True)
    
    # Process each frame that doesn't have faces yet
    for frame_path in equirect_frames:
        frame_base = frame_path.stem
        
        # Skip if this frame already has faces
        if frame_base in existing_faces:
            continue
        
        logger.info(f"Processing frame: {frame_base}")
        
        # Create cubemap file path
        cubemap_path = cubemap_dir / f"{frame_base}_cube.jpg"
        
        # If cubemap doesn't exist, create it
        if not cubemap_path.exists():
            logger.info(f"Creating cubemap for {frame_base}")
            
            # Read the equirectangular frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Could not read equirectangular frame {frame_path}")
                continue
            
            # First create cubemap directory if needed
            if not cubemap_dir.exists():
                cubemap_dir.mkdir(exist_ok=True)
            
            # Try to convert to cubemap
            try:
                # Use our enhanced converter
                faces = converter.convert_frame(frame)
                
                # If we got multiple faces, combine them into a cubemap
                if len(faces) == 6:
                    # Create a horizontal strip of faces
                    face_height, face_width = faces[0].shape[:2]
                    cubemap = np.zeros((face_height, face_width * 6, 3), dtype=np.uint8)
                    
                    for i, face in enumerate(faces):
                        cubemap[:, i*face_width:(i+1)*face_width] = face
                    
                    # Save the cubemap
                    cv2.imwrite(str(cubemap_path), cubemap)
                    logger.info(f"Created cubemap for {frame_base}")
                else:
                    logger.warning(f"Converter returned {len(faces)} faces instead of 6")
                    continue
                    
            except Exception as e:
                logger.error(f"Error creating cubemap for {frame_base}: {e}")
                continue
        
        # Extract faces from the cubemap
        if cubemap_path.exists():
            logger.info(f"Extracting faces from cubemap: {cubemap_path.name}")
            try:
                faces_saved = cubemap_converter.process_file(str(cubemap_path), str(faces_dir))
                if faces_saved > 0:
                    logger.info(f"Extracted {faces_saved} faces from {cubemap_path.name}")
                    fixed_frames += 1
                else:
                    logger.warning(f"No faces were extracted from {cubemap_path.name}")
            except Exception as e:
                logger.error(f"Error extracting faces from {cubemap_path.name}: {e}")
    
    return fixed_frames


def repair_output_directory(directory, logger):
    """Repair the output directory structure.
    
    Args:
        directory: Path to the output directory
        logger: Logger instance
        
    Returns:
        Number of repaired frames
    """
    # Check the directory structure
    info = check_output_structure(directory, logger)
    
    if not info['valid']:
        logger.error(f"Invalid output directory: {directory}")
        return 0
    
    # Fix permissions on the directory
    try:
        import stat
        dir_path = Path(directory)
        
        # Set permissions on the main directory
        os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
        logger.info(f"Set permissions on directory: {dir_path}")
        
        # Set permissions on all subdirectories
        for subdir in dir_path.glob('**/'):
            if subdir.is_dir():
                os.chmod(subdir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
                logger.debug(f"Set permissions on subdirectory: {subdir}")
    except Exception as e:
        logger.warning(f"Could not set permissions: {e}")
    
    # Process incomplete videos
    fixed_frames = 0
    for video_info in info['incomplete_videos']:
        fixed = repair_video_directory(video_info, logger)
        fixed_frames += fixed
        if fixed > 0:
            info['repaired_videos'] += 1
    
    logger.info(f"Repaired {info['repaired_videos']} videos with {fixed_frames} frames")
    return fixed_frames


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Repair Frame Extractor output directories')
    parser.add_argument('--directory', '-d', default=None, help='Output directory to repair')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Default directory if not specified
    if args.directory is None:
        # Try default Mac Desktop path
        default_path = os.path.expanduser("~/Desktop/FrameExtractor_Output/360split")
        if os.path.exists(default_path):
            args.directory = default_path
            logger.info(f"Using default path: {default_path}")
        else:
            logger.error("No directory specified and default path not found")
            print("Please specify an output directory with --directory")
            return 1
    
    # Repair the output directory
    logger.info(f"Starting repair of directory: {args.directory}")
    fixed_frames = repair_output_directory(args.directory, logger)
    
    # Report results
    if fixed_frames > 0:
        logger.info(f"Successfully repaired {fixed_frames} frames")
        print(f"SUCCESS: Repaired {fixed_frames} frames in {args.directory}")
        return 0
    else:
        logger.info("No frames needed repair")
        print(f"No frames needed repair in {args.directory}")
        return 0


if __name__ == "__main__":
    exit(main()) 