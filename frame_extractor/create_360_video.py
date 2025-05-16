#!/usr/bin/env python3
"""
Generate a synthetic 360° video for testing the Frame Extractor application.
"""

import os
import cv2
import numpy as np
import subprocess
from pathlib import Path


def create_equirectangular_frame(width=1920, height=960, frame_num=0, total_frames=30):
    """Create an equirectangular frame (2:1 aspect ratio).
    
    Args:
        width: Frame width (typically 2x height for equirectangular)
        height: Frame height
        frame_num: Current frame number for animation
        total_frames: Total frames in the video
    
    Returns:
        NumPy array with the frame
    """
    # Create a blank frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a latitude/longitude grid
    for y in range(height):
        # Convert y to latitude (-90 to 90 degrees)
        lat = 90 - (y * 180.0 / height)
        
        for x in range(width):
            # Convert x to longitude (-180 to 180 degrees)
            lon = (x * 360.0 / width) - 180
            
            # Assign colors based on position
            # Blue varies with latitude
            b = int(((lat + 90) / 180) * 255)
            # Green varies with longitude
            g = int(((lon + 180) / 360) * 255)
            # Red varies with time (animation)
            r = int((frame_num / total_frames) * 255)
            
            frame[y, x] = [b, g, r]
    
    # Add grid lines for latitude and longitude
    for lat_line in range(0, height, height // 9):  # Every 20 degrees
        cv2.line(frame, (0, lat_line), (width, lat_line), (255, 255, 255), 1)
        
    for lon_line in range(0, width, width // 12):  # Every 30 degrees
        cv2.line(frame, (lon_line, 0), (lon_line, height), (255, 255, 255), 1)
    
    # Add cardinal directions
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # North
    cv2.putText(frame, "N", (width // 2, 50), font, 2, (255, 255, 255), 2)
    # South
    cv2.putText(frame, "S", (width // 2, height - 50), font, 2, (255, 255, 255), 2)
    # East
    cv2.putText(frame, "E", (width * 3 // 4, height // 2), font, 2, (255, 255, 255), 2)
    # West
    cv2.putText(frame, "W", (width // 4, height // 2), font, 2, (255, 255, 255), 2)
    
    # Add frame information
    cv2.putText(frame, f"Frame: {frame_num}", (50, 50), font, 1, (255, 255, 255), 2)
    
    # Add an animated element
    circle_x = int(width/2 + (width/3) * np.sin(frame_num * 2 * np.pi / total_frames))
    circle_y = int(height/2 + (height/4) * np.cos(frame_num * 2 * np.pi / total_frames))
    cv2.circle(frame, (circle_x, circle_y), 30, (0, 0, 255), -1)
    
    return frame


def create_360_video(output_path, width=1920, height=960, seconds=5, fps=30):
    """Create a synthetic 360° video.
    
    Args:
        output_path: Path to save the video
        width: Video width
        height: Video height
        seconds: Video duration in seconds
        fps: Frames per second
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = seconds * fps
    
    for i in range(total_frames):
        # Create equirectangular frame
        frame = create_equirectangular_frame(width, height, i, total_frames)
        
        # Write to video
        out.write(frame)
        
        print(f"\rCreating 360° frame {i+1}/{total_frames}", end="", flush=True)
    
    out.release()
    print(f"\nVideo saved to {output_path}")


def add_360_metadata(input_path, output_path=None):
    """Add 360° metadata to a video using FFmpeg.
    
    Args:
        input_path: Path to the input video
        output_path: Path to the output video with metadata (if None, will overwrite input)
    
    Returns:
        Path to the output video
    """
    if output_path is None:
        output_path = input_path + ".temp.mp4"
        will_replace = True
    else:
        will_replace = False
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "copy", 
        "-metadata:s:v", "spherical=true",
        "-metadata:s:v", "stereo=monoscopic",
        "-metadata:s:v", "projection=equirectangular",
        output_path
    ]
    
    try:
        print("Adding 360° metadata...")
        subprocess.run(cmd, check=True, capture_output=True)
        print("Metadata added successfully")
        
        if will_replace:
            os.replace(output_path, input_path)
            return input_path
        else:
            return output_path
    
    except (subprocess.SubprocessError, OSError) as e:
        print(f"Error adding 360° metadata: {e}")
        return input_path


if __name__ == "__main__":
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 360° test video
    video_path = os.path.join(output_dir, "test_360.mp4")
    create_360_video(video_path)
    
    # Add 360° metadata
    add_360_metadata(video_path) 