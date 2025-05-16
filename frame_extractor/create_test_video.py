#!/usr/bin/env python3
"""
Generate a test video for testing the Frame Extractor application.
"""

import os
import cv2
import numpy as np
from pathlib import Path


def create_test_video(output_path, width=1280, height=720, seconds=5, fps=30):
    """Create a test video with some moving shapes.
    
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
    
    # Create frames
    frames = []
    total_frames = seconds * fps
    
    for i in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background with gradient
        for y in range(height):
            for x in range(width):
                b = int(x * 255 / width)
                g = int(y * 255 / height)
                r = int(255 - (x * 255 / width))
                frame[y, x] = [b, g, r]
        
        # Add timestamp
        time_str = f"Frame: {i}/{total_frames-1} - Time: {i/fps:.2f}s"
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add moving circle
        radius = 50
        circle_x = int(width/2 + (width/3) * np.sin(i * 2 * np.pi / total_frames))
        circle_y = int(height/2 + (height/3) * np.cos(i * 2 * np.pi / total_frames))
        cv2.circle(frame, (circle_x, circle_y), radius, (0, 0, 255), -1)
        
        # Add static rectangles in the corners
        cv2.rectangle(frame, (10, 10), (110, 110), (0, 255, 0), -1)
        cv2.rectangle(frame, (width-110, 10), (width-10, 110), (255, 0, 0), -1)
        cv2.rectangle(frame, (10, height-110), (110, height-10), (255, 255, 0), -1)
        cv2.rectangle(frame, (width-110, height-110), (width-10, height-10), (0, 255, 255), -1)
        
        # Add some text that changes every second
        second = i // fps
        if second % 2 == 0:
            cv2.putText(frame, "SHARP TEXT", (width//2 - 100, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Add slightly blurred text
            cv2.putText(frame, "BLURRY TEXT", (width//2 - 100, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        # Every 30 frames, make the image blurry
        if i % 30 == 0:
            frame = cv2.GaussianBlur(frame, (15, 15), 5)
        
        out.write(frame)
        
        print(f"\rCreating frame {i+1}/{total_frames}", end="", flush=True)
    
    out.release()
    print(f"\nVideo saved to {output_path}")


if __name__ == "__main__":
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a basic test video
    create_test_video(os.path.join(output_dir, "test_video.mp4"))
    
    # Create a shorter video with more blur
    create_test_video(os.path.join(output_dir, "blurry_video.mp4"), 
                     seconds=2, 
                     width=640,
                     height=480) 