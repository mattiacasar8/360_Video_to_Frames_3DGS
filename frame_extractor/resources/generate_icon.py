#!/usr/bin/env python3
"""
Generate a simple icon for the Frame Extractor application.
This creates a PNG image that can be converted to an ICNS file using 
the macOS iconutil tool.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path


def create_app_icon(output_path, size=1024):
    """Create a simple application icon for the Frame Extractor."""
    # Create a blank square canvas with the desired size
    icon = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Background color (dark blue)
    bg_color = (50, 30, 120)  # BGR
    icon[:] = bg_color
    
    # Create a circular gradient background
    center = size // 2
    radius = size // 2
    for y in range(size):
        for x in range(size):
            # Calculate distance from center
            dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if dist < radius:
                # Create a gradient that's darker at the edges
                ratio = 1.0 - (dist / radius) ** 2
                # Slightly brighten the bg_color based on the ratio
                icon[y, x] = [
                    min(int(c + (255 - c) * ratio * 0.5), 255) for c in bg_color
                ]
    
    # Draw a camera aperture-like shape
    aperture_color = (240, 240, 240)  # White-ish
    aperture_radius = int(size * 0.4)
    aperture_thickness = int(size * 0.05)
    cv2.circle(icon, (center, center), aperture_radius, aperture_color, aperture_thickness)
    
    # Draw aperture blades
    n_blades = 6
    inner_radius = aperture_radius - aperture_thickness // 2
    outer_radius = inner_radius + int(size * 0.15)
    
    for i in range(n_blades):
        angle = 2 * np.pi * i / n_blades
        x1 = center + int(inner_radius * np.cos(angle))
        y1 = center + int(inner_radius * np.sin(angle))
        x2 = center + int(outer_radius * np.cos(angle))
        y2 = center + int(outer_radius * np.sin(angle))
        cv2.line(icon, (x1, y1), (x2, y2), aperture_color, aperture_thickness)
    
    # Draw frame extraction symbol (play button with frames)
    frame_color = (50, 200, 100)  # Green-ish
    triangle_size = int(size * 0.2)
    
    # Draw a play triangle in the center
    triangle_pts = np.array([
        [center, center - triangle_size//2],
        [center + triangle_size//2, center],
        [center, center + triangle_size//2]
    ], np.int32)
    cv2.fillPoly(icon, [triangle_pts], frame_color)
    
    # Add small frame rectangles
    rect_size = int(size * 0.08)
    rect_offset = int(size * 0.15)
    
    # Draw three frame rectangles trailing to the right
    for i in range(3):
        offset = rect_offset + i * rect_size//2
        alpha = 1.0 - i * 0.3
        rect_color = tuple(int(c * alpha) for c in frame_color)
        rect_x = center + offset
        rect_y = center - rect_size//2
        cv2.rectangle(icon, 
                     (rect_x, rect_y), 
                     (rect_x + rect_size, rect_y + rect_size), 
                     rect_color, 
                     -1)  # Filled rectangle
        # Add a white border
        cv2.rectangle(icon, 
                     (rect_x, rect_y), 
                     (rect_x + rect_size, rect_y + rect_size), 
                     (255, 255, 255), 
                     max(1, int(size * 0.005)))  # Thin border
    
    # Apply a subtle gaussian blur to smooth everything
    icon = cv2.GaussianBlur(icon, (5, 5), 0)
    
    # Save the icon
    cv2.imwrite(output_path, icon)
    print(f"Icon saved to {output_path}")
    
    return output_path


def convert_to_icns(png_path):
    """
    Convert the PNG to ICNS format using macOS iconutil.
    
    This requires creating a temporary iconset directory with 
    multiple sizes of the icon.
    """
    try:
        # Create the iconset directory
        png_path = Path(png_path)
        iconset_path = png_path.parent / "AppIcon.iconset"
        os.makedirs(iconset_path, exist_ok=True)
        
        # Read the source image
        img = cv2.imread(str(png_path))
        
        # Create different icon sizes required for macOS
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        for size in sizes:
            # Standard resolution
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(iconset_path / f"icon_{size}x{size}.png"), resized)
            
            # High resolution (2x) - required for Retina displays
            if size <= 512:  # Don't need 2048x2048
                high_res_size = size * 2
                high_res = cv2.resize(img, (high_res_size, high_res_size), 
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(iconset_path / f"icon_{size}x{size}@2x.png"), high_res)
        
        # Convert the iconset to icns using iconutil
        icns_path = png_path.parent / "icon.icns"
        os.system(f"iconutil -c icns {iconset_path} -o {icns_path}")
        print(f"ICNS file saved to {icns_path}")
        
        # Clean up the iconset directory
        import shutil
        shutil.rmtree(iconset_path)
        
        return str(icns_path)
    
    except Exception as e:
        print(f"Error converting to ICNS: {e}")
        print("You may need to manually convert the PNG to ICNS format.")
        return None


if __name__ == "__main__":
    # Base directory is the script's directory
    base_dir = Path(__file__).parent
    
    # Create the icon
    png_path = os.path.join(base_dir, "app_icon.png")
    create_app_icon(png_path)
    
    # Try to convert to ICNS
    convert_to_icns(png_path) 