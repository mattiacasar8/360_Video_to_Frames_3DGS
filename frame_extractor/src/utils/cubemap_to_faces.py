"""Utility to convert cubemap images to individual faces."""

import os
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

class CubemapToFaces:
    """Utility class to convert cubemap images to individual faces."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the converter.
        
        Args:
            debug_mode: Whether to enable debug mode for additional diagnostics
        """
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode
        
    def process_directory(self, input_dir: str, output_dir: str) -> int:
        """Process all cubemap images in a directory.
        
        Args:
            input_dir: Directory containing cubemap images
            output_dir: Directory to save face images
            
        Returns:
            Number of successfully processed images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all cubemap images
        cubemap_files = list(input_path.glob("*_cube.jpg"))
        self.logger.info(f"Found {len(cubemap_files)} cubemap images")
        
        processed_count = 0
        for cubemap_file in cubemap_files:
            faces_saved = self.process_file(str(cubemap_file), output_dir)
            if faces_saved > 0:
                processed_count += 1
        
        return processed_count
        
    def process_file(self, cubemap_path: str, output_dir: str) -> int:
        """Process a single cubemap image.
        
        Args:
            cubemap_path: Path to the cubemap image
            output_dir: Directory to save face images
            
        Returns:
            Number of faces successfully extracted (6 if successful, 0 if failed)
        """
        try:
            cubemap_file = Path(cubemap_path)
            output_path = Path(output_dir)
            
            # Create output directory if it doesn't exist and set permissions
            self._ensure_directory(output_path)
            
            # Read the cubemap image
            cubemap = cv2.imread(str(cubemap_file))
            if cubemap is None:
                self.logger.warning(f"Failed to read cubemap image: {cubemap_file}")
                return 0
            
            # Log cubemap details for debugging
            height, width, channels = cubemap.shape
            self.logger.info(f"Cubemap dimensions: {width}x{height}x{channels}")
            
            # Ensure width is divisible by 6 for proper face extraction
            if width % 6 != 0:
                self.logger.warning(f"Cubemap width {width} is not divisible by 6. Auto-correcting dimensions.")
                # Adjust width to be divisible by 6
                new_width = (width // 6) * 6
                cubemap = cv2.resize(cubemap, (new_width, height))
                width = new_width
                self.logger.info(f"Adjusted cubemap dimensions to: {width}x{height}")
            
            # Save debug image if in debug mode
            if self.debug_mode:
                debug_path = output_path / f"debug_{cubemap_file.stem}.jpg"
                cv2.imwrite(str(debug_path), cubemap)
                self.logger.info(f"Saved debug image to {debug_path}")
            
            # Get base name without _cube.jpg
            base_name = cubemap_file.stem
            if base_name.endswith("_cube"):
                base_name = base_name[:-5]  # Remove _cube suffix
            
            # Split the cubemap into 6 individual faces
            face_width = width // 6
            
            if face_width <= 0:
                self.logger.error(f"Invalid cubemap dimensions: {cubemap.shape}, face_width would be {width / 6}")
                return 0
            
            # Extract and save individual faces
            face_names = ["lateral_1", "lateral_2", "top", "bottom", "lateral_3", "lateral_4"]
            
            saved_faces = 0
            faces_to_verify = []
            
            for i, face_name in enumerate(face_names):
                try:
                    if i * face_width >= width:
                        self.logger.error(f"Face index {i} exceeds cubemap width")
                        continue
                        
                    x = i * face_width
                    end_x = min(x + face_width, width)
                    
                    if x >= end_x:
                        self.logger.error(f"Invalid face slice coordinates: x={x}, end_x={end_x}")
                        continue
                    
                    face = cubemap[:, x:end_x]
                    
                    # Check that the face is valid
                    if face.size == 0:
                        self.logger.error(f"Extracted face has zero size: {face.shape}")
                        continue
                    
                    # Create a filename that includes frame number for uniqueness
                    face_path = output_path / f"{base_name}_{face_name}.jpg"
                    
                    # Log face info for debugging
                    self.logger.info(f"Face {i} ({face_name}): {face.shape} from x={x} to x={end_x}")
                    
                    # Try multiple methods to save the face
                    if self._save_face_with_fallbacks(face, face_path):
                        saved_faces += 1
                        faces_to_verify.append(face_path)
                    else:
                        self.logger.error(f"All save methods failed for face {face_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing face {face_name}: {e}")
            
            # Verify all faces exist and are readable
            verified_faces = 0
            for face_path in faces_to_verify:
                if face_path.exists():
                    try:
                        # Try to read the face to make sure it's valid
                        test_img = cv2.imread(str(face_path))
                        if test_img is not None:
                            verified_faces += 1
                        else:
                            self.logger.warning(f"Face file exists but could not be read: {face_path}")
                    except Exception as e:
                        self.logger.warning(f"Error verifying face {face_path}: {e}")
            
            # If verified faces don't match saved faces, update the count
            if verified_faces != saved_faces:
                self.logger.warning(f"Only {verified_faces}/{saved_faces} faces could be verified")
                saved_faces = verified_faces
            
            self.logger.info(f"Processed {cubemap_file.name}: extracted {saved_faces}/6 faces")
            return saved_faces
            
        except Exception as e:
            self.logger.error(f"Error processing {cubemap_path}: {e}")
            return 0
    
    def _save_face_with_fallbacks(self, face: np.ndarray, face_path: Path) -> bool:
        """Save a face image using multiple fallback methods if needed.
        
        Args:
            face: Face image as NumPy array
            face_path: Path to save the face
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Method 1: Direct OpenCV write
        try:
            if cv2.imwrite(str(face_path), face):
                self.logger.debug(f"Saved face to {face_path} using direct write")
                return True
        except Exception as e:
            self.logger.debug(f"Direct write failed: {e}")
        
        # Method 2: Write to temp file then copy
        try:
            import tempfile
            import shutil
            
            # Create a unique temp path
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
            
            # Write to temp location first
            if cv2.imwrite(temp_path, face):
                # Then copy to destination
                shutil.copy2(temp_path, str(face_path))
                os.unlink(temp_path)
                self.logger.debug(f"Saved face to {face_path} using temp file method")
                return True
        except Exception as e:
            self.logger.debug(f"Temp file write failed: {e}")
        
        # Method 3: Try with PIL
        try:
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            img.save(str(face_path), "JPEG")
            self.logger.debug(f"Saved face to {face_path} using PIL")
            return True
        except Exception as e:
            self.logger.debug(f"PIL write failed: {e}")
        
        # Method 4: Last resort - binary write
        try:
            # Encode the image
            _, buffer = cv2.imencode('.jpg', face)
            # Write binary data
            with open(face_path, 'wb') as f:
                f.write(buffer)
            self.logger.debug(f"Saved face to {face_path} using binary write")
            return True
        except Exception as e:
            self.logger.debug(f"Binary write failed: {e}")
        
        return False

    def validate_cubemap(self, cubemap_path: str) -> Tuple[bool, str]:
        """Validate a cubemap image.
        
        Args:
            cubemap_path: Path to the cubemap image
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Read the cubemap image
            cubemap = cv2.imread(str(cubemap_path))
            if cubemap is None:
                return False, f"Could not read image at {cubemap_path}"
            
            # Check dimensions
            height, width, channels = cubemap.shape
            if width % 6 != 0:
                return False, f"Cubemap width {width} is not divisible by 6"
            
            # Check if each face has reasonable dimensions
            face_width = width // 6
            if face_width < 10 or height < 10:
                return False, f"Cubemap face dimensions too small: {face_width}x{height}"
            
            return True, "Cubemap is valid"
            
        except Exception as e:
            return False, f"Error validating cubemap: {e}"

    def _ensure_directory(self, directory_path):
        """Ensure directory exists with proper permissions.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            True if directory is accessible, False otherwise
        """
        try:
            dir_path = Path(directory_path)
            
            # Create directory if it doesn't exist
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Set permissions to ensure we can write to it
            import stat
            try:
                current_mode = dir_path.stat().st_mode
                permissions = stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH  # 0o775
                
                # Only change if permissions are different
                if (current_mode & 0o777) != (permissions & 0o777):
                    os.chmod(dir_path, permissions)
                    self.logger.info(f"Updated permissions for directory: {dir_path}")
            except Exception as e:
                self.logger.warning(f"Could not set permissions on directory {dir_path}: {e}")
            
            # Verify we can write to the directory
            test_file = dir_path / ".write_test"
            try:
                with open(test_file, 'w') as f:
                    f.write("Test write access")
                test_file.unlink()
                return True
            except Exception as e:
                self.logger.warning(f"Directory {dir_path} is not writable: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error ensuring directory {directory_path}: {e}")
            return False

def main():
    """Command-line interface for the cubemap to faces converter."""
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert cubemap images to individual faces')
    parser.add_argument('input_dir', help='Directory containing cubemap images')
    parser.add_argument('output_dir', help='Directory to save face images')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Process the directory
    converter = CubemapToFaces(debug_mode=args.debug)
    processed = converter.process_directory(args.input_dir, args.output_dir)
    
    print(f"Successfully processed {processed} cubemap images")

if __name__ == "__main__":
    main() 