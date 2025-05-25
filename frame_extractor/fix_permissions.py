#!/usr/bin/env python3
"""Utility script to fix permissions for Frame Extractor output directories."""

import os
import argparse
import stat
import shutil
from pathlib import Path
import logging

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('fix_permissions')

def fix_directory_permissions(directory_path, recursive=True, logger=None):
    """Fix permissions for the given directory.
    
    Args:
        directory_path: Path to the directory to fix permissions for
        recursive: Whether to fix permissions recursively
        logger: Logger instance
        
    Returns:
        Tuple of (success, message)
    """
    if logger is None:
        logger = logging.getLogger('fix_permissions')
    
    try:
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            return False, f"Directory does not exist: {dir_path}"
        
        if not dir_path.is_dir():
            return False, f"Not a directory: {dir_path}"
        
        # Fix permissions for the directory itself
        logger.info(f"Fixing permissions for directory: {dir_path}")
        
        # Set read/write/execute permissions for owner and group, read/execute for others
        # This is equivalent to chmod 775
        os.chmod(
            dir_path, 
            stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH
        )
        
        if recursive:
            # Process all files and subdirectories
            for item in dir_path.glob('**/*'):
                if item.is_dir():
                    # Set directory permissions: rwxrwxr-x (775)
                    logger.info(f"Fixing permissions for subdirectory: {item}")
                    os.chmod(
                        item, 
                        stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH
                    )
                else:
                    # Set file permissions: rw-rw-r-- (664)
                    logger.info(f"Fixing permissions for file: {item}")
                    os.chmod(
                        item, 
                        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH
                    )
        
        return True, f"Successfully fixed permissions for {dir_path}"
    
    except Exception as e:
        return False, f"Error fixing permissions: {str(e)}"

def verify_directory_access(directory_path, logger=None):
    """Verify access to the directory.
    
    Args:
        directory_path: Path to the directory to verify access for
        logger: Logger instance
        
    Returns:
        Dict with access details
    """
    if logger is None:
        logger = logging.getLogger('fix_permissions')
    
    dir_path = Path(directory_path)
    access_info = {
        'exists': dir_path.exists(),
        'is_dir': dir_path.is_dir() if dir_path.exists() else False,
        'readable': os.access(dir_path, os.R_OK) if dir_path.exists() else False,
        'writable': os.access(dir_path, os.W_OK) if dir_path.exists() else False,
        'executable': os.access(dir_path, os.X_OK) if dir_path.exists() else False,
        'owner': None,
        'permissions': None
    }
    
    if access_info['exists']:
        try:
            stat_info = os.stat(dir_path)
            access_info['owner'] = stat_info.st_uid
            access_info['permissions'] = stat.filemode(stat_info.st_mode)
        except Exception as e:
            logger.warning(f"Could not get detailed stat info: {e}")
    
    return access_info

def test_write_access(directory_path, logger=None):
    """Test write access to a directory by creating and deleting a test file.
    
    Args:
        directory_path: Path to the directory to test
        logger: Logger instance
        
    Returns:
        Tuple of (success, message)
    """
    if logger is None:
        logger = logging.getLogger('fix_permissions')
    
    dir_path = Path(directory_path)
    test_file = dir_path / "permission_test.txt"
    
    try:
        # Try writing to a test file
        logger.info(f"Testing write access to {dir_path}")
        with open(test_file, 'w') as f:
            f.write("Testing permissions")
        
        # Try reading the file
        with open(test_file, 'r') as f:
            content = f.read()
            
        # Try deleting the file
        test_file.unlink()
        
        return True, "Write test successful"
    
    except Exception as e:
        if test_file.exists():
            try:
                test_file.unlink()
            except:
                pass
        
        return False, f"Write test failed: {str(e)}"

def fix_output_directories(base_output_dir=None, logger=None):
    """Fix permissions for all output directories used by the Frame Extractor.
    
    Args:
        base_output_dir: Path to the base output directory (if None, use 'output-frames')
        logger: Logger instance
        
    Returns:
        Tuple of (success, message)
    """
    if logger is None:
        logger = logging.getLogger('fix_permissions')
    
    # Default output directory
    if base_output_dir is None:
        base_output_dir = Path("output-frames")
    else:
        base_output_dir = Path(base_output_dir)
    
    # Create output directory if it doesn't exist
    if not base_output_dir.exists():
        try:
            base_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {base_output_dir}")
        except Exception as e:
            return False, f"Failed to create output directory: {str(e)}"
    
    # Fix permissions for base output directory
    success, message = fix_directory_permissions(base_output_dir, True, logger)
    if not success:
        return False, message
    
    # Test write access to the directory
    success, message = test_write_access(base_output_dir, logger)
    if not success:
        return False, f"After fixing permissions, write test still failed: {message}"
    
    return True, "Successfully fixed permissions for all output directories"

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Fix permissions for Frame Extractor output directories')
    parser.add_argument('--output-dir', help='Path to the output directory (default: output-frames)', default=None)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Fix permissions
    success, message = fix_output_directories(args.output_dir, logger)
    
    if success:
        logger.info(message)
        print("SUCCESS: Permissions fixed successfully. Try running the extraction again.")
        return 0
    else:
        logger.error(message)
        print(f"ERROR: {message}")
        
        # Check if running as root would help
        if os.geteuid() != 0:  # Not root
            print("\nSuggestion: Try running this script with sudo for full permissions")
            print("  sudo python fix_permissions.py")
        
        return 1

if __name__ == "__main__":
    exit(main()) 