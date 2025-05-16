#!/usr/bin/env python3
"""Main entry point for the Frame Extractor application."""

import sys
import os
import argparse
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication, Qt

from .ui.main_window import MainWindow
from .core.video_processor import VideoProcessor


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(os.path.expanduser('~'), 'frame_extractor.log'))
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract frames from videos for 3D Gaussian Splatting')
    
    parser.add_argument('--cli', action='store_true', help='Run in command-line mode')
    parser.add_argument('--input', '-i', nargs='+', help='Input video files')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--blur-threshold', type=float, default=100.0, help='Blur detection threshold (Laplacian)')
    parser.add_argument('--max-frames', type=int, default=20, help='Maximum frames to select per video')
    parser.add_argument('--min-frame-gap', type=int, default=5, help='Minimum gap between selected frames')
    parser.add_argument('--frame-interval', type=int, default=15, help='Frame extraction interval')
    parser.add_argument('--is-360', action='store_true', help='Input videos are 360 degree')
    
    return parser.parse_args()


def run_cli(args):
    """Run the application in command-line mode."""
    logger = logging.getLogger('cli')
    
    if not args.input:
        logger.error("No input files specified")
        return 1
    
    if not args.output:
        logger.error("No output directory specified")
        return 1
    
    # Ensure input files exist
    for input_file in args.input:
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return 1
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = VideoProcessor(
        output_dir=args.output,
        blur_threshold=args.blur_threshold,
        max_frames=args.max_frames,
        min_frame_gap=args.min_frame_gap,
        frame_interval=args.frame_interval
    )
    
    # Process videos
    is_360_list = [args.is_360] * len(args.input)
    stats = processor.process_videos(args.input, is_360_list)
    
    # Print statistics
    logger.info("Processing complete!")
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Blurry frames removed: {stats['blurry_frames']}")
    logger.info(f"Frames selected: {stats['selected_frames']}")
    logger.info(f"Total views saved: {stats['saved_frames']}")
    logger.info(f"Videos processed: {stats['videos_processed']}")
    
    return 0


def run_gui():
    """Run the application in GUI mode."""
    # Enable high DPI scaling
    # In newer PyQt6 versions, high DPI scaling is enabled by default
    app = QApplication(sys.argv)
    app.setApplicationName("Frame Extractor")
    app.setOrganizationName("3D Gaussian Splatting Tools")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    return app.exec()


def main():
    """Main entry point for the application."""
    args = parse_args()
    
    if args.cli:
        # Command line mode
        print("Running in command line mode")
        
        if not args.input:
            print("Error: No input files specified")
            sys.exit(1)
        
        if not args.output:
            # Use current directory as default
            args.output = os.path.join(os.getcwd(), "output")
        
        # Initialize the processor with command line arguments
        processor = VideoProcessor(
            output_dir=args.output,
            blur_threshold=args.blur_threshold,
            max_frames=args.max_frames,
            min_frame_gap=args.min_frame_gap,
            frame_interval=args.frame_interval
        )
        
        # Report settings
        print(f"\n--- Processing Configuration ---")
        print(f"Output directory: {args.output}")
        print(f"Blur threshold (Laplacian): {args.blur_threshold}")
        print(f"Maximum frames to select: {args.max_frames}")
        print(f"Minimum frame gap: {args.min_frame_gap}")
        print(f"Frame interval: {args.frame_interval}")
        print(f"Videos are 360°: {'Yes' if args.is_360 else 'No'}")
        print(f"----------------------------\n")
        
        # Process all input videos
        is_360_list = [args.is_360] * len(args.input)
        stats = processor.process_videos(args.input, is_360_list)
        
        # Print results
        print("\nProcessing complete")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Blurry frames removed: {stats['blurry_frames']}")
        print(f"Frames selected: {stats['selected_frames']}")
        print(f"Total views saved: {stats['saved_frames']}")
        print(f"Output directory: {args.output}")
    else:
        # GUI mode
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    sys.exit(main()) 