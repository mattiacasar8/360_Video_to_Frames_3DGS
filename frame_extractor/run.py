#!/usr/bin/env python3
"""
Run script for the Frame Extractor application.

This script simplifies running the application during development.
"""

import sys
import os

# Add the parent directory to the Python path to allow importing the src module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main


if __name__ == "__main__":
    sys.exit(main()) 