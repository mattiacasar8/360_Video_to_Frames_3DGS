#!/usr/bin/env python3
"""
Script to package the Frame Extractor application for macOS using py2app.

Usage:
    python package_macos.py py2app
"""

import sys
import os
from setuptools import setup

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APP = ['run.py']
DATA_FILES = []

OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'resources/icon.icns',
    'plist': {
        'CFBundleName': 'Frame Extractor',
        'CFBundleDisplayName': 'Frame Extractor',
        'CFBundleIdentifier': 'com.frame-extractor.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': 'Copyright © 2023. All rights reserved.',
        'NSHighResolutionCapable': True,
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'LSMinimumSystemVersion': '10.15',
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Video File',
                'CFBundleTypeExtensions': ['mp4', 'mov', 'avi', 'mkv'],
                'CFBundleTypeRole': 'Viewer',
            }
        ],
    },
    'packages': [
        'src',
        'PyQt6',
        'cv2',
        'numpy',
        'skimage',
        'scipy',
    ],
    # Add resources like icons
    'resources': ['resources'],
    # Exclude unnecessary packages to reduce size
    'excludes': [
        'tkinter',
        'matplotlib',
        'PyQt5',
        'PyQt6.QtWebEngineCore',
        'pandas',
    ],
    # Enable high resolution mode
    'high_resolution': True,
    # Optimize Python bytecode
    'optimize': 2,
    # Add binary dependencies
    'frameworks': [],
    # Extension to include FFmpeg
    'extra_scripts': [],
}


def ensure_resources_directory():
    """Create resources directory if it doesn't exist."""
    resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
        print(f"Created resources directory at {resources_dir}")
        print("Note: You should add an icon.icns file to this directory before packaging.")


if __name__ == '__main__':
    ensure_resources_directory()
    setup(
        name='Frame Extractor',
        app=APP,
        data_files=DATA_FILES,
        options={'py2app': OPTIONS},
        setup_requires=['py2app'],
    ) 