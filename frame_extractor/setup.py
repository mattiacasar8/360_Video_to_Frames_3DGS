from setuptools import setup, find_packages

setup(
    name="frame_extractor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "scikit-image>=0.18.0",
        "PyQt6>=6.0.0",
        "ffmpeg-python>=0.2.0",
    ],
    entry_points={
        "console_scripts": [
            "frame-extractor=src.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for extracting frames from videos for 3D Gaussian Splatting",
    keywords="video, frame extraction, 3D, gaussian splatting, photogrammetry",
    python_requires=">=3.8",
) 