# Add this content to setup.py
from setuptools import setup, find_packages

setup(
    name="hemsam",
    version="1.0.0",
    author="Your Name",
    description="HEMSAM: Hybrid Enhanced Multi-Scale SAM for Plant Disease Detection",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "albumentations>=1.3.0",
        "segment-anything",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
    ],
)