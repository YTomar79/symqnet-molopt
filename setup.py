#!/usr/bin/env python3
"""
Setup script for SymQNet Molecular Optimization CLI

This package provides a command-line interface for molecular Hamiltonian 
parameter estimation using trained SymQNet neural networks.
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
def read_requirements():
    requirements_file = this_directory / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # Fallback requirements if file doesn't exist
        return [
            "torch>=1.12.0",
            "numpy>=1.21.0",
            "qiskit>=0.39.0",
            "openfermion>=1.5.0",
            "scipy>=1.9.0",
            "click>=8.0.0",
            "tqdm>=4.64.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "torch-geometric>=2.2.0",
            "pandas>=1.4.0",
            "tensorboard>=2.8.0",
            "gym>=0.26.0"
        ]

# Package metadata
setup(
    name="symqnet-molopt",
    version="1.0.0",
    author="SymQNet Research Team",
    author_email="research@symqnet.ai",
    description="Molecular Hamiltonian parameter estimation using SymQNet neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/symqnet/molecular-optimization",
    project_urls={
        "Bug Tracker": "https://github.com/symqnet/molecular-optimization/issues",
        "Documentation": "https://symqnet.readthedocs.io/",
        "Source Code": "https://github.com/symqnet/molecular-optimization",
    },
    
    # Package discovery
    packages=find_packages(where="."),
    package_dir={"": "."},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "symqnet_molopt": [
            "examples/*.json",
            "models/*.pth",
            "scripts/*.py",
            "*.md",
            "requirements.txt"
        ],
    },
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "symqnet-molopt=cli:main",
            "symqnet-validate=scripts.validate_installation:main",
            "symqnet-test=scripts.test_models:main",
            "symqnet-examples=scripts.create_examples:main",
        ],
    },
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Extra dependencies for development
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17",
        ],
        "gpu": [
            "torch>=1.12.0+cu116",
            "torch-geometric>=2.2.0+cu116",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
            "plotly>=5.0",
        ]
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    
    # Keywords for discoverability
    keywords=[
        "quantum computing",
        "molecular simulation", 
        "neural networks",
        "hamiltonian estimation",
        "symqnet",
        "quantum chemistry",
        "machine learning",
        "reinforcement learning"
    ],
    
    # License
    license="MIT",
    
    # Additional metadata
    zip_safe=False,  # Required for accessing package data
    
    # Platform support
    platforms=["any"],
    
    # Minimum setuptools version
    setup_requires=["setuptools>=45", "wheel"],
)

# Post-installation message
print("""
🎉 SymQNet Molecular Optimization CLI installed successfully!

Quick start:
  1. Validate installation: symqnet-validate
  2. Create examples: symqnet-examples  
  3. Test models: symqnet-test
  4. Run optimization: symqnet-molopt --help

Documentation: https://symqnet.readthedocs.io/
Support: https://github.com/symqnet/molecular-optimization/issues
""")
