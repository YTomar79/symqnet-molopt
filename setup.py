#!/usr/bin/env python3
"""
Setup script for SymQNet Molecular Optimization CLI
FIXED: CLI module name conflict resolved by renaming cli.py to symqnet_cli.py

This package provides a command-line interface for molecular Hamiltonian 
parameter estimation using trained SymQNet neural networks.
"""

from setuptools import setup, find_packages
from pathlib import Path
import os
import glob

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
            "scipy>=1.9.0",
            "click>=8.0.0",
            "tqdm>=4.64.0",
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
            "gym>=0.26.0"
        ]

# Helper function to get data files safely
def get_data_files():
    """Get data files that actually exist"""
    data_files = []
    
    # Examples
    if Path("examples").exists():
        example_files = glob.glob("examples/*.json")
        if example_files:
            data_files.append(("examples", example_files))
    
    # Models (if any)
    if Path("models").exists():
        model_files = glob.glob("models/*.pth")
        if model_files:
            data_files.append(("models", model_files))
    
    # Scripts (if any)
    if Path("scripts").exists():
        script_files = glob.glob("scripts/*.py")
        if script_files:
            data_files.append(("scripts", script_files))
    
    return data_files

# Package metadata
setup(
    name="symqnet-molopt",
    version="1.0.5",  # 🔧 FIX: Increment version for CLI conflict fix
    author="YTomar79",
    author_email="yashm.tomar@gmail.com",  # Update with your actual email
    description="Molecular Hamiltonian parameter estimation using SymQNet neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YTomar79/symqnet-molopt",
    project_urls={
        "Bug Tracker": "https://github.com/YTomar79/symqnet-molopt/issues",
        "Documentation": "https://github.com/YTomar79/symqnet-molopt#readme",
        "Source Code": "https://github.com/YTomar79/symqnet-molopt",
    },
    
    # 🔧 FIX: Updated py_modules to use symqnet_cli instead of cli
    py_modules=[
        "symqnet_cli",      # 🔧 RENAMED: cli.py → symqnet_cli.py
        "architectures", 
        "hamiltonian_parser",
        "measurement_simulator",
        "policy_engine",
        "bootstrap_estimator",
        "utils",
        "add_hamiltonian"
    ],
    
    # Include non-Python files
    include_package_data=True,
    
    # Use proper data_files function
    data_files=get_data_files(),
    
    # Simplified package_data
    package_data={
        "": [
            "*.md",
            "*.txt",
            "LICENSE"
        ],
    },
    
    # 🔧 FIX: Updated entry points to use symqnet_cli module
    entry_points={
        "console_scripts": [
            "symqnet-molopt=symqnet_cli:main",     # 🔧 FIXED: Use symqnet_cli instead of cli
            "symqnet-add=add_hamiltonian:main",    # This one is fine
        ],
    },
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Corrected extras_require with valid specifiers
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.950"
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17"
        ],
        "gpu": [
            "torch>=1.12.0",  # Fixed: Removed invalid +cu118
            "torch-geometric>=2.2.0"
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
            "plotly>=5.0"
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
    
    # Keywords as proper list of strings
    keywords=[
        "quantum-computing",
        "molecular-simulation", 
        "neural-networks",
        "hamiltonian-estimation",
        "symqnet",
        "quantum-chemistry",
        "machine-learning",
        "reinforcement-learning"
    ],
    
    # License
    license="MIT",
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    setup_requires=["setuptools>=45", "wheel"],
)
