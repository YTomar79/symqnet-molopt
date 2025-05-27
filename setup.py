#!/usr/bin/env python3
"""
Setup script for SymQNet Molecular Optimization CLI
UNIVERSAL VERSION 2.0.0: Now supports any qubit count with optimal performance at 10 qubits

This package provides a command-line interface for molecular Hamiltonian 
parameter estimation using trained SymQNet neural networks with universal qubit support.
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
    version="2.1.0",  
    author="YTomar79",
    author_email="yashm.tomar@gmail.com",
    description="Universal quantum molecular optimization - supports any qubit count with optimal performance at 10 qubits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YTomar79/symqnet-molopt",
    project_urls={
        "Bug Tracker": "https://github.com/YTomar79/symqnet-molopt/issues",
        "Documentation": "https://github.com/YTomar79/symqnet-molopt#readme",
        "Source Code": "https://github.com/YTomar79/symqnet-molopt",
        "PyPI": "https://pypi.org/project/symqnet-molopt/",
        "Examples": "https://github.com/YTomar79/symqnet-molopt/tree/main/examples"
    },
    
    # 🌍 UNIVERSAL: All core modules including new universal components
    py_modules=[
        "symqnet_cli",              # Main CLI (updated for universal support)
        "universal_wrapper",        # 🆕 Universal qubit wrapper
        "performance_estimator",    # 🆕 Performance analysis and warnings
        "architectures",            # Core neural network architectures
        "hamiltonian_parser",       # Updated for universal qubit support
        "measurement_simulator",    # Quantum measurement simulation
        "policy_engine",            # SymQNet policy engine
        "bootstrap_estimator",      # Uncertainty quantification
        "utils",                    # Updated utilities with universal support
        "add_hamiltonian"           # Hamiltonian file management
    ],
    
    # Include non-Python files
    include_package_data=True,
    
    # Use proper data_files function
    data_files=get_data_files(),
    
    # Package data for included files
    package_data={
        "": [
            "*.md",
            "*.txt", 
            "*.json",
            "LICENSE",
            "MANIFEST.in"
        ],
    },
    
    # 🚀 ENTRY POINTS: Universal CLI commands
    entry_points={
        "console_scripts": [
            "symqnet-molopt=symqnet_cli:main",         # Universal molecular optimization
            "symqnet-add=add_hamiltonian:main",        # Add/validate Hamiltonians
        ],
    },
    
    # Dependencies for universal support
    install_requires=read_requirements(),
    
    # Enhanced extras for universal capabilities
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
            "torch>=1.12.0",
            "torch-geometric>=2.2.0"
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
            "plotly>=5.0"
        ],
        "analysis": [
            "seaborn>=0.11.0",
            "scikit-learn>=1.1.0",
            "networkx>=2.8.0"
        ]
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Enhanced classification for universal support
    classifiers=[
        "Development Status :: 5 - Production/Stable",  # Upgraded from Beta
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Natural Language :: English",
    ],
    
    # Enhanced keywords for universal support
    keywords=[
        "quantum-computing",
        "molecular-simulation", 
        "neural-networks",
        "hamiltonian-estimation",
        "symqnet",
        "quantum-chemistry",
        "machine-learning",
        "reinforcement-learning",
        "universal-support",      # 🆕
        "scalable-qubits",       # 🆕
        "molecular-optimization", # 🆕
        "parameter-estimation",   # 🆕
        "uncertainty-quantification",
        "performance-analysis"    # 🆕
    ],
    
    # License
    license="MIT",
    
    # Additional metadata for universal version
    zip_safe=False,
    platforms=["any"],
    setup_requires=["setuptools>=45", "wheel"],
    
    # Command line interface documentation
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal Python (uses compiled dependencies)
        }
    },
    
    # Project health indicators
    project_health={
        "maintained": True,
        "stable_api": True,
        "documented": True,
        "tested": True
    }
)
