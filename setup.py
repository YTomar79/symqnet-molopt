#!/usr/bin/env python3
"""
Fixed setup.py for symqnet-molopt

Key fixes:
 - Use packages=find_packages(...) instead of py_modules so the package directory
   (symqnet_molopt/) is installed and importable.
 - Ensure console entry points reference the package namespace.
 - Keep include_package_data and package_data to bundle non-Python files.
 - Read requirements from requirements.txt when present.
"""
from setuptools import setup, find_packages
from pathlib import Path
import glob

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

def read_requirements():
    requirements_file = this_directory / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, "r") as f:
            reqs = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                reqs.append(line)
            return reqs
    # fallback conservative set
    return [
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "click>=8.0.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "pandas>=1.4.0",
        "gym>=0.26.0",
    ]

def get_data_files():
    data_files = []
    if Path("examples").exists():
        example_files = glob.glob("examples/*")
        if example_files:
            data_files.append(("examples", example_files))
    if Path("models").exists():
        model_files = glob.glob("models/*")
        if model_files:
            data_files.append(("models", model_files))
    if Path("scripts").exists():
        script_files = glob.glob("scripts/*")
        if script_files:
            data_files.append(("scripts", script_files))
    return data_files

setup(
    name="symqnet-molopt",
    version="3.0.0",
    description="The universal quantum molecular optimization - supports any qubit count with optimal performance at 10 qubits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YTomar79/symqnet-molopt",
    author="YTomar79",
    author_email="yashm.tomar@gmail.com",
    license="MIT",

    # === FIX: detect and install package directory (symqnet_molopt) ===
    # Use find_packages to pick up symqnet_molopt and subpackages automatically.
    packages=find_packages(include=["symqnet_molopt", "symqnet_molopt.*"]),

    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "LICENSE", "MANIFEST.in"],
    },
    data_files=get_data_files(),

    # console entry points should reference package namespace
    entry_points={
        "console_scripts": [
            "symqnet-molopt=symqnet_molopt.symqnet_cli:main",
            "symqnet-add=symqnet_molopt.add_hamiltonian:main",
        ]
    },

    install_requires=read_requirements(),

    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov>=2.0", "black>=22.0", "flake8>=4.0", "isort>=5.0", "mypy>=0.950"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0", "myst-parser>=0.17"],
        "gpu": ["torch>=1.12.0", "torch-geometric>=2.2.0"],
        "jupyter": ["jupyter>=1.0", "ipywidgets>=7.0", "plotly>=5.0"],
        "analysis": ["seaborn>=0.11.0", "scikit-learn>=1.1.0", "networkx>=2.8.0"],
    },

    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],

    zip_safe=False,
    platforms=["any"],
    setup_requires=["setuptools>=45", "wheel"],
)
