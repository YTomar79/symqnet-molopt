#!/usr/bin/env python3
"""
setup.py for SymQNet-MolOpt (v3.0.21) - FIXED MODEL SHIPPING
"""
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent.resolve()  

def read_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        return [
            "torch>=1.12.0",
            "numpy>=1.21.0", 
            "scipy>=1.9.0",
            "click>=8.0.0",
        ]
    lines = [
        ln.strip()
        for ln in req.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    return lines

def collect_data_files(subdir: str, pattern: str = "*"):
    """Return a [(subdir, [files…])] entry if files exist, else []."""
    base = ROOT / subdir
    if not base.is_dir():
        return []
    rel_paths = [str(p.relative_to(ROOT)) for p in base.glob(pattern) if p.is_file()]
    return [(subdir, rel_paths)] if rel_paths else []

# Package / module layout detection
PKG_DIR = ROOT / "symqnet_molopt"
HAS_PKG = PKG_DIR.is_dir() and (PKG_DIR / "__init__.py").exists()

if HAS_PKG:
    # proper package layout
    packages = find_packages(include=["symqnet_molopt", "symqnet_molopt.*"])
    py_modules = []
    console_scripts = [
        "symqnet-molopt=symqnet_molopt.symqnet_cli:main",
        "symqnet-add=symqnet_molopt.add_hamiltonian:main",
    ]
    # Package data for package layout
    model_files = []
    model_subdir = PKG_DIR / "models"
    if model_subdir.is_dir():
        model_files = [
            str(p.relative_to(PKG_DIR))
            for p in model_subdir.glob("*.pth")
        ]
    package_data = {
        "symqnet_molopt": model_files + ["*.md", "*.json", "LICENSE*"],
    }
    data_files = []
else:
    #  flat-layout with proper model shipping
    packages = []
    py_modules = [
        "symqnet_cli",
        "add_hamiltonian", 
        "architectures",
        "bootstrap_estimator",
        "hamiltonian_parser",
        "measurement_simulator",
        "performance_estimator",
        "policy_engine", 
        "universal_wrapper",
        "utils",
    ]
    console_scripts = [
        "symqnet-molopt=symqnet_cli:main",
        "symqnet-add=add_hamiltonian:main",
    ]
    package_data = {}
    
    # Ship models via data_files for flat layout
    data_files = []
    models_dir = ROOT / "models"
    if models_dir.is_dir():
        model_file_paths = [str(p.relative_to(ROOT)) for p in models_dir.glob("*.pth")]
        if model_file_paths:
            data_files.append(("models", model_file_paths))
            print(f" Found {len(model_file_paths)} model files to include: {model_file_paths}")
    
    # Add examples
    data_files.extend(collect_data_files("examples", "*.json"))

# Long description
README = ROOT / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="symqnet-molopt",
    version="3.0.21",
    description="SymQNet Molecular Optimization via Hamiltonian Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="YTomar79",
    author_email="yashm.tomar@gmail.com",
    url="https://github.com/YTomar79/symqnet-molopt",
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    # Layout
    packages=packages,
    py_modules=py_modules,
    entry_points={"console_scripts": console_scripts},
    # Resource files  
    include_package_data=True,
    package_data=package_data,
    data_files=data_files,  
    # Dependencies
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0", 
            "isort>=5.0",
            "mypy>=0.950",
        ],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0", "myst-parser>=0.17"],
        "gpu": ["torch>=1.12.0", "torch-geometric>=2.2.0"], 
        "analysis": ["pandas>=1.4.0", "seaborn>=0.11.0", "scikit-learn>=1.1.0"],
    },
    zip_safe=False,
)
