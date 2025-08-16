#!/usr/bin/env python3
"""
Robust setup.py for symqnet-molopt
Automatically detects whether code is packaged as a package directory
(symqnet_molopt/) or as top-level single-file modules and configures
packages/py_modules and console entry points accordingly.
"""
from setuptools import setup, find_packages
from pathlib import Path
import glob
import sys

ROOT = Path(__file__).parent

# read long description if present
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

def read_requirements():
    req_file = ROOT / "requirements.txt"
    if req_file.exists():
        reqs = []
        for line in req_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
        return reqs
    # conservative fallback
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
    if (ROOT / "examples").exists():
        files = [str(p) for p in (ROOT / "examples").glob("*") if p.is_file()]
        if files:
            data_files.append(("examples", files))
    if (ROOT / "models").exists():
        files = [str(p) for p in (ROOT / "models").glob("*") if p.is_file()]
        if files:
            data_files.append(("models", files))
    if (ROOT / "scripts").exists():
        files = [str(p) for p in (ROOT / "scripts").glob("*") if p.is_file()]
        if files:
            data_files.append(("scripts", files))
    return data_files

# ---------- auto-detect layout ----------
pkg_dir = ROOT / "symqnet_molopt"
has_package = pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists()

# find top-level python modules (excluding setup.py)
top_py = [p for p in ROOT.glob("*.py") if p.name not in ("setup.py",)]
top_modules = [p.stem for p in top_py]

# Default placeholders
packages = []
py_modules = []
console_entry = None

if has_package:
    # Package layout: install the package and use package entry points
    packages = find_packages(include=["symqnet_molopt", "symqnet_molopt.*"])
    # prefer to point console entry to symqnet_molopt.symqnet_cli:main if that module exists
    if (pkg_dir / "symqnet_cli.py").exists() or (pkg_dir / "cli.py").exists():
        # attempt to point to symqnet_cli
        console_entry = [
            "symqnet-molopt=symqnet_molopt.symqnet_cli:main",
            "symqnet-add=symqnet_molopt.add_hamiltonian:main",
        ]
    else:
        # fallback: point to package-level entry if __init__ defines main() or expose
        console_entry = [
            "symqnet-molopt=symqnet_molopt.__main__:main" if (pkg_dir / "__main__.py").exists() else "symqnet-molopt=symqnet_molopt:main",
            "symqnet-add=symqnet_molopt.add_hamiltonian:main",
        ]
else:
    # No package directory. Look for top-level module named symqnet_cli.py
    if "symqnet_cli" in top_modules:
        py_modules = top_modules  # install any top-level .py present
        console_entry = [
            "symqnet-molopt=symqnet_cli:main",
            "symqnet-add=add_hamiltonian:main" if "add_hamiltonian" in top_modules else "symqnet-add=add_hamiltonian:main",
        ]
    else:
        # Last resort: install any top-level modules, but warn user
        if top_modules:
            py_modules = top_modules
            console_entry = []
        else:
            print("ERROR: No package directory 'symqnet_molopt/' or top-level modules found.", file=sys.stderr)
            print("Please ensure your Python sources are placed in symqnet_molopt/ or as top-level .py files.", file=sys.stderr)
            raise SystemExit(1)

# ---------- common metadata ----------
setup(
    name="symqnet-molopt",
    version="3.0.3",
    author="YTomar79",
    author_email="yashm.tomar@gmail.com",
    description="The universal quantum molecular optimization - supports any qubit count with optimal performance at 10 qubits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YTomar79/symqnet-molopt",
    packages=packages if packages else None,
    py_modules=py_modules if py_modules else None,
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "LICENSE", "MANIFEST.in"],
    },
    data_files=get_data_files(),
    entry_points={"console_scripts": console_entry} if console_entry else {},
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    zip_safe=False,
    platforms=["any"],
    setup_requires=["setuptools>=45", "wheel"],
)
