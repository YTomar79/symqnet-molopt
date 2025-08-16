#!/usr/bin/env python3
"""
setup.py for SymQNet-MolOpt (v3.0.6)

Key fixes
──────────
1. Uses Path(__file__) (the earlier draft had Path(**file**)).
2. Embeds the two weight files (*.pth) in the wheel so they are always
   available via importlib.resources — no matter where the user runs.
3. Works whether the project is laid out as a real package
   (symqnet_molopt/ …) or as a flat collection of modules at repo root.
"""

from pathlib import Path
from setuptools import setup, find_packages


ROOT = Path(__file__).parent.resolve()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        # minimal hard requirements
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


# ──────────────────────────────────────────────────────────────────────────────
# Package / module layout detection
# ──────────────────────────────────────────────────────────────────────────────
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
else:
    # flat-layout fallback
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

# ──────────────────────────────────────────────────────────────────────────────
# ✨  Ship the pretrained weights inside the package  ✨
#     Wheel users will always get them, importlib.resources works.
# ──────────────────────────────────────────────────────────────────────────────
model_files = []
model_pkg_root = PKG_DIR if HAS_PKG else ROOT  # relative to chosen package
model_subdir = model_pkg_root / "models"
if model_subdir.is_dir():
    model_files = [
        str(p.relative_to(model_pkg_root))  # path *inside* the package
        for p in model_subdir.glob("*.pth")
    ]

# Extra non-code resources you want to bundle (optional)
extra_data_patterns = ["*.md", "*.json", "LICENSE*"]

package_data = {
    # If we have a real package, put resources there; otherwise nothing.
    "symqnet_molopt": model_files + extra_data_patterns if HAS_PKG else [],
}

# Top-level examples furnished as install-time data_files (wheel-safe)
data_files = collect_data_files("examples", "*.json")

# ──────────────────────────────────────────────────────────────────────────────
# Long description
# ──────────────────────────────────────────────────────────────────────────────
README = ROOT / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

# ──────────────────────────────────────────────────────────────────────────────
# setup()
# ──────────────────────────────────────────────────────────────────────────────
setup(
    name="symqnet-molopt",
    version="3.0.9",
    description="Universal SymQNet Molecular Optimisation (supports any qubit count)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="YTomar79",
    author_email="yashm.tomar@gmail.com",
    url="https://github.com/YTomar79/symqnet-molopt",
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
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
