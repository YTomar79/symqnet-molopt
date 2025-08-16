#!/usr/bin/env python3
"""
Fixed setup.py for symqnet-molopt
- Fixes Path(__file__) typo
- Excludes unwanted top-level directories from package discovery
- Removes deprecated license classifier
- Uses explicit package discovery to avoid flat-layout issues
"""
from setuptools import setup, find_packages
from pathlib import Path
import glob
import sys

ROOT = Path(__file__).parent.resolve()  # Fixed: was Path(file)



# long description
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

py_modules = [
    "symqnet_cli",          # CLI entry point   (symqnet-molopt)
    "add_hamiltonian",      # secondary CLI     (symqnet-add)
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
    """
    Return a list suitable for setuptools' data_files argument:
      [("target_dir", ["rel/path/one", "rel/path/two"]), ...]
    All file paths are made relative to ROOT and converted to POSIX style.
    """
    ret = []
    for sub in ("examples", "models", "scripts", "outputs"):  # Added outputs here
        d = ROOT / sub
        if d.exists() and d.is_dir():
            files = [f for f in sorted(d.iterdir()) if f.is_file()]
            if not files:
                continue
            # make paths relative to ROOT and POSIX-style (no absolute paths)
            rel_paths = [str(p.relative_to(ROOT).as_posix()) for p in files]
            ret.append((sub, rel_paths))
    return ret

# detect package layout
pkg_dir = ROOT / "symqnet_molopt"
has_package = pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists()

# decide entry points (prefer package namespace)
if has_package:
    console_entry = [
        "symqnet-molopt=symqnet_molopt.symqnet_cli:main",
        "symqnet-add=symqnet_molopt.add_hamiltonian:main",
    ]
else:
    # fallback to top-level modules if present
    top_py = [p.stem for p in ROOT.glob("*.py") if p.name not in ("setup.py",)]
    if "symqnet_cli" in top_py:
        console_entry = [
            "symqnet-molopt=symqnet_cli:main",
            "symqnet-add=add_hamiltonian:main" if "add_hamiltonian" in top_py else "symqnet-add=add_hamiltonian:main",
        ]
    else:
        console_entry = []

# Package metadata
setup(
    name="symqnet-molopt",
    version="3.0.4",
    description="The universal quantum molecular optimization - supports any qubit count with optimal performance at 10 qubits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YTomar79/symqnet-molopt",
    author="YTomar79",
    author_email="yashm.tomar@gmail.com",
    license="MIT",
    license_files=("LICENSE",),
    # FIXED: Use explicit exclusion to avoid flat-layout conflicts
    py_modules=py_modules,
    entry_points={"console_scripts": console_scripts},

    include_package_data=True,
    package_data={
        # include markdown/json/LICENSE etc.
        "": ["*.md", "*.txt", "*.json", "LICENSE", "MANIFEST.in"],

    },
    # data_files must use relative paths (converted by get_data_files)
    data_files=get_data_files(),
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
        # Removed deprecated license classifier
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    zip_safe=False,
    platforms=["any"],
    setup_requires=["setuptools>=45", "wheel"],
)
