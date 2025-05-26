# SymQNet-MolOpt: Universal Quantum Molecular Optimization

Neural network-guided parameter estimation for molecular Hamiltonians with **universal qubit support**.


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Universal Support](https://img.shields.io/badge/qubits-2%2B%20supported-green.svg)](https://github.com/YTomar79/symqnet-molopt)

## 🚀 Installation

pip install symqnet-molopt

 

## 🌍 Universal Qubit Support

**SymQNet-MolOpt now supports any molecular system** with intelligent performance scaling:

| Qubit Count | Performance | Status | Use Case |
|-------------|-------------|---------|----------|
| 2-6 qubits  | 85-95%     | ✅ Good | Small molecules (H₂, LiH) |
| 8-12 qubits | 95-100%    | ⭐ Excellent | Medium molecules (H₂O, NH₃) |
| **10 qubits** | **100%**   | 🎯 **OPTIMAL** | **Maximum accuracy** |
| 14-18 qubits | 75-90%     | ⚠️ Moderate | Large molecules (C₂H₄) |


## 📖 How to Use

### Quick Start
Works with any molecular system!
symqnet-molopt --hamiltonian your_molecule.json --output results.json

 

### 🧪 Example Systems

#### Small Molecule (4 qubits)
symqnet-molopt
--hamiltonian examples/H2_4q.json
--output h2_results.json
--shots 1024

 

#### Medium Molecule (10 qubits)
symqnet-molopt
--hamiltonian examples/H2O_10q.json
--output h2o_results.json
--shots 1024
--n-rollouts 5
--max-steps 50

 

#### Large Molecule (15 qubits)
symqnet-molopt
--hamiltonian examples/large_mol_15q.json
--output large_results.json
--shots 2048
--n-rollouts 8

 

### 🔧 Create Example Files
Generate examples for different system sizes
symqnet-examples

Validate any molecular system
symqnet-add my_molecule.json --validate-only

 

### 📊 Performance Analysis
Get detailed performance analysis
symqnet-molopt
--hamiltonian molecule.json
--output results.json
--show-performance-analysis

 

## 🎛️ Command Reference

### Core Commands
| Command | Description | Qubit Support |
|---------|-------------|---------------|
| `symqnet-molopt` | Run molecular parameter optimization | **Any count ≥2** |
| `symqnet-add` | Add or validate a Hamiltonian file | **Universal** |
| `symqnet-examples` | Generate example Hamiltonians | **Multiple sizes** |
| `symqnet-validate` | Validate installation | **System check** |

### Key Parameters
| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `--hamiltonian` | Input Hamiltonian file (any qubit count) | **Required** |
| `--output` | Output results file | **Required** |
| `--shots` | Shots per measurement | Auto-scales with system size |
| `--n-rollouts` | Number of optimization runs | Auto-scales with system size |
| `--max-steps` | Max steps per rollout | 50 (standard) |
| `--show-performance-analysis` | Show performance expectations | Recommended for new systems |
| `--no-performance-warnings` | Disable performance warnings | For batch processing |

## 🎯 Performance Optimization

### Automatic Parameter Scaling
SymQNet automatically adjusts parameters based on your system size:

4-qubit system: CLI automatically increases shots/rollouts
symqnet-molopt --hamiltonian small_mol_4q.json --output results.json

10-qubit system: Optimal parameters used
symqnet-molopt --hamiltonian optimal_mol_10q.json --output results.json

16-qubit system: CLI recommends higher accuracy settings
symqnet-molopt --hamiltonian large_mol_16q.json --output results.json

 


 

## 📁 Hamiltonian Format

Your molecular system files should specify the qubit count:

{
"format": "openfermion",
"molecule": "H2O",
"n_qubits": 10,
"pauli_terms": [
{"coefficient": -74.943, "pauli_string": "IIIIIIIIII"},
{"coefficient": 0.342, "pauli_string": "IIIIIIIIIZ"}
]
}


**Supported qubit counts:** 2, 3, 4, ..., 10 ⭐, ..., 20, 25, 30+

## 📊 Results Format

{
"symqnet_results": {
"coupling_parameters": [
{
"index": 0,
"mean": 0.213400,
"confidence_interval": [0.208900, 0.217900],
"uncertainty": 0.004500
}
],
"field_parameters": [...],
"total_uncertainty": 0.085600
},
"hamiltonian_info": {
"molecule": "H2O",
"n_qubits": 10,
"performance_optimal": true
},
"performance_analysis": {
"expected_performance": 1.0,
"performance_level": "optimal"
}
}

 

## ⚡ Quick Examples

Validate any system
symqnet-add examples/molecule_8q.json --validate-only

Quick test (any qubit count)
symqnet-molopt --hamiltonian examples/test.json --output output.json --shots 512

High accuracy run (auto-optimized parameters)
symqnet-molopt --hamiltonian molecule.json --output results.json --shots 2048 --n-rollouts 10

Batch processing (no warnings)
symqnet-molopt --hamiltonian large_system.json --output results.json --no-performance-warnings

 

## 🔧 Requirements

- **Python 3.8+**
- **PyTorch 1.12+**
- **NumPy, SciPy, Click**
- **Universal qubit support** (2+ qubits)


## 🌍 Universal Capabilities

### ✅ What Works
- **Any molecular system** with ≥2 qubits
- **Automatic performance scaling** 
- **Intelligent parameter recommendations**
- **Performance transparency**
- **Backward compatibility** with existing 10-qubit workflows


 

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/YTomar79/symqnet-molopt/issues)
- **Help:** Run any command with `--help` for usage info
- **Performance Questions:** Use `--show-performance-analysis` for system-specific guidance

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🎉 Universal SymQNet

**Install once, optimize any molecular system:**

pip install symqnet-molopt

symqnet-molopt --hamiltonian your_molecule.json --output results.json

 ⚛️🚀
