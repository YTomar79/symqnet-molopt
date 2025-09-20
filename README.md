# SymQNet-MolOpt: Hamiltonian Parameter Estimation

SymQNet-MolOpt provides efficient, uncertainty-aware estimation of Hamiltonian parameters for molecular and spin-system models.
It is designed for sample-efficient optimization and reports confidence intervals for each parameter.

---

## 🚀 Installation

```bash
pip install SymQNet-MolOpt
```

---

## 📚 Usage

### Core Command

```bash
SymQNet-MolOpt --hamiltonian input.json --output results.json
```

**Arguments:**

* `--hamiltonian`: Path to a JSON Hamiltonian (OpenFermion-like format).
* `--output`: File to save results (JSON).
* `--shots`: Number of measurement shots (default auto-scales).
* `--n-rollouts`: Number of independent rollouts (default: 5).
* `--max-steps`: Max optimization steps per rollout (default: 50).

---

## 🤬 Examples

**Small Molecule (H₂, 4 qubits)**

```bash
SymQNet-MolOpt --hamiltonian examples/H2_4q.json --output h2_results.json --shots 1024
```

**Water Molecule (H₂O, 10 qubits)**

```bash
SymQNet-MolOpt --hamiltonian examples/H2O_10q.json --output h2o_results.json --shots 1024 --n-rollouts 5 --max-steps 50
```

**Ising Chain (12 qubits)**

```bash
SymQNet-MolOpt --hamiltonian examples/ising_12q.json --output ising_results.json --shots 1024
```

---

## 📁 Input Format

Hamiltonians are specified in JSON:

```json
{
  "format": "openfermion",
  "system": "H2O",
  "n_qubits": 10,
  "pauli_terms": [
    {"coefficient": -74.943, "pauli_string": "IIIIIIIIII"},
    {"coefficient": 0.342, "pauli_string": "IIIIIIIIIZ"}
  ]
}
```

---

## 📊 Output Format

Results include estimated parameters with uncertainties:

```json
{
  "symqnet_results": {
    "coupling_parameters": [
      {
        "index": 0,
        "mean": 0.2134,
        "confidence_interval": [0.2089, 0.2179],
        "uncertainty": 0.0045
      }
    ],
    "field_parameters": [...],
    "total_uncertainty": 0.0856
  },
  "hamiltonian_info": {
    "system": "H2O",
    "n_qubits": 10
  }
}
```

---

## 🔧 Requirements

* Python 3.8+
* PyTorch 1.12+
* NumPy, SciPy, Click

---

## 🎯 Summary

* Works with molecules, Ising, and Heisenberg systems.
* Estimates Hamiltonian parameters efficiently.
* Reports 95% confidence intervals for uncertainty quantification.
