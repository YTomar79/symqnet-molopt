# SymQNet-MolOpt: Hamiltonian Parameter Estimation

SymQNet-MolOpt provides efficient, uncertainty-aware estimation of Hamiltonian parameters for 1D and 2D molecular models and ultimately much more efficient molecular optimization.
It is designed to use a Bayseian estimator, guided by a reinforcement-learning (RL) agent that acts as a scientist, to estimate Hamiltonians optimally that can be used for methods such as VQE-based molecular optimization and bring bigger, downstream optimization gains. 

---

## Installation

```bash
pip install SymQNet-MolOpt
```

---

## Usage

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

## Examples

**Water Molecule (H₂O, 10 qubits)**

```bash
SymQNet-MolOpt --hamiltonian examples/H2O_10q.json --output h2o_results.json --shots 1024 --n-rollouts 5 --max-steps 50
```

**Ising Chain (12 qubits)**

```bash
SymQNet-MolOpt --hamiltonian examples/ising_12q.json --output ising_results.json --shots 1024
```
(You need to create your own JSON Hamiltonian.)

---

## Input Format

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

## Output Format

Results include estimated parameters with uncertainties plus experiment/config metadata:

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
    "total_uncertainty": 0.0856,
    "avg_measurements_used": 512.0,
    "confidence_level": 0.95,
    "n_rollouts": 5
  },
  "hamiltonian_info": {
    "molecule": "H2O",
    "n_qubits": 10,
    "n_pauli_terms": 512,
    "format": "openfermion",
    "optimal_qubits": 10,
    "performance_optimal": true
  },
  "experimental_config": {
    "shots": 1024,
    "max_steps": 50,
    "n_rollouts": 5,
    "confidence": 0.95,
    "device": "cpu",
    "seed": 42
  },
  "metadata": {
    "generated_by": "Universal SymQNet Molecular Optimization CLI",
    "version": "2.0.12",
    "model_constraint": "Trained optimally for 10 qubits, supports any qubit count",
    "timestamp": "2024-01-01T00:00:00",
    "parameter_count": {
      "coupling": 9,
      "field": 10,
      "total": 19
    },
    "parameter_extraction_fixed": true
  },
  "performance_analysis": {
    "expected_performance": 1.0,
    "performance_level": "optimal",
    "optimal_qubits": 10,
    "universal_symqnet_version": "2.0.0"
  },
  "universal_wrapper": {
    "original_qubits": 12,
    "normalized_to": 10,
    "expected_performance": 0.85,
    "normalization_applied": true,
    "optimal_at": 10,
    "fixed_parameter_extraction": true
  },
  "validation": {
    "has_ground_truth": false
  }
}
```

---


## Checkpoint Schema

A supported checkpoint is a dict-based bundle with these required fields:

* `model_state_dict`: PyTorch state dict for the policy network.
* `meta_dim`: Total metadata vector length.
* `shots_encoding`: `null` or a dict with a `type` field indicating shot-conditioning.
* `n_qubits`: Qubit count the model was trained on.
* `M_evo`: Number of discrete evolution times.
* `rollout_steps`: Temporal context length (T).

Optional, but recommended, format guards:

* `checkpoint_format`: must be `symqnet-ppo-v2`.
* `checkpoint_version`: must be `1`.

## Metadata Vector Layout

The metadata vector follows `MetadataLayout.from_problem(n_qubits, M_evo)` and is ordered as:

1. **Qubit selection (one-hot):** `n_qubits` slots.
2. **Measurement basis (one-hot):** 3 slots for X/Y/Z.
3. **Evolution time (one-hot):** `M_evo` slots.
4. **Shots slot:** single slot used if `shots_encoding` is enabled.
5. **Posterior mean (θ):** `2 * n_qubits - 1` slots.
6. **Posterior covariance features:** `theta_dim + 8` slots.
7. **Posterior Fisher diagonal:** `theta_dim` slots.

The total metadata dimension is `n_qubits + 3 + M_evo + 1 + theta_dim + (theta_dim + 8) + theta_dim`.

## Shot Conditioning

When shot conditioning is enabled (`shots_encoding` is a dict), the CLI and policy engine
normalize the user-provided shot budget into the shots slot using:

```
log1p(shots) / log1p(1_000_000)
```

If `shots_encoding` is `null`, the shots slot remains zero and the model ignores shot conditioning.

## Outputs (High-Level)

The CLI emits a JSON bundle with:

* `symqnet_results` (parameter estimates + uncertainty summaries)
* `hamiltonian_info`
* `experimental_config`
* `metadata`
* Optional `performance_analysis`, `universal_wrapper`, and `validation` blocks

See `README.md` for a full output example.

---

## Requirements

* Python 3.8+
* PyTorch 1.12+
* NumPy, SciPy, Click

