# SymQNet Architecture Contract

This document captures the model-architecture contract for SymQNet-MolOpt checkpoints and
runtime metadata. It is intended as a stable reference when producing or consuming checkpoints
with the CLI.

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
