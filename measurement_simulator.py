"""
Measurement Simulator for symbolic quantum measurements
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MeasurementSimulator:
    """Simulates quantum measurements for molecular Hamiltonians."""

    DEFAULT_MAX_MATRIX_MB = 512
    DEFAULT_MAX_HILBERT_DIM = 2 ** 14
    WARN_HILBERT_DIM = 2 ** 12
    
    def __init__(self, hamiltonian_data: Dict[str, Any], shots: int, 
                 device: torch.device, noise_prob: float = 0.02):
        self.hamiltonian_data = hamiltonian_data
        self.shots = shots
        self.device = device
        self.noise_prob = noise_prob
        self.n_qubits = hamiltonian_data["n_qubits"]

        capability_limits = hamiltonian_data.get("capability_limits", {})
        max_matrix_mb = capability_limits.get("max_matrix_mb", self.DEFAULT_MAX_MATRIX_MB)
        max_hilbert_dim = capability_limits.get("max_hilbert_dim", self.DEFAULT_MAX_HILBERT_DIM)
        self.validate_capability(self.n_qubits, max_matrix_mb, max_hilbert_dim)

        dim = 2 ** self.n_qubits
        memory_mb = self._estimate_matrix_mb(dim)
        if dim > self.WARN_HILBERT_DIM:
            logger.warning(
                "Large Hilbert space (%s). Measurements may be slow or memory-intensive.",
                dim
            )
        if memory_mb > 100:  # Warn for >100MB
            logger.warning(f"Large Hamiltonian matrix: {memory_mb:.1f} MB")
        
        # Pauli matrices
        self.pauli_matrices = self._get_pauli_matrices()
        
        # Build full Hamiltonian matrix
        self.hamiltonian_matrix = self._build_hamiltonian_matrix()
        self._hamiltonian_eigvals, self._hamiltonian_eigvecs = np.linalg.eigh(
            self.hamiltonian_matrix
        )
        
        # fix was to match policy_engine.py time range
        n_time_points = max(2, min(10, self.n_qubits))
        self.evolution_times = np.linspace(0.1, 1.0, n_time_points)  # Was 0.1 to 2.0
        
        # Precompute evolution operators
        self._precompute_evolution_operators()
        
        logger.info(f"Initialized simulator for {self.n_qubits}-qubit system")

    @classmethod
    def _estimate_matrix_mb(cls, dim: int) -> float:
        """Estimate Hamiltonian matrix memory (Complex128)."""
        return (dim * dim * 16) / (1024 * 1024)

    @classmethod
    def check_capability(
        cls,
        n_qubits: int,
        max_matrix_mb: Optional[float] = None,
        max_hilbert_dim: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check whether simulator can handle the requested qubit count."""
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            return False, f"Invalid qubit count: {n_qubits}"

        max_matrix_mb = cls.DEFAULT_MAX_MATRIX_MB if max_matrix_mb is None else max_matrix_mb
        max_hilbert_dim = cls.DEFAULT_MAX_HILBERT_DIM if max_hilbert_dim is None else max_hilbert_dim

        dim = 2 ** n_qubits
        if dim > max_hilbert_dim:
            return (
                False,
                f"Hilbert space dimension {dim} exceeds limit {max_hilbert_dim}."
            )
        memory_mb = cls._estimate_matrix_mb(dim)
        if memory_mb > max_matrix_mb:
            return (
                False,
                f"Hamiltonian matrix requires {memory_mb:.1f} MB, exceeds limit {max_matrix_mb:.1f} MB."
            )
        return True, None

    @classmethod
    def validate_capability(
        cls,
        n_qubits: int,
        max_matrix_mb: Optional[float] = None,
        max_hilbert_dim: Optional[int] = None,
    ) -> None:
        """Raise if simulator cannot handle the requested qubit count."""
        supported, reason = cls.check_capability(n_qubits, max_matrix_mb, max_hilbert_dim)
        if not supported:
            raise ValueError(
                f"MeasurementSimulator cannot support {n_qubits} qubits. {reason}"
            )
    
    def _get_pauli_matrices(self) -> Dict[str, np.ndarray]:
        """Get Pauli matrices."""
        return {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
    
    def _build_hamiltonian_matrix(self) -> np.ndarray:
        """Build the full Hamiltonian matrix from Pauli terms."""
        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        for term in self.hamiltonian_data['pauli_terms']:
            coeff = term['coefficient']
            pauli_indices = term['pauli_indices']
            
            # Build tensor product of Pauli operators
            term_matrix = self._build_pauli_operator(pauli_indices)
            H += coeff * term_matrix
        
        return H
    
    def _build_pauli_operator(self, pauli_indices: List[Tuple[int, str]]) -> np.ndarray:
        """Build full Pauli operator from indices."""
        # Start with identity on all qubits
        operators = ['I'] * self.n_qubits
        
        # Set specified Pauli operators
        for qubit_idx, pauli in pauli_indices:
            if not 0 <= qubit_idx < self.n_qubits:
                raise ValueError(f"Invalid qubit index {qubit_idx} for {self.n_qubits}-qubit system")
            if pauli not in ['I', 'X', 'Y', 'Z']:
                raise ValueError(f"Invalid Pauli operator '{pauli}'")
            operators[qubit_idx] = pauli
        
        # Build tensor product
        result = self.pauli_matrices[operators[0]]
        for i in range(1, self.n_qubits):
            result = np.kron(result, self.pauli_matrices[operators[i]])
        
        return result
    
    def _precompute_evolution_operators(self):
        """Precompute time evolution operators U(t) = exp(-iHt)."""
        self.evolution_operators = {}
        
        for t in self.evolution_times:
            phases = np.exp(-1j * self._hamiltonian_eigvals * t)
            U = (self._hamiltonian_eigvecs * phases) @ self._hamiltonian_eigvecs.conj().T
            self.evolution_operators[t] = U
        
        logger.debug(f"Precomputed {len(self.evolution_operators)} evolution operators")
    
    def get_initial_measurement(self) -> np.ndarray:
        """Get initial measurement (ground state)."""
        # Start with ground state |000...........0>
        psi0 = np.zeros(2 ** self.n_qubits, dtype=complex)
        psi0[0] = 1.0
        
        # Measure all qubits in Z basis (individual measurements)
        expectations = self._measure_state_individual(psi0, ['Z'] * self.n_qubits)
        return self._add_shot_noise(expectations)
    
    def get_thermal_initial_measurement(self, temperature: float = 0.1) -> np.ndarray:
        """Get thermal initial state for more realistic simulation."""
        
        # Compute thermal state: ρ = exp(-βH) / Tr[exp(-βH)]
        beta = 1.0 / max(temperature, 1e-6)  # Avoid division by zero
        
        # Compute thermal populations
        thermal_pops = np.exp(
            -beta * (self._hamiltonian_eigvals - self._hamiltonian_eigvals.min())
        )
        thermal_pops /= np.sum(thermal_pops)
        
        # Sample from thermal distribution
        state_idx = np.random.choice(len(self._hamiltonian_eigvals), p=thermal_pops)
        psi_thermal = self._hamiltonian_eigvecs[:, state_idx]
        
        return self._measure_state_individual(psi_thermal, ['Z'] * self.n_qubits)
    
    def execute_measurement(self, qubit_indices: List[int], 
                          pauli_operators: List[str], 
                          evolution_time: float) -> Dict[str, Any]:
        """
        Execute a symbolic measurement.
        
        Args:
            qubit_indices: Which qubits to measure
            pauli_operators: Pauli operators for each qubit
            evolution_time: Time evolution before measurement
        
        Returns:
            Dictionary with measurement results
        """
        
        if not qubit_indices or not pauli_operators:
            raise ValueError("Must specify at least one qubit and operator")
        
        if len(qubit_indices) != len(pauli_operators):
            raise ValueError(
                f"Mismatch: {len(qubit_indices)} qubits but {len(pauli_operators)} operators"
            )
        
        for qubit_idx in qubit_indices:
            if not 0 <= qubit_idx < self.n_qubits:
                raise ValueError(f"Invalid qubit index {qubit_idx} for {self.n_qubits}-qubit system")
        
        for op in pauli_operators:
            if op not in ['X', 'Y', 'Z']:
                raise ValueError(f"Invalid Pauli operator '{op}'. Must be X, Y, or Z")
        
        if evolution_time < 0:
            raise ValueError(f"Evolution time must be non-negative, got {evolution_time}")
        
        # Get evolution operator
        if evolution_time not in self.evolution_operators:
            # Find closest precomputed time
            closest_time = min(self.evolution_times, 
                             key=lambda t: abs(t - evolution_time))
            U = self.evolution_operators[closest_time]
            if abs(evolution_time - closest_time) > 0.1:
                logger.warning(f"Using closest time {closest_time} for requested {evolution_time}")
        else:
            U = self.evolution_operators[evolution_time]
        
        # Start with ground state
        psi0 = np.zeros(2 ** self.n_qubits, dtype=complex)
        psi0[0] = 1.0
        
        # Apply time evolution
        psi_t = U @ psi0
        
        if len(qubit_indices) == 1:
            # Single-qubit measurement
            measurement_ops = ['I'] * self.n_qubits
            measurement_ops[qubit_indices[0]] = pauli_operators[0]
            expectation_values = self._measure_state_individual(psi_t, measurement_ops)
        else:
            # Joint measurement
            pauli_indices = list(zip(qubit_indices, pauli_operators))
            joint_expectation = self._measure_state_joint(psi_t, pauli_indices)
            # Return as array with joint result in first position, rest unmeasured
            expectation_values = np.full(self.n_qubits, np.nan)
            expectation_values[0] = joint_expectation
        
        # Add shot noise
        noisy_expectations = self._add_shot_noise(expectation_values)
        measurement_mask = ~np.isnan(expectation_values)
        
        return {
            'qubit_indices': qubit_indices,
            'pauli_operators': pauli_operators,
            'evolution_time': evolution_time,
            'expectation_values': noisy_expectations,
            'ideal_expectation_values': expectation_values,
            'measurement_mask': measurement_mask,
            'shots_used': self.shots,
            'measurement_type': 'joint' if len(qubit_indices) > 1 else 'individual'
        }
    
    def _measure_state_individual(self, psi: np.ndarray, 
                                measurement_ops: List[str]) -> np.ndarray:
        """Measure quantum state with individual Pauli operators."""
        
        expectations = []
        
        for i, op in enumerate(measurement_ops):
            if op == 'I':
                expectations.append(np.nan)  # Unmeasured qubit
                continue
            
            # Build single-qubit measurement operator
            pauli_indices = [(i, op)]
            M = self._build_pauli_operator(pauli_indices)
            
            # Compute expectation value
            exp_val = np.real(np.conj(psi) @ M @ psi)
            expectations.append(exp_val)
        
        return np.array(expectations)
    
    def _measure_state_joint(self, psi: np.ndarray, 
                           pauli_indices: List[Tuple[int, str]]) -> float:
        """Measure quantum state with joint Pauli operator (tensor product)."""
        
        if not pauli_indices:
            return 1.0  # All identity
        
        # Build joint measurement operator (tensor product)
        M = self._build_pauli_operator(pauli_indices)
        
        # Compute joint expectation value
        joint_exp_val = np.real(np.conj(psi) @ M @ psi)
        
        return joint_exp_val
    
    def _add_shot_noise(self, expectations: np.ndarray) -> np.ndarray:
        """Add finite shot noise to expectation values."""
        
        noisy_expectations = np.full_like(expectations, np.nan, dtype=float)
        
        for i, exp_val in enumerate(expectations):
            # Skip if measurement is not taken
            if np.isnan(exp_val):
                continue
            
            # Clamp expectation value to valid range
            exp_val = np.clip(exp_val, -1.0, 1.0)
            
            # Convert expectation value to probability
            p_plus = (1 + exp_val) / 2
            p_plus = np.clip(p_plus, 0.0, 1.0)  # Ensure valid probability
            
            # Sample measurement outcomes
            outcomes = np.random.choice([-1, 1], size=self.shots, 
                                      p=[1-p_plus, p_plus])
            
            # Add bit flip noise
            flip_mask = np.random.random(self.shots) < self.noise_prob
            outcomes[flip_mask] *= -1
            
            # Compute noisy expectation value
            noisy_expectations[i] = np.mean(outcomes)
        
        return noisy_expectations
    
    def get_symbolic_measurements(self) -> List[Dict[str, Any]]:
        """Get list of available symbolic measurements."""
        
        measurements = []
        
        # Single-qubit measurements
        for qubit in range(self.n_qubits):
            for pauli in ['X', 'Y', 'Z']:
                measurements.append({
                    'type': 'single_qubit',
                    'qubits': [qubit],
                    'operators': [pauli],
                    'description': f'{pauli}_{qubit}'
                })
        
        # Two-qubit measurements (joint)
        for q1 in range(self.n_qubits):
            for q2 in range(q1 + 1, self.n_qubits):
                for p1 in ['X', 'Y', 'Z']:
                    for p2 in ['X', 'Y', 'Z']:
                        measurements.append({
                            'type': 'two_qubit',
                            'qubits': [q1, q2],
                            'operators': [p1, p2],
                            'description': f'{p1}{p2}_{q1}{q2}'
                        })
        
        return measurements
    
    def compute_fidelity(self, target_state: np.ndarray) -> float:
        """Compute fidelity between current ground state and target state."""
        
        # Current ground state
        psi0 = np.zeros(2 ** self.n_qubits, dtype=complex)
        psi0[0] = 1.0
        
        # Compute fidelity |⟨ψ_target|ψ_0⟩|²
        overlap = np.abs(np.vdot(target_state, psi0))**2
        
        return overlap
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the simulated quantum system."""
        
        return {
            'n_qubits': self.n_qubits,
            'hilbert_space_dim': 2 ** self.n_qubits,
            'n_pauli_terms': len(self.hamiltonian_data['pauli_terms']),
            'shots': self.shots,
            'noise_prob': self.noise_prob,
            'evolution_times': self.evolution_times.tolist(),
            'device': str(self.device),
            'hamiltonian_trace': np.real(np.trace(self.hamiltonian_matrix)),
            'hamiltonian_norm': np.linalg.norm(self.hamiltonian_matrix)
        }
