"""
Measurement Simulator for symbolic quantum measurements
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.linalg import expm

logger = logging.getLogger(__name__)

class MeasurementSimulator:
    """Simulates quantum measurements for molecular Hamiltonians."""
    
    def __init__(self, hamiltonian_data: Dict[str, Any], shots: int, 
                 device: torch.device, noise_prob: float = 0.02):
        self.hamiltonian_data = hamiltonian_data
        self.shots = shots
        self.device = device
        self.noise_prob = noise_prob
        self.n_qubits = hamiltonian_data['n_qubits']
        
        # Pauli matrices
        self.pauli_matrices = self._get_pauli_matrices()
        
        # Build full Hamiltonian matrix
        self.hamiltonian_matrix = self._build_hamiltonian_matrix()
        
        # Evolution times
        self.evolution_times = np.linspace(0.1, 2.0, 10)
        
        # Precompute evolution operators
        self._precompute_evolution_operators()
        
        logger.info(f"Initialized simulator for {self.n_qubits}-qubit system")
    
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
            U = expm(-1j * self.hamiltonian_matrix * t)
            self.evolution_operators[t] = U
    
    def get_initial_measurement(self) -> np.ndarray:
        """Get initial measurement (ground state or random)."""
        # Start with ground state |000...0>
        psi0 = np.zeros(2 ** self.n_qubits, dtype=complex)
        psi0[0] = 1.0
        
        # Measure all qubits in Z basis
        return self._measure_state(psi0, measurement_ops=['Z'] * self.n_qubits)
    
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
        
        # Get evolution operator
        if evolution_time not in self.evolution_operators:
            # Find closest precomputed time
            closest_time = min(self.evolution_times, 
                             key=lambda t: abs(t - evolution_time))
            U = self.evolution_operators[closest_time]
        else:
            U = self.evolution_operators[evolution_time]
        
        # Start with ground state
        psi0 = np.zeros(2 ** self.n_qubits, dtype=complex)
        psi0[0] = 1.0
        
        # Apply time evolution
        psi_t = U @ psi0
        
        # Construct measurement operators
        measurement_ops = ['I'] * self.n_qubits
        for qubit_idx, pauli_op in zip(qubit_indices, pauli_operators):
            measurement_ops[qubit_idx] = pauli_op
        
        # Perform measurement
        expectation_values = self._measure_state(psi_t, measurement_ops)
        
        # Add shot noise
        noisy_expectations = self._add_shot_noise(expectation_values)
        
        return {
            'qubit_indices': qubit_indices,
            'pauli_operators': pauli_operators,
            'evolution_time': evolution_time,
            'expectation_values': noisy_expectations,
            'ideal_expectation_values': expectation_values,
            'shots_used': self.shots
        }
    
    def _measure_state(self, psi: np.ndarray, 
                      measurement_ops: List[str]) -> np.ndarray:
        """Measure quantum state with given operators."""
        
        expectations = []
        
        for i, op in enumerate(measurement_ops):
            if op == 'I':
                expectations.append(1.0)  # Identity always gives 1
                continue
            
            # Build measurement operator
            ops = ['I'] * self.n_qubits
            ops[i] = op
            M = self._build_pauli_operator([(i, op)])
            
            # Compute expectation value
            exp_val = np.real(np.conj(psi) @ M @ psi)
            expectations.append(exp_val)
        
        return np.array(expectations)
    
    def _add_shot_noise(self, expectations: np.ndarray) -> np.ndarray:
        """Add finite shot noise to expectation values."""
        
        noisy_expectations = np.zeros_like(expectations)
        
        for i, exp_val in enumerate(expectations):
            # Convert expectation value to probability
            p_plus = (1 + exp_val) / 2
            
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
        
        # Two-qubit measurements
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
