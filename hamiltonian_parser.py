"""
Hamiltonian Parser for OpenFermion/Qiskit molecular Hamiltonians
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class HamiltonianParser:
    """Parse molecular Hamiltonians from various formats."""
    
    def __init__(self):
        self.supported_formats = ['openfermion', 'qiskit', 'custom']
    
    def load_hamiltonian(self, file_path: Path) -> Dict[str, Any]:
        """
        Load molecular Hamiltonian from JSON file.
        
        Expected JSON format:
        {
            "format": "openfermion",  # or "qiskit", "custom"
            "molecule": "LiH",
            "basis": "sto-3g",
            "n_qubits": 4,
            "pauli_terms": [
                {
                    "coefficient": 0.5,
                    "pauli_string": "ZZII"
                },
                ...
            ],
            "true_parameters": {  # Optional, for validation
                "coupling": [0.5, 0.7, ...],
                "field": [0.2, 0.3, ...]
            }
        }
        """
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate format
        if 'format' not in data:
            raise ValueError("Hamiltonian file must specify 'format' field")
        
        if data['format'] not in self.supported_formats:
            raise ValueError(f"Unsupported format: {data['format']}. "
                           f"Supported: {self.supported_formats}")
        
        # Parse based on format
        if data['format'] == 'openfermion':
            return self._parse_openfermion(data)
        elif data['format'] == 'qiskit':
            return self._parse_qiskit(data)
        elif data['format'] == 'custom':
            return self._parse_custom(data)
    
    def _parse_openfermion(self, data: Dict) -> Dict[str, Any]:
        """Parse OpenFermion-style Hamiltonian."""
        
        required_fields = ['n_qubits', 'pauli_terms']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        n_qubits = data['n_qubits']
        pauli_terms = []
        
        for term in data['pauli_terms']:
            if 'coefficient' not in term or 'pauli_string' not in term:
                raise ValueError("Each Pauli term must have 'coefficient' and 'pauli_string'")
            
            coeff = complex(term['coefficient'])
            pauli_str = term['pauli_string']
            
            if len(pauli_str) != n_qubits:
                raise ValueError(f"Pauli string length {len(pauli_str)} != n_qubits {n_qubits}")
            
            # Convert to standardized format
            pauli_terms.append({
                'coefficient': coeff,
                'pauli_indices': self._pauli_string_to_indices(pauli_str),
                'original_string': pauli_str
            })
        
        # Extract coupling and field structure
        structure = self._analyze_hamiltonian_structure(pauli_terms, n_qubits)
        
        return {
            'format': 'openfermion',
            'molecule': data.get('molecule', 'unknown'),
            'basis': data.get('basis', 'unknown'),
            'n_qubits': n_qubits,
            'pauli_terms': pauli_terms,
            'structure': structure,
            'true_parameters': data.get('true_parameters', None)
        }
    
    def _parse_qiskit(self, data: Dict) -> Dict[str, Any]:
        """Parse Qiskit-style Hamiltonian."""
        # Similar to OpenFermion but with Qiskit-specific conventions
        return self._parse_openfermion(data)  # For now, same format
    
    def _parse_custom(self, data: Dict) -> Dict[str, Any]:
        """Parse custom Hamiltonian format."""
        return self._parse_openfermion(data)  # For now, same format
    
    def _pauli_string_to_indices(self, pauli_str: str) -> List[Tuple[int, str]]:
        """Convert Pauli string like 'XYZI' to [(0,'X'), (1,'Y'), (2,'Z')]."""
        indices = []
        for i, pauli in enumerate(pauli_str):
            if pauli.upper() in ['X', 'Y', 'Z']:
                indices.append((i, pauli.upper()))
        return indices
    
    def _analyze_hamiltonian_structure(self, pauli_terms: List[Dict], 
                                     n_qubits: int) -> Dict[str, Any]:
        """Analyze Hamiltonian to identify coupling and field terms."""
        
        coupling_terms = []  # ZZ, XX, YY interactions
        field_terms = []     # Single-qubit X, Y, Z terms
        other_terms = []
        
        for term in pauli_terms:
            indices = term['pauli_indices']
            
            if len(indices) == 1:
                # Single-qubit term (field)
                field_terms.append(term)
            elif len(indices) == 2:
                # Two-qubit term (potential coupling)
                coupling_terms.append(term)
            else:
                # Multi-qubit term
                other_terms.append(term)
        
        return {
            'coupling_terms': coupling_terms,
            'field_terms': field_terms,
            'other_terms': other_terms,
            'n_coupling_params': len(coupling_terms),
            'n_field_params': len(field_terms)
        }

    @staticmethod
    def create_example_hamiltonian(molecule: str = "H2", n_qubits: int = 4) -> Dict:
        """Create an example molecular Hamiltonian for testing."""
        
        # Example H2 molecule Hamiltonian
        if molecule == "H2" and n_qubits == 4:
            pauli_terms = [
                {"coefficient": -1.0523732, "pauli_string": "IIII"},
                {"coefficient": 0.39793742, "pauli_string": "IIIZ"},
                {"coefficient": -0.39793742, "pauli_string": "IIZI"},
                {"coefficient": -0.01128010, "pauli_string": "IIZZ"},
                {"coefficient": 0.18093119, "pauli_string": "IXIX"},
                {"coefficient": 0.18093119, "pauli_string": "IYIY"}
            ]
        elif molecule == "LiH" and n_qubits == 6:
            pauli_terms = [
                {"coefficient": -7.8384, "pauli_string": "IIIIII"},
                {"coefficient": 0.1809, "pauli_string": "IIIIIZ"},
                {"coefficient": 0.1809, "pauli_string": "IIIIZI"},
                {"coefficient": -0.2436, "pauli_string": "IIIZII"},
                {"coefficient": 0.1665, "pauli_string": "IIXIIX"},
                {"coefficient": 0.1665, "pauli_string": "IIYIIY"},
                {"coefficient": 0.1743, "pauli_string": "IZIIZI"}
            ]
        else:
            raise ValueError(f"Example for {molecule} with {n_qubits} qubits not available")
        
        return {
            "format": "custom",
            "molecule": molecule,
            "basis": "sto-3g",
            "n_qubits": n_qubits,
            "pauli_terms": pauli_terms,
            "true_parameters": {
                "coupling": [0.18, 0.18, 0.17],
                "field": [0.40, -0.40, -0.24]
            }
        }
