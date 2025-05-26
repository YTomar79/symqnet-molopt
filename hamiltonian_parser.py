"""
Hamiltonian Parser for OpenFermion/Qiskit molecular Hamiltonians
STRICT 10-QUBIT VALIDATION
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class HamiltonianParser:
    """Parse molecular Hamiltonians from various formats."""
    
    # STRICT CONSTRAINT: SymQNet only supports exactly 10 qubits
    SUPPORTED_QUBITS = 10
    
    def __init__(self):
        self.supported_formats = ['openfermion', 'qiskit', 'custom']
    
    def load_hamiltonian(self, file_path: Path) -> Dict[str, Any]:
        """
        Load molecular Hamiltonian from JSON file.
        
        IMPORTANT: SymQNet-MolOpt only supports exactly 10-qubit systems.
        """
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate format
        if 'format' not in data:
            raise ValueError("Hamiltonian file must specify 'format' field")
        
        if data['format'] not in self.supported_formats:
            raise ValueError(f"Unsupported format: {data['format']}. "
                           f"Supported: {self.supported_formats}")
        
        # STRICT QUBIT VALIDATION
        n_qubits = data.get('n_qubits', 0)
        if n_qubits != self.SUPPORTED_QUBITS:
            raise ValueError(
                f"❌ SymQNet-MolOpt is only trained for {self.SUPPORTED_QUBITS} qubits. "
                f"Your Hamiltonian has {n_qubits} qubits.\n\n"
                f"💡 To use this tool:\n"
                f"   • Use a {self.SUPPORTED_QUBITS}-qubit molecular Hamiltonian\n"
                f"   • Try the provided examples: H2O_10q.json\n"
                f"   • Map your molecule to a {self.SUPPORTED_QUBITS}-qubit representation"
            )
        
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
        
        # Double-check qubit constraint (should already be caught above)
        assert n_qubits == self.SUPPORTED_QUBITS, "Internal error: qubit validation failed"
        
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
        return self._parse_openfermion(data)  # Same validation applies
    
    def _parse_custom(self, data: Dict) -> Dict[str, Any]:
        """Parse custom Hamiltonian format."""
        return self._parse_openfermion(data)  # Same validation applies
    
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
    def create_example_hamiltonian(molecule: str = "H2O", n_qubits: int = 10) -> Dict:
        """Create an example molecular Hamiltonian (MUST be 10 qubits)."""
        
        if n_qubits != 10:
            raise ValueError(
                f"❌ SymQNet-MolOpt only supports 10-qubit systems. "
                f"Cannot create {n_qubits}-qubit example."
            )
        
        # Only provide 10-qubit examples
        if molecule == "H2O" and n_qubits == 10:
            pauli_terms = [
                {"coefficient": -74.9431, "pauli_string": "IIIIIIIIII"},
                {"coefficient": 0.3421, "pauli_string": "IIIIIIIIIZ"},
                {"coefficient": 0.3421, "pauli_string": "IIIIIIIIZI"},
                {"coefficient": -0.4523, "pauli_string": "IIIIIIIIZZ"},
                {"coefficient": 0.2134, "pauli_string": "IIIIIIXIIX"},
                {"coefficient": 0.2134, "pauli_string": "IIIIIIYIIY"},
                {"coefficient": 0.1876, "pauli_string": "IIIIZIIZII"},
                {"coefficient": -0.0934, "pauli_string": "IIIZIIIZII"},
                {"coefficient": 0.1623, "pauli_string": "IIXIIIIIIX"},
                {"coefficient": 0.1623, "pauli_string": "IIYIIIIIIY"}
            ]
        else:
            raise ValueError(
                f"❌ Only 10-qubit H2O example available. "
                f"SymQNet-MolOpt is trained specifically for 10-qubit systems."
            )
        
        return {
            "format": "custom",
            "molecule": molecule,
            "basis": "sto-3g",
            "n_qubits": n_qubits,
            "pauli_terms": pauli_terms,
            "true_parameters": {
                "coupling": [0.2134, 0.2134, -0.4523, 0.1876, -0.0934, 0.1623, 0.1623, 0.0823, -0.0456],
                "field": [0.3421, 0.3421, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "description": f"{molecule} molecule optimized for SymQNet 10-qubit training"
        }
