"""
Hamiltonian Parser for OpenFermion/Qiskit molecular Hamiltonians
UNIVERSAL QUBIT SUPPORT with optimal performance at 10 qubits
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings

logger = logging.getLogger(__name__)

class HamiltonianParser:
    """Parse molecular Hamiltonians from various formats with universal qubit support."""
    
    # Universal support - optimal at 10 qubits, works with any viable count
    OPTIMAL_QUBITS = 10
    MIN_VIABLE_QUBITS = 2
    
    def __init__(self):
        self.supported_formats = ['openfermion', 'qiskit', 'custom', 'universal']
        logger.info(f"Hamiltonian parser initialized - optimal at {self.OPTIMAL_QUBITS} qubits")
    
    def load_hamiltonian(self, file_path: Path, warn_performance: bool = True) -> Dict[str, Any]:
        """
        Load molecular Hamiltonian from JSON file with universal qubit support.
        
        Args:
            file_path: Path to Hamiltonian JSON file
            warn_performance: Whether to warn about non-optimal qubit counts
            
        Returns:
            Parsed Hamiltonian data with performance metadata
        """
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate format
        if 'format' not in data:
            raise ValueError("Hamiltonian file must specify 'format' field")
        
        if data['format'] not in self.supported_formats:
            raise ValueError(f"Unsupported format: {data['format']}. "
                           f"Supported: {self.supported_formats}")
        
        # Universal qubit validation with performance guidance
        n_qubits = data.get('n_qubits', 0)
        
        if n_qubits < self.MIN_VIABLE_QUBITS:
            raise ValueError(
                f"❌ VALIDATION FAILED: Minimum {self.MIN_VIABLE_QUBITS} qubits required.\n"
                f"   Your Hamiltonian: {n_qubits} qubits\n"
                f"   Minimum viable: {self.MIN_VIABLE_QUBITS} qubits\n\n"
                f"💡 SymQNet-MolOpt requires at least {self.MIN_VIABLE_QUBITS} qubits for meaningful molecular representation."
            )
        
        # Performance guidance instead of hard constraint
        if warn_performance and n_qubits != self.OPTIMAL_QUBITS:
            performance_factor = self._estimate_performance_factor(n_qubits)
            
            if performance_factor < 0.7:  # Significant degradation
                logger.warning(
                    f"⚠️  PERFORMANCE WARNING: {n_qubits}-qubit system will operate at "
                    f"{performance_factor:.1%} of optimal performance.\n"
                    f"   Optimal qubit count: {self.OPTIMAL_QUBITS}\n"
                    f"   Consider system size optimization for better accuracy."
                )
            else:
                logger.info(
                    f"📊 System: {n_qubits} qubits ({performance_factor:.1%} of optimal performance)"
                )
        
        # Parse based on format
        if data['format'] in ['openfermion', 'qiskit', 'custom', 'universal']:
            return self._parse_universal(data)
        else:
            raise ValueError(f"Unsupported format: {data['format']}")
    
    def _parse_universal(self, data: Dict) -> Dict[str, Any]:
        """Parse Hamiltonian in universal format supporting any qubit count."""
        
        required_fields = ['n_qubits', 'pauli_terms']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        n_qubits = data['n_qubits']
        
        # Validate structure
        pauli_terms = self._validate_and_process_pauli_terms(data['pauli_terms'], n_qubits)
        
        # Analyze Hamiltonian structure
        structure = self._analyze_hamiltonian_structure(pauli_terms, n_qubits)
        
        # Add performance metadata
        performance_metadata = self._get_performance_metadata(n_qubits)
        
        return {
            'format': data.get('format', 'universal'),
            'molecule': data.get('molecule', 'unknown'),
            'basis': data.get('basis', 'unknown'),
            'geometry': data.get('geometry', 'unknown'),
            'n_qubits': n_qubits,
            'pauli_terms': pauli_terms,
            'structure': structure,
            'true_parameters': data.get('true_parameters', None),
            'reference_energy': data.get('reference_energy', None),
            'description': data.get('description', f'{n_qubits}-qubit molecular system'),
            'performance_metadata': performance_metadata,
            'universal_compatible': True
        }
    
    def _validate_and_process_pauli_terms(self, pauli_terms: List[Dict], n_qubits: int) -> List[Dict]:
        """Validate and process Pauli terms for universal compatibility."""
        
        processed_terms = []
        
        for i, term in enumerate(pauli_terms):
            if 'coefficient' not in term:
                raise ValueError(f"Pauli term {i} missing 'coefficient'")
            
            # Handle both string and index representations
            if 'pauli_string' in term:
                pauli_str = term['pauli_string']
                
                if len(pauli_str) != n_qubits:
                    raise ValueError(
                        f"Pauli term {i}: string length {len(pauli_str)} != n_qubits {n_qubits}\n"
                        f"Pauli string: '{pauli_str}'"
                    )
                
                # Convert to standardized format
                processed_terms.append({
                    'coefficient': complex(term['coefficient']),
                    'pauli_indices': self._pauli_string_to_indices(pauli_str),
                    'pauli_string': pauli_str,
                    'description': term.get('description', f'Pauli term {i}')
                })
                
            elif 'pauli_indices' in term:
                # Already in index format
                indices = term['pauli_indices']
                pauli_str = self._indices_to_pauli_string(indices, n_qubits)
                
                processed_terms.append({
                    'coefficient': complex(term['coefficient']),
                    'pauli_indices': indices,
                    'pauli_string': pauli_str,
                    'description': term.get('description', f'Pauli term {i}')
                })
                
            else:
                raise ValueError(
                    f"Pauli term {i} must have either 'pauli_string' or 'pauli_indices'"
                )
        
        return processed_terms
    
    def _pauli_string_to_indices(self, pauli_str: str) -> List[Tuple[int, str]]:
        """Convert Pauli string like 'XYZI' to [(0,'X'), (1,'Y'), (2,'Z')]."""
        indices = []
        for i, pauli in enumerate(pauli_str):
            if pauli.upper() in ['X', 'Y', 'Z']:
                indices.append((i, pauli.upper()))
        return indices
    
    def _indices_to_pauli_string(self, indices: List[Tuple[int, str]], n_qubits: int) -> str:
        """Convert indices like [(0,'X'), (1,'Y')] to Pauli string 'XYI...'."""
        pauli_str = ['I'] * n_qubits
        for qubit_idx, pauli_op in indices:
            if 0 <= qubit_idx < n_qubits:
                pauli_str[qubit_idx] = pauli_op.upper()
        return ''.join(pauli_str)
    
    def _analyze_hamiltonian_structure(self, pauli_terms: List[Dict], 
                                     n_qubits: int) -> Dict[str, Any]:
        """Analyze Hamiltonian structure for universal systems."""
        
        coupling_terms = []  # Two-qubit interactions
        field_terms = []     # Single-qubit terms
        multi_qubit_terms = []  # Higher-order terms
        identity_terms = []  # Constant terms
        
        total_coupling_strength = 0.0
        total_field_strength = 0.0
        
        for term in pauli_terms:
            indices = term['pauli_indices']
            coeff_magnitude = abs(term['coefficient'])
            
            if len(indices) == 0:
                # Identity term (constant energy)
                identity_terms.append(term)
            elif len(indices) == 1:
                # Single-qubit term (field)
                field_terms.append(term)
                total_field_strength += coeff_magnitude
            elif len(indices) == 2:
                # Two-qubit term (coupling)
                coupling_terms.append(term)
                total_coupling_strength += coeff_magnitude
            else:
                # Multi-qubit term
                multi_qubit_terms.append(term)
        
        # Estimate expected parameter counts for universal systems
        expected_coupling = n_qubits - 1  # Typical linear chain connectivity
        expected_field = n_qubits         # One field per qubit
        
        structure = {
            'coupling_terms': coupling_terms,
            'field_terms': field_terms,
            'multi_qubit_terms': multi_qubit_terms,
            'identity_terms': identity_terms,
            'n_coupling_terms': len(coupling_terms),
            'n_field_terms': len(field_terms),
            'n_multi_qubit_terms': len(multi_qubit_terms),
            'expected_coupling_params': expected_coupling,
            'expected_field_params': expected_field,
            'total_coupling_strength': total_coupling_strength,
            'total_field_strength': total_field_strength,
            'complexity_score': self._calculate_complexity_score(pauli_terms, n_qubits)
        }
        
        return structure
    
    def _calculate_complexity_score(self, pauli_terms: List[Dict], n_qubits: int) -> float:
        """Calculate complexity score for the Hamiltonian."""
        
        # Factors contributing to complexity
        term_count_factor = len(pauli_terms) / n_qubits  # Terms per qubit
        
        # Weight by term order (higher-order terms increase complexity)
        weighted_complexity = 0.0
        for term in pauli_terms:
            order = len(term['pauli_indices'])
            weight = order ** 1.5  # Higher-order terms weighted more
            weighted_complexity += weight
        
        normalized_complexity = weighted_complexity / n_qubits
        
        return min(normalized_complexity, 10.0)  # Cap at 10.0
    
    def _estimate_performance_factor(self, n_qubits: int) -> float:
        """Estimate expected performance factor for given qubit count."""
        
        if n_qubits == self.OPTIMAL_QUBITS:
            return 1.0
        
        distance = abs(n_qubits - self.OPTIMAL_QUBITS)
        
        if n_qubits < self.OPTIMAL_QUBITS:
            # Small systems: information loss from padding
            return max(0.95 ** (distance * 0.8), 0.7)
        else:
            # Large systems: information loss from compression
            return max(0.90 ** (distance * 1.2), 0.4)
    
    def _get_performance_metadata(self, n_qubits: int) -> Dict[str, Any]:
        """Get performance metadata for the system."""
        
        performance_factor = self._estimate_performance_factor(n_qubits)
        
        if performance_factor >= 0.95:
            performance_level = "optimal"
        elif performance_factor >= 0.85:
            performance_level = "excellent"
        elif performance_factor >= 0.70:
            performance_level = "good"
        elif performance_factor >= 0.50:
            performance_level = "moderate"
        else:
            performance_level = "poor"
        
        return {
            'optimal_qubits': self.OPTIMAL_QUBITS,
            'performance_factor': performance_factor,
            'performance_level': performance_level,
            'is_optimal': n_qubits == self.OPTIMAL_QUBITS,
            'degradation_reason': self._get_degradation_reason(n_qubits)
        }
    
    def _get_degradation_reason(self, n_qubits: int) -> Optional[str]:
        """Get reason for performance degradation."""
        
        if n_qubits == self.OPTIMAL_QUBITS:
            return None
        elif n_qubits < self.OPTIMAL_QUBITS:
            return f"Information dilution from padding {n_qubits} to {self.OPTIMAL_QUBITS} qubits"
        else:
            return f"Information loss from compressing {n_qubits} to {self.OPTIMAL_QUBITS} qubits"
    
    def create_example_hamiltonian(self, molecule: str = "H2O", n_qubits: int = 10,
                                 complexity: str = "moderate") -> Dict[str, Any]:
        """
        Create example molecular Hamiltonian for any qubit count.
        
        Args:
            molecule: Molecule name
            n_qubits: Number of qubits (any viable count ≥ 2)
            complexity: Complexity level ('simple', 'moderate', 'complex')
            
        Returns:
            Hamiltonian dictionary
        """
        
        if n_qubits < self.MIN_VIABLE_QUBITS:
            raise ValueError(
                f"Minimum {self.MIN_VIABLE_QUBITS} qubits required for meaningful molecular representation"
            )
        
        # Generate examples based on complexity and qubit count
        if complexity == "simple":
            return self._create_simple_example(molecule, n_qubits)
        elif complexity == "moderate":
            return self._create_moderate_example(molecule, n_qubits)
        elif complexity == "complex":
            return self._create_complex_example(molecule, n_qubits)
        else:
            raise ValueError(f"Unsupported complexity: {complexity}. Use 'simple', 'moderate', or 'complex'")
    
    def _create_simple_example(self, molecule: str, n_qubits: int) -> Dict[str, Any]:
        """Create simple Hamiltonian with basic terms."""
        
        # Simple linear chain with nearest-neighbor interactions
        pauli_terms = []
        
        # Constant term
        pauli_terms.append({
            "coefficient": -2.0 * n_qubits,  # Scale with system size
            "pauli_string": "I" * n_qubits,
            "description": "Constant energy term"
        })
        
        # Single-qubit Z terms (fields)
        for i in range(n_qubits):
            coeff = 0.5 * (1 - 0.1 * i)  # Decreasing field strength
            pauli_str = "I" * i + "Z" + "I" * (n_qubits - i - 1)
            pauli_terms.append({
                "coefficient": coeff,
                "pauli_string": pauli_str,
                "description": f"Z field on qubit {i}"
            })
        
        # Nearest-neighbor ZZ couplings
        for i in range(n_qubits - 1):
            coeff = 0.3 * (1 - 0.05 * i)  # Decreasing coupling strength
            pauli_str = "I" * i + "ZZ" + "I" * (n_qubits - i - 2)
            pauli_terms.append({
                "coefficient": coeff,
                "pauli_string": pauli_str,
                "description": f"ZZ coupling between qubits {i}-{i+1}"
            })
        
        # Generate realistic true parameters
        coupling_params = [0.3 * (1 - 0.05 * i) for i in range(n_qubits - 1)]
        field_params = [0.5 * (1 - 0.1 * i) for i in range(n_qubits)]
        
        return {
            "format": "universal",
            "molecule": f"{molecule}_simple",
            "basis": "minimal",
            "geometry": "linear_chain",
            "n_qubits": n_qubits,
            "pauli_terms": pauli_terms,
            "true_parameters": {
                "coupling": coupling_params,
                "field": field_params
            },
            "reference_energy": -2.0 * n_qubits + sum(field_params),
            "description": f"Simple {molecule} model with {n_qubits} qubits - linear chain connectivity"
        }
    
    def _create_moderate_example(self, molecule: str, n_qubits: int) -> Dict[str, Any]:
        """Create moderate complexity Hamiltonian."""
        
        pauli_terms = []
        
        # Constant term
        pauli_terms.append({
            "coefficient": -5.0 * np.sqrt(n_qubits),
            "pauli_string": "I" * n_qubits,
            "description": "Constant energy term"
        })
        
        # Single-qubit terms (X, Y, Z)
        for i in range(n_qubits):
            # Z field
            z_coeff = 0.8 * np.cos(2 * np.pi * i / n_qubits)
            if abs(z_coeff) > 0.1:  # Only include significant terms
                pauli_str = "I" * i + "Z" + "I" * (n_qubits - i - 1)
                pauli_terms.append({
                    "coefficient": z_coeff,
                    "pauli_string": pauli_str,
                    "description": f"Z field on qubit {i}"
                })
            
            # X field (transverse field)
            if i % 3 == 0:  # Every third qubit
                x_coeff = 0.2
                pauli_str = "I" * i + "X" + "I" * (n_qubits - i - 1)
                pauli_terms.append({
                    "coefficient": x_coeff,
                    "pauli_string": pauli_str,
                    "description": f"X field on qubit {i}"
                })
        
        # Two-qubit interactions
        for i in range(n_qubits - 1):
            # ZZ nearest-neighbor
            zz_coeff = 0.4 * np.exp(-0.1 * i)
            pauli_str = "I" * i + "ZZ" + "I" * (n_qubits - i - 2)
            pauli_terms.append({
                "coefficient": zz_coeff,
                "pauli_string": pauli_str,
                "description": f"ZZ coupling {i}-{i+1}"
            })
            
            # XX coupling (every other pair)
            if i % 2 == 0:
                xx_coeff = 0.15
                pauli_str = "I" * i + "XX" + "I" * (n_qubits - i - 2)
                pauli_terms.append({
                    "coefficient": xx_coeff,
                    "pauli_string": pauli_str,
                    "description": f"XX coupling {i}-{i+1}"
                })
        
        # Some longer-range interactions
        for i in range(n_qubits - 2):
            if i % 4 == 0:  # Sparse long-range
                lr_coeff = 0.05
                pauli_str = "I" * i + "Z" + "I" + "Z" + "I" * (n_qubits - i - 3)
                pauli_terms.append({
                    "coefficient": lr_coeff,
                    "pauli_string": pauli_str,
                    "description": f"Long-range ZZ coupling {i}-{i+2}"
                })
        
        # Generate realistic parameters
        coupling_params = [0.4 * np.exp(-0.1 * i) for i in range(n_qubits - 1)]
        field_params = [0.8 * np.cos(2 * np.pi * i / n_qubits) for i in range(n_qubits)]
        
        return {
            "format": "universal",
            "molecule": f"{molecule}_moderate",
            "basis": "sto-3g",
            "geometry": "optimized",
            "n_qubits": n_qubits,
            "pauli_terms": pauli_terms,
            "true_parameters": {
                "coupling": coupling_params,
                "field": field_params
            },
            "reference_energy": -5.0 * np.sqrt(n_qubits) + 0.5 * sum(field_params),
            "description": f"Moderate {molecule} model with {n_qubits} qubits - mixed connectivity"
        }
    
    def _create_complex_example(self, molecule: str, n_qubits: int) -> Dict[str, Any]:
        """Create complex Hamiltonian with many-body terms."""
        
        pauli_terms = []
        
        # Constant term
        pauli_terms.append({
            "coefficient": -10.0 * n_qubits,
            "pauli_string": "I" * n_qubits,
            "description": "Constant energy term"
        })
        
        # Rich single-qubit structure
        for i in range(n_qubits):
            for pauli_op in ['X', 'Y', 'Z']:
                coeff_map = {'X': 0.3, 'Y': 0.2, 'Z': 1.0}
                base_coeff = coeff_map[pauli_op]
                
                # Modulated coefficients
                coeff = base_coeff * np.sin(np.pi * i / n_qubits) * np.cos(2 * np.pi * i / n_qubits)
                
                if abs(coeff) > 0.05:  # Only significant terms
                    pauli_str = "I" * i + pauli_op + "I" * (n_qubits - i - 1)
                    pauli_terms.append({
                        "coefficient": coeff,
                        "pauli_string": pauli_str,
                        "description": f"{pauli_op} term on qubit {i}"
                    })
        
        # Rich two-qubit structure
        for i in range(n_qubits):
            for j in range(i + 1, min(i + 4, n_qubits)):  # Up to 3-qubit range
                distance = j - i
                decay = np.exp(-0.3 * distance)
                
                for pauli_pair in ['ZZ', 'XX', 'YY', 'XY', 'XZ']:
                    base_strength = {'ZZ': 0.5, 'XX': 0.3, 'YY': 0.3, 'XY': 0.1, 'XZ': 0.1}
                    coeff = base_strength[pauli_pair] * decay
                    
                    if abs(coeff) > 0.02:
                        # Build Pauli string
                        pauli_str = ['I'] * n_qubits
                        pauli_str[i] = pauli_pair[0]
                        pauli_str[j] = pauli_pair[1]
                        pauli_string = ''.join(pauli_str)
                        
                        pauli_terms.append({
                            "coefficient": coeff,
                            "pauli_string": pauli_string,
                            "description": f"{pauli_pair} coupling {i}-{j}"
                        })
        
        # Some three-body terms (sparse)
        if n_qubits >= 6:
            for i in range(0, n_qubits - 2, 3):  # Every third position
                if i + 2 < n_qubits:
                    coeff = 0.02  # Small three-body interaction
                    pauli_str = ['I'] * n_qubits
                    pauli_str[i] = 'Z'
                    pauli_str[i + 1] = 'Z' 
                    pauli_str[i + 2] = 'Z'
                    pauli_string = ''.join(pauli_str)
                    
                    pauli_terms.append({
                        "coefficient": coeff,
                        "pauli_string": pauli_string,
                        "description": f"ZZZ three-body term {i}-{i+1}-{i+2}"
                    })
        
        # Generate complex parameter structure
        coupling_params = []
        for i in range(n_qubits - 1):
            param = 0.5 * np.exp(-0.2 * i) * (1 + 0.3 * np.sin(2 * np.pi * i / n_qubits))
            coupling_params.append(param)
        
        field_params = []
        for i in range(n_qubits):
            param = 1.0 * np.sin(np.pi * i / n_qubits) * np.cos(2 * np.pi * i / n_qubits)
            field_params.append(param)
        
        return {
            "format": "universal",
            "molecule": f"{molecule}_complex",
            "basis": "cc-pvdz",
            "geometry": "complex_optimized",
            "n_qubits": n_qubits,
            "pauli_terms": pauli_terms,
            "true_parameters": {
                "coupling": coupling_params,
                "field": field_params
            },
            "reference_energy": -10.0 * n_qubits + 0.3 * sum(field_params),
            "description": f"Complex {molecule} model with {n_qubits} qubits - full connectivity with many-body terms"
        }
    
    def validate_hamiltonian_universal(self, hamiltonian_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate Hamiltonian with universal support."""
        
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['n_qubits', 'pauli_terms']
        for field in required_fields:
            if field not in hamiltonian_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        n_qubits = hamiltonian_data['n_qubits']
        
        # Universal validation
        if n_qubits < self.MIN_VIABLE_QUBITS:
            errors.append(f"Minimum {self.MIN_VIABLE_QUBITS} qubits required, got {n_qubits}")
        
        # Performance warnings
        if n_qubits != self.OPTIMAL_QUBITS:
            performance = self._estimate_performance_factor(n_qubits)
            warnings.append(
                f"Non-optimal qubit count: {n_qubits} qubits "
                f"(expected {performance:.1%} performance vs optimal {self.OPTIMAL_QUBITS} qubits)"
            )
        
        # Validate Pauli terms structure
        try:
            self._validate_and_process_pauli_terms(hamiltonian_data['pauli_terms'], n_qubits)
        except ValueError as e:
            errors.append(f"Pauli terms validation failed: {e}")
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
        
        return len(errors) == 0, errors
    
    def get_normalization_info(self, n_qubits: int) -> Dict[str, Any]:
        """Get information about normalization needed for universal compatibility."""
        
        if n_qubits == self.OPTIMAL_QUBITS:
            return {
                'normalization_needed': False,
                'target_qubits': self.OPTIMAL_QUBITS,
                'performance_factor': 1.0
            }
        
        performance = self._estimate_performance_factor(n_qubits)
        
        return {
            'normalization_needed': True,
            'original_qubits': n_qubits,
            'target_qubits': self.OPTIMAL_QUBITS,
            'performance_factor': performance,
            'normalization_type': 'padding' if n_qubits < self.OPTIMAL_QUBITS else 'compression',
            'scaling_factor': np.sqrt(self.OPTIMAL_QUBITS / n_qubits)
        }
    
    def suggest_system_optimization(self, n_qubits: int) -> str:
        """Suggest how to optimize system for better performance."""
        
        if n_qubits == self.OPTIMAL_QUBITS:
            return "✅ System already at optimal qubit count for maximum performance"
        
        performance = self._estimate_performance_factor(n_qubits)
        
        if n_qubits < self.OPTIMAL_QUBITS:
            diff = self.OPTIMAL_QUBITS - n_qubits
            return (
                f"📊 Current: {n_qubits} qubits ({performance:.1%} performance)\n"
                f"🎯 Optimal: {self.OPTIMAL_QUBITS} qubits (100% performance)\n\n"
                f"💡 To reach optimal performance (+{diff} qubits):\n"
                f"   • Expand active space to include {diff} more orbitals\n"
                f"   • Use larger basis set (STO-3G → cc-pVDZ)\n"
                f"   • Include virtual orbitals\n"
                f"   • Add ancilla qubits for error correction\n\n"
                f"📈 Expected improvement: {performance:.1%} → 100% performance"
            )
        else:
            diff = n_qubits - self.OPTIMAL_QUBITS
            return (
                f"📊 Current: {n_qubits} qubits ({performance:.1%} performance)\n"
                f"🎯 Optimal: {self.OPTIMAL_QUBITS} qubits (100% performance)\n\n"
                f"💡 To reach optimal performance (-{diff} qubits):\n"
                f"   • Freeze {diff} core orbitals\n"
                f"   • Use smaller active space\n"
                f"   • Apply symmetry reduction\n"
                f"   • Use effective Hamiltonian approximation\n\n"
                f"📈 Expected improvement: {performance:.1%} → 100% performance"
            )
    
    def create_universal_examples(self, output_dir: Path = Path("examples")) -> None:
        """Create a suite of universal examples for different qubit counts."""
        
        output_dir.mkdir(exist_ok=True)
        
        # Examples highlighting 10-qubit optimum
        examples = [
            ("H2", 4, "simple"),
            ("LiH", 6, "simple"),
            ("BeH2", 8, "moderate"),
            ("H2O", 10, "moderate"),     # OPTIMAL
            ("NH3", 12, "moderate"),
            ("CH4", 14, "complex"),
            ("C2H4", 16, "complex")
        ]
        
        print(f"🌍 Creating universal SymQNet examples in {output_dir}/")
        print("=" * 60)
        
        for molecule, n_qubits, complexity in examples:
            try:
                hamiltonian = self.create_example_hamiltonian(molecule, n_qubits, complexity)
                
                filename = f"{molecule}_{n_qubits}q.json"
                filepath = output_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(hamiltonian, f, indent=2)
                
                performance = self._estimate_performance_factor(n_qubits)
                optimal_marker = " ⭐ OPTIMAL" if n_qubits == self.OPTIMAL_QUBITS else ""
                
                print(f"✅ {filename:<20} ({n_qubits:2d} qubits, {performance:.1%} performance){optimal_marker}")
                
            except Exception as e:
                print(f"❌ Failed to create {molecule}_{n_qubits}q: {e}")
        
        print("=" * 60)
        print(f"📚 Universal examples created - test with:")
        print(f"   symqnet-molopt --hamiltonian examples/H2O_10q.json --output results.json  ⭐")
        print(f"   symqnet-molopt --hamiltonian examples/NH3_12q.json --output results.json")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_hamiltonian_universal(file_path: Path, warn_performance: bool = True) -> Dict[str, Any]:
    """Convenience function to load Hamiltonian with universal support."""
    parser = HamiltonianParser()
    return parser.load_hamiltonian(file_path, warn_performance=warn_performance)

def create_example_hamiltonian(molecule: str = "H2O", n_qubits: int = 10, 
                              complexity: str = "moderate") -> Dict[str, Any]:
    """Convenience function to create example Hamiltonian."""
    parser = HamiltonianParser()
    return parser.create_example_hamiltonian(molecule, n_qubits, complexity)

def validate_hamiltonian_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Convenience function to validate Hamiltonian file."""
    try:
        parser = HamiltonianParser()
        hamiltonian = parser.load_hamiltonian(file_path, warn_performance=False)
        return parser.validate_hamiltonian_universal(hamiltonian)
    except Exception as e:
        return False, [str(e)]
