"""
Utility functions for SymQNet molecular optimization
ENHANCED WITH 10-QUBIT VALIDATION
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch

logger = logging.getLogger(__name__)

# STRICT CONSTRAINT: SymQNet only supports exactly 10 qubits
SUPPORTED_QUBITS = 10

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_inputs(hamiltonian_path: Path, shots: int, confidence: float,
                   max_steps: int, n_rollouts: int):
    """Validate CLI input parameters with 10-qubit constraint."""
    
    if not hamiltonian_path.exists():
        raise ValueError(f"Hamiltonian file not found: {hamiltonian_path}")
    
    # STRICT QUBIT VALIDATION - Check file before proceeding
    try:
        with open(hamiltonian_path, 'r') as f:
            data = json.load(f)
        
        n_qubits = data.get('n_qubits', 0)
        if n_qubits != SUPPORTED_QUBITS:
            raise ValueError(
                f"❌ VALIDATION FAILED: SymQNet-MolOpt only supports {SUPPORTED_QUBITS}-qubit systems.\n"
                f"   Your Hamiltonian: {n_qubits} qubits\n"
                f"   Required: {SUPPORTED_QUBITS} qubits\n\n"
                f"💡 Solutions:\n"
                f"   • Use: symqnet-examples to create valid 10-qubit examples\n"
                f"   • Try: examples/H2O_10q.json\n"
                f"   • Map your molecule to 10 qubits using Jordan-Wigner encoding"
            )
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {hamiltonian_path}")
    except KeyError:
        raise ValueError(f"Hamiltonian file missing 'n_qubits' field: {hamiltonian_path}")
    
    if shots <= 0:
        raise ValueError("Number of shots must be positive")
    
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    
    if max_steps <= 0:
        raise ValueError("Maximum steps must be positive")
    
    if n_rollouts <= 0:
        raise ValueError("Number of rollouts must be positive")
    
    logger.debug("✅ Input validation passed - 10-qubit constraint satisfied")

def save_results(results: Dict[str, Any], hamiltonian_data: Dict[str, Any],
                config: Dict[str, Any], output_path: Path):
    """Save estimation results to JSON file."""
    
    # Double-check qubit constraint in results
    n_qubits = hamiltonian_data.get('n_qubits', 0)
    if n_qubits != SUPPORTED_QUBITS:
        logger.warning(f"⚠️  Unexpected qubit count in results: {n_qubits} != {SUPPORTED_QUBITS}")
    
    output_data = {
        'symqnet_results': {
            'coupling_parameters': [
                {
                    'index': i,
                    'mean': float(mean),
                    'confidence_interval': [float(ci_low), float(ci_high)],
                    'uncertainty': float(ci_high - ci_low)
                }
                for i, (mean, ci_low, ci_high) in enumerate(results['coupling_parameters'])
            ],
            'field_parameters': [
                {
                    'index': i,
                    'mean': float(mean),
                    'confidence_interval': [float(ci_low), float(ci_high)],
                    'uncertainty': float(ci_high - ci_low)
                }
                for i, (mean, ci_low, ci_high) in enumerate(results['field_parameters'])
            ],
            'total_uncertainty': float(results['total_uncertainty']),
            'avg_measurements_used': float(results['avg_measurements']),
            'confidence_level': float(results['confidence_level']),
            'n_rollouts': int(results['n_rollouts'])
        },
        'hamiltonian_info': {
            'molecule': hamiltonian_data.get('molecule', 'unknown'),
            'n_qubits': hamiltonian_data['n_qubits'],
            'n_pauli_terms': len(hamiltonian_data['pauli_terms']),
            'format': hamiltonian_data['format'],
            'supported_qubits': SUPPORTED_QUBITS,
            'validation_passed': hamiltonian_data['n_qubits'] == SUPPORTED_QUBITS
        },
        'experimental_config': config,
        'metadata': {
            'generated_by': 'SymQNet Molecular Optimization CLI',
            'version': '1.0.0',
            'model_constraint': f'Trained for exactly {SUPPORTED_QUBITS} qubits',
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
        }
    }
    
    # Add true parameters if available (for validation)
    if hamiltonian_data.get('true_parameters'):
        output_data['validation'] = {
            'true_coupling': hamiltonian_data['true_parameters'].get('coupling', []),
            'true_field': hamiltonian_data['true_parameters'].get('field', [])
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def validate_hamiltonian_data(data: Dict[str, Any]) -> bool:
    """Validate loaded Hamiltonian data structure with 10-qubit constraint."""
    
    required_fields = ['n_qubits', 'pauli_terms', 'format']
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # STRICT QUBIT VALIDATION
    n_qubits = data['n_qubits']
    if n_qubits != SUPPORTED_QUBITS:
        logger.error(
            f"❌ INVALID QUBIT COUNT: {n_qubits} qubits. "
            f"SymQNet-MolOpt only supports {SUPPORTED_QUBITS} qubits."
        )
        return False
    
    # Validate Pauli terms
    for i, term in enumerate(data['pauli_terms']):
        if 'coefficient' not in term:
            logger.error(f"Pauli term {i} missing coefficient")
            return False
        
        if 'pauli_string' not in term and 'pauli_indices' not in term:
            logger.error(f"Pauli term {i} missing operator specification")
            return False
        
        # Check Pauli string length matches qubit count
        if 'pauli_string' in term:
            pauli_len = len(term['pauli_string'])
            if pauli_len != n_qubits:
                logger.error(f"Pauli term {i}: string length {pauli_len} != {n_qubits} qubits")
                return False
    
    logger.debug(f"✅ Hamiltonian data validation passed - {n_qubits} qubits confirmed")
    return True

def create_molecular_hamiltonian_examples():
    """Create ONLY 10-qubit molecular Hamiltonian examples."""
    
    from hamiltonian_parser import HamiltonianParser
    
    # Only create 10-qubit examples
    examples = [('H2O', 10)]  # Only valid combination
    
    for molecule, n_qubits in examples:
        try:
            data = HamiltonianParser.create_example_hamiltonian(molecule, n_qubits)
            
            filename = f"{molecule}_{n_qubits}q.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Created {filename} ({n_qubits} qubits)")
            
        except ValueError as e:
            print(f"❌ Cannot create {molecule}_{n_qubits}q: {e}")

def print_qubit_constraint_info():
    """Print information about the 10-qubit constraint."""
    
    print(f"""
🎯 SYMQNET-MOLOPT QUBIT CONSTRAINT
{'='*50}
✅ Supported qubits: {SUPPORTED_QUBITS}
❌ Other qubit counts: Not supported

💡 Why only {SUPPORTED_QUBITS} qubits?
   • SymQNet was trained specifically on {SUPPORTED_QUBITS}-qubit systems
   • Neural network architecture is optimized for this size
   • Ensures reliable and accurate parameter estimation

🚀 Available examples:
   • H2O_{SUPPORTED_QUBITS}q.json - Water molecule
   • Run 'symqnet-examples' to create examples

📚 To use your molecule:
   • Map to {SUPPORTED_QUBITS} qubits using Jordan-Wigner encoding
   • Use active space approximation
   • Freeze core orbitals to reduce qubit count
{'='*50}
""")

# Additional validation functions
def check_model_compatibility(n_qubits: int) -> bool:
    """Check if qubit count is compatible with trained model."""
    return n_qubits == SUPPORTED_QUBITS

def suggest_qubit_mapping(current_qubits: int) -> str:
    """Suggest how to map current system to 10 qubits."""
    
    if current_qubits == SUPPORTED_QUBITS:
        return "✅ Perfect! Your system matches the supported qubit count."
    
    elif current_qubits < SUPPORTED_QUBITS:
        diff = SUPPORTED_QUBITS - current_qubits
        return (
            f"💡 Your system has {current_qubits} qubits. To use SymQNet-MolOpt:\n"
            f"   • Add {diff} ancilla qubits with identity operations\n"
            f"   • Extend to {SUPPORTED_QUBITS} qubits with padding\n"
            f"   • Use larger active space if possible"
        )
    
    else:
        diff = current_qubits - SUPPORTED_QUBITS  
        return (
            f"💡 Your system has {current_qubits} qubits. To use SymQNet-MolOpt:\n"
            f"   • Reduce by {diff} qubits using active space approximation\n"
            f"   • Freeze {diff} core orbitals\n"
            f"   • Use smaller basis set\n"
            f"   • Apply symmetry reduction techniques"
        )
