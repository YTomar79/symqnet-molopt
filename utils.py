"""
Utility functions for SymQNet molecular optimization
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch

logger = logging.getLogger(__name__)

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
    """Validate CLI input parameters."""
    
    if not hamiltonian_path.exists():
        raise ValueError(f"Hamiltonian file not found: {hamiltonian_path}")
    
    if shots <= 0:
        raise ValueError("Number of shots must be positive")
    
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    
    if max_steps <= 0:
        raise ValueError("Maximum steps must be positive")
    
    if n_rollouts <= 0:
        raise ValueError("Number of rollouts must be positive")
    
    logger.debug("Input validation passed")

def save_results(results: Dict[str, Any], hamiltonian_data: Dict[str, Any],
                config: Dict[str, Any], output_path: Path):
    """Save estimation results to JSON file."""
    
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
            'format': hamiltonian_data['format']
        },
        'experimental_config': config,
        'metadata': {
            'generated_by': 'SymQNet Molecular Optimization CLI',
            'version': '1.0.0',
            'timestamp': str(pd.Timestamp.now())  # You'll need to import pandas
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

def load_symqnet_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load SymQNet model checkpoint with error handling."""
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        if 'episode' in checkpoint:
            logger.info(f"Checkpoint from episode {checkpoint['episode']}")
        
        return checkpoint
    
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint {checkpoint_path}: {e}")

def create_molecular_hamiltonian_examples():
    """Create example molecular Hamiltonian files for testing."""
    
    from hamiltonian_parser import HamiltonianParser
    
    examples = ['H2', 'LiH']
    qubits = [4, 6]
    
    for molecule, n_qubits in zip(examples, qubits):
        data = HamiltonianParser.create_example_hamiltonian(molecule, n_qubits)
        
        filename = f"{molecule}_{n_qubits}q.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Created example: {filename}")

# Data validation utilities
def validate_hamiltonian_data(data: Dict[str, Any]) -> bool:
    """Validate loaded Hamiltonian data structure."""
    
    required_fields = ['n_qubits', 'pauli_terms', 'format']
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate Pauli terms
    for i, term in enumerate(data['pauli_terms']):
        if 'coefficient' not in term:
            logger.error(f"Pauli term {i} missing coefficient")
            return False
        
        if 'pauli_string' not in term and 'pauli_indices' not in term:
            logger.error(f"Pauli term {i} missing operator specification")
            return False
    
    logger.debug("Hamiltonian data validation passed")
    return True

def estimate_runtime(n_qubits: int, max_steps: int, n_rollouts: int, 
                    shots: int) -> Dict[str, float]:
    """Estimate computational runtime."""
    
    # Rough estimates based on system complexity
    base_time_per_step = 0.1  # seconds
    qubit_scaling = n_qubits ** 1.5
    shot_scaling = np.log10(shots)
    
    time_per_step = base_time_per_step * qubit_scaling * shot_scaling
    total_time = time_per_step * max_steps * n_rollouts
    
    return {
        'estimated_total_seconds': total_time,
        'estimated_total_minutes': total_time / 60,
        'time_per_rollout_seconds': time_per_step * max_steps,
        'scaling_factors': {
            'qubits': qubit_scaling,
            'shots': shot_scaling
        }
    }
