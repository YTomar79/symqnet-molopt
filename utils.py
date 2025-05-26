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
from datetime import datetime  # 🔧 FIX: Use datetime instead of pandas

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
    """Save estimation results to JSON file with robust error handling."""
    
    try:
        # Double-check qubit constraint in results
        n_qubits = hamiltonian_data.get('n_qubits', 0)
        if n_qubits != SUPPORTED_QUBITS:
            logger.warning(f"⚠️  Unexpected qubit count in results: {n_qubits} != {SUPPORTED_QUBITS}")
        
        # 🔧 FIX: Robust parameter processing with error handling
        coupling_parameters = []
        if 'coupling_parameters' in results and results['coupling_parameters']:
            for i, param_data in enumerate(results['coupling_parameters']):
                try:
                    if isinstance(param_data, (tuple, list)) and len(param_data) >= 3:
                        mean, ci_low, ci_high = param_data[:3]
                        coupling_parameters.append({
                            'index': i,
                            'mean': float(mean),
                            'confidence_interval': [float(ci_low), float(ci_high)],
                            'uncertainty': float(ci_high - ci_low) / 2.0  # ✅ FIXED: Divide by 2
                        })
                    else:
                        logger.warning(f"Invalid coupling parameter format at index {i}: {param_data}")
                except Exception as e:
                    logger.warning(f"Error processing coupling parameter {i}: {e}")
        
        field_parameters = []
        if 'field_parameters' in results and results['field_parameters']:
            for i, param_data in enumerate(results['field_parameters']):
                try:
                    if isinstance(param_data, (tuple, list)) and len(param_data) >= 3:
                        mean, ci_low, ci_high = param_data[:3]
                        field_parameters.append({
                            'index': i,
                            'mean': float(mean),
                            'confidence_interval': [float(ci_low), float(ci_high)],
                            'uncertainty': float(ci_high - ci_low) / 2.0  # ✅ FIXED: Divide by 2
                        })
                    else:
                        logger.warning(f"Invalid field parameter format at index {i}: {param_data}")
                except Exception as e:
                    logger.warning(f"Error processing field parameter {i}: {e}")
        
        # 🔧 FIX: Robust data extraction with defaults
        output_data = {
            'symqnet_results': {
                'coupling_parameters': coupling_parameters,
                'field_parameters': field_parameters,
                'total_uncertainty': float(results.get('total_uncertainty', 0.0)),
                'avg_measurements_used': float(results.get('avg_measurements', 0.0)),  # ✅ Match bootstrap key
                'confidence_level': float(results.get('confidence_level', 0.95)),
                'n_rollouts': int(results.get('n_rollouts', 0))
            },
            'hamiltonian_info': {
                'molecule': hamiltonian_data.get('molecule', 'unknown'),
                'n_qubits': hamiltonian_data.get('n_qubits', 0),
                'n_pauli_terms': len(hamiltonian_data.get('pauli_terms', [])),
                'format': hamiltonian_data.get('format', 'unknown'),
                'supported_qubits': SUPPORTED_QUBITS,
                'validation_passed': hamiltonian_data.get('n_qubits') == SUPPORTED_QUBITS
            },
            'experimental_config': {
                'shots': config.get('shots', 0),
                'max_steps': config.get('max_steps', 0),
                'n_rollouts': config.get('n_rollouts', 0),
                'confidence': float(config.get('confidence', 0.95)),
                'device': config.get('device', 'cpu'),
                'seed': config.get('seed', 42)
            },
            'metadata': {
                'generated_by': 'SymQNet Molecular Optimization CLI',
                'version': '1.0.3',  # ✅ Updated version
                'model_constraint': f'Trained for exactly {SUPPORTED_QUBITS} qubits',
                'timestamp': datetime.now().isoformat(),
                'parameter_count': {
                    'coupling': len(coupling_parameters),
                    'field': len(field_parameters),
                    'total': len(coupling_parameters) + len(field_parameters)
                }
            }
        }
        
        # Add true parameters if available (for validation)
        if hamiltonian_data.get('true_parameters'):
            output_data['validation'] = {
                'true_coupling': hamiltonian_data['true_parameters'].get('coupling', []),
                'true_field': hamiltonian_data['true_parameters'].get('field', []),
                'has_ground_truth': True
            }
        else:
            output_data['validation'] = {'has_ground_truth': False}
        
        # 🔧 ROBUST: Create output directory and handle permissions
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied creating directory: {output_path.parent}")
            raise
        
        # 🔧 ROBUST: Save with proper error handling
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            file_size = output_path.stat().st_size
            logger.info(f"✅ Results saved to {output_path} ({file_size} bytes)")
            
            # 🔧 ADD: Log summary of what was saved
            logger.info(f"📊 Saved {len(coupling_parameters)} coupling + {len(field_parameters)} field parameters")
            
        except PermissionError:
            logger.error(f"Permission denied writing to: {output_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to write JSON file: {e}")
            raise
            
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        # 🔧 ADD: Try to save minimal results as fallback
        try:
            fallback_data = {
                'error': str(e),
                'partial_results': str(results)[:500],  # First 500 chars
                'timestamp': datetime.now().isoformat()
            }
            fallback_path = output_path.with_suffix('.error.json')
            with open(fallback_path, 'w') as f:
                json.dump(fallback_data, f, indent=2)
            logger.info(f"💾 Saved error info to {fallback_path}")
        except:
            pass
        raise

# 🔧 ADD: Function to verify JSON output structure
def verify_json_output(output_path: Path) -> bool:
    """Verify that the JSON output has correct structure."""
    
    if not output_path.exists():
        logger.error(f"Output file does not exist: {output_path}")
        return False
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Check required top-level keys
        required_keys = ['symqnet_results', 'hamiltonian_info', 'experimental_config', 'metadata']
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key in JSON output: {key}")
                return False
        
        # Check symqnet_results structure
        symqnet = data['symqnet_results']
        if 'coupling_parameters' not in symqnet or 'field_parameters' not in symqnet:
            logger.error("Missing parameter arrays in symqnet_results")
            return False
        
        # Check parameter structure
        for param_list in [symqnet['coupling_parameters'], symqnet['field_parameters']]:
            for param in param_list:
                required_param_keys = ['index', 'mean', 'confidence_interval', 'uncertainty']
                for param_key in required_param_keys:
                    if param_key not in param:
                        logger.error(f"Missing key in parameter: {param_key}")
                        return False
        
        logger.info("✅ JSON output structure verified")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return False
    except Exception as e:
        logger.error(f"Error verifying JSON output: {e}")
        return False


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

# 🔧 ADD: Additional utility functions
def format_parameter_results(coupling_params: List[float], field_params: List[float], 
                           uncertainties: List[float] = None) -> str:
    """Format parameter results for display."""
    
    result_str = "🎯 PARAMETER ESTIMATION RESULTS\n"
    result_str += "=" * 40 + "\n"
    
    result_str += "\n📊 COUPLING PARAMETERS (J):\n"
    for i, param in enumerate(coupling_params):
        if uncertainties and i < len(uncertainties):
            result_str += f"  J_{i}: {param:.6f} ± {uncertainties[i]:.6f}\n"
        else:
            result_str += f"  J_{i}: {param:.6f}\n"
    
    result_str += "\n🧲 FIELD PARAMETERS (h):\n"
    for i, param in enumerate(field_params):
        uncertainty_idx = len(coupling_params) + i
        if uncertainties and uncertainty_idx < len(uncertainties):
            result_str += f"  h_{i}: {param:.6f} ± {uncertainties[uncertainty_idx]:.6f}\n"
        else:
            result_str += f"  h_{i}: {param:.6f}\n"
    
    return result_str

def estimate_computation_time(n_rollouts: int, max_steps: int, shots: int) -> str:
    """Estimate computation time for given parameters."""
    
    # Rough estimates based on typical performance
    seconds_per_measurement = shots / 1000.0  # Rough estimate
    total_measurements = n_rollouts * max_steps
    estimated_seconds = total_measurements * seconds_per_measurement
    
    if estimated_seconds < 60:
        return f"~{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"~{estimated_seconds/60:.1f} minutes"
    else:
        return f"~{estimated_seconds/3600:.1f} hours"

def validate_output_path(output_path: Path) -> bool:
    """Validate that output path is writable."""
    
    try:
        # Try to create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to create a test file
        test_file = output_path.parent / ".test_write"
        test_file.touch()
        test_file.unlink()
        
        return True
    except (PermissionError, OSError) as e:
        logger.error(f"Cannot write to output path {output_path}: {e}")
        return False
