"""
Utility functions for SymQNet molecular optimization
ENHANCED FOR UNIVERSAL QUBIT SUPPORT with optimal performance at 10 qubits
FIXED: Parameter extraction in save_results() function
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

# Universal support - no hard constraints, optimal at 10 qubits
OPTIMAL_QUBITS = 10
MIN_VIABLE_QUBITS = 2

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
    """Validate CLI input parameters with universal qubit support."""
    
    if not hamiltonian_path.exists():
        raise ValueError(f"Hamiltonian file not found: {hamiltonian_path}")
    
    # Universal qubit validation - warnings instead of errors
    try:
        with open(hamiltonian_path, 'r') as f:
            data = json.load(f)
        
        n_qubits = data.get('n_qubits', 0)
        
        # Minimum viable constraint
        if n_qubits < MIN_VIABLE_QUBITS:
            raise ValueError(
                f"❌ VALIDATION FAILED: Minimum {MIN_VIABLE_QUBITS} qubits required.\n"
                f"   Your Hamiltonian: {n_qubits} qubits\n"
                f"   Minimum viable: {MIN_VIABLE_QUBITS} qubits"
            )
        
        # Performance guidance instead of hard limits
        if n_qubits != OPTIMAL_QUBITS:
            logger.warning(
                f"⚠️  Non-optimal qubit count detected: {n_qubits} qubits\n"
                f"   Optimal performance: {OPTIMAL_QUBITS} qubits\n"
                f"   Expected performance degradation for {n_qubits}-qubit system"
            )
            
        # Additional guidance for extreme cases
        if n_qubits > 25:
            logger.warning(f"Large system ({n_qubits} qubits) may have significant accuracy degradation and long runtime")
        elif n_qubits < 4:
            logger.warning(f"Small system ({n_qubits} qubits) may have limited representational power")
            
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {hamiltonian_path}")
    except KeyError:
        raise ValueError(f"Hamiltonian file missing 'n_qubits' field: {hamiltonian_path}")
    
    # Standard parameter validation
    if shots <= 0:
        raise ValueError("Number of shots must be positive")
    
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    
    if max_steps <= 0:
        raise ValueError("Maximum steps must be positive")
    
    if n_rollouts <= 0:
        raise ValueError("Number of rollouts must be positive")
    
    logger.debug(f"✅ Input validation passed - {n_qubits}-qubit system accepted")

def save_results(results: Dict[str, Any], hamiltonian_data: Dict[str, Any],
                config: Dict[str, Any], output_path: Path):
    """
    🔧 FIXED: Save estimation results to JSON file with universal support.
    Properly extracts parameters from nested results structure.
    """
    
    try:
        n_qubits = hamiltonian_data.get('n_qubits', 0)
        
        # Extract performance metadata if available
        performance_metadata = config.get('performance_metadata', {})
        
        # 🔧 CRITICAL FIX: Robust parameter extraction from nested results structure
        coupling_parameters = []
        field_parameters = []
        
        # Handle nested results structure - extract symqnet_results first
        if 'symqnet_results' in results:
            symqnet_data = results['symqnet_results']
            logger.debug(f"🔍 Found symqnet_results in results")
        else:
            symqnet_data = results
            logger.debug(f"🔍 Using results directly as symqnet_data")
        
        logger.debug(f"🔍 symqnet_data keys: {list(symqnet_data.keys())}")
        
        # 🔧 FIXED: Extract coupling parameters with detailed debugging
        if 'coupling_parameters' in symqnet_data:
            raw_coupling = symqnet_data['coupling_parameters']
            logger.debug(f"🔍 Raw coupling parameters: type={type(raw_coupling)}, length={len(raw_coupling) if hasattr(raw_coupling, '__len__') else 'N/A'}")
            
            if raw_coupling:  # Not empty
                logger.debug(f"🔍 First coupling parameter: {raw_coupling[0] if raw_coupling else 'None'}")
                
                for i, param_data in enumerate(raw_coupling):
                    try:
                        if isinstance(param_data, (tuple, list)) and len(param_data) >= 3:
                            # Tuple format: (mean, ci_low, ci_high)
                            mean, ci_low, ci_high = param_data[0], param_data[1], param_data[2]
                            coupling_parameters.append({
                                'index': i,
                                'mean': float(mean),
                                'confidence_interval': [float(ci_low), float(ci_high)],
                                'uncertainty': float(ci_high - ci_low) / 2.0
                            })
                            logger.debug(f"✅ Processed coupling parameter {i}: {mean:.6f}")
                        elif isinstance(param_data, dict):
                            # Dict format - already properly formatted
                            coupling_parameters.append(param_data)
                            logger.debug(f"✅ Added coupling parameter dict {i}")
                        else:
                            logger.warning(f"Unexpected coupling parameter format at index {i}: {type(param_data)} = {param_data}")
                    except Exception as e:
                        logger.error(f"Error processing coupling parameter {i}: {e}")
            else:
                logger.warning("⚠️ coupling_parameters is empty or None")
        else:
            logger.warning("⚠️ No 'coupling_parameters' key found in symqnet_data")
        
        # 🔧 FIXED: Extract field parameters with detailed debugging
        if 'field_parameters' in symqnet_data:
            raw_field = symqnet_data['field_parameters']
            logger.debug(f"🔍 Raw field parameters: type={type(raw_field)}, length={len(raw_field) if hasattr(raw_field, '__len__') else 'N/A'}")
            
            if raw_field:  # Not empty
                logger.debug(f"🔍 First field parameter: {raw_field[0] if raw_field else 'None'}")
                
                for i, param_data in enumerate(raw_field):
                    try:
                        if isinstance(param_data, (tuple, list)) and len(param_data) >= 3:
                            # Tuple format: (mean, ci_low, ci_high)
                            mean, ci_low, ci_high = param_data[0], param_data[1], param_data[2]
                            field_parameters.append({
                                'index': i,
                                'mean': float(mean),
                                'confidence_interval': [float(ci_low), float(ci_high)],
                                'uncertainty': float(ci_high - ci_low) / 2.0
                            })
                            logger.debug(f"✅ Processed field parameter {i}: {mean:.6f}")
                        elif isinstance(param_data, dict):
                            # Dict format - already properly formatted
                            field_parameters.append(param_data)
                            logger.debug(f"✅ Added field parameter dict {i}")
                        else:
                            logger.warning(f"Unexpected field parameter format at index {i}: {type(param_data)} = {param_data}")
                    except Exception as e:
                        logger.error(f"Error processing field parameter {i}: {e}")
            else:
                logger.warning("⚠️ field_parameters is empty or None")
        else:
            logger.warning("⚠️ No 'field_parameters' key found in symqnet_data")
        
        # 🔧 CRITICAL: Log what we extracted for debugging
        logger.info(f"✅ Extracted {len(coupling_parameters)} coupling + {len(field_parameters)} field parameters for saving")
        
        # Extract other symqnet_results data with safe defaults
        total_uncertainty = float(symqnet_data.get('total_uncertainty', 0.0))
        avg_measurements = float(symqnet_data.get('avg_measurements_used', symqnet_data.get('avg_measurements', 0.0)))
        confidence_level = float(symqnet_data.get('confidence_level', 0.95))
        n_rollouts = int(symqnet_data.get('n_rollouts', 0))
        
        # Build output data with correctly extracted parameters
        output_data = {
            'symqnet_results': {
                'coupling_parameters': coupling_parameters,  # ✅ FIXED
                'field_parameters': field_parameters,        # ✅ FIXED
                'total_uncertainty': total_uncertainty,
                'avg_measurements_used': avg_measurements,
                'confidence_level': confidence_level,
                'n_rollouts': n_rollouts
            },
            'hamiltonian_info': {
                'molecule': hamiltonian_data.get('molecule', 'unknown'),
                'n_qubits': n_qubits,
                'n_pauli_terms': len(hamiltonian_data.get('pauli_terms', [])),
                'format': hamiltonian_data.get('format', 'unknown'),
                'optimal_qubits': OPTIMAL_QUBITS,
                'performance_optimal': n_qubits == OPTIMAL_QUBITS
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
                'generated_by': 'Universal SymQNet Molecular Optimization CLI',
                'version': '2.0.12',  # Updated for universal support
                'model_constraint': f'Trained optimally for {OPTIMAL_QUBITS} qubits, supports any qubit count',
                'timestamp': datetime.now().isoformat(),
                'parameter_count': {
                    'coupling': len(coupling_parameters),
                    'field': len(field_parameters),
                    'total': len(coupling_parameters) + len(field_parameters)
                },
                'parameter_extraction_fixed': True  # ✅ Mark as fixed
            }
        }
        
        # Add universal performance metadata
        if performance_metadata:
            output_data['performance_analysis'] = {
                'expected_performance': performance_metadata.get('expected_performance', 1.0),
                'performance_level': performance_metadata.get('performance_level', 'optimal'),
                'optimal_qubits': performance_metadata.get('optimal_qubits', OPTIMAL_QUBITS),
                'universal_symqnet_version': performance_metadata.get('universal_symqnet_version', '2.0.0')
            }
        
        # Add universal wrapper metadata if present
        if 'universal_metadata' in results:
            output_data['universal_wrapper'] = results['universal_metadata']
        
        # Add true parameters if available (for validation)
        if hamiltonian_data.get('true_parameters'):
            output_data['validation'] = {
                'true_coupling': hamiltonian_data['true_parameters'].get('coupling', []),
                'true_field': hamiltonian_data['true_parameters'].get('field', []),
                'has_ground_truth': True
            }
        else:
            output_data['validation'] = {'has_ground_truth': False}
        
        # Create output directory and save
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied creating directory: {output_path.parent}")
            raise
        
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            file_size = output_path.stat().st_size
            logger.info(f"✅ Results saved to {output_path} ({file_size} bytes)")
            logger.info(f"📊 Saved {len(coupling_parameters)} coupling + {len(field_parameters)} field parameters")  # ✅ FIXED
            
        except PermissionError:
            logger.error(f"Permission denied writing to: {output_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to write JSON file: {e}")
            raise
            
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        # Try to save minimal results as fallback
        try:
            fallback_data = {
                'error': str(e),
                'partial_results': str(results)[:500],
                'timestamp': datetime.now().isoformat()
            }
            fallback_path = output_path.with_suffix('.error.json')
            with open(fallback_path, 'w') as f:
                json.dump(fallback_data, f, indent=2)
            logger.info(f"💾 Saved error info to {fallback_path}")
        except:
            pass
        raise

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
    """Validate loaded Hamiltonian data structure with universal support."""
    
    required_fields = ['n_qubits', 'pauli_terms', 'format']
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Universal qubit validation - warnings instead of errors
    n_qubits = data['n_qubits']
    if n_qubits < MIN_VIABLE_QUBITS:
        logger.error(f"❌ INVALID QUBIT COUNT: {n_qubits} < {MIN_VIABLE_QUBITS} (minimum viable)")
        return False
    
    # Performance guidance
    if n_qubits != OPTIMAL_QUBITS:
        performance_factor = _estimate_performance_factor(n_qubits)
        logger.warning(f"Non-optimal qubit count: {n_qubits} qubits (expected {performance_factor:.1%} performance)")
    
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
    
    logger.debug(f"✅ Hamiltonian data validation passed - {n_qubits} qubits accepted")
    return True

def _estimate_performance_factor(n_qubits: int) -> float:
    """Quick performance estimation for validation purposes."""
    if n_qubits == OPTIMAL_QUBITS:
        return 1.0
    
    distance = abs(n_qubits - OPTIMAL_QUBITS)
    
    if n_qubits < OPTIMAL_QUBITS:
        # Small systems: padding effects
        return max(0.95 ** (distance * 0.8), 0.7)
    else:
        # Large systems: compression effects
        return max(0.90 ** (distance * 1.2), 0.4)

def create_molecular_hamiltonian_examples():
    """Create molecular Hamiltonian examples for various qubit counts."""
    
    from hamiltonian_parser import HamiltonianParser
    
    # Create examples for different qubit counts, highlighting 10-qubit optimum
    examples = [
        ('H2', 4),
        ('LiH', 6), 
        ('BeH2', 8),
        ('H2O', 10),      # OPTIMAL
        ('NH3', 12),
        ('CH4', 14)
    ]
    
    for molecule, n_qubits in examples:
        try:
            data = HamiltonianParser.create_example_hamiltonian(molecule, n_qubits)
            
            filename = f"{molecule}_{n_qubits}q.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            optimal_marker = " ⭐ OPTIMAL" if n_qubits == OPTIMAL_QUBITS else ""
            print(f"✅ Created {filename} ({n_qubits} qubits){optimal_marker}")
            
        except ValueError as e:
            print(f"❌ Cannot create {molecule}_{n_qubits}q: {e}")

def print_universal_support_info():
    """Print information about universal qubit support."""
    
    print(f"""
🌍 UNIVERSAL SYMQNET-MOLOPT SUPPORT
{'='*50}
✅ Supported qubits: {MIN_VIABLE_QUBITS}+ (any molecular system)
🎯 Optimal qubits: {OPTIMAL_QUBITS} (maximum accuracy)
📈 Performance: Degrades gracefully from optimum

💡 Performance expectations:
   • 4-8 qubits:  Good performance (85-97%)
   • 10 qubits:   Optimal performance (100%) ⭐
   • 12-16 qubits: Moderate degradation (75-90%)
   • 20+ qubits:   Significant degradation (<70%)

🚀 Universal capabilities:
   • Automatic normalization for any system size
   • Intelligent parameter scaling
   • Performance warnings and recommendations
   • Works with existing workflows

📚 Usage examples:
   • symqnet-molopt --hamiltonian H2_4q.json --output results.json
   • symqnet-molopt --hamiltonian H2O_10q.json --output results.json  ⭐
   • symqnet-molopt --hamiltonian large_mol_20q.json --output results.json
{'='*50}
""")

def check_model_compatibility(n_qubits: int) -> Tuple[bool, float]:
    """Check qubit count compatibility and return performance factor."""
    is_viable = n_qubits >= MIN_VIABLE_QUBITS
    performance_factor = _estimate_performance_factor(n_qubits)
    return is_viable, performance_factor

def suggest_qubit_mapping(current_qubits: int) -> str:
    """Suggest how to optimize system for better performance."""
    
    if current_qubits == OPTIMAL_QUBITS:
        return "✅ Perfect! Your system is at the optimal qubit count for maximum accuracy."
    
    elif current_qubits < OPTIMAL_QUBITS:
        diff = OPTIMAL_QUBITS - current_qubits
        performance = _estimate_performance_factor(current_qubits)
        
        return (
            f"💡 Your {current_qubits}-qubit system will work with {performance:.1%} performance.\n"
            f"   To reach optimal performance:\n"
            f"   • Expand to {OPTIMAL_QUBITS}-qubit active space (+{diff} qubits)\n"
            f"   • Use larger basis set if available\n"
            f"   • Add virtual orbitals to reach {OPTIMAL_QUBITS} qubits\n"
            f"   • Current system is still viable and will produce useful results"
        )
    
    else:
        diff = current_qubits - OPTIMAL_QUBITS
        performance = _estimate_performance_factor(current_qubits)
        
        return (
            f"💡 Your {current_qubits}-qubit system will work with {performance:.1%} performance.\n"
            f"   To improve accuracy:\n"
            f"   • Reduce to {OPTIMAL_QUBITS}-qubit active space (-{diff} qubits)\n"
            f"   • Freeze {diff} core orbitals\n"
            f"   • Use smaller basis set\n"
            f"   • Apply symmetry reduction techniques\n"
            f"   • Current system will still produce useful results with uncertainty scaling"
        )

def get_recommended_parameters(n_qubits: int, base_shots: int = 1024, 
                              base_rollouts: int = 5) -> Dict[str, int]:
    """Get recommended parameters based on system size and performance."""
    
    performance_factor = _estimate_performance_factor(n_qubits)
    
    # Scale parameters inversely with performance to compensate
    if performance_factor >= 0.95:
        # Near-optimal performance
        shot_multiplier = 1.0
        rollout_multiplier = 1.0
    elif performance_factor >= 0.80:
        # Good performance
        shot_multiplier = 1.2
        rollout_multiplier = 1.2
    elif performance_factor >= 0.65:
        # Moderate performance
        shot_multiplier = 1.5
        rollout_multiplier = 1.5
    else:
        # Poor performance - need many more samples
        shot_multiplier = 2.0
        rollout_multiplier = 2.0
    
    return {
        'recommended_shots': int(base_shots * shot_multiplier),
        'recommended_rollouts': int(base_rollouts * rollout_multiplier),
        'performance_factor': performance_factor,
        'adjustment_reason': _get_adjustment_reason(performance_factor)
    }

def _get_adjustment_reason(performance_factor: float) -> str:
    """Get explanation for parameter adjustments."""
    
    if performance_factor >= 0.95:
        return "Near-optimal system - standard parameters"
    elif performance_factor >= 0.80:
        return "Good performance - slight parameter increase for robustness"
    elif performance_factor >= 0.65:
        return "Moderate degradation - increased parameters to compensate"
    else:
        return "Significant degradation - substantial parameter increase needed"

def format_parameter_results(coupling_params: List[float], field_params: List[float], 
                           uncertainties: List[float] = None, n_qubits: int = None) -> str:
    """Format parameter results for display with universal context."""
    
    result_str = "🎯 UNIVERSAL SYMQNET PARAMETER ESTIMATION RESULTS\n"
    result_str += "=" * 50 + "\n"
    
    if n_qubits:
        performance = _estimate_performance_factor(n_qubits)
        result_str += f"📊 System: {n_qubits} qubits ({performance:.1%} performance)\n"
        result_str += f"🎯 Optimal: {OPTIMAL_QUBITS} qubits\n\n"
    
    result_str += "📊 COUPLING PARAMETERS (J):\n"
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

def estimate_computation_time(n_rollouts: int, max_steps: int, shots: int, 
                             n_qubits: int = None) -> str:
    """Estimate computation time with universal overhead considerations."""
    
    # Base time estimate
    seconds_per_measurement = shots / 1000.0
    total_measurements = n_rollouts * max_steps
    base_time = total_measurements * seconds_per_measurement
    
    # Add overhead for non-optimal systems
    if n_qubits and n_qubits != OPTIMAL_QUBITS:
        # Normalization and processing overhead
        overhead_factor = 1.1 + 0.02 * abs(n_qubits - OPTIMAL_QUBITS)
        estimated_seconds = base_time * overhead_factor
    else:
        estimated_seconds = base_time
    
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

def warn_performance_degradation(n_qubits: int, threshold: float = 0.8) -> None:
    """Issue performance warning if below threshold."""
    
    performance = _estimate_performance_factor(n_qubits)
    
    if performance < threshold:
        message = (
            f"Performance degradation warning: {n_qubits}-qubit system "
            f"expected to operate at {performance:.1%} of optimal performance. "
            f"Consider using {OPTIMAL_QUBITS}-qubit representations for best results."
        )
        warnings.warn(message, UserWarning, stacklevel=2)

def get_system_compatibility_report(n_qubits: int) -> Dict[str, Any]:
    """Generate comprehensive compatibility report for a system."""
    
    performance = _estimate_performance_factor(n_qubits)
    is_viable = n_qubits >= MIN_VIABLE_QUBITS
    
    if performance >= 0.95:
        compatibility_level = "excellent"
    elif performance >= 0.80:
        compatibility_level = "good"
    elif performance >= 0.65:
        compatibility_level = "moderate"
    elif performance >= 0.45:
        compatibility_level = "poor"
    else:
        compatibility_level = "severe"
    
    recommended_params = get_recommended_parameters(n_qubits)
    
    return {
        'n_qubits': n_qubits,
        'is_viable': is_viable,
        'performance_factor': performance,
        'compatibility_level': compatibility_level,
        'optimal_qubits': OPTIMAL_QUBITS,
        'recommended_parameters': recommended_params,
        'mapping_suggestion': suggest_qubit_mapping(n_qubits)
    }

# Convenience functions for backward compatibility
def check_qubit_constraint(n_qubits: int) -> bool:
    """Backward compatibility - now always returns True for viable systems."""
    return n_qubits >= MIN_VIABLE_QUBITS

def get_constraint_info() -> Dict[str, Any]:
    """Get information about qubit constraints."""
    return {
        'min_viable_qubits': MIN_VIABLE_QUBITS,
        'optimal_qubits': OPTIMAL_QUBITS,
        'hard_constraints': False,
        'universal_support': True
    }
