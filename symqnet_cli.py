#!/usr/bin/env python3
"""
SymQNet Molecular Optimization CLI - CORRECTED UNIVERSAL VERSION

Fixed the parameter extraction issue by using proven working rollout logic.

Usage:
    symqnet-molopt --hamiltonian molecule.json --shots 1024 --output results.json
"""

import click
import json
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import sys
import os
import warnings

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core imports (always available)
from hamiltonian_parser import HamiltonianParser
from measurement_simulator import MeasurementSimulator
from policy_engine import PolicyEngine
from bootstrap_estimator import BootstrapEstimator
from utils import setup_logging, validate_inputs, save_results

# 🔧 GRACEFUL IMPORT HANDLING FOR UNIVERSAL COMPONENTS
UNIVERSAL_MODE = False
try:
    from universal_wrapper import UniversalSymQNetWrapper
    from performance_estimator import PerformanceEstimator, get_performance_warning
    UNIVERSAL_MODE = True
    logger = logging.getLogger(__name__)
    logger.info("🌍 Universal mode available")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️  Universal components not available: {e}")
    logger.warning("🔧 Falling back to 10-qubit-only mode")
    
    # Fallback implementations
    class PerformanceEstimator:
        def __init__(self, optimal_qubits=10):
            self.optimal_qubits = optimal_qubits
        
        def estimate_performance(self, n_qubits):
            from dataclasses import dataclass
            from enum import Enum
            
            class PerformanceLevel(Enum):
                OPTIMAL = "optimal"
                POOR = "poor"
            
            @dataclass
            class PerformanceReport:
                performance_factor: float = 1.0 if n_qubits == 10 else 0.0
                level: PerformanceLevel = PerformanceLevel.OPTIMAL if n_qubits == 10 else PerformanceLevel.POOR
                recommendations: List[str] = None
                
                def __post_init__(self):
                    if self.recommendations is None:
                        if n_qubits != 10:
                            self.recommendations = [f"Use exactly 10-qubit systems for this version"]
                        else:
                            self.recommendations = []
            
            return PerformanceReport()
        
        def get_recommended_parameters(self, n_qubits):
            return {'shots': 1024, 'n_rollouts': 5}
    
    def get_performance_warning(n_qubits, optimal_qubits=10):
        if n_qubits != optimal_qubits:
            return f"Only {optimal_qubits}-qubit systems supported in fallback mode"
        return None

# Architecture imports
try:
    from architectures import (
        VariationalAutoencoder,
        FixedSymQNetWithEstimator,
        GraphEmbed,
        TemporalContextualAggregator,
        PolicyValueHead,
        SpinChainEnv
    )
except ImportError as e:
    logger.warning(f"⚠️  Architecture imports failed: {e}")
    # Continue anyway - let it fail later with clearer error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_model_files(model_path: Optional[Path], vae_path: Optional[Path]) -> Tuple[Path, Path]:
    """🔧 FIXED: Ensure model files exist, provide helpful error messages if not"""
    
    # Default paths
    default_model_path = Path("models/FINAL_FIXED_SYMQNET.pth")
    default_vae_path = Path("models/vae_M10_f.pth")
    
    # Use provided paths or defaults
    final_model_path = model_path or default_model_path
    final_vae_path = vae_path or default_vae_path
    
    # Check if files exist and provide helpful error messages
    if not final_model_path.exists():
        raise click.ClickException(
            f"❌ Model file not found: {final_model_path}\n\n"
            f"💡 SOLUTIONS:\n"
            f"   1. Create models/ directory in current location\n"
            f"   2. Download model files from your GitHub repository\n"
            f"   3. Provide custom path: --model-path /path/to/your/model.pth\n"
            f"   4. Ensure you're running from the correct directory\n\n"
            f"🔍 Current working directory: {os.getcwd()}\n"
            f"📂 Expected model location: {final_model_path.absolute()}"
        )
    
    if not final_vae_path.exists():
        raise click.ClickException(
            f"❌ VAE file not found: {final_vae_path}\n\n"
            f"💡 SOLUTIONS:\n"
            f"   1. Create models/ directory in current location\n"
            f"   2. Download VAE files from your GitHub repository\n"
            f"   3. Provide custom path: --vae-path /path/to/your/vae.pth\n"
            f"   4. Ensure you're running from the correct directory\n\n"
            f"🔍 Current working directory: {os.getcwd()}\n"
            f"📂 Expected VAE location: {final_vae_path.absolute()}"
        )
    
    return final_model_path, final_vae_path


def find_hamiltonian_file(hamiltonian_path: Path) -> Path:
    """Find Hamiltonian file in examples or user directories"""
    
    # If absolute path or relative path that exists, use as-is
    if hamiltonian_path.is_absolute() or hamiltonian_path.exists():
        return hamiltonian_path
    
    # Check user directory first
    user_path = Path("user_hamiltonians") / hamiltonian_path
    if user_path.exists():
        logger.info(f"Found in user directory: {user_path}")
        return user_path
    
    # Check examples directory
    examples_path = Path("examples") / hamiltonian_path
    if examples_path.exists():
        logger.info(f"Found in examples directory: {examples_path}")
        return examples_path
    
    # Not found
    raise ValueError(
        f"Hamiltonian file not found: {hamiltonian_path}\n"
        f"Searched in:\n"
        f"  • Current directory\n"
        f"  • user_hamiltonians/\n"
        f"  • examples/\n\n"
        f"Use 'symqnet-add {hamiltonian_path}' to add your file to the system."
    )


def validate_hamiltonian_universal(hamiltonian_path: Path) -> Dict[str, any]:
    """Validate Hamiltonian with universal support or fallback"""
    
    try:
        with open(hamiltonian_path, 'r') as f:
            hamiltonian_data = json.load(f)
        
        n_qubits = hamiltonian_data.get('n_qubits', 0)
        
        if UNIVERSAL_MODE:
            # Universal validation - any qubit count ≥2
            if n_qubits < 2:
                raise ValueError(f"Minimum 2 qubits required, got {n_qubits}")
            
            # Performance guidance
            if n_qubits > 25:
                logger.warning(f"Large system ({n_qubits} qubits) may have very long runtime")
            
            logger.info(f"✅ Validated: {n_qubits}-qubit Hamiltonian (Universal mode)")
            
        else:
            # Fallback mode - only 10 qubits
            if n_qubits != 10:
                raise ValueError(
                    f"❌ FALLBACK MODE: Only 10-qubit systems supported.\n"
                    f"   Your Hamiltonian: {n_qubits} qubits\n"
                    f"   Required: exactly 10 qubits\n\n"
                    f"💡 To enable universal support:\n"
                    f"   • Install universal components (universal_wrapper.py, performance_estimator.py)\n"
                    f"   • Or use a 10-qubit molecular representation"
                )
            
            logger.info(f"✅ Validated: {n_qubits}-qubit Hamiltonian (Fallback mode)")
        
        return hamiltonian_data
        
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {hamiltonian_path}")
    except FileNotFoundError:
        raise ValueError(f"Hamiltonian file not found: {hamiltonian_path}")


def run_optimization_universal(hamiltonian_data, model_path, vae_path, device, shots, n_rollouts, max_steps, warn_performance=True):
    """
    🔧 FIXED: Run optimization with working parameter extraction
    Uses proven rollout logic for both universal and fallback modes
    """
    
    original_qubits = hamiltonian_data['n_qubits']
    
    if UNIVERSAL_MODE:
        # 🔧 FIXED: Use working rollout logic with universal normalization
        logger.info("🌍 Using Universal SymQNet with FIXED parameter extraction")
        
        # Step 1: Normalize Hamiltonian for 10-qubit processing
        universal_wrapper = UniversalSymQNetWrapper(
            trained_model_path=model_path,
            trained_vae_path=vae_path,
            device=device
        )
        
        # Get normalized hamiltonian
        normalized_hamiltonian = universal_wrapper._normalize_hamiltonian(hamiltonian_data)
        logger.info(f"🔄 Normalized {original_qubits}-qubit → 10-qubit system")
        
        # Step 2: Use the PROVEN WORKING rollout logic on normalized system
        policy = PolicyEngine(model_path, vae_path, device)
        simulator = MeasurementSimulator(normalized_hamiltonian, shots, device)
        estimator = BootstrapEstimator()
        
        # Step 3: Run rollouts with the WORKING logic
        rollout_results = []
        for i in range(n_rollouts):
            logger.info(f"🔄 Universal rollout {i+1}/{n_rollouts}")
            policy.reset()
            
            measurements = []
            parameter_estimates = []
            current_measurement = simulator.get_initial_measurement()
            
            for step in range(max_steps):
                # 🔧 THIS IS THE WORKING LOGIC FROM FALLBACK MODE
                action_info = policy.get_action(current_measurement)
                measurement_result = simulator.execute_measurement(
                    qubit_indices=action_info['qubits'],
                    pauli_operators=action_info['operators'],
                    evolution_time=action_info['time']
                )
                
                measurements.append(measurement_result)
                
                # 🔧 CRITICAL: This call actually works!
                param_estimate = policy.get_parameter_estimate()
                parameter_estimates.append(param_estimate)
                
                current_measurement = measurement_result['expectation_values']
                
                # Debug: Check if we're getting real parameters
                if step == 0:
                    logger.debug(f"🧪 First parameter estimate: shape={param_estimate.shape}, "
                                f"range=[{param_estimate.min():.6f}, {param_estimate.max():.6f}]")
                
                if step > 5 and policy.has_converged(parameter_estimates):
                    logger.debug(f"Converged at step {step}")
                    break
            
            # Store rollout results
            rollout_results.append({
                'rollout_id': i,
                'measurements': measurements,
                'parameter_estimates': parameter_estimates,
                'final_estimate': parameter_estimates[-1] if parameter_estimates else None,
                'convergence_step': step
            })
            
            # Debug final estimate
            if parameter_estimates:
                final_est = parameter_estimates[-1]
                logger.debug(f"Rollout {i} final estimate: shape={final_est.shape}, "
                           f"non-zero: {not np.allclose(final_est, 0)}")
        
        # Step 4: Bootstrap analysis on 10-qubit results
        logger.info("📊 Computing confidence intervals on normalized results...")
        bootstrap_results = estimator.compute_intervals(rollout_results)
        
        # Step 5: Denormalize results back to original system
        logger.info(f"🔄 Denormalizing results: 10-qubit → {original_qubits}-qubit")
        normalized_results = {
            'symqnet_results': bootstrap_results,
            'rollout_results': rollout_results
        }
        
        final_results = universal_wrapper._denormalize_results(normalized_results, original_qubits)
        
        # Add universal metadata
        final_results['universal_metadata'] = {
            'original_qubits': original_qubits,
            'normalized_to': 10,
            'expected_performance': universal_wrapper._calculate_performance_factor(original_qubits),
            'normalization_applied': True,
            'optimal_at': 10,
            'fixed_parameter_extraction': True
        }
        
        return final_results
    
    else:
        # Fallback to original implementation (this already works)
        logger.info("🔧 Using fallback 10-qubit implementation")
        
        # Initialize components directly
        policy = PolicyEngine(model_path, vae_path, device)
        simulator = MeasurementSimulator(hamiltonian_data, shots, device)
        estimator = BootstrapEstimator()
        
        # Run rollouts
        rollout_results = []
        for i in range(n_rollouts):
            logger.info(f"  Rollout {i+1}/{n_rollouts}")
            policy.reset()
            
            measurements = []
            parameter_estimates = []
            current_measurement = simulator.get_initial_measurement()
            
            for step in range(max_steps):
                action_info = policy.get_action(current_measurement)
                measurement_result = simulator.execute_measurement(
                    qubit_indices=action_info['qubits'],
                    pauli_operators=action_info['operators'],
                    evolution_time=action_info['time']
                )
                
                measurements.append(measurement_result)
                param_estimate = policy.get_parameter_estimate()
                parameter_estimates.append(param_estimate)
                current_measurement = measurement_result['expectation_values']
                
                if step > 5 and policy.has_converged(parameter_estimates):
                    break
            
            rollout_results.append({
                'rollout_id': i,
                'final_estimate': parameter_estimates[-1] if parameter_estimates else None,
                'convergence_step': step
            })
        
        # Bootstrap analysis
        logger.info("📊 Computing confidence intervals...")
        bootstrap_results = estimator.compute_intervals(rollout_results)
        
        return {
            'symqnet_results': bootstrap_results,
            'rollout_results': rollout_results,
            'fallback_mode': True
        }


def print_performance_info(n_qubits: int, performance_estimator: PerformanceEstimator):
    """Print performance information and recommendations"""
    
    print("\n" + "="*60)
    if UNIVERSAL_MODE:
        print("🌍 UNIVERSAL SYMQNET PERFORMANCE ANALYSIS")
    else:
        print("🔧 FALLBACK MODE ANALYSIS")
    print("="*60)
    
    report = performance_estimator.estimate_performance(n_qubits)
    
    print(f"📊 System Size: {n_qubits} qubits")
    print(f"🎯 Optimal Size: {performance_estimator.optimal_qubits} qubits")
    
    if UNIVERSAL_MODE:
        print(f"📈 Expected Performance: {report.performance_factor:.1%} of optimal")
        print(f"🏷️  Performance Level: {report.level.value.upper()}")
        
        if n_qubits == performance_estimator.optimal_qubits:
            print("\n✨ Running at optimal performance!")
        else:
            if hasattr(report, 'uncertainty_scaling'):
                print(f"\n📏 Uncertainty Scaling: {report.uncertainty_scaling:.1f}x")
    else:
        if n_qubits == 10:
            print("📈 Expected Performance: 100% (exact training match)")
            print("✨ Running in supported mode!")
        else:
            print("📈 Expected Performance: Not supported in fallback mode")
            print("⚠️  Only 10-qubit systems supported")
    
    # Show recommendations
    if report.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"   {i}. {rec}")


def print_summary(results: Dict, n_qubits: int, performance_factor: float):
    """Print a formatted summary of results with performance context."""
    
    print("\n" + "="*60)
    print("🎯 SYMQNET MOLECULAR OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"🧪 System: {n_qubits} qubits")
    print(f"📊 Performance: {performance_factor:.1%} of optimal")
    
    if 'fallback_mode' in results:
        print("🔧 Mode: Fallback (10-qubit only)")
    elif UNIVERSAL_MODE:
        print("🌍 Mode: Universal (FIXED)")
    
    # Extract results from nested structure if needed
    if 'symqnet_results' in results:
        symqnet_results = results['symqnet_results']
    else:
        symqnet_results = results
    
    if 'coupling_parameters' in symqnet_results:
        coupling_params = symqnet_results['coupling_parameters']
        if coupling_params and len(coupling_params) > 0:
            if isinstance(coupling_params[0], tuple):
                # Tuple format: (mean, ci_low, ci_high)
                coupling_count = len(coupling_params)
                print(f"\n⚛️  COUPLING PARAMETERS ({coupling_count} estimated):")
                for i, (mean, ci_low, ci_high) in enumerate(coupling_params):
                    uncertainty = (ci_high - ci_low) / 2
                    print(f"  J_{i}: {mean:8.6f} ± {uncertainty:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
            else:
                # Dict format
                coupling_count = len(coupling_params)
                print(f"\n⚛️  COUPLING PARAMETERS ({coupling_count} estimated):")
                for param in coupling_params:
                    i = param.get('index', 0)
                    mean = param.get('mean', 0)
                    uncertainty = param.get('uncertainty', 0)
                    ci = param.get('confidence_interval', [0, 0])
                    print(f"  J_{i}: {mean:8.6f} ± {uncertainty:.6f} [{ci[0]:.6f}, {ci[1]:.6f}]")
    
    if 'field_parameters' in symqnet_results:
        field_params = symqnet_results['field_parameters']
        if field_params and len(field_params) > 0:
            if isinstance(field_params[0], tuple):
                # Tuple format: (mean, ci_low, ci_high)
                field_count = len(field_params)
                print(f"\n🧲 FIELD PARAMETERS ({field_count} estimated):")
                for i, (mean, ci_low, ci_high) in enumerate(field_params):
                    uncertainty = (ci_high - ci_low) / 2
                    print(f"  h_{i}: {mean:8.6f} ± {uncertainty:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
            else:
                # Dict format
                field_count = len(field_params)
                print(f"\n🧲 FIELD PARAMETERS ({field_count} estimated):")
                for param in field_params:
                    i = param.get('index', 0)
                    mean = param.get('mean', 0)
                    uncertainty = param.get('uncertainty', 0)
                    ci = param.get('confidence_interval', [0, 0])
                    print(f"  h_{i}: {mean:8.6f} ± {uncertainty:.6f} [{ci[0]:.6f}, {ci[1]:.6f}]")
    
    if 'total_uncertainty' in symqnet_results:
        print(f"\n📏 Total Parameter Uncertainty: {symqnet_results['total_uncertainty']:.6f}")
    
    if 'n_rollouts' in symqnet_results:
        print(f"🔄 Rollouts Completed: {symqnet_results['n_rollouts']}")
    
    print("="*60)


def get_recommended_params_for_system(n_qubits: int, 
                                     user_shots: int, 
                                     user_rollouts: int,
                                     performance_estimator: PerformanceEstimator) -> Dict[str, int]:
    """Get recommended parameters based on system size and user preferences"""
    
    # Get performance-based recommendations
    recommended = performance_estimator.get_recommended_parameters(n_qubits)
    
    # Respect user choices but warn if they seem too low
    final_shots = max(user_shots, int(recommended['shots'] * 0.8))
    final_rollouts = max(user_rollouts, int(recommended['n_rollouts'] * 0.8))
    
    if user_shots < recommended['shots'] or user_rollouts < recommended['n_rollouts']:
        logger.warning(f"Parameters may be too low for {n_qubits}-qubit system. "
                      f"Recommended: shots={recommended['shots']}, rollouts={recommended['n_rollouts']}")
    
    return {
        'shots': final_shots,
        'n_rollouts': final_rollouts,
        'recommended_shots': recommended['shots'],
        'recommended_rollouts': recommended['n_rollouts']
    }


@click.command()
@click.option('--hamiltonian', '-h', 
              type=click.Path(path_type=Path),
              required=True,
              help='Path to molecular Hamiltonian JSON file')
@click.option('--shots', '-s', 
              type=int, 
              default=1024,
              help='Number of measurement shots per observable (default: 1024)')
@click.option('--output', '-o', 
              type=click.Path(path_type=Path),
              required=True,
              help='Output JSON file for estimates and uncertainties')
@click.option('--model-path', '-m',
              type=click.Path(path_type=Path),  # 🔧 FIXED: Removed exists=True
              default=None,  # 🔧 FIXED: Made optional with None default
              help='Path to trained SymQNet model (default: models/FINAL_FIXED_SYMQNET.pth)')
@click.option('--vae-path', '-v',
              type=click.Path(path_type=Path),  # 🔧 FIXED: Removed exists=True
              default=None,  # 🔧 FIXED: Made optional with None default
              help='Path to pre-trained VAE (default: models/vae_M10_f.pth)')
@click.option('--max-steps', '-t',
              type=int,
              default=50,
              help='Maximum measurement steps per rollout (default: 50)')
@click.option('--n-rollouts', '-r',
              type=int,
              default=10,
              help='Number of policy rollouts for averaging (default: 10)')
@click.option('--confidence', '-c',
              type=float,
              default=0.95,
              help='Confidence level for uncertainty intervals (default: 0.95)')
@click.option('--device', '-d',
              type=click.Choice(['cpu', 'cuda', 'auto']),
              default='auto',
              help='Compute device (default: auto)')
@click.option('--seed', 
              type=int,
              default=42,
              help='Random seed for reproducibility')
@click.option('--verbose', '-V',
              is_flag=True,
              help='Enable verbose logging')
@click.option('--no-performance-warnings',
              is_flag=True,
              help='Disable performance degradation warnings')
@click.option('--show-performance-analysis',
              is_flag=True,
              help='Show detailed performance analysis')
def main(hamiltonian: Path, shots: int, output: Path, model_path: Optional[Path], 
         vae_path: Optional[Path], max_steps: int, n_rollouts: int, confidence: float,
         device: str, seed: int, verbose: bool, no_performance_warnings: bool,
         show_performance_analysis: bool):
    """
    SymQNet Molecular Optimization CLI - FIXED UNIVERSAL VERSION
    
    🌍 Universal support (any qubit count) with WORKING parameter extraction
    🔧 Fallback to 10-qubit-only mode if universal components unavailable
    
    Examples:
        symqnet-molopt --hamiltonian H2O_10q.json --output results.json
        symqnet-molopt --hamiltonian molecule.json --output results.json --shots 2048
    """
    
    # Setup logging first
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging(verbose)

    # Display mode information
    if UNIVERSAL_MODE:
        logger.info("🌍 Universal SymQNet mode enabled (FIXED)")
        logger.info("   Supports any qubit count ≥2 with optimal performance at 10 qubits")
    else:
        logger.info("🔧 Fallback mode active")
        logger.info("   Supports exactly 10-qubit systems only")

    # 🔧 FIXED: Handle model file paths gracefully
    try:
        model_path, vae_path = ensure_model_files(model_path, vae_path)
        logger.info(f"✅ Using model: {model_path}")
        logger.info(f"✅ Using VAE: {vae_path}")
    except click.ClickException:
        raise  # Re-raise click exceptions as-is
    except Exception as e:
        raise click.ClickException(f"❌ Model setup failed: {e}")

    # Find hamiltonian file early
    try:
        hamiltonian_path = find_hamiltonian_file(hamiltonian)
    except ValueError as e:
        raise click.ClickException(str(e))
    
    # Validate with mode-appropriate constraints
    try:
        hamiltonian_data = validate_hamiltonian_universal(hamiltonian_path)
        n_qubits = hamiltonian_data['n_qubits']
        
    except ValueError as e:
        raise click.ClickException(str(e))
    
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize performance estimator
    performance_estimator = PerformanceEstimator(optimal_qubits=10)
    
    # Performance analysis and warnings
    if not no_performance_warnings:
        warning = get_performance_warning(n_qubits, optimal_qubits=10)
        if warning:
            logger.warning(warning)
    
    if show_performance_analysis:
        print_performance_info(n_qubits, performance_estimator)
    
    # Get recommended parameters based on system size
    param_recommendations = get_recommended_params_for_system(
        n_qubits, shots, n_rollouts, performance_estimator
    )
    
    # Use recommended parameters if significantly different
    if param_recommendations['shots'] > shots:
        logger.info(f"Increasing shots: {shots} → {param_recommendations['shots']} (recommended for {n_qubits} qubits)")
        shots = param_recommendations['shots']
    
    if param_recommendations['n_rollouts'] > n_rollouts:
        logger.info(f"Increasing rollouts: {n_rollouts} → {param_recommendations['n_rollouts']} (recommended for {n_qubits} qubits)")
        n_rollouts = param_recommendations['n_rollouts']
    
    try:
        # Validate inputs
        validate_inputs(hamiltonian_path, shots, confidence, max_steps, n_rollouts)
        
        # 1. Parse Hamiltonian
        logger.info("🔍 Parsing molecular Hamiltonian...")
        parser = HamiltonianParser()
        hamiltonian_data = parser.load_hamiltonian(hamiltonian_path, warn_performance=(not no_performance_warnings))
        logger.info(f"Loaded {hamiltonian_data['n_qubits']}-qubit Hamiltonian "
                   f"with {len(hamiltonian_data['pauli_terms'])} terms")
        
        # 2. Run Parameter Estimation (Universal or Fallback) - FIXED VERSION
        performance_report = performance_estimator.estimate_performance(n_qubits)
        logger.info(f"🎯 Expected performance: {performance_report.performance_factor:.1%} of optimal")
        
        logger.info(f"🚀 Running FIXED parameter estimation...")
        logger.info(f"📊 Configuration: {shots} shots, {n_rollouts} rollouts, {max_steps} max steps")
        
        final_results = run_optimization_universal(
            hamiltonian_data=hamiltonian_data,
            model_path=model_path,
            vae_path=vae_path,
            device=device,
            shots=shots,
            n_rollouts=n_rollouts,
            max_steps=max_steps,
            warn_performance=(not no_performance_warnings)
        )
        
        # 3. Save Results
        logger.info(f"💾 Saving results to {output}")
        save_results(
            results=final_results,
            hamiltonian_data=hamiltonian_data,
            config={
                'shots': shots,
                'max_steps': max_steps,
                'n_rollouts': n_rollouts,
                'confidence': confidence,
                'seed': seed,
                'mode': 'universal_fixed' if UNIVERSAL_MODE else 'fallback',
                'performance_metadata': {
                    'expected_performance': performance_report.performance_factor,
                    'performance_level': performance_report.level.value,
                    'optimal_qubits': 10,
                    'universal_mode': UNIVERSAL_MODE,
                    'parameter_extraction_fixed': True
                }
            },
            output_path=output
        )
        
        # Print summary with performance context
        print_summary(final_results, n_qubits, performance_report.performance_factor)
        
        # Final performance note
        if n_qubits != 10:
            if UNIVERSAL_MODE:
                print(f"\n💡 NOTE: For optimal accuracy, consider using 10-qubit molecular representations.")
                print(f"   Current system ({n_qubits} qubits) operates at {performance_report.performance_factor:.1%} of optimal performance.")
            else:
                print(f"\n💡 NOTE: Install universal components to support {n_qubits}-qubit systems.")
                print(f"   Current fallback mode only supports exactly 10-qubit systems.")
        
        success_msg = "✅ FIXED Universal molecular optimization completed successfully!" if UNIVERSAL_MODE else "✅ Molecular optimization completed successfully!"
        logger.info(success_msg)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()
