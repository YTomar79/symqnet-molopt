#!/usr/bin/env python3
"""
SymQNet Molecular Optimization CLI - Universal Version

Supports any qubit count with optimal performance at 10 qubits.

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

from hamiltonian_parser import HamiltonianParser
from measurement_simulator import MeasurementSimulator
from policy_engine import PolicyEngine
from bootstrap_estimator import BootstrapEstimator
from utils import setup_logging, validate_inputs, save_results, suggest_qubit_mapping
from universal_wrapper import UniversalSymQNetWrapper
from performance_estimator import PerformanceEstimator, get_performance_warning

# 🔥 IMPORT YOUR EXACT ARCHITECTURES 🔥
from architectures import (
    VariationalAutoencoder,
    FixedSymQNetWithEstimator,
    GraphEmbed,
    TemporalContextualAggregator,
    PolicyValueHead,
    SpinChainEnv
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def validate_hamiltonian_basic(hamiltonian_path: Path) -> Dict[str, any]:
    """Basic validation of Hamiltonian file with universal support"""
    
    try:
        with open(hamiltonian_path, 'r') as f:
            hamiltonian_data = json.load(f)
        
        n_qubits = hamiltonian_data.get('n_qubits', 0)
        
        # Basic validation - no hard limits
        if n_qubits < 2:
            raise ValueError(f"Minimum 2 qubits required, got {n_qubits}")
        
        # Performance guidance instead of hard constraints
        if n_qubits > 25:
            logger.warning(f"Large system ({n_qubits} qubits) may have very long runtime")
        
        logger.info(f"✅ Validated: {n_qubits}-qubit Hamiltonian")
        return hamiltonian_data
        
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {hamiltonian_path}")
    except FileNotFoundError:
        raise ValueError(f"Hamiltonian file not found: {hamiltonian_path}")


def run_single_rollout(policy, simulator, max_steps: int, rollout_id: int):
    """Run a single policy rollout to estimate Hamiltonian parameters."""
    
    measurements = []
    parameter_estimates = []
    
    # Initial measurement
    current_measurement = simulator.get_initial_measurement()
    
    for step in range(max_steps):
        # Get action from policy
        action_info = policy.get_action(current_measurement)
        
        # Execute measurement
        measurement_result = simulator.execute_measurement(
            qubit_indices=action_info['qubits'],
            pauli_operators=action_info['operators'],
            evolution_time=action_info['time']
        )
        
        measurements.append({
            'step': step,
            'action': action_info,
            'result': measurement_result
        })
        
        # Get parameter estimate from policy
        param_estimate = policy.get_parameter_estimate()
        parameter_estimates.append(param_estimate)
        
        # Update current measurement for next step
        current_measurement = measurement_result['expectation_values']
        
        # Early stopping if converged
        if step > 5 and policy.has_converged(parameter_estimates):
            logger.debug(f"Rollout {rollout_id} converged at step {step}")
            break
    
    return {
        'rollout_id': rollout_id,
        'measurements': measurements,
        'parameter_estimates': parameter_estimates,
        'final_estimate': parameter_estimates[-1] if parameter_estimates else None,
        'convergence_step': step
    }


def print_performance_info(n_qubits: int, performance_estimator: PerformanceEstimator):
    """Print performance information and recommendations"""
    
    print("\n" + "="*60)
    print("🌍 UNIVERSAL SYMQNET PERFORMANCE ANALYSIS")
    print("="*60)
    
    report = performance_estimator.estimate_performance(n_qubits)
    
    print(f"📊 System Size: {n_qubits} qubits")
    print(f"🎯 Optimal Size: {performance_estimator.optimal_qubits} qubits")
    print(f"📈 Expected Performance: {report.performance_factor:.1%} of optimal")
    print(f"🏷️  Performance Level: {report.level.value.upper()}")
    
    if report.warning_message:
        print(f"\n⚠️  {report.warning_message}")
    
    if n_qubits == performance_estimator.optimal_qubits:
        print("\n✨ Running at optimal performance!")
    else:
        print(f"\n📏 Uncertainty Scaling: {report.uncertainty_scaling:.1f}x")
        print(f"⚡ Computational Overhead: {report.computational_overhead:.1f}x")
    
    # Show top recommendations
    if report.recommendations:
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations[:3], 1):  # Show top 3
            print(f"   {i}. {rec}")
        
        if len(report.recommendations) > 3:
            print(f"   ... and {len(report.recommendations)-3} more recommendations")


def print_summary(results: Dict, n_qubits: int, performance_factor: float):
    """Print a formatted summary of results with performance context."""
    
    print("\n" + "="*60)
    print("🎯 SYMQNET MOLECULAR OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"🧪 System: {n_qubits} qubits")
    print(f"📊 Performance: {performance_factor:.1%} of optimal")
    
    if 'coupling_parameters' in results:
        coupling_count = len(results['coupling_parameters'])
        print(f"\n⚛️  COUPLING PARAMETERS ({coupling_count} estimated):")
        for i, (mean, ci_low, ci_high) in enumerate(results['coupling_parameters']):
            uncertainty = (ci_high - ci_low) / 2
            print(f"  J_{i}: {mean:8.6f} ± {uncertainty:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
    
    if 'field_parameters' in results:
        field_count = len(results['field_parameters'])
        print(f"\n🧲 FIELD PARAMETERS ({field_count} estimated):")
        for i, (mean, ci_low, ci_high) in enumerate(results['field_parameters']):
            uncertainty = (ci_high - ci_low) / 2
            print(f"  h_{i}: {mean:8.6f} ± {uncertainty:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
    
    if 'total_uncertainty' in results:
        print(f"\n📏 Total Parameter Uncertainty: {results['total_uncertainty']:.6f}")
    
    if 'avg_measurements' in results:
        print(f"📐 Average Measurements Used: {results['avg_measurements']:.1f}")
    
    if 'n_rollouts' in results:
        print(f"🔄 Rollouts Completed: {results['n_rollouts']}")
    
    print("="*60)


def get_recommended_params_for_system(n_qubits: int, 
                                     user_shots: int, 
                                     user_rollouts: int,
                                     performance_estimator: PerformanceEstimator) -> Dict[str, int]:
    """Get recommended parameters based on system size and user preferences"""
    
    # Get performance-based recommendations
    recommended = performance_estimator.get_recommended_parameters(n_qubits)
    
    # Respect user choices but warn if they seem too low
    final_shots = max(user_shots, int(recommended['shots'] * 0.8))  # At least 80% of recommended
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
              help='Path to molecular Hamiltonian JSON file (any qubit count supported)')
@click.option('--shots', '-s', 
              type=int, 
              default=1024,
              help='Number of measurement shots per observable (default: 1024)')
@click.option('--output', '-o', 
              type=click.Path(path_type=Path),
              required=True,
              help='Output JSON file for estimates and uncertainties')
@click.option('--model-path', '-m',
              type=click.Path(exists=True, path_type=Path),
              default='models/FINAL_FIXED_SYMQNET.pth',
              help='Path to trained SymQNet model')
@click.option('--vae-path', '-v',
              type=click.Path(exists=True, path_type=Path),
              default='models/vae_M10_f.pth',
              help='Path to pre-trained VAE')
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
              help='Random seed for reproducibility (default: 42)')
@click.option('--verbose', '-V',
              is_flag=True,
              help='Enable verbose logging')
@click.option('--no-performance-warnings',
              is_flag=True,
              help='Disable performance degradation warnings')
@click.option('--show-performance-analysis',
              is_flag=True,
              help='Show detailed performance analysis')
def main(hamiltonian: Path, shots: int, output: Path, model_path: Path, 
         vae_path: Path, max_steps: int, n_rollouts: int, confidence: float,
         device: str, seed: int, verbose: bool, no_performance_warnings: bool,
         show_performance_analysis: bool):
    """
    Universal SymQNet Molecular Optimization CLI
    
    🌍 Supports any qubit count with optimal performance at 10 qubits.
    
    Performance degrades gracefully for non-10-qubit systems:
    • 4-8 qubits: Good performance (85-97%)
    • 10 qubits: Optimal performance (100%) 
    • 12-16 qubits: Moderate degradation (75-90%)
    • 20+ qubits: Significant degradation (<70%)
    
    Examples:
        symqnet-molopt --hamiltonian H2_4q.json --output results.json
        symqnet-molopt --hamiltonian BeH2_12q.json --output results.json --shots 2048
    """
    
    # Setup logging first
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging(verbose)

    # Find hamiltonian file early
    try:
        hamiltonian_path = find_hamiltonian_file(hamiltonian)
    except ValueError as e:
        raise click.ClickException(str(e))
    
    # Universal validation - no hard qubit constraints
    try:
        hamiltonian_data = validate_hamiltonian_basic(hamiltonian_path)
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
        # Validate inputs (updated to support any qubit count)
        validate_inputs(hamiltonian_path, shots, confidence, max_steps, n_rollouts)
        
        # 1. Parse Hamiltonian
        logger.info("🔍 Parsing molecular Hamiltonian...")
        parser = HamiltonianParser()
        hamiltonian_data = parser.load_hamiltonian(hamiltonian_path)
        logger.info(f"Loaded {hamiltonian_data['n_qubits']}-qubit Hamiltonian "
                   f"with {len(hamiltonian_data['pauli_terms'])} terms")
        
        # 2. Initialize Universal SymQNet
        logger.info("🌍 Loading Universal SymQNet...")
        universal_wrapper = UniversalSymQNetWrapper(
            trained_model_path=model_path,
            trained_vae_path=vae_path,
            device=device
        )
        
        performance_report = performance_estimator.estimate_performance(n_qubits)
        logger.info(f"🎯 Expected performance: {performance_report.performance_factor:.1%} of optimal")
        
        # 3. Initialize Measurement Simulator
        logger.info("⚛️  Setting up measurement simulator...")
        simulator = MeasurementSimulator(
            hamiltonian_data=hamiltonian_data,
            shots=shots,
            device=device
        )
        
        # 4. Run Universal Parameter Estimation
        logger.info(f"🚀 Running universal parameter estimation...")
        logger.info(f"📊 Configuration: {shots} shots, {n_rollouts} rollouts, {max_steps} max steps")
        
        # Use the universal wrapper for parameter estimation
        final_results = universal_wrapper.estimate_parameters(
            hamiltonian_data=hamiltonian_data,
            shots=shots,
            n_rollouts=n_rollouts,
            max_steps=max_steps,
            warn_degradation=(not no_performance_warnings)
        )
        
        # 5. Save Results
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
                'performance_metadata': {
                    'expected_performance': performance_report.performance_factor,
                    'performance_level': performance_report.level.value,
                    'optimal_qubits': 10,
                    'universal_symqnet_version': '1.0.0'
                }
            },
            output_path=output
        )
        
        # Extract results for summary
        symqnet_results = final_results.get('symqnet_results', {})
        
        # Print summary with performance context
        print_summary(symqnet_results, n_qubits, performance_report.performance_factor)
        
        # Final performance note
        if n_qubits != 10:
            print(f"\n💡 NOTE: For optimal accuracy, consider using 10-qubit molecular representations.")
            print(f"   Current system ({n_qubits} qubits) operates at {performance_report.performance_factor:.1%} of optimal performance.")
        
        logger.info("✅ Universal molecular optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()
