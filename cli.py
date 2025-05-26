#!/usr/bin/env python3
"""
SymQNet Molecular Optimization CLI

Usage:
    symqnet-molopt --hamiltonian LiH.json --shots 300 --output estimate.json
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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hamiltonian_parser import HamiltonianParser
from measurement_simulator import MeasurementSimulator
from policy_engine import PolicyEngine
from bootstrap_estimator import BootstrapEstimator
from utils import setup_logging, validate_inputs, save_results

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

@click.command()
@click.option('--hamiltonian', '-h', 
              type=click.Path(exists=True, path_type=Path),
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
def main(hamiltonian: Path, shots: int, output: Path, model_path: Path, 
         vae_path: Path, max_steps: int, n_rollouts: int, confidence: float,
         device: str, seed: int, verbose: bool):
    """
    SymQNet Molecular Optimization CLI
    
    Estimates molecular Hamiltonian parameters using trained SymQNet policy.
    """
    
    # Setup
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging(verbose)
    
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:
        # Validate inputs
        validate_inputs(hamiltonian, shots, confidence, max_steps, n_rollouts)
        
        # 1. Parse Hamiltonian
        logger.info("🔍 Parsing molecular Hamiltonian...")
        parser = HamiltonianParser()
        hamiltonian_data = parser.load_hamiltonian(hamiltonian)
        logger.info(f"Loaded {hamiltonian_data['n_qubits']}-qubit Hamiltonian "
                   f"with {len(hamiltonian_data['pauli_terms'])} terms")
        
        # 2. Initialize Policy Engine
        logger.info("🤖 Loading SymQNet policy...")
        policy = PolicyEngine(
            model_path=model_path,
            vae_path=vae_path,
            device=device
        )
        
        # 3. Initialize Measurement Simulator
        logger.info("⚛️  Setting up measurement simulator...")
        simulator = MeasurementSimulator(
            hamiltonian_data=hamiltonian_data,
            shots=shots,
            device=device
        )
        
        # 4. Run Policy Rollouts
        logger.info(f"🎯 Running {n_rollouts} policy rollouts...")
        estimates = []
        
        for rollout in range(n_rollouts):
            logger.info(f"  Rollout {rollout + 1}/{n_rollouts}")
            
            # Reset policy buffer for new rollout
            policy.reset()
            
            # Run single rollout
            estimate = run_single_rollout(
                policy=policy,
                simulator=simulator,
                max_steps=max_steps,
                rollout_id=rollout
            )
            estimates.append(estimate)
        
        # 5. Bootstrap Uncertainty Estimation
        logger.info("📊 Computing confidence intervals...")
        bootstrap = BootstrapEstimator(confidence_level=confidence)
        final_results = bootstrap.compute_intervals(estimates)
        
        # 6. Save Results
        logger.info(f"💾 Saving results to {output}")
        save_results(
            results=final_results,
            hamiltonian_data=hamiltonian_data,
            config={
                'shots': shots,
                'max_steps': max_steps,
                'n_rollouts': n_rollouts,
                'confidence': confidence,
                'seed': seed
            },
            output_path=output
        )
        
        # Print summary
        print_summary(final_results)
        
        logger.info("✅ Molecular optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise click.ClickException(str(e))

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

def print_summary(results: Dict):
    """Print a formatted summary of results."""
    
    print("\n" + "="*60)
    print("🎯 SYMQNET MOLECULAR OPTIMIZATION RESULTS")
    print("="*60)
    
    if 'coupling_parameters' in results:
        print("\n📊 COUPLING PARAMETERS (J):")
        for i, (mean, ci_low, ci_high) in enumerate(results['coupling_parameters']):
            print(f"  J_{i}: {mean:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
    
    if 'field_parameters' in results:
        print("\n🧲 FIELD PARAMETERS (h):")
        for i, (mean, ci_low, ci_high) in enumerate(results['field_parameters']):
            print(f"  h_{i}: {mean:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
    
    if 'total_uncertainty' in results:
        print(f"\n📏 Total Parameter Uncertainty: {results['total_uncertainty']:.6f}")
    
    if 'avg_measurements' in results:
        print(f"📐 Average Measurements Used: {results['avg_measurements']:.1f}")
    
    print("="*60)

if __name__ == '__main__':
    main()
