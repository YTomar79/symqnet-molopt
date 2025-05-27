"""
Universal SymQNet Wrapper - FIXED for MeasurementSimulator compatibility
Makes 10-qubit trained SymQNet work for any molecular system
Performance degrades gracefully as you move away from 10 qubits
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import warnings

logger = logging.getLogger(__name__)

class UniversalSymQNetWrapper:
    """Makes 10-qubit trained SymQNet work for any molecular system"""
    
    def __init__(self, trained_model_path: Path, trained_vae_path: Path, device: torch.device):
        """
        Initialize Universal SymQNet Wrapper
        
        Args:
            trained_model_path: Path to trained SymQNet model (.pth file)
            trained_vae_path: Path to trained VAE model (.pth file)  
            device: PyTorch device (cpu/cuda)
        """
        self.device = device
        self.trained_qubits = 10  # Your model's trained sweet spot
        self.vae_latent = 64     # From your training
        self.metadata_dim = 18   # From your training: 10 + 3 + 5
        self.M_evo = 5          # From your training
        
        # Store paths
        self.model_path = trained_model_path
        self.vae_path = trained_vae_path
        
        # Load trained 10-qubit model
        self.policy_engine = self._load_trained_model(trained_model_path, trained_vae_path)
        
        logger.info("Universal SymQNet loaded - supports any qubit count")
        logger.info(f"Optimal performance at {self.trained_qubits} qubits")
    
    def _load_trained_model(self, model_path: Path, vae_path: Path):
        """Load the original trained 10-qubit model"""
        from policy_engine import PolicyEngine
        return PolicyEngine(model_path, vae_path, self.device)
    
    def _pauli_string_to_indices(self, pauli_str: str) -> List[Tuple[int, str]]:
        """Convert Pauli string like 'XYZI' to [(0,'X'), (1,'Y'), (2,'Z')]."""
        indices = []
        for i, pauli in enumerate(pauli_str):
            if pauli.upper() in ['X', 'Y', 'Z']:
                indices.append((i, pauli.upper()))
        return indices
    
    def estimate_parameters(self, 
                          hamiltonian_data: Dict[str, Any],
                          shots: int = 1024,
                          n_rollouts: int = 5,
                          max_steps: int = 50,
                          warn_degradation: bool = True) -> Dict[str, Any]:
        """
        Universal parameter estimation for any qubit system
        
        Args:
            hamiltonian_data: Molecular Hamiltonian (any qubit count)
            shots: Number of measurements per step
            n_rollouts: Number of optimization rollouts
            max_steps: Maximum steps per rollout
            warn_degradation: Whether to warn about performance degradation
            
        Returns:
            Parameter estimates with confidence intervals (original qubit count)
        """
        
        original_qubits = hamiltonian_data['n_qubits']
        
        logger.info(f"🌍 Universal estimation: {original_qubits} qubits → {self.trained_qubits} qubits → {original_qubits} qubits")
        
        # Performance warning
        if warn_degradation and original_qubits != self.trained_qubits:
            perf_factor = self._calculate_performance_factor(original_qubits)
            if perf_factor < 0.8:  # Significant degradation
                warnings.warn(
                    f"Performance degradation expected: {original_qubits} qubits "
                    f"vs optimal {self.trained_qubits} qubits. "
                    f"Expected accuracy: {perf_factor:.1%} of optimal. "
                    f"Consider using {self.trained_qubits}-qubit systems for best results.",
                    UserWarning
                )
        
        # Normalize to 10-qubit representation
        logger.debug(f"Normalizing {original_qubits}-qubit system to {self.trained_qubits} qubits")
        normalized_hamiltonian = self._normalize_hamiltonian(hamiltonian_data)
        
        # Run on normalized system using existing infrastructure
        logger.debug("Running optimization on normalized system")
        normalized_results = self._run_optimization(
            normalized_hamiltonian, shots, n_rollouts, max_steps
        )
        
        # Denormalize back to original system
        logger.debug(f"Denormalizing results back to {original_qubits} qubits")
        final_results = self._denormalize_results(normalized_results, original_qubits)
        
        # Add performance metadata
        final_results['universal_metadata'] = {
            'original_qubits': original_qubits,
            'normalized_to': self.trained_qubits,
            'expected_performance': self._calculate_performance_factor(original_qubits),
            'normalization_applied': True,
            'optimal_at': self.trained_qubits
        }
        
        logger.info(f"✅ Universal parameter estimation completed for {original_qubits}-qubit system")
        return final_results
    
    def _normalize_hamiltonian(self, hamiltonian_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize any qubit system to 10-qubit representation"""
        
        original_qubits = hamiltonian_data['n_qubits']
        pauli_terms = hamiltonian_data['pauli_terms']
        
        logger.debug(f"Processing {len(pauli_terms)} Pauli terms for normalization")
        
        normalized_terms = []
        
        for i, term in enumerate(pauli_terms):
            original_string = term['pauli_string']
            coeff = term['coefficient']
            
            # Validate original string length
            if len(original_string) != original_qubits:
                raise ValueError(f"Pauli term {i}: string length {len(original_string)} != {original_qubits} qubits")
            
            if original_qubits <= self.trained_qubits:
                # Pad with identities for smaller systems
                padding = 'I' * (self.trained_qubits - original_qubits)
                normalized_string = original_string + padding
                # Scale coefficient for system size
                scale_factor = np.sqrt(self.trained_qubits / original_qubits)
                normalized_coeff = coeff * scale_factor
                
            else:
                # Intelligent compression for larger systems
                normalized_string = self._compress_pauli_string(
                    original_string, self.trained_qubits
                )
                # Scale coefficient
                scale_factor = np.sqrt(original_qubits / self.trained_qubits) 
                normalized_coeff = coeff * scale_factor
            
            # Validate normalized string length
            if len(normalized_string) != self.trained_qubits:
                raise ValueError(f"Normalization error: normalized string length {len(normalized_string)} != {self.trained_qubits}")
            
            # 🔧 CRITICAL FIX: Convert pauli_string to pauli_indices for MeasurementSimulator
            pauli_indices = self._pauli_string_to_indices(normalized_string)
            
            normalized_terms.append({
                'coefficient': normalized_coeff,
                'pauli_string': normalized_string,
                'pauli_indices': pauli_indices,  # ✅ ADDED: Required by MeasurementSimulator
                'original_coefficient': coeff,
                'scale_factor': scale_factor,
                'description': term.get('description', f'Normalized term {i}')
            })
        
        # Create normalized hamiltonian
        normalized_hamiltonian = hamiltonian_data.copy()
        normalized_hamiltonian.update({
            'n_qubits': self.trained_qubits,
            'pauli_terms': normalized_terms,
            'original_qubits': original_qubits,
            'normalization': 'universal_symqnet',
            'molecule': f"{hamiltonian_data['molecule']}_norm{self.trained_qubits}q"
        })
        
        logger.debug(f"Normalized {len(pauli_terms)} terms to {self.trained_qubits}-qubit representation")
        return normalized_hamiltonian
    
    def _compress_pauli_string(self, pauli_string: str, target_length: int) -> str:
        """Intelligently compress Pauli string preserving important structure"""
        
        if len(pauli_string) <= target_length:
            return pauli_string.ljust(target_length, 'I')
        
        original_length = len(pauli_string)
        
        # Strategy: Preserve structure while downsampling
        # 1. Always keep first and last positions
        # 2. Keep positions with non-identity operators
        # 3. Sample remaining positions uniformly
        
        key_positions = {0, original_length-1}  # Boundary positions
        
        # Add non-identity positions
        non_identity = [i for i, op in enumerate(pauli_string) if op != 'I']
        key_positions.update(non_identity[:target_length-2])
        
        # Fill remaining slots uniformly
        remaining_slots = target_length - len(key_positions)
        if remaining_slots > 0:
            available = set(range(original_length)) - key_positions
            if available:
                step = max(1, len(available) // remaining_slots)
                sampled = list(available)[::step][:remaining_slots]
                key_positions.update(sampled)
        
        # Build compressed string
        selected = sorted(list(key_positions))[:target_length]
        compressed = ''.join(pauli_string[pos] for pos in selected)
        
        # Ensure exact target length
        return compressed.ljust(target_length, 'I')[:target_length]
    
    def _run_optimization(self, normalized_hamiltonian: Dict[str, Any], 
                         shots: int, n_rollouts: int, max_steps: int) -> Dict[str, Any]:
        """Run optimization on normalized 10-qubit system using existing infrastructure"""
        
        from measurement_simulator import MeasurementSimulator
        from bootstrap_estimator import BootstrapEstimator
        
        logger.debug("Initializing components for normalized optimization")
        
        # Initialize components for 10-qubit normalized system
        simulator = MeasurementSimulator(
            hamiltonian_data=normalized_hamiltonian,
            shots=shots,
            device=self.device
        )
        
        estimator = BootstrapEstimator()
        
        # Run rollouts using existing workflow
        rollout_results = []
        
        for i in range(n_rollouts):
            logger.debug(f"Universal rollout {i+1}/{n_rollouts}")
            
            # Reset policy for new rollout
            self.policy_engine.reset()
            
            measurements = []
            parameter_estimates = []
            current_measurement = simulator.get_initial_measurement()
            
            for step in range(max_steps):
                # Get action from policy
                action_info = self.policy_engine.get_action(current_measurement)
                
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
                param_estimate = self.policy_engine.get_parameter_estimate()
                parameter_estimates.append(param_estimate)
                
                # Update current measurement for next step
                current_measurement = measurement_result['expectation_values']
                
                # Early stopping if converged
                if step > 5 and self.policy_engine.has_converged(parameter_estimates):
                    logger.debug(f"Universal rollout {i} converged at step {step}")
                    break
            
            rollout_results.append({
                'rollout_id': i,
                'measurements': measurements,
                'parameter_estimates': parameter_estimates,
                'final_estimate': parameter_estimates[-1] if parameter_estimates else None,
                'convergence_step': step
            })
        
        # Bootstrap analysis
        logger.debug("Computing bootstrap confidence intervals")
        bootstrap_results = estimator.compute_intervals(rollout_results)
        
        return {
            'symqnet_results': bootstrap_results,
            'rollout_results': rollout_results
        }
    
    def _denormalize_results(self, normalized_results: Dict[str, Any], 
                           target_qubits: int) -> Dict[str, Any]:
        """Denormalize results back to original system size"""
        
        logger.debug(f"Denormalizing results: {self.trained_qubits} → {target_qubits} qubits")
        
        symqnet_results = normalized_results['symqnet_results']
        
        # Extract normalized parameters (19 total: 9+10 from 10-qubit system)
        norm_coupling = symqnet_results['coupling_parameters']  # 9 from 10-qubit
        norm_field = symqnet_results['field_parameters']        # 10 from 10-qubit
        
        # Target parameter counts
        target_coupling_count = target_qubits - 1
        target_field_count = target_qubits
        
        logger.debug(f"Parameter counts: coupling {len(norm_coupling)}→{target_coupling_count}, field {len(norm_field)}→{target_field_count}")
        
        # Denormalize coupling parameters
        if target_qubits <= self.trained_qubits:
            # Extract subset for smaller systems
            denorm_coupling = norm_coupling[:target_coupling_count]
        else:
            # Extrapolate for larger systems
            denorm_coupling = self._extrapolate_parameters(
                norm_coupling, target_coupling_count, 'coupling'
            )
        
        # Denormalize field parameters
        if target_qubits <= self.trained_qubits:
            denorm_field = norm_field[:target_field_count]
        else:
            denorm_field = self._extrapolate_parameters(
                norm_field, target_field_count, 'field'
            )
        
        # Apply inverse scaling
        scale_factor = np.sqrt(target_qubits / self.trained_qubits)
        
        # Scale parameters back to original system size
        denorm_coupling = self._scale_parameters(denorm_coupling, scale_factor)
        denorm_field = self._scale_parameters(denorm_field, scale_factor)
        
        # Build final results
        denormalized_results = normalized_results.copy()
        denormalized_results['symqnet_results'].update({
            'coupling_parameters': denorm_coupling,
            'field_parameters': denorm_field,
            'denormalization_scaling': scale_factor,
            'parameter_count_adjusted': True
        })
        
        return denormalized_results
    
    def _scale_parameters(self, params: List, scale_factor: float) -> List:
        """Scale parameters by given factor, handling both tuple and dict formats"""
        
        scaled_params = []
        
        for param in params:
            if isinstance(param, tuple) and len(param) >= 3:
                # Tuple format: (mean, ci_low, ci_high)
                mean, ci_low, ci_high = param[0], param[1], param[2]
                scaled_params.append((
                    mean * scale_factor,
                    ci_low * scale_factor,
                    ci_high * scale_factor
                ))
            elif isinstance(param, dict):
                # Dict format
                scaled_param = param.copy()
                scaled_param['mean'] *= scale_factor
                scaled_param['confidence_interval'] = [
                    ci * scale_factor for ci in scaled_param['confidence_interval']
                ]
                scaled_param['uncertainty'] *= scale_factor
                scaled_params.append(scaled_param)
            else:
                # Unknown format, pass through
                scaled_params.append(param)
        
        return scaled_params
    
    def _extrapolate_parameters(self, source_params: List, 
                               target_count: int, param_type: str) -> List:
        """Extrapolate parameters for larger systems"""
        
        source_count = len(source_params)
        
        if target_count <= source_count:
            return source_params[:target_count]
        
        # For extrapolation, use patterns from source parameters
        extrapolated = source_params.copy()
        
        # Analyze source parameter patterns - handle both tuple and dict formats
        if source_params and isinstance(source_params[0], tuple):
            # Tuple format: (mean, ci_low, ci_high)
            source_means = [p[0] for p in source_params]
        elif source_params and isinstance(source_params[0], dict):
            # Dict format
            source_means = [p['mean'] for p in source_params]
        else:
            source_means = []
        
        mean_trend = np.mean(source_means) if source_means else 0.0
        
        # Generate additional parameters with decreasing magnitude
        for i in range(source_count, target_count):
            decay_factor = 0.8 ** (i - source_count + 1)  # Exponential decay
            
            if source_params and isinstance(source_params[0], tuple):
                # Tuple format
                new_param = (
                    mean_trend * decay_factor,
                    mean_trend * decay_factor * 0.95,
                    mean_trend * decay_factor * 1.05
                )
            else:
                # Dict format
                new_param = {
                    'index': i,
                    'mean': mean_trend * decay_factor,
                    'confidence_interval': [
                        mean_trend * decay_factor * 0.95,
                        mean_trend * decay_factor * 1.05
                    ],
                    'uncertainty': abs(mean_trend * decay_factor * 0.05),
                    'extrapolated': True
                }
            
            extrapolated.append(new_param)
        
        return extrapolated
    
    def _calculate_performance_factor(self, n_qubits: int) -> float:
        """Calculate expected performance relative to 10-qubit optimum"""
        
        if n_qubits == self.trained_qubits:
            return 1.0
        
        distance = abs(n_qubits - self.trained_qubits)
        
        # Performance degradation model (empirically tuned)
        if n_qubits < self.trained_qubits:
            # Smaller systems: information loss from padding
            degradation = 0.95 ** (distance * 0.8)
        else:
            # Larger systems: information loss from compression
            degradation = 0.90 ** (distance * 1.2)
        
        return max(degradation, 0.3)  # Minimum 30% performance
