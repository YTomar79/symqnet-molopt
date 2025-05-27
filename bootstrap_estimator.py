"""
Bootstrap estimator for uncertainty quantification - ENHANCED DEBUG VERSION
This version will show us exactly what's wrong with the rollout data
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class BootstrapEstimator:
    """Bootstrap-based uncertainty estimation for parameter estimates."""
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = 1.0 - confidence_level
        
    def compute_intervals(self, estimates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute confidence intervals from multiple rollout estimates.
        ENHANCED: Detailed debugging to find the issue
        """
        
        logger.info(f"Computing {self.confidence_level:.1%} confidence intervals "
                   f"from {len(estimates)} rollouts")
        
        # 🔍 ENHANCED: Debug what we actually received
        logger.info("🔍 DEBUGGING ROLLOUT DATA:")
        for i, estimate in enumerate(estimates):
            logger.info(f"  Rollout {i}:")
            logger.info(f"    Keys: {list(estimate.keys())}")
            
            # Check final_estimate
            final_est = estimate.get('final_estimate')
            logger.info(f"    final_estimate: {final_est}")
            logger.info(f"    final_estimate type: {type(final_est)}")
            
            if final_est is not None:
                if hasattr(final_est, 'shape'):
                    logger.info(f"    final_estimate shape: {final_est.shape}")
                elif hasattr(final_est, '__len__'):
                    logger.info(f"    final_estimate length: {len(final_est)}")
                
                # Check if it's all zeros
                try:
                    final_array = np.array(final_est)
                    if final_array.size > 0:
                        all_zero = np.allclose(final_array, 0, atol=1e-10)
                        logger.info(f"    final_estimate all zeros: {all_zero}")
                        if not all_zero:
                            logger.info(f"    final_estimate range: [{final_array.min():.6f}, {final_array.max():.6f}]")
                except:
                    logger.info(f"    Cannot convert final_estimate to array")
            
            # Check parameter_estimates
            param_ests = estimate.get('parameter_estimates', [])
            logger.info(f"    parameter_estimates length: {len(param_ests)}")
            
            if param_ests:
                logger.info(f"    First param_estimate: {param_ests[0]}")
                logger.info(f"    Last param_estimate: {param_ests[-1]}")
                
                # Check if any are non-zero
                non_zero_count = 0
                for pe in param_ests:
                    if pe is not None:
                        try:
                            pe_array = np.array(pe)
                            if pe_array.size > 0 and not np.allclose(pe_array, 0, atol=1e-10):
                                non_zero_count += 1
                        except:
                            pass
                logger.info(f"    Non-zero parameter estimates: {non_zero_count}/{len(param_ests)}")
            
            # Check convergence step
            conv_step = estimate.get('convergence_step', 0)
            logger.info(f"    convergence_step: {conv_step}")
        
        # Validate input
        if not estimates:
            raise ValueError("No rollout estimates provided")
        
        # Extract final parameter estimates from each rollout
        final_estimates = []
        convergence_steps = []
        
        for i, estimate in enumerate(estimates):
            final_est = estimate.get('final_estimate')
            
            if final_est is not None:
                try:
                    # Convert to numpy array
                    final_array = np.array(final_est)
                    
                    # Check if it's valid (not empty, not all zeros)
                    if final_array.size > 0:
                        if not np.allclose(final_array, 0, atol=1e-10):
                            final_estimates.append(final_array)
                            convergence_steps.append(estimate.get('convergence_step', 0))
                            logger.info(f"✅ Rollout {i}: Valid final estimate with {final_array.size} parameters")
                        else:
                            logger.warning(f"⚠️ Rollout {i}: final_estimate is all zeros - skipping")
                    else:
                        logger.warning(f"⚠️ Rollout {i}: final_estimate is empty - skipping")
                except Exception as e:
                    logger.warning(f"⚠️ Rollout {i}: Cannot process final_estimate: {e}")
            else:
                logger.warning(f"⚠️ Rollout {i}: No final_estimate - checking parameter_estimates...")
                
                # 🔧 FALLBACK: Try to extract from parameter_estimates
                param_ests = estimate.get('parameter_estimates', [])
                if param_ests:
                    for pe in reversed(param_ests):  # Start from last
                        if pe is not None:
                            try:
                                pe_array = np.array(pe)
                                if pe_array.size > 0 and not np.allclose(pe_array, 0, atol=1e-10):
                                    final_estimates.append(pe_array)
                                    convergence_steps.append(len(param_ests))
                                    logger.info(f"✅ Rollout {i}: Extracted from parameter_estimates")
                                    break
                            except:
                                continue
        
        logger.info(f"🎯 SUMMARY: Found {len(final_estimates)} valid parameter estimates from {len(estimates)} rollouts")
        
        if not final_estimates:
            logger.error("❌ NO VALID PARAMETER ESTIMATES FOUND!")
            logger.error("This means the policy engine is not generating meaningful parameters")
            logger.error("Possible causes:")
            logger.error("1. Neural network not loaded properly")
            logger.error("2. Input tensor shapes are wrong") 
            logger.error("3. Model weights are corrupted")
            logger.error("4. get_parameter_estimate() always returns zeros")
            
            # Return empty results instead of crashing
            return {
                'coupling_parameters': [],
                'field_parameters': [],
                'n_rollouts': len(estimates),
                'avg_measurements': np.mean([e.get('convergence_step', 0) for e in estimates]),
                'std_measurements': 0.0,
                'confidence_level': self.confidence_level,
                'total_uncertainty': 0.0,
                'error': 'No valid parameter estimates found - check policy engine',
                'debug_info': {
                    'rollouts_received': len(estimates),
                    'rollouts_with_final_estimate': sum(1 for e in estimates if e.get('final_estimate') is not None),
                    'rollouts_with_param_estimates': sum(1 for e in estimates if e.get('parameter_estimates')),
                    'avg_param_estimates_per_rollout': np.mean([len(e.get('parameter_estimates', [])) for e in estimates])
                }
            }
        
        # Check minimum samples for reliable bootstrap
        if len(final_estimates) < 3:
            logger.warning(
                f"Only {len(final_estimates)} valid rollouts found. "
                f"Bootstrap confidence intervals may be unreliable."
            )
        
        final_estimates = np.array(final_estimates)  # [n_rollouts, n_params]
        
        logger.info(f"📊 Final estimates array shape: {final_estimates.shape}")
        
        # Flexible parameter count handling
        n_params = final_estimates.shape[1]
        
        if n_params == 19:
            n_qubits = 10  # Standard 10-qubit system
            n_coupling = 9
            n_field = 10
        else:
            # Try to infer from parameter count
            # For n-qubit system: (n-1) coupling + n field = 2n-1 parameters
            if n_params % 2 == 1:
                n_qubits = (n_params + 1) // 2
                n_coupling = n_qubits - 1
                n_field = n_qubits
            else:
                n_qubits = n_params // 2  # Approximate
                n_coupling = n_qubits // 2
                n_field = n_qubits - n_coupling
            
            logger.warning(f"Non-standard parameter count {n_params}, inferring {n_qubits} qubits")
        
        # Split into coupling and field parameters
        coupling_estimates = final_estimates[:, :n_coupling]
        field_estimates = final_estimates[:, n_coupling:n_coupling + n_field]
        
        logger.info(f"💫 Parameter split: {n_coupling} coupling + {n_field} field")
        
        # Compute bootstrap confidence intervals
        coupling_results = self._bootstrap_parameters(coupling_estimates, "coupling")
        field_results = self._bootstrap_parameters(field_estimates, "field")
        
        # Overall statistics
        results = {
            'coupling_parameters': coupling_results,
            'field_parameters': field_results,
            'n_rollouts': len(estimates),
            'avg_measurements': np.mean(convergence_steps),
            'std_measurements': np.std(convergence_steps),
            'confidence_level': self.confidence_level,
            'total_uncertainty': self._compute_total_uncertainty(final_estimates),
            'parameter_extraction_success': True,
            'detected_qubits': n_qubits
        }
        
        logger.info(f"✅ Successfully extracted {len(coupling_results)} coupling + {len(field_results)} field parameters")
        return results
    
    def _bootstrap_parameters(self, estimates: np.ndarray, 
                            param_type: str) -> List[Tuple[float, float, float]]:
        """Bootstrap confidence intervals for a set of parameters."""
        
        if estimates.size == 0:
            logger.warning(f"No {param_type} parameters to bootstrap")
            return []
        
        n_rollouts, n_params = estimates.shape
        results = []
        
        logger.info(f"🔢 Bootstrapping {n_params} {param_type} parameters from {n_rollouts} rollouts")
        
        for param_idx in range(n_params):
            param_values = estimates[:, param_idx]
            
            # Bootstrap resampling
            bootstrap_means = []
            for _ in range(self.n_bootstrap):
                # Resample with replacement
                bootstrap_sample = np.random.choice(param_values, 
                                                  size=n_rollouts, 
                                                  replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            bootstrap_means = np.array(bootstrap_means)
            
            # Compute confidence interval
            mean_estimate = np.mean(param_values)
            ci_low = np.percentile(bootstrap_means, 100 * self.alpha / 2)
            ci_high = np.percentile(bootstrap_means, 100 * (1 - self.alpha / 2))
            
            results.append((mean_estimate, ci_low, ci_high))
            
            logger.debug(f"{param_type} parameter {param_idx}: "
                        f"{mean_estimate:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
        
        return results
    
    def _compute_total_uncertainty(self, estimates: np.ndarray) -> float:
        """Compute overall uncertainty metric."""
        
        if estimates.size == 0:
            return 0.0
        
        # Use coefficient of variation as uncertainty measure
        means = np.mean(estimates, axis=0)
        stds = np.std(estimates, axis=0)
        
        # Avoid division by zero
        cv = np.divide(stds, np.abs(means), 
                      out=np.zeros_like(stds), 
                      where=np.abs(means) > 1e-10)
        
        return float(np.mean(cv))
    
    def bayesian_update(self, prior_estimates: List[np.ndarray], 
                       new_estimates: List[np.ndarray]) -> Dict[str, Any]:
        """Bayesian update of parameter estimates (optional advanced feature)."""
        
        # Simple Bayesian updating assuming Gaussian priors and likelihoods
        prior_mean = np.mean(prior_estimates, axis=0)
        prior_var = np.var(prior_estimates, axis=0)
        
        new_mean = np.mean(new_estimates, axis=0)
        new_var = np.var(new_estimates, axis=0)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        prior_var = np.maximum(prior_var, epsilon)
        new_var = np.maximum(new_var, epsilon)
        
        # Bayesian update formulas
        posterior_var = 1.0 / (1.0/prior_var + 1.0/new_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + new_mean/new_var)
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_var': posterior_var,
            'posterior_std': np.sqrt(posterior_var)
        }
