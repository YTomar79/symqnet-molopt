"""
Bootstrap estimator for uncertainty quantification
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
        
        Args:
            estimates: List of estimates from different rollouts
            
        Returns:
            Dictionary with mean estimates and confidence intervals
        """
        
        logger.info(f"Computing {self.confidence_level:.1%} confidence intervals "
                   f"from {len(estimates)} rollouts")
        
        # Validate input
        if not estimates:
            raise ValueError("No rollout estimates provided")
        
        # Extract final parameter estimates from each rollout
        final_estimates = []
        convergence_steps = []
        
        for i, estimate in enumerate(estimates):
            if estimate.get('final_estimate') is not None:
                final_estimates.append(estimate['final_estimate'])
                convergence_steps.append(estimate.get('convergence_step', 0))
            else:
                logger.warning(f"Rollout {i} has no final estimate - skipping")
        
        if not final_estimates:
            raise ValueError("No valid parameter estimates found in any rollout")
        
        # Check minimum samples for reliable bootstrap
        if len(final_estimates) < 3:
            logger.warning(
                f"Only {len(final_estimates)} valid rollouts found. "
                f"Bootstrap confidence intervals may be unreliable."
            )
        
        final_estimates = np.array(final_estimates)  # [n_rollouts, n_params]
        
        # Validate parameter count for 10-qubit system
        expected_params = 19  # 9 coupling + 10 field
        if final_estimates.shape[1] != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameters for 10-qubit system, "
                f"got {final_estimates.shape[1]}. Check your parameter estimation."
            )
        
        # Split into coupling and field parameters
        n_qubits = 10
        n_coupling = n_qubits - 1  # 9
        
        coupling_estimates = final_estimates[:, :n_coupling]
        field_estimates = final_estimates[:, n_coupling:]
        
        # Validate field parameter count
        if field_estimates.shape[1] != n_qubits:
            logger.warning(
                f"Expected {n_qubits} field parameters, got {field_estimates.shape[1]}"
            )
        
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
            'total_uncertainty': self._compute_total_uncertainty(final_estimates)
        }
        
        return results
    
    def _bootstrap_parameters(self, estimates: np.ndarray, 
                            param_type: str) -> List[Tuple[float, float, float]]:
        """Bootstrap confidence intervals for a set of parameters."""
        
        n_rollouts, n_params = estimates.shape
        results = []
        
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
