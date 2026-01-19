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
        ENHANCED: Detailed debugging to find the issue
        """
        
        logger.info(f"Computing {self.confidence_level:.1%} confidence intervals "
                   f"from {len(estimates)} rollouts")
        
        # 🔍 ENHANCED: Debug what we actually received
        logger.info("🔍 DEBUGGING ROLLOUT DATA:")
        for i, estimate in enumerate(estimates):
            logger.info(f"  Rollout {i}:")
            logger.info(f"    Keys: {list(estimate.keys())}")

            posterior_mean = estimate.get('smc_posterior_mean')
            if posterior_mean is None:
                posterior_mean = estimate.get('posterior_mean')
            posterior_cov = estimate.get('smc_posterior_cov')
            if posterior_cov is None:
                posterior_cov = estimate.get('posterior_cov')

            logger.info(f"    smc_posterior_mean: {posterior_mean}")
            logger.info(f"    smc_posterior_cov: {posterior_cov}")
            logger.info(f"    smc_posterior_mean type: {type(posterior_mean)}")
            logger.info(f"    smc_posterior_cov type: {type(posterior_cov)}")

            if posterior_mean is not None:
                if hasattr(posterior_mean, 'shape'):
                    logger.info(f"    smc_posterior_mean shape: {posterior_mean.shape}")
                elif hasattr(posterior_mean, '__len__'):
                    logger.info(f"    smc_posterior_mean length: {len(posterior_mean)}")

                try:
                    mean_array = np.array(posterior_mean)
                    if mean_array.size > 0:
                        mean_min = mean_array.min()
                        mean_max = mean_array.max()
                        logger.info(f"    smc_posterior_mean range: [{mean_min:.6f}, {mean_max:.6f}]")
                except Exception:
                    logger.info("    Cannot convert smc_posterior_mean to array")

            if posterior_cov is not None:
                try:
                    cov_array = np.array(posterior_cov)
                    logger.info(f"    smc_posterior_cov shape: {cov_array.shape}")
                except Exception:
                    logger.info("    Cannot convert smc_posterior_cov to array")

            # Check convergence step
            conv_step = estimate.get('convergence_step', 0)
            logger.info(f"    convergence_step: {conv_step}")
        
        # Validate input
        if not estimates:
            raise ValueError("No rollout estimates provided")
        
        # Extract posterior means and covariances from each rollout
        final_estimates = []
        convergence_steps = []
        rollouts_with_valid_posterior = 0
        rollouts_missing_posterior = 0
        rollouts_degenerate = 0
        rollouts_high_variance = 0

        min_variance = 1e-12
        max_variance = 1e3
        
        for i, estimate in enumerate(estimates):
            posterior_mean = estimate.get('smc_posterior_mean')
            if posterior_mean is None:
                posterior_mean = estimate.get('posterior_mean')
            posterior_cov = estimate.get('smc_posterior_cov')
            if posterior_cov is None:
                posterior_cov = estimate.get('posterior_cov')

            if posterior_mean is None or posterior_cov is None:
                if posterior_mean is None and estimate.get('final_estimate') is not None:
                    logger.warning(
                        f" Rollout {i}: Missing SMC posterior mean; "
                        "falling back to final_estimate (deprecated)."
                    )
                    posterior_mean = estimate.get('final_estimate')
                if posterior_cov is None:
                    logger.warning(f" Rollout {i}: Missing SMC posterior covariance - skipping")
                rollouts_missing_posterior += 1
                if posterior_cov is None:
                    continue

            try:
                mean_array = np.array(posterior_mean, dtype=float).reshape(-1)
                cov_array = np.array(posterior_cov, dtype=float)
            except Exception as e:
                logger.warning(f" Rollout {i}: Cannot process posterior mean/cov: {e}")
                rollouts_missing_posterior += 1
                continue

            if mean_array.size == 0:
                logger.warning(f" Rollout {i}: Posterior mean is empty - skipping")
                rollouts_missing_posterior += 1
                continue

            if cov_array.ndim != 2 or cov_array.shape[0] != cov_array.shape[1]:
                logger.warning(f" Rollout {i}: Posterior covariance shape invalid - skipping")
                rollouts_missing_posterior += 1
                continue

            if cov_array.shape[0] != mean_array.size:
                logger.warning(
                    f" Rollout {i}: Posterior covariance dimension mismatch "
                    f"({cov_array.shape[0]} vs {mean_array.size}) - skipping"
                )
                rollouts_missing_posterior += 1
                continue

            if not np.isfinite(mean_array).all() or not np.isfinite(cov_array).all():
                logger.warning(f" Rollout {i}: Non-finite posterior mean/cov - skipping")
                rollouts_missing_posterior += 1
                continue

            variances = np.diag(cov_array)
            if np.any(variances < 0):
                logger.warning(f" Rollout {i}: Negative variances in posterior covariance - skipping")
                rollouts_degenerate += 1
                continue

            mean_variance = float(np.mean(variances))
            if mean_variance <= min_variance:
                logger.warning(f" Rollout {i}: Degenerate posterior covariance (variance too small) - skipping")
                rollouts_degenerate += 1
                continue

            if mean_variance >= max_variance:
                logger.warning(f" Rollout {i}: High-variance posterior (variance={mean_variance:.3e}) - skipping")
                rollouts_high_variance += 1
                continue

            final_estimates.append(mean_array)
            convergence_steps.append(estimate.get('convergence_step', 0))
            logger.info(f" Rollout {i}: Valid posterior mean with {mean_array.size} parameters")
            rollouts_with_valid_posterior += 1
        
        logger.info(f"🎯 SUMMARY: Found {len(final_estimates)} valid parameter estimates from {len(estimates)} rollouts")

        if rollouts_with_valid_posterior == 0:
            message = (
                "All rollouts produced invalid SMC posterior summaries. "
                "Posterior mean/covariance were missing, degenerate, or high-variance. "
                "This indicates SMC inference is unstable (metadata mismatch, invalid "
                "inputs, or collapsed particle weights)."
            )
            logger.error(message)
            raise ValueError(message)
        
        if not final_estimates:
            logger.error(" NO VALID PARAMETER ESTIMATES FOUND!")
            logger.error("This means the SMC posterior is not producing valid mean/covariance")
            logger.error("Possible causes:")
            logger.error("1. SMC particle filter collapsed or diverged")
            logger.error("2. Invalid metadata or measurement inputs")
            logger.error("3. Covariance is degenerate or numerically unstable")
            logger.error("4. Posterior variance is unbounded or too noisy")
            
            # Return empty results instead of crashing
            return {
                'coupling_parameters': [],
                'field_parameters': [],
                'n_rollouts': len(estimates),
                'avg_measurements': np.mean([e.get('convergence_step', 0) for e in estimates]),
                'std_measurements': 0.0,
                'confidence_level': self.confidence_level,
                'total_uncertainty': 0.0,
                'error': 'No valid SMC posterior summaries found - check SMC filter',
                'debug_info': {
                    'rollouts_received': len(estimates),
                    'rollouts_missing_posterior': rollouts_missing_posterior,
                    'rollouts_degenerate': rollouts_degenerate,
                    'rollouts_high_variance': rollouts_high_variance,
                    'rollouts_with_posterior_mean': sum(
                        1 for e in estimates
                        if e.get('smc_posterior_mean') is not None or e.get('posterior_mean') is not None
                    ),
                    'rollouts_with_posterior_cov': sum(
                        1 for e in estimates
                        if e.get('smc_posterior_cov') is not None or e.get('posterior_cov') is not None
                    )
                }
            }
        
        # Check minimum samples for reliable bootstrap
        if len(final_estimates) < 3:
            logger.warning(
                f"Only {len(final_estimates)} valid rollouts found. "
                f"Bootstrap confidence intervals may be unreliable."
            )
        
        final_estimates = np.array(final_estimates)  # [n_rollouts, n_params]
        
        logger.info(f" Final estimates array shape: {final_estimates.shape}")
        
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
        
        logger.info(f" Successfully extracted {len(coupling_results)} coupling + {len(field_results)} field parameters")
        return results
    
    def _bootstrap_parameters(self, estimates: np.ndarray, 
                            param_type: str) -> List[Tuple[float, float, float]]:
        """Bootstrap confidence intervals for a set of parameters."""
        
        if estimates.size == 0:
            logger.warning(f"No {param_type} parameters to bootstrap")
            return []
        
        n_rollouts, n_params = estimates.shape
        results = []
        
        logger.info(f" Bootstrapping {n_params} {param_type} parameters from {n_rollouts} rollouts")
        
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
