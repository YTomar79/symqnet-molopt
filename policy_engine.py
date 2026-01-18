"""
Policy Engine for SymQNet integration 
Fixed tensor shapes, error checking, and parameter extraction.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging


from architectures import (
    VariationalAutoencoder, 
    GraphEmbed,
    TemporalContextualAggregator, 
    PolicyValueHead,
    FixedSymQNetWithEstimator
)

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when policy inference fails and rollouts should abort."""

class PolicyEngine:
    """Integrates trained SymQNet for molecular Hamiltonian estimation."""
    
    def __init__(self, model_path: Path, vae_path: Path, device: torch.device, shots: Optional[int] = None):
        self.device = device
        self.model_path = model_path
        self.vae_path = vae_path
        self.shots = shots
        
        # Load models
        self._load_models()
        
        # Initialize buffers
        self.reset()
        
        logger.info("Policy engine initialized successfully")

    def set_shots(self, shots: Optional[int]) -> None:
        """Update the shot budget for metadata conditioning."""
        self.shots = shots
    
    def _load_models(self):
        """Load pre-trained VAE and SymQNet models with EXACT architecture matching."""
        
        self.vae = VariationalAutoencoder(M=10, L=64).to(self.device)
        vae_state = torch.load(self.vae_path, map_location=self.device, weights_only=False)
        self.vae.load_state_dict(vae_state)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        
        #Inspect checkpoint to determine EXACT architecture
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get the actual state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        logger.info(f"🔍 Checkpoint contains {len(state_dict)} parameters")
        logger.info(f"🔍 Keys: {list(state_dict.keys())[:10]}...")
        
        #  PARAMETERS
        self.n_qubits = 10
        self.T = 10
        self.M_evo = 5
        self.A = self.n_qubits * 3 * self.M_evo  # 150 actions

        inferred_m_evo = self._infer_m_evo(state_dict, self.n_qubits)
        if inferred_m_evo is not None and inferred_m_evo != self.M_evo:
            logger.info(
                "🔧 Overriding M_evo from checkpoint: %s -> %s",
                self.M_evo,
                inferred_m_evo,
            )
            self.M_evo = inferred_m_evo
            self.A = self.n_qubits * 3 * self.M_evo

        default_meta_dim = self.n_qubits + 3 + self.M_evo
        self.meta_dim = self._infer_meta_dim(state_dict, base_latent_dim=64, default_meta_dim=default_meta_dim)
        self.include_shots = self.meta_dim == default_meta_dim + 1
        if not self.include_shots and self.meta_dim == default_meta_dim:
            logger.info("🔧 Checkpoint metadata excludes shot conditioning.")
        elif not self.include_shots and self.meta_dim != default_meta_dim:
            logger.warning(
                "⚠️ Unrecognized metadata size (%s); proceeding with include_shots=%s",
                self.meta_dim,
                self.include_shots,
            )
        
        is_simple_estimator = self._detect_simple_estimator(state_dict)
        
        if is_simple_estimator:
            logger.info("🎯 Detected estimator-only model; loading full architecture with partial weights")
            self._create_minimal_model(state_dict, self.n_qubits, self.M_evo, self.A, self.meta_dim)
        else:
            logger.info("🎯 Detected full trained model")
            self._create_full_model(state_dict, self.n_qubits, self.T, self.A, self.M_evo, self.meta_dim)
        
        self.symqnet.eval()
        logger.info(" Models loaded with EXACT architecture match")
    
    def _detect_simple_estimator(self, state_dict):
        """Detect if this is a simple estimator or full model."""
        
        has_graph_embed = any('graph_embed' in key for key in state_dict.keys())
        has_temp_agg = any('temp_agg' in key for key in state_dict.keys())
        has_policy = any('policy_value' in key for key in state_dict.keys())
        estimator_keys = [key for key in state_dict.keys() if 'estimator' in key]
        
        logger.info(f"Architecture detection:")
        logger.info(f" Graph embed: {has_graph_embed}")
        logger.info(f" Temporal agg: {has_temp_agg}")
        logger.info(f" Policy head: {has_policy}")
        logger.info(f" Estimator keys: {len(estimator_keys)}")
        
        is_simple = (
            not has_graph_embed and
            not has_temp_agg and
            not has_policy and
            len(estimator_keys) > 0
        )
        
        return is_simple

    def _infer_m_evo(self, state_dict, n_qubits: int) -> Optional[int]:
        """Infer M_evo from policy head weights when present."""
        action_dim = None
        if 'policy_value.policy_fc.weight' in state_dict:
            action_dim = state_dict['policy_value.policy_fc.weight'].shape[0]
        elif 'policy_value.policy_fc.bias' in state_dict:
            action_dim = state_dict['policy_value.policy_fc.bias'].shape[0]

        if action_dim is None:
            return None

        denom = n_qubits * 3
        if action_dim % denom != 0:
            logger.warning(
                "⚠️ Cannot infer M_evo: action_dim=%s not divisible by %s",
                action_dim,
                denom,
            )
            return None

        return int(action_dim // denom)

    def _infer_meta_dim(self, state_dict, base_latent_dim: int, default_meta_dim: int) -> int:
        """Infer metadata dimension from checkpoint shapes."""
        embed_dim = None
        candidates = [
            ("temp_agg.conv1.weight", lambda w: w.shape[0]),
            ("graph_embed.phi_e_layers.0.0.weight", lambda w: w.shape[0]),
            ("policy_value.shared_fc.weight", lambda w: w.shape[1]),
            ("estimator.0.weight", lambda w: w.shape[1]),
            ("estimator.weight", lambda w: w.shape[1]),
        ]

        for key, extractor in candidates:
            if key in state_dict:
                embed_dim = int(extractor(state_dict[key]))
                logger.info("🔍 Inferred embedding dim from %s: %s", key, embed_dim)
                break

        if embed_dim is None:
            logger.warning("⚠️ Unable to infer embedding dim; using default metadata size.")
            return default_meta_dim

        meta_dim = embed_dim - base_latent_dim
        if meta_dim <= 0:
            logger.warning(
                "⚠️ Invalid inferred metadata size (%s); using default %s.",
                meta_dim,
                default_meta_dim,
            )
            return default_meta_dim

        return meta_dim
    
    def _create_minimal_model(self, state_dict, n_qubits, M_evo, A, meta_dim):
        """Create minimal model matching training's estimator architecture."""
        
        estimator_keys = [key for key in state_dict.keys() if 'estimator' in key]
        is_mlp_estimator = any('estimator.0.' in key or 'estimator.2.' in key or 'estimator.4.' in key 
                              for key in estimator_keys)
        
        class MinimalSymQNet(nn.Module):
            def __init__(self, vae, n_qubits, device, is_mlp, meta_dim):
                super().__init__()
                self.vae = vae
                self.device = device
                self.n_qubits = n_qubits
                
                input_dim = 64 + meta_dim  # VAE + metadata
                output_dim = 2 * n_qubits - 1  # J + h parameters
                
                if is_mlp:
                    self.estimator = nn.Sequential(
                        nn.Linear(input_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, output_dim)
                    )
                else:
                    self.estimator = nn.Linear(input_dim, output_dim)
                
                self.step_count = 0
                
            def forward(self, obs, metadata, deterministic_inference: bool = False):
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)  # [10] -> [1, 10]
                if metadata.dim() == 1:
                    metadata = metadata.unsqueeze(0)  # [meta] -> [1, meta]
                
                # VAE encoding
                with torch.no_grad():
                    mu_z, logvar_z = self.vae.encode(obs)
                    z = self.vae.reparameterize(mu_z, logvar_z)  # [1, 64]
                
                # Concatenate with metadata
                z_with_meta = torch.cat([z, metadata], dim=-1)  # [1, L + meta]
                
                # Estimate parameters
                theta_hat = self.estimator(z_with_meta)  # [1, 2*n_qubits-1]
                

                theta_hat = theta_hat.squeeze(0)  # [1, 2*n_qubits-1] -> [2*n_qubits-1]
                
                # Create dummy policy outputs
                action_probs = torch.ones(A, device=self.device) / A
                dummy_dist = torch.distributions.Categorical(probs=action_probs)
                dummy_value = torch.tensor(0.0, device=self.device)
                
                return dummy_dist, dummy_value, theta_hat
            
            def reset_buffer(self):
                self.step_count = 0
        
        self.symqnet = MinimalSymQNet(
            self.vae,
            n_qubits,
            self.T,
            A,
            M_evo,
            meta_dim,
            allow_partial=True,
        )
    
    def _create_full_model(self, state_dict, n_qubits, T, A, M_evo, meta_dim, allow_partial: bool = False):
        """Create full model matching EXACT training architecture."""
        
        # EXACT graph connectivity from training
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=self.device) * 0.1
        
        self.symqnet = FixedSymQNetWithEstimator(
            vae=self.vae,
            n_qubits=n_qubits,
            L=64,  # BASE VAE dimension, internal arch adds metadata
            edge_index=edge_index,
            edge_attr=edge_attr,
            T=T,
            A=A,
            M_evo=M_evo,
            K_gnn=2,
            meta_dim=meta_dim,
        ).to(self.device)
        
        # Load with architecture matching
        missing_keys, unexpected_keys = self.symqnet.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            mismatch_message = (
                "Full model load encountered checkpoint mismatch. "
                f"Missing keys ({len(missing_keys)}): {missing_keys}. "
                f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}."
            )
            if allow_partial:
                logger.warning(mismatch_message)
            else:
                logger.error(mismatch_message)
                raise RuntimeError(mismatch_message)

        logger.info("Full model loaded with correct dimensions")
    
    def reset(self):
        """Reset policy state for new rollout."""
        if hasattr(self.symqnet, 'reset_buffer'):
            self.symqnet.reset_buffer()
        self.step_count = 0
        self.parameter_history = []
        self.zero_theta_steps = 0
        self.convergence_threshold = 1e-7
        self.convergence_window = 10
        self.last_action = None
        
        logger.debug(" Policy engine state reset")
    
    def get_action(self, current_measurement: np.ndarray) -> Dict[str, Any]:
        """Get next measurement action from policy with EXACT metadata."""
        
        if np.isnan(current_measurement).any():
            current_measurement = np.nan_to_num(current_measurement, nan=0.0)
        
        if len(current_measurement) != self.n_qubits:
            padded_measurement = np.zeros(self.n_qubits)
            min_len = min(len(current_measurement), self.n_qubits)
            padded_measurement[:min_len] = current_measurement[:min_len]
            current_measurement = padded_measurement
        
        obs_tensor = torch.from_numpy(current_measurement).float().to(self.device)  # [n_qubits]
        metadata = self._create_metadata(self.last_action)  # [meta_dim]
        
        logger.debug(f" Input shapes: obs={obs_tensor.shape}, metadata={metadata.shape}")
        
        try:
            with torch.no_grad():
                dist, value, theta_estimate = self.symqnet(obs_tensor, metadata)
                

                if theta_estimate is None:
                    logger.error(" theta_estimate is None!")
                    theta_estimate = torch.zeros(2 * self.n_qubits - 1, device=self.device)
                
                if theta_estimate.numel() == 0:
                    logger.error(" theta_estimate is empty!")
                    theta_estimate = torch.zeros(2 * self.n_qubits - 1, device=self.device)
                
                # Convert to numpy and validate
                theta_np = theta_estimate.detach().cpu().numpy()

                if theta_np.ndim > 1:
                    theta_np = np.squeeze(theta_np)

                expected_dim = 2 * self.n_qubits - 1
                if theta_np.size == expected_dim and theta_np.shape != (expected_dim,):
                    theta_np = theta_np.reshape(expected_dim)

                if theta_np.shape != (expected_dim,):
                    logger.error(f" Wrong parameter shape: {theta_np.shape} (size={theta_np.size})")
                    theta_np = np.zeros(expected_dim)
                
                if np.allclose(theta_np, 0, atol=1e-10):
                    logger.warning(f" All parameters are zero at step {self.step_count}")
                    self.zero_theta_steps += 1
                    if self.zero_theta_steps >= 3:
                        raise InferenceError(
                            "theta_estimate has been all zeros for multiple steps. "
                            "This indicates inference is broken (model load mismatch, "
                            "wrong input shape, or corrupted weights). The MAE is not "
                            "expected to vary with shots until inference is fixed."
                        )
                else:
                    logger.debug(f" Got non-zero parameters: range [{theta_np.min():.6f}, {theta_np.max():.6f}]")
                    self.zero_theta_steps = 0
                
                self.parameter_history.append(theta_np)
                
                # Generate action
                action_idx = dist.sample().item()
                action_info = self._decode_action(action_idx)
                self.last_action = action_info
                
        except Exception as e:
            if isinstance(e, InferenceError):
                raise
            logger.exception(" Error in get_action: inference failed.", exc_info=e)
            logger.error(
                "Inference failure detected; MAE will be invalid if inference is broken."
            )
            raise InferenceError(
                "Inference failed in PolicyEngine.get_action; aborting rollout. "
                "MAE will be invalid if inference is broken."
            ) from e
        
        self.step_count += 1
        return action_info
    
    def _create_metadata(self, action_info: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Create metadata tensor based on the most recent action."""
        n_qubits = self.n_qubits
        M_evo = self.M_evo
        metadata = torch.zeros(self.meta_dim, device=self.device)
        required_dim = n_qubits + 3 + M_evo
        if metadata.numel() < required_dim:
            logger.warning(
                "⚠️ Metadata dim (%s) smaller than required (%s); returning zeros.",
                metadata.numel(),
                required_dim,
            )
            return metadata
        
        if action_info:
            qubits = action_info.get('qubits') or []
            basis_idx = action_info.get('basis_idx')
            time_idx = action_info.get('time_idx')

            if qubits:
                qi = max(0, min(qubits[0], n_qubits - 1))
                metadata[qi] = 1.0
            if basis_idx is not None:
                bi = max(0, min(basis_idx, 2))
                metadata[n_qubits + bi] = 1.0
            if time_idx is not None:
                ti = max(0, min(time_idx, M_evo - 1))
                metadata[n_qubits + 3 + ti] = 1.0
        elif self.step_count > 0:
            qi = self.step_count % n_qubits
            bi = 2  # prefer Z measurements
            ti = self.step_count % M_evo

            metadata[qi] = 1.0
            metadata[n_qubits + bi] = 1.0
            metadata[n_qubits + 3 + ti] = 1.0

        if self.include_shots and metadata.numel() > required_dim:
            metadata[-1] = self._normalize_shots()

        return metadata

    def _normalize_shots(self) -> float:
        """Normalize shot count into [0, 1] for metadata conditioning."""
        if self.shots is None:
            return 0.0
        shot_value = max(0, int(self.shots))
        return float(np.log1p(shot_value) / np.log1p(1_000_000))
    
    def _decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode integer action EXACTLY as in training."""
        M_evo = self.M_evo
        
        action_idx = max(0, min(action_idx, self.A - 1))
        
        time_idx = action_idx % M_evo
        action_idx //= M_evo
        
        basis_idx = action_idx % 3
        qubit_idx = action_idx // 3
        
        qubit_idx = min(qubit_idx, self.n_qubits - 1)
        basis_idx = min(basis_idx, 2)
        time_idx = min(time_idx, M_evo - 1)
        
        basis_map = {0: 'X', 1: 'Y', 2: 'Z'}
        time_map = np.linspace(0.1, 1.0, M_evo)
        
        return {
            'qubits': [qubit_idx],
            'operators': [basis_map[basis_idx]],
            'time': time_map[time_idx],
            'basis_idx': basis_idx,
            'time_idx': time_idx,
            'description': f"{basis_map[basis_idx]}_{qubit_idx}_t{time_idx}"
        }
    
    def get_parameter_estimate(self) -> np.ndarray:
        """Get current parameter estimate from policy."""
        if self.parameter_history:
            estimate = self.parameter_history[-1]
            logger.debug(f" Returning parameter estimate: shape={estimate.shape}, "
                        f"range=[{estimate.min():.6f}, {estimate.max():.6f}]")
            return estimate
        else:
            logger.warning(" No parameter history, returning zeros")
            return np.zeros(2 * self.n_qubits - 1)
    
    def has_converged(self, parameter_estimates: List[np.ndarray]) -> bool:
        """Check if parameter estimates have converged."""
        if len(parameter_estimates) < self.convergence_window:
            return False
        
        recent_estimates = np.array(parameter_estimates[-self.convergence_window:])
        variance = np.var(recent_estimates, axis=0)
        
        return np.all(variance < self.convergence_threshold)
