"""
Policy Engine for SymQNet integration 
Fixed tensor shapes, error checking, and parameter extraction.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


from architectures import (
    VariationalAutoencoder,
    FixedSymQNetWithEstimator,
    MetadataLayout,
    SpinChainEnv,
)
from smc_filter import SMCParticleFilter, covariance_to_features

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when policy inference fails and rollouts should abort."""

class PolicyEngine:
    """Integrates trained SymQNet for molecular Hamiltonian estimation."""

    CHECKPOINT_FORMAT = "symqnet-ppo-v2"
    CHECKPOINT_VERSION = 1
    
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
        
        # Inspect checkpoint metadata to determine EXACT architecture
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        checkpoint_data = self._validate_checkpoint(checkpoint)
        state_dict = checkpoint_data["model_state_dict"]
        
        logger.info(f"🔍 Checkpoint contains {len(state_dict)} parameters")
        logger.info(f"🔍 Keys: {list(state_dict.keys())[:10]}...")
        
        #  PARAMETERS (explicit from checkpoint metadata)
        self.n_qubits = self._coerce_positive_int(checkpoint_data["n_qubits"], "n_qubits")
        self.T = self._coerce_positive_int(checkpoint_data["rollout_steps"], "rollout_steps")
        self.M_evo = self._coerce_positive_int(checkpoint_data["M_evo"], "M_evo")
        self.A = self.n_qubits * 3 * self.M_evo  # 150 actions
        self.meta_dim = self._coerce_positive_int(checkpoint_data["meta_dim"], "meta_dim")
        self.shots_encoding = checkpoint_data["shots_encoding"]
        self.include_shots = bool(self.shots_encoding)
        logger.info("🔍 Metadata: meta_dim=%s, include_shots=%s", self.meta_dim, self.include_shots)

        self.metadata_layout = MetadataLayout.from_problem(self.n_qubits, self.M_evo)
        if self.meta_dim != self.metadata_layout.meta_dim:
            raise ValueError(
                "Checkpoint metadata dimension mismatch: "
                f"checkpoint meta_dim={self.meta_dim}, "
                f"expected={self.metadata_layout.meta_dim} for "
                f"{self.n_qubits} qubits and M_evo={self.M_evo}."
            )
        self.theta_dim = self.metadata_layout.theta_dim
        self.cov_feat_dim = self.metadata_layout.cov_feat_dim
        self.theta_slot0 = self.metadata_layout.theta_slot0
        self.cov_slot0 = self.metadata_layout.cov_slot0
        self.fisher_slot0 = self.metadata_layout.fisher_slot0
        self.shots_slot = self.metadata_layout.shots_slot
        
        is_simple_estimator = self._detect_simple_estimator(state_dict)
        
        if is_simple_estimator:
            logger.info("🎯 Detected estimator-only model; loading full architecture with partial weights")
            self._create_minimal_model(state_dict, self.n_qubits, self.M_evo, self.A, self.meta_dim)
        else:
            logger.info("🎯 Detected full trained model")
            self._create_full_model(state_dict, self.n_qubits, self.T, self.A, self.M_evo, self.meta_dim)
        
        self.symqnet.eval()
        logger.info(" Models loaded with EXACT architecture match")

        self.belief_env = SpinChainEnv(
            N=self.n_qubits,
            M_evo=self.M_evo,
            T=self.T,
            device=self.device,
            resample_each_reset=False,
        )
        self.smc = SMCParticleFilter(self.belief_env, device=self.device)

    def _validate_checkpoint(self, checkpoint: Any) -> Dict[str, Any]:
        """Validate checkpoint schema and return normalized checkpoint data."""
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported checkpoint format: expected dict-based checkpoint.")

        if "model_state_dict" not in checkpoint:
            legacy_message = (
                "Legacy checkpoint detected (missing 'model_state_dict' and metadata). "
                "Please re-export using a versioned checkpoint with explicit fields "
                "(model_state_dict, meta_dim, shots_encoding, n_qubits, M_evo, rollout_steps)."
            )
            raise ValueError(legacy_message)

        required_keys = {
            "model_state_dict",
            "meta_dim",
            "shots_encoding",
            "n_qubits",
            "M_evo",
            "rollout_steps",
        }
        missing = sorted(required_keys - checkpoint.keys())
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")

        has_format = "checkpoint_format" in checkpoint
        has_version = "checkpoint_version" in checkpoint
        if has_format != has_version:
            raise ValueError(
                "Checkpoint must include both 'checkpoint_format' and 'checkpoint_version'."
            )
        if has_format and has_version:
            if checkpoint["checkpoint_format"] != self.CHECKPOINT_FORMAT:
                raise ValueError(
                    f"Unsupported checkpoint format: {checkpoint['checkpoint_format']}. "
                    f"Expected '{self.CHECKPOINT_FORMAT}'."
                )
            if checkpoint["checkpoint_version"] != self.CHECKPOINT_VERSION:
                raise ValueError(
                    f"Unsupported checkpoint version: {checkpoint['checkpoint_version']}. "
                    f"Expected {self.CHECKPOINT_VERSION}."
                )
        else:
            logger.warning(
                "⚠️ Legacy checkpoint without format/version; proceeding with validated metadata."
            )

        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint 'model_state_dict' must be a state dict.")

        shots_encoding = checkpoint["shots_encoding"]
        if shots_encoding is not None and not isinstance(shots_encoding, dict):
            raise ValueError("Checkpoint 'shots_encoding' must be a dict or None.")
        if isinstance(shots_encoding, dict) and "type" not in shots_encoding:
            raise ValueError("Checkpoint 'shots_encoding' dict must include a 'type' field.")

        return checkpoint

    def _coerce_positive_int(self, value: Any, name: str) -> int:
        """Convert checkpoint metadata to positive int values."""
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint field '{name}' must be an integer.") from exc
        if int_value <= 0:
            raise ValueError(f"Checkpoint field '{name}' must be > 0.")
        return int_value
    
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

    def _create_minimal_model(self, state_dict, n_qubits, M_evo, A, meta_dim):
        """Create minimal model matching training's estimator architecture."""
        
        class MinimalSymQNet(nn.Module):
            def __init__(self, vae, n_qubits, device, meta_dim):
                super().__init__()
                self.vae = vae
                self.device = device
                self.n_qubits = n_qubits
                self.meta_dim = meta_dim
            
            def forward(self, obs, metadata, deterministic_inference: bool = False):
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)  # [10] -> [1, 10]
                if metadata.dim() == 1:
                    metadata = metadata.unsqueeze(0)  # [meta] -> [1, meta]
                
                if metadata.shape[-1] != self.meta_dim:
                    raise ValueError(
                        f"Metadata dimension mismatch: expected {self.meta_dim}, "
                        f"got {metadata.shape[-1]}"
                    )

                # VAE encoding
                with torch.no_grad():
                    mu_z, logvar_z = self.vae.encode(obs)
                    z = self.vae.reparameterize(mu_z, logvar_z)  # [1, 64]
                
                # Create dummy policy outputs
                action_probs = torch.ones(A, device=self.device) / A
                dummy_dist = torch.distributions.Categorical(probs=action_probs)
                dummy_value = torch.tensor(0.0, device=self.device)

                return dummy_dist, dummy_value
            
            def reset_buffer(self):
                return None
        
        self.symqnet = MinimalSymQNet(
            self.vae,
            n_qubits,
            self.device,
            meta_dim,
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
        self.convergence_threshold = 1e-7
        self.convergence_window = 10
        self.last_action = None
        if hasattr(self, "theta_dim") and hasattr(self, "cov_feat_dim"):
            self.prev_theta_feat = torch.zeros(self.theta_dim, device=self.device)
            self.prev_cov_feat = torch.zeros(self.cov_feat_dim, device=self.device)
            self.prev_fisher_feat = torch.zeros(self.theta_dim, device=self.device)
        if hasattr(self, "smc") and self.smc is not None:
            self.smc.reset()
        
        logger.debug(" Policy engine state reset")

    def _update_belief(self, obs_tensor: torch.Tensor) -> None:
        """Update SMC belief using the latest observation and previous action."""
        if not hasattr(self, "smc") or self.smc is None:
            return
        if self.last_action is None:
            return

        info = {
            "qubit_idx": int(self.last_action.get("qubits", [0])[0]),
            "basis_idx": int(self.last_action.get("basis_idx", 0)),
            "time_idx": int(self.last_action.get("time_idx", 0)),
            "shots": int(self.shots or 1),
        }

        theta_mean, theta_cov = self.smc.update(obs_tensor, info)
        if theta_mean.numel() != self.theta_dim:
            raise InferenceError(
                f"Posterior mean shape mismatch: expected {self.theta_dim}, "
                f"got {theta_mean.numel()}"
            )
        if torch.isnan(theta_mean).any() or torch.isnan(theta_cov).any():
            raise InferenceError("SMC posterior contains NaNs; check metadata inputs.")

        self.prev_theta_feat = theta_mean.detach()
        self.prev_cov_feat = covariance_to_features(theta_cov).detach()
        precision = torch.linalg.pinv(theta_cov)
        self.prev_fisher_feat = torch.diag(precision).detach()
        self.parameter_history.append(theta_mean.detach().cpu().numpy())
    
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
        self._update_belief(obs_tensor)
        metadata = self._create_metadata(self.last_action)  # [meta_dim]
        
        logger.debug(f" Input shapes: obs={obs_tensor.shape}, metadata={metadata.shape}")
        
        try:
            with torch.no_grad():
                dist, value = self.symqnet(obs_tensor, metadata)
                
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
        if metadata.numel() != self.meta_dim:
            raise ValueError(
                f"Metadata size mismatch: expected {self.meta_dim}, got {metadata.numel()}"
            )
        
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

        if self.include_shots:
            metadata[self.shots_slot] = self._normalize_shots()

        theta_slice = slice(self.theta_slot0, self.theta_slot0 + self.theta_dim)
        cov_slice = slice(self.cov_slot0, self.cov_slot0 + self.cov_feat_dim)
        fisher_slice = slice(self.fisher_slot0, self.fisher_slot0 + self.theta_dim)

        if self.prev_theta_feat.numel() != self.theta_dim:
            raise ValueError("Posterior mean feature dimension mismatch.")
        if self.prev_cov_feat.numel() != self.cov_feat_dim:
            raise ValueError("Posterior covariance feature dimension mismatch.")
        if self.prev_fisher_feat.numel() != self.theta_dim:
            raise ValueError("Posterior fisher feature dimension mismatch.")

        metadata[theta_slice] = self.prev_theta_feat
        metadata[cov_slice] = self.prev_cov_feat
        metadata[fisher_slice] = self.prev_fisher_feat

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
