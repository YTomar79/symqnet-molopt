"""
Policy Engine for SymQNet integration using EXACT architectures
FIXED for PyTorch 2.6 weights_only issue
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# 🔥 IMPORT YOUR EXACT ARCHITECTURES 🔥
from architectures import VariationalAutoencoder, FixedSymQNetWithEstimator

logger = logging.getLogger(__name__)

class PolicyEngine:
    """Integrates trained SymQNet for molecular Hamiltonian estimation."""
    
    def __init__(self, model_path: Path, vae_path: Path, device: torch.device):
        self.device = device
        self.model_path = model_path
        self.vae_path = vae_path
        
        # Load models
        self._load_models()
        
        # Initialize buffers
        self.reset()
        
        logger.info("Policy engine initialized successfully")
    
    def _load_models(self):
        """Load pre-trained VAE and SymQNet models with smart architecture detection."""
        
        # 🔥 Load VAE separately (as it was trained)
        self.vae = VariationalAutoencoder(M=10, L=64).to(self.device)
        # 🔧 FIX: Add weights_only=False for PyTorch 2.6
        vae_state = torch.load(self.vae_path, map_location=self.device, weights_only=False)
        self.vae.load_state_dict(vae_state)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        
        # 🔧 FIXED: Inspect checkpoint first to determine architecture  
        # 🔧 FIX: Add weights_only=False for PyTorch 2.6
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get the actual state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        logger.info(f"🔍 Checkpoint contains {len(state_dict)} parameters")
        logger.info(f"🔍 Keys: {list(state_dict.keys())}")
        
        # Model parameters
        n_qubits = 10
        L = 64  # Base latent dimension
        T = 10
        M_evo = 5
        A = n_qubits * 3 * M_evo  # actions
        
        # 🔧 SMART DETECTION: Check if this is a simple estimator-only model
        is_simple_estimator = (
            len(state_dict) <= 3 and  # Very few parameters
            any('estimator' in key for key in state_dict.keys()) and  # Has estimator
            not any('vae' in key for key in state_dict.keys()) and  # No VAE
            not any('graph_embed' in key for key in state_dict.keys())  # No graph components
        )
        
        if is_simple_estimator:
            logger.info("🎯 Detected simple estimator-only model - creating minimal architecture")
            self._create_minimal_model(state_dict, n_qubits, L, M_evo, A)
        else:
            logger.info("🎯 Detected full model - creating complete architecture")
            self._create_full_model(state_dict, n_qubits, L, T, A, M_evo)
        
        self.symqnet.eval()
        logger.info("✅ Models loaded successfully")
    
    def _create_minimal_model(self, state_dict, n_qubits, L, M_evo, A):
        """Create minimal model for estimator-only checkpoints."""
        
        class MinimalSymQNet(nn.Module):
            def __init__(self, vae, input_dim, output_dim, device):
                super().__init__()
                self.vae = vae
                self.device = device
                self.estimator = nn.Linear(input_dim, output_dim)
                self.step_count = 0
                
            def forward(self, obs, metadata):
                # Encode observation
                with torch.no_grad():
                    _, _, _, z = self.vae(obs)
                
                # Concatenate with metadata
                combined = torch.cat([z, metadata], dim=-1)
                
                # Estimate parameters
                theta_hat = self.estimator(combined)
                
                # Create simple policy outputs (random but valid)
                action_probs = torch.ones(A, device=self.device) / A  # Uniform distribution
                dummy_dist = torch.distributions.Categorical(probs=action_probs)
                dummy_value = torch.tensor(0.0, device=self.device)
                
                return dummy_dist, dummy_value, theta_hat
            
            def reset_buffer(self):
                self.step_count = 0
        
        # Create minimal model
        input_dim = L + n_qubits + 3 + M_evo  # z + metadata = 64 + 18 = 82
        output_dim = 2 * n_qubits - 1  # 19 parameters
        
        self.symqnet = MinimalSymQNet(self.vae, input_dim, output_dim, self.device).to(self.device)
        
        # Load estimator weights
        estimator_state = {}
        for key, value in state_dict.items():
            if 'estimator' in key:
                # Handle different possible key formats
                if key == 'estimator.weight':
                    estimator_state['weight'] = value
                elif key == 'estimator.bias':
                    estimator_state['bias'] = value
                else:
                    # Remove 'estimator.' prefix if present
                    new_key = key.replace('estimator.', '')
                    estimator_state[new_key] = value
        
        # Load the estimator weights
        self.symqnet.estimator.load_state_dict(estimator_state)
        logger.info("✅ Loaded minimal estimator model")
    
    def _create_full_model(self, state_dict, n_qubits, L, T, A, M_evo):
        """Create full model for complete checkpoints."""
        
        # Graph connectivity
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=self.device) * 0.1
        
        # Create full model
        self.symqnet = FixedSymQNetWithEstimator(
            vae=self.vae,
            n_qubits=n_qubits,
            L=L,
            edge_index=edge_index,
            edge_attr=edge_attr,
            T=T,
            A=A,
            M_evo=M_evo,
            K_gnn=2
        ).to(self.device)
        
        # Load with strict=False to handle missing keys
        missing_keys, unexpected_keys = self.symqnet.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} keys (using random init)")
        if unexpected_keys:
            logger.warning(f"Ignoring {len(unexpected_keys)} unexpected keys")
        
        logger.info("✅ Loaded full model with available weights")
    
    def reset(self):
        """Reset policy state for new rollout."""
        if hasattr(self.symqnet, 'reset_buffer'):
            self.symqnet.reset_buffer()
        self.step_count = 0
        self.parameter_history = []
        self.convergence_threshold = 1e-4
        self.convergence_window = 5
    
    def get_action(self, current_measurement: np.ndarray) -> Dict[str, Any]:
        """Get next measurement action from policy."""
        
        # Convert measurement to tensor
        if len(current_measurement) != 10:
            # Pad or truncate to 10 elements
            padded_measurement = np.zeros(10)
            min_len = min(len(current_measurement), 10)
            padded_measurement[:min_len] = current_measurement[:min_len]
            current_measurement = padded_measurement
        
        obs_tensor = torch.from_numpy(current_measurement).float().to(self.device)
        
        # Create metadata
        metadata = self._create_metadata()
        
        with torch.no_grad():
            # Get action from policy
            dist, value, theta_estimate = self.symqnet(obs_tensor, metadata)
            
            # Sample action
            action_idx = dist.sample().item()
            
            # Decode action
            action_info = self._decode_action(action_idx)
            
            # Store parameter estimate
            self.parameter_history.append(theta_estimate.cpu().numpy())
        
        self.step_count += 1
        
        return action_info
    
    def _create_metadata(self) -> torch.Tensor:
        """Create metadata tensor."""
        n_qubits = 10
        M_evo = 5
        meta_dim = n_qubits + 3 + M_evo  # 18
        metadata = torch.zeros(meta_dim, device=self.device)
        
        # Set some reasonable defaults based on step
        if self.step_count > 0:
            qi = self.step_count % n_qubits  # cycle through qubits
            bi = 2  # prefer Z measurements
            ti = self.step_count % M_evo  # cycle through times
            
            metadata[qi] = 1.0
            metadata[n_qubits + bi] = 1.0  
            metadata[n_qubits + 3 + ti] = 1.0
        
        return metadata
    
    def _decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode integer action."""
        M_evo = 5
        
        # Ensure valid range
        action_idx = max(0, min(action_idx, 149))
        
        time_idx = action_idx % M_evo
        action_idx //= M_evo
        
        basis_idx = action_idx % 3
        qubit_idx = action_idx // 3
        
        # Clamp to valid ranges
        qubit_idx = min(qubit_idx, 9)
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
            return self.parameter_history[-1]
        else:
            return np.zeros(19)
    
    def has_converged(self, parameter_estimates: List[np.ndarray]) -> bool:
        """Check if parameter estimates have converged."""
        if len(parameter_estimates) < self.convergence_window:
            return False
        
        recent_estimates = np.array(parameter_estimates[-self.convergence_window:])
        variance = np.var(recent_estimates, axis=0)
        
        return np.all(variance < self.convergence_threshold)
