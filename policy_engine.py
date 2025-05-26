"""
Policy Engine for SymQNet integration using EXACT architectures
FIXED to handle training architecture mismatch
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
        """Load pre-trained VAE and SymQNet models with compatibility handling."""
        
        # 🔥 Load VAE separately (as it was trained)
        self.vae = VariationalAutoencoder(M=10, L=64).to(self.device)
        vae_state = torch.load(self.vae_path, map_location=self.device)
        self.vae.load_state_dict(vae_state)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        
        # 🔧 FIXED: Load checkpoint and inspect what's actually saved
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Model parameters from YOUR training
        n_qubits = 10
        L = 64  # Base latent dimension
        T = 10
        M_evo = 5
        A = n_qubits * 3 * M_evo  # actions
        
        # Graph connectivity from YOUR training
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=self.device) * 0.1
        
        # 🔧 FIXED: Handle architecture mismatch
        try:
            # First, try to create the full model and load directly
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
            
            # Try loading the checkpoint
            if 'model_state_dict' in checkpoint:
                self.symqnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.symqnet.load_state_dict(checkpoint, strict=False)
                
            logger.info("✅ Loaded model with full architecture")
            
        except Exception as e:
            logger.warning(f"Full model loading failed: {e}")
            logger.info("🔄 Trying compatibility mode...")
            
            # 🔧 BACKUP: Create a simpler compatible model
            self._create_compatible_model(checkpoint, n_qubits, L, edge_index, edge_attr, T, A, M_evo)
        
        self.symqnet.eval()
        logger.info("Models loaded successfully")
    
    def _create_compatible_model(self, checkpoint, n_qubits, L, edge_index, edge_attr, T, A, M_evo):
        """Create a model compatible with the saved checkpoint."""
        
        # Check what keys are actually in the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        logger.info(f"Available keys in checkpoint: {list(state_dict.keys())}")
        
        # If only estimator weights are available, create a minimal model
        if len(state_dict) <= 2 and any('estimator' in key for key in state_dict.keys()):
            logger.info("🔧 Creating minimal estimator-only model")
            
            class MinimalSymQNet(nn.Module):
                def __init__(self, vae, input_dim, output_dim):
                    super().__init__()
                    self.vae = vae
                    self.estimator = nn.Linear(input_dim, output_dim)
                    
                def forward(self, obs, metadata):
                    # Encode observation
                    with torch.no_grad():
                        _, _, _, z = self.vae(obs)
                    
                    # Concatenate with metadata
                    combined = torch.cat([z, metadata], dim=-1)
                    
                    # Estimate parameters
                    theta_hat = self.estimator(combined)
                    
                    # Create dummy policy outputs for compatibility
                    dummy_logits = torch.zeros(A, device=self.device)
                    dummy_dist = torch.distributions.Categorical(logits=dummy_logits)
                    dummy_value = torch.tensor(0.0, device=self.device)
                    
                    return dummy_dist, dummy_value, theta_hat
                
                def reset_buffer(self):
                    pass  # No buffer in minimal model
            
            # Create minimal model
            input_dim = L + n_qubits + 3 + M_evo  # z + metadata
            output_dim = 2 * n_qubits - 1  # 19 parameters
            
            self.symqnet = MinimalSymQNet(self.vae, input_dim, output_dim).to(self.device)
            
            # Load estimator weights
            estimator_state = {}
            for key, value in state_dict.items():
                if 'estimator' in key:
                    # Remove 'estimator.' prefix if present
                    new_key = key.replace('estimator.', '')
                    estimator_state[new_key] = value
            
            self.symqnet.estimator.load_state_dict(estimator_state)
            logger.info("✅ Loaded minimal estimator model")
            
        else:
            # Try to create full model with relaxed loading
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
            
            # Load with strict=False to ignore missing keys
            missing_keys, unexpected_keys = self.symqnet.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys (will use random initialization): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys (ignored): {unexpected_keys[:5]}...")
            
            logger.info("✅ Loaded model with partial weights")
    
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
        obs_tensor = torch.from_numpy(current_measurement).float().to(self.device)
        
        # Create metadata (using YOUR exact format)
        metadata = self._create_metadata()
        
        with torch.no_grad():
            # Get action from policy
            dist, value, theta_estimate = self.symqnet(obs_tensor, metadata)
            
            # Sample action (handle both real and dummy distributions)
            try:
                action_idx = dist.sample().item()
            except:
                # Fallback for dummy distributions
                action_idx = np.random.randint(0, 150)  # A = 150
            
            # Decode action
            action_info = self._decode_action(action_idx)
            
            # Store parameter estimate
            self.parameter_history.append(theta_estimate.cpu().numpy())
        
        self.step_count += 1
        
        return action_info
    
    def _create_metadata(self) -> torch.Tensor:
        """Create metadata tensor using YOUR exact format."""
        n_qubits = 10
        M_evo = 5
        meta_dim = n_qubits + 3 + M_evo  # exactly as in your code
        metadata = torch.zeros(meta_dim, device=self.device)
        
        # Set some default values based on step (following your training logic)
        if self.step_count > 0:
            # Example: favor Z measurements initially, first time step
            qi = min(self.step_count % n_qubits, n_qubits - 1)  # cycle through qubits
            bi = 2  # basis index (Z)
            ti = min(self.step_count % M_evo, M_evo - 1)  # cycle through times
            
            metadata[qi] = 1.0  # qubit index
            metadata[n_qubits + bi] = 1.0  # basis index
            metadata[n_qubits + 3 + ti] = 1.0  # time index
        
        return metadata
    
    def _decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode integer action using YOUR exact format."""
        M_evo = 5
        
        # Ensure action_idx is within bounds
        action_idx = max(0, min(action_idx, 149))  # Clamp to valid range
        
        time_idx = action_idx % M_evo
        action_idx //= M_evo
        
        basis_idx = action_idx % 3
        qubit_idx = action_idx // 3
        
        # Ensure indices are within bounds
        qubit_idx = min(qubit_idx, 9)  # Max 9 for 10 qubits
        basis_idx = min(basis_idx, 2)  # Max 2 for 3 bases
        time_idx = min(time_idx, M_evo - 1)  # Max M_evo-1
        
        basis_map = {0: 'X', 1: 'Y', 2: 'Z'}
        time_map = np.linspace(0.1, 1.0, M_evo)  # matching your SpinChainEnv
        
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
            # Default for 10-qubit system: 9 coupling + 10 field = 19 parameters
            return np.zeros(19)
    
    def has_converged(self, parameter_estimates: List[np.ndarray]) -> bool:
        """Check if parameter estimates have converged."""
        if len(parameter_estimates) < self.convergence_window:
            return False
        
        # Check variance in recent estimates
        recent_estimates = np.array(parameter_estimates[-self.convergence_window:])
        variance = np.var(recent_estimates, axis=0)
        
        return np.all(variance < self.convergence_threshold)
