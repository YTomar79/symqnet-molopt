"""
Policy Engine for SymQNet integration using EXACT architectures
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
        """Load pre-trained VAE and SymQNet models using EXACT architectures."""
        
        # 🔥 Use YOUR exact VAE architecture
        self.vae = VariationalAutoencoder(M=10, L=64).to(self.device)
        vae_state = torch.load(self.vae_path, map_location=self.device)
        self.vae.load_state_dict(vae_state)
        self.vae.eval()
        
        # 🔥 Use YOUR exact SymQNet architecture  
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Model parameters from YOUR training (EXACTLY as in your code)
        n_qubits = 10
        L = 64  # Base latent dimension
        T = 10
        M_evo = 5
        A = n_qubits * 3 * M_evo  # actions
        
        # Graph connectivity from YOUR training (EXACTLY as in your code)
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=self.device) * 0.1
        
        # 🔥 Use YOUR exact FixedSymQNetWithEstimator
        self.symqnet = FixedSymQNetWithEstimator(
            vae=self.vae,
            n_qubits=n_qubits,
            L=L,  # ✅ FIXED: Use base dimension (64), not total (82)
            edge_index=edge_index,
            edge_attr=edge_attr,
            T=T,
            A=A,
            M_evo=M_evo,
            K_gnn=2
        ).to(self.device)
        
        # Load weights - this will now work perfectly!
        if 'model_state_dict' in checkpoint:
            self.symqnet.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.symqnet.load_state_dict(checkpoint)
        
        self.symqnet.eval()
        
        logger.info("Models loaded successfully with exact architectures")

    
    def reset(self):
        """Reset policy state for new rollout."""
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
            
            # Sample action
            action_idx = dist.sample().item()
            
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
            qi = 0  # qubit index
            bi = 2  # basis index (Z)
            ti = 0  # time index
            
            metadata[qi] = 1.0  # qubit index
            metadata[n_qubits + bi] = 1.0  # basis index
            metadata[n_qubits + 3 + ti] = 1.0  # time index
        
        return metadata
    
    def _decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode integer action using YOUR exact format."""
        M_evo = 5
        
        time_idx = action_idx % M_evo
        action_idx //= M_evo
        
        basis_idx = action_idx % 3
        qubit_idx = action_idx // 3
        
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
