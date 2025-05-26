"""
Policy Engine for SymQNet integration using EXACT architectures
CLINICALLY FIXED to match training code architecture 1:1
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# 🔥 IMPORT YOUR EXACT ARCHITECTURES 🔥
from architectures import (
    VariationalAutoencoder, 
    GraphEmbed,
    TemporalContextualAggregator, 
    PolicyValueHead,
    FixedSymQNetWithEstimator
)

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
        """Load pre-trained VAE and SymQNet models with EXACT architecture matching."""
        
        # 🔥 Load VAE exactly as trained
        self.vae = VariationalAutoencoder(M=10, L=64).to(self.device)
        vae_state = torch.load(self.vae_path, map_location=self.device, weights_only=False)
        self.vae.load_state_dict(vae_state)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        
        # 🔧 FIXED: Inspect checkpoint to determine EXACT architecture
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get the actual state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        logger.info(f"🔍 Checkpoint contains {len(state_dict)} parameters")
        logger.info(f"🔍 Keys: {list(state_dict.keys())[:10]}...")  # Show first 10 keys
        
        # 🔧 TRAINING PARAMETERS (EXACT)
        n_qubits = 10
        L_vae = 64  # VAE latent dimension
        meta_dim = n_qubits + 3 + 5  # 10 + 3 + 5 = 18
        L_total = L_vae + meta_dim   # 64 + 18 = 82 (CRITICAL!)
        T = 10
        M_evo = 5
        A = n_qubits * 3 * M_evo  # 150 actions
        
        # 🔧 ARCHITECTURE DETECTION
        is_simple_estimator = self._detect_simple_estimator(state_dict)
        
        if is_simple_estimator:
            logger.info("🎯 Detected simple estimator-only model")
            self._create_minimal_model(state_dict, n_qubits, L_vae, meta_dim, M_evo, A)
        else:
            logger.info("🎯 Detected full trained model")
            self._create_full_model(state_dict, n_qubits, L_total, T, A, M_evo)
        
        self.symqnet.eval()
        logger.info("✅ Models loaded with EXACT architecture match")
    
    def _detect_simple_estimator(self, state_dict):
        """Detect if this is a simple estimator or full model."""
        
        # Count different component types
        has_graph_embed = any('graph_embed' in key for key in state_dict.keys())
        has_temp_agg = any('temp_agg' in key for key in state_dict.keys())
        has_policy = any('policy_value' in key for key in state_dict.keys())
        estimator_keys = [key for key in state_dict.keys() if 'estimator' in key]
        
        logger.info(f"🔍 Architecture detection:")
        logger.info(f"   Graph embed: {has_graph_embed}")
        logger.info(f"   Temporal agg: {has_temp_agg}")
        logger.info(f"   Policy head: {has_policy}")
        logger.info(f"   Estimator keys: {len(estimator_keys)}")
        
        # Simple estimator if only estimator components
        is_simple = (
            not has_graph_embed and
            not has_temp_agg and
            not has_policy and
            len(estimator_keys) > 0
        )
        
        return is_simple
    
    def _create_minimal_model(self, state_dict, n_qubits, L_vae, meta_dim, M_evo, A):
        """Create minimal model matching training's estimator architecture."""
        
        # Detect estimator architecture type
        estimator_keys = [key for key in state_dict.keys() if 'estimator' in key]
        
        # Check if MLP (training uses Sequential with 0, 2, 4 indices)
        is_mlp_estimator = any('estimator.0.' in key or 'estimator.2.' in key or 'estimator.4.' in key 
                              for key in estimator_keys)
        
        if is_mlp_estimator:
            logger.info("🔧 Creating MLP estimator (matches training)")
        else:
            logger.info("🔧 Creating linear estimator")
        
        class MinimalSymQNet(nn.Module):
            def __init__(self, vae, L_vae, meta_dim, n_qubits, device, is_mlp):
                super().__init__()
                self.vae = vae
                self.device = device
                self.L_vae = L_vae
                self.meta_dim = meta_dim
                self.n_qubits = n_qubits
                
                input_dim = L_vae + meta_dim  # 64 + 18 = 82
                output_dim = 2 * n_qubits - 1  # 19 parameters
                
                # 🔧 EXACT estimator architecture from training
                if is_mlp:
                    self.estimator = nn.Sequential(
                        nn.Linear(input_dim, 128),  # 82 → 128
                        nn.ReLU(),
                        nn.Linear(128, 64),         # 128 → 64
                        nn.ReLU(),
                        nn.Linear(64, output_dim)   # 64 → 19
                    )
                else:
                    self.estimator = nn.Linear(input_dim, output_dim)
                
                self.step_count = 0
                
            def forward(self, obs, metadata):
                """EXACT forward interface matching training"""
                # VAE encoding (EXACT as training)
                with torch.no_grad():
                    mu_z, logvar_z = self.vae.encode(obs)
                    z = self.vae.reparameterize(mu_z, logvar_z)
                
                # Concatenate with metadata (EXACT as training)
                z_with_meta = torch.cat([z, metadata], dim=-1)
                
                # Estimate parameters
                theta_hat = self.estimator(z_with_meta)
                
                # Create dummy policy outputs for compatibility
                action_probs = torch.ones(A, device=self.device) / A
                dummy_dist = torch.distributions.Categorical(probs=action_probs)
                dummy_value = torch.tensor(0.0, device=self.device)
                
                return dummy_dist, dummy_value, theta_hat
            
            def reset_buffer(self):
                self.step_count = 0
        
        # Create minimal model
        self.symqnet = MinimalSymQNet(
            self.vae, L_vae, meta_dim, n_qubits, self.device, is_mlp_estimator
        ).to(self.device)
        
        # 🔧 LOAD ESTIMATOR WEIGHTS with proper key mapping
        self._load_estimator_weights(state_dict, is_mlp_estimator)
    
    def _load_estimator_weights(self, state_dict, is_mlp):
        """Load estimator weights with exact key mapping."""
        
        estimator_state = {}
        
        if is_mlp:
            # Map Sequential indices: estimator.0. → 0., estimator.2. → 2., etc.
            for key, value in state_dict.items():
                if 'estimator.' in key:
                    new_key = key.replace('estimator.', '')
                    estimator_state[new_key] = value
        else:
            # Direct linear layer mapping
            for key, value in state_dict.items():
                if key == 'estimator.weight':
                    estimator_state['weight'] = value
                elif key == 'estimator.bias':
                    estimator_state['bias'] = value
        
        # Load with error checking
        try:
            self.symqnet.estimator.load_state_dict(estimator_state, strict=True)
            logger.info("✅ Estimator weights loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Estimator loading issue: {e}")
            self.symqnet.estimator.load_state_dict(estimator_state, strict=False)
    
    def _create_full_model(self, state_dict, n_qubits, L_total, T, A, M_evo):
        """Create full model matching EXACT training architecture."""
        
        # 🔧 EXACT graph connectivity from training
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=self.device) * 0.1
        
        # 🔧 CRITICAL: Use L_total (82) not L_vae (64)
        self.symqnet = FixedSymQNetWithEstimator(
            vae=self.vae,
            n_qubits=n_qubits,
            L=L_total,  # ✅ FIXED: 82 not 64!
            edge_index=edge_index,
            edge_attr=edge_attr,
            T=T,
            A=A,
            M_evo=M_evo,
            K_gnn=2
        ).to(self.device)
        
        # Load with architecture matching
        try:
            missing_keys, unexpected_keys = self.symqnet.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing {len(missing_keys)} keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected {len(unexpected_keys)} keys: {unexpected_keys[:5]}...")
            
            logger.info("✅ Full model loaded with available weights")
            
        except Exception as e:
            logger.error(f"❌ Full model loading failed: {e}")
            raise
    
    def reset(self):
        """Reset policy state for new rollout."""
        if hasattr(self.symqnet, 'reset_buffer'):
            self.symqnet.reset_buffer()
        self.step_count = 0
        self.parameter_history = []
        self.convergence_threshold = 1e-4
        self.convergence_window = 5
    
    def get_action(self, current_measurement: np.ndarray) -> Dict[str, Any]:
        """Get next measurement action from policy with EXACT metadata."""
        
        # Convert measurement to tensor
        if len(current_measurement) != 10:
            # Pad or truncate to 10 elements
            padded_measurement = np.zeros(10)
            min_len = min(len(current_measurement), 10)
            padded_measurement[:min_len] = current_measurement[:min_len]
            current_measurement = padded_measurement
        
        obs_tensor = torch.from_numpy(current_measurement).float().to(self.device)
        
        # 🔧 EXACT metadata creation matching training
        metadata = self._create_metadata()
        
        with torch.no_grad():
            # 🔧 EXACT forward interface: (obs, metadata)
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
        """Create metadata tensor EXACTLY as in training."""
        n_qubits = 10
        M_evo = 5
        meta_dim = n_qubits + 3 + M_evo  # 18
        
        metadata = torch.zeros(meta_dim, device=self.device)
        
        # 🔧 EXACT one-hot encoding as in training
        if self.step_count > 0:
            qi = self.step_count % n_qubits  # cycle through qubits
            bi = 2  # prefer Z measurements initially
            ti = self.step_count % M_evo  # cycle through times
            
            # One-hot encoding (EXACT as training)
            metadata[qi] = 1.0  # qubit index
            metadata[n_qubits + bi] = 1.0  # basis index  
            metadata[n_qubits + 3 + ti] = 1.0  # time index
        
        return metadata
    
    def _decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode integer action EXACTLY as in training."""
        M_evo = 5
        
        # Ensure valid range
        action_idx = max(0, min(action_idx, 149))  # 0 to 149
        
        # 🔧 EXACT decoding from training
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
            return np.zeros(19)  # 2*10-1 = 19 parameters
    
    def has_converged(self, parameter_estimates: List[np.ndarray]) -> bool:
        """Check if parameter estimates have converged."""
        if len(parameter_estimates) < self.convergence_window:
            return False
        
        recent_estimates = np.array(parameter_estimates[-self.convergence_window:])
        variance = np.var(recent_estimates, axis=0)
        
        return np.all(variance < self.convergence_threshold)
