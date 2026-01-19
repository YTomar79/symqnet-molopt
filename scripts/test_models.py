#!/usr/bin/env python3
"""
Test model loading with exact architectures

This script specifically tests that trained models can be loaded
correctly with exact architectures from the source code.
"""

import sys
import os
import torch
import numpy as np
import traceback
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures import MetadataLayout, SpinChainEnv
from smc_filter import SMCParticleFilter, covariance_to_features


def normalize_shots(shots: int) -> float:
    """Normalize shot count into [0, 1] for metadata conditioning."""
    shot_value = max(0, int(shots))
    return float(np.log1p(shot_value) / np.log1p(1_000_000))


def init_smc_belief(n_qubits: int, M_evo: int, device: torch.device) -> SMCParticleFilter:
    """Initialize a compact SMC belief filter for metadata updates."""
    belief_env = SpinChainEnv(
        N=n_qubits,
        M_evo=M_evo,
        T=10,
        device=device,
        resample_each_reset=False,
    )
    return SMCParticleFilter(belief_env, n_particles=4, device=device)


def build_metadata_with_belief(
    layout: MetadataLayout,
    smc: SMCParticleFilter,
    obs: torch.Tensor,
    action_info: dict,
    shots: int,
) -> torch.Tensor:
    """Build metadata using SMC belief updates and the latest action."""
    metadata = torch.zeros(layout.meta_dim, device=obs.device)
    qubit_idx = int(action_info.get("qubit_idx", 0))
    basis_idx = int(action_info.get("basis_idx", 2))
    time_idx = int(action_info.get("time_idx", 0))

    metadata[qubit_idx] = 1.0
    metadata[layout.n_qubits + basis_idx] = 1.0
    metadata[layout.n_qubits + 3 + time_idx] = 1.0
    metadata[layout.shots_slot] = normalize_shots(shots)

    theta_mean, theta_cov = smc.update(
        obs,
        {
            "qubit_idx": qubit_idx,
            "basis_idx": basis_idx,
            "time_idx": time_idx,
            "shots": shots,
        },
    )

    theta_slice = slice(layout.theta_slot0, layout.theta_slot0 + layout.theta_dim)
    cov_slice = slice(layout.cov_slot0, layout.cov_slot0 + layout.cov_feat_dim)
    fisher_slice = slice(layout.fisher_slot0, layout.fisher_slot0 + layout.theta_dim)

    metadata[theta_slice] = theta_mean
    metadata[cov_slice] = covariance_to_features(theta_cov)
    metadata[fisher_slice] = torch.diag(torch.linalg.pinv(theta_cov))
    return metadata

def print_test_header(title):
    """Print formatted test header"""
    print(f" {title}")

def print_test_result(test_name, success, details=""):
    """Print test result"""
    status = " PASS" if success else " FAIL"
    print(f"{test_name:<40} | {status}")
    if details:
        print(f"    └─ {details}")

def test_architecture_imports():
    """Test importing all architectures from  exact code"""
    print_test_header("Architecture Import Tests")
    
    try:
        from architectures import (
            VariationalAutoencoder,
            GraphEmbed, 
            TemporalContextualAggregator,
            PolicyValueHead,
            FixedSymQNetWithEstimator,
            SpinChainEnv,
            get_pauli_matrices,
            kron_n,
            generate_measurement_pair
        )
        print_test_result("Import all architectures", True, "All classes imported successfully")
        return True
    except ImportError as e:
        print_test_result("Import all architectures", False, f"Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print_test_result("Import all architectures", False, f"Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_vae_creation():
    """Test VAE creation and basic functionality"""
    print_test_header("VAE Architecture Tests")
    
    try:
        from architectures import VariationalAutoencoder
        
        # Test VAE creation
        M, L = 10, 64
        vae = VariationalAutoencoder(M=M, L=L)
        print_test_result("VAE creation", True, f"M={M}, L={L}")
        
        # Test VAE forward pass
        test_input = torch.randn(M)
        recon, mu, logvar, z = vae(test_input)
        
        # Check output shapes
        shapes_correct = (
            recon.shape == torch.Size([M]) and
            mu.shape == torch.Size([L]) and 
            logvar.shape == torch.Size([L]) and
            z.shape == torch.Size([L])
        )
        print_test_result("VAE forward pass", shapes_correct, 
                         f"Shapes: recon{recon.shape}, mu{mu.shape}, z{z.shape}")
        
        # Test encoding specifically
        mu_test, logvar_test = vae.encode(test_input)
        z_test = vae.reparameterize(mu_test, logvar_test)
        encoding_ok = mu_test.shape == torch.Size([L]) and z_test.shape == torch.Size([L])
        print_test_result("VAE encoding", encoding_ok, f"Latent dim: {L}")
        
        return shapes_correct and encoding_ok
        
    except Exception as e:
        print_test_result("VAE tests", False, f"Error: {e}")
        traceback.print_exc()
        return False

def test_graph_embed():
    """Test GraphEmbed layer"""
    print_test_header("GraphEmbed Tests")
    
    try:
        from architectures import GraphEmbed
        
        # Parameters matching  training
        n_qubits = 10
        L = 64
        K = 2
        
        # Create graph connectivity (chain)
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32) * 0.1
        
        # Create GraphEmbed
        graph_embed = GraphEmbed(
            n_qubits=n_qubits,
            L=L,
            edge_index=edge_index,
            edge_attr=edge_attr,
            K=K,
            use_global_node=False
        )
        print_test_result("GraphEmbed creation", True, f"n_qubits={n_qubits}, L={L}, K={K}")
        
        # Test forward pass
        z_input = torch.randn(L)
        z_graph = graph_embed(z_input)
        
        shape_ok = z_graph.shape == torch.Size([L])
        print_test_result("GraphEmbed forward", shape_ok, f"Input{z_input.shape} -> Output{z_graph.shape}")
        
        # Test batched input
        z_batch = torch.randn(5, L)
        z_graph_batch = graph_embed(z_batch)
        batch_ok = z_graph_batch.shape == torch.Size([5, L])
        print_test_result("GraphEmbed batched", batch_ok, f"Batch input{z_batch.shape} -> {z_graph_batch.shape}")
        
        return shape_ok and batch_ok
        
    except Exception as e:
        print_test_result("GraphEmbed tests", False, f"Error: {e}")
        traceback.print_exc()
        return False

def test_temporal_aggregator():
    """Test TemporalContextualAggregator"""
    print_test_header("Temporal Aggregator Tests")
    
    try:
        from architectures import TemporalContextualAggregator
        
        L, T = 64, 10
        temp_agg = TemporalContextualAggregator(L=L, T=T)
        print_test_result("TemporalAggregator creation", True, f"L={L}, T={T}")
        
        # Test with temporal buffer
        buffer = torch.randn(T, L)
        c_t = temp_agg(buffer)
        
        shape_ok = c_t.shape == torch.Size([L])
        print_test_result("Temporal aggregation", shape_ok, f"Buffer{buffer.shape} -> Context{c_t.shape}")
        
        # Test batched
        buffer_batch = torch.randn(3, T, L) 
        c_t_batch = temp_agg(buffer_batch)
        batch_ok = c_t_batch.shape == torch.Size([3, L])
        print_test_result("Temporal batched", batch_ok, f"Batch{buffer_batch.shape} -> {c_t_batch.shape}")
        
        return shape_ok and batch_ok
        
    except Exception as e:
        print_test_result("Temporal aggregator tests", False, f"Error: {e}")
        traceback.print_exc()
        return False

def test_policy_value_head():
    """Test PolicyValueHead"""
    print_test_header("Policy Value Head Tests")
    
    try:
        from architectures import PolicyValueHead
        
        L = 64
        A = 150  # n_qubits * 3 * M_evo = 10 * 3 * 5
        
        policy_head = PolicyValueHead(L=L, A=A)
        print_test_result("PolicyValueHead creation", True, f"L={L}, A={A}")
        
        # Test forward pass
        c_t = torch.randn(L)
        dist, value = policy_head(c_t)
        
        # Check distribution and value
        dist_ok = hasattr(dist, 'sample') and hasattr(dist, 'log_prob')
        value_ok = value.shape == torch.Size([])
        print_test_result("Policy forward pass", dist_ok and value_ok, 
                         f"Dist type: {type(dist).__name__}, Value shape: {value.shape}")
        
        # Test action sampling
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        action_ok = action.dtype == torch.int64 and 0 <= action.item() < A
        print_test_result("Action sampling", action_ok, f"Action: {action.item()}, LogProb: {log_prob.item():.4f}")
        
        return dist_ok and value_ok and action_ok
        
    except Exception as e:
        print_test_result("Policy value head tests", False, f"Error: {e}")
        traceback.print_exc()
        return False

def test_complete_symqnet():
    """Test complete FixedSymQNetWithEstimator"""
    print_test_header("Complete SymQNet Tests")
    
    try:
        from architectures import VariationalAutoencoder, FixedSymQNetWithEstimator
        
        # Create VAE first
        vae = VariationalAutoencoder(M=10, L=64)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        
        # SymQNet parameters matching  training
        n_qubits = 10
        L = 64  
        T = 10
        M_evo = 5
        A = n_qubits * 3 * M_evo
        
        # Graph connectivity
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32) * 0.1
        
        # Create complete SymQNet
        symqnet = FixedSymQNetWithEstimator(
            vae=vae,
            n_qubits=n_qubits,
            L=L,
            edge_index=edge_index,
            edge_attr=edge_attr,
            T=T,
            A=A,
            M_evo=M_evo,
            K_gnn=2
        )
        print_test_result("SymQNet creation", True, f"n_qubits={n_qubits}, A={A}")
        
        # Test forward pass with SMC belief metadata
        obs = torch.tanh(torch.randn(10))
        layout = MetadataLayout.from_problem(n_qubits, M_evo)
        smc = init_smc_belief(n_qubits, M_evo, obs.device)
        action_info = {"qubit_idx": 0, "basis_idx": 2, "time_idx": 0}
        metadata = build_metadata_with_belief(layout, smc, obs, action_info, shots=256)

        dist, value = symqnet(obs, metadata)
        
        # Check outputs
        dist_ok = hasattr(dist, 'sample')
        value_ok = value.shape == torch.Size([])
        
        print_test_result("SymQNet forward pass", dist_ok and value_ok,
                         f"Value: {value.shape}")
        
        # Test multiple steps (ring buffer)
        symqnet.reset_buffer()
        for step in range(5):
            obs_step = torch.tanh(torch.randn(10))
            action_info = {
                "qubit_idx": step % n_qubits,
                "basis_idx": step % 3,
                "time_idx": step % M_evo,
            }
            metadata_step = build_metadata_with_belief(
                layout, smc, obs_step, action_info, shots=256
            )
            dist_step, value_step = symqnet(obs_step, metadata_step)
        
        buffer_ok = len(symqnet.zG_history) == 5
        print_test_result("Ring buffer functionality", buffer_ok, f"Buffer length: {len(symqnet.zG_history)}")
        
        return dist_ok and value_ok and buffer_ok
        
    except Exception as e:
        print_test_result("Complete SymQNet tests", False, f"Error: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test loading  actual trained models"""
    print_test_header("Trained Model Loading Tests")
    
    device = torch.device('cpu')
    
    # Test VAE loading
    vae_ok = False
    try:
        from architectures import VariationalAutoencoder
        
        if not Path('models/vae_M10_f.pth').exists():
            print_test_result("VAE file check", False, "models/vae_M10_f.pth not found")
            return False
        
        vae = VariationalAutoencoder(M=10, L=64).to(device)
        vae_state = torch.load('models/vae_M10_f.pth', map_location=device)
        vae.load_state_dict(vae_state)
        vae.eval()
        
        # Test loaded VAE
        test_input = torch.randn(10)
        with torch.no_grad():
            mu, logvar = vae.encode(test_input)
            z = vae.reparameterize(mu, logvar)
        
        vae_ok = z.shape == torch.Size([64])
        print_test_result("VAE model loading", vae_ok, f"Loaded and tested, z.shape: {z.shape}")
        
    except Exception as e:
        print_test_result("VAE model loading", False, f"Error: {e}")
    
    # Test SymQNet loading
    symqnet_ok = False
    try:
        from architectures import FixedSymQNetWithEstimator
        
        if not Path('models/FINAL_FIXED_SYMQNET.pth').exists():
            print_test_result("SymQNet file check", False, "models/FINAL_FIXED_SYMQNET.pth not found")
            return vae_ok
        
        checkpoint = torch.load('models/FINAL_FIXED_SYMQNET.pth', map_location=device)
        print_test_result("SymQNet checkpoint load", True, f"Keys: {list(checkpoint.keys())}")
        
        # Model parameters from  exact training
        n_qubits = 10
        L = 64
        T = 10
        M_evo = 5
        A = n_qubits * 3 * M_evo
        
        # Graph connectivity
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=device) * 0.1
        
        symqnet = FixedSymQNetWithEstimator(
            vae=vae,
            n_qubits=n_qubits,
            L=L,
            edge_index=edge_index,
            edge_attr=edge_attr,
            T=T,
            A=A,
            M_evo=M_evo,
            K_gnn=2
        ).to(device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            symqnet.load_state_dict(checkpoint['model_state_dict'])
        else:
            symqnet.load_state_dict(checkpoint)
        
        symqnet.eval()
        print_test_result("SymQNet weight loading", True, "Weights loaded successfully")
        
        # Test full forward pass with SMC belief metadata
        obs = torch.tanh(torch.randn(10, device=device))
        layout = MetadataLayout.from_problem(n_qubits, M_evo)
        smc = init_smc_belief(n_qubits, M_evo, device)
        action_info = {"qubit_idx": 0, "basis_idx": 2, "time_idx": 0}
        metadata = build_metadata_with_belief(layout, smc, obs, action_info, shots=256)

        with torch.no_grad():
            dist, value = symqnet(obs, metadata)
        
        symqnet_ok = (
            hasattr(dist, 'sample') and
            value.shape == torch.Size([])
        )
        print_test_result("SymQNet full test", symqnet_ok, 
                         f"Output shapes: value{value.shape}")
        
    except Exception as e:
        print_test_result("SymQNet model loading", False, f"Error: {e}")
        traceback.print_exc()
    
    return vae_ok and symqnet_ok

def test_environment():
    """Test SpinChainEnv"""
    print_test_header("Environment Tests")
    
    try:
        from architectures import SpinChainEnv
        
        # Create environment
        env = SpinChainEnv(N=10, M_evo=5, T=8, device=torch.device('cpu'))
        print_test_result("Environment creation", True, "SpinChainEnv created")
        
        # Test reset
        obs = env.reset()
        reset_ok = isinstance(obs, np.ndarray) and obs.shape == (10,)
        print_test_result("Environment reset", reset_ok, f"Obs shape: {obs.shape}")
        
        # Test step
        action = 0
        obs2, reward, done, info = env.step(action)
        step_ok = (
            isinstance(obs2, np.ndarray) and
            isinstance(reward, (int, float)) and
            isinstance(done, bool) and
            isinstance(info, dict)
        )
        print_test_result("Environment step", step_ok, f"Action: {action}, Reward: {reward}")
        
        # Check info structure
        info_ok = all(key in info for key in ['J_true', 'h_true', 'qubit_idx', 'basis_idx', 'time_idx'])
        print_test_result("Environment info", info_ok, f"Info keys: {list(info.keys())}")
        
        return reset_ok and step_ok and info_ok
        
    except Exception as e:
        print_test_result("Environment tests", False, f"Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all model tests"""

    print(f"Testing from: {Path.cwd()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run all tests
    tests = [
        ("Architecture Imports", test_architecture_imports),
        ("VAE Creation", test_vae_creation),
        ("GraphEmbed", test_graph_embed),
        ("Temporal Aggregator", test_temporal_aggregator),
        ("Policy Value Head", test_policy_value_head),
        ("Complete SymQNet", test_complete_symqnet),
        ("Trained Model Loading", test_model_loading),
        ("Environment", test_environment)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_test_result(test_name, False, f"Test crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print_test_header("summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"  
        print(f"{test_name:<25} | {status}")
    
    print(f"\n OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:

        print("The architectures and trained models are working correctly.")


    else:
        print("  Some tests failed. Please check the errors above.")
        
        if not results.get("Architecture Imports", True):
            print("\n Fix architecture imports:")

        
        if not results.get("Trained Model Loading", True):
            print("\nFix model loading:")

    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
