#!/usr/bin/env python3
"""
Unit tests for SymQNet architectures

Tests all neural network components with correct constraints:
- L=82 total dimension (64 latent + 18 metadata) for internal components
- L=64 base latent dimension for VAE and SymQNet initialization  
- 10-qubit maximum system size
- Proper tensor shapes throughout
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures import (
    VariationalAutoencoder,
    GraphEmbed,
    TemporalContextualAggregator,
    PolicyValueHead,
    FixedSymQNetWithEstimator,
    SpinChainEnv,
    get_pauli_matrices,
    kron_n,
    generate_measurement_pair,
    MeasurementDataset
)


class TestVariationalAutoencoder:
    """Test VAE architecture"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.M = 10  # Input dimension (measurement dimension)
        self.L = 64  # Latent dimension
        self.device = torch.device('cpu')
        
    def test_vae_creation(self):
        """Test VAE can be created with correct dimensions"""
        vae = VariationalAutoencoder(M=self.M, L=self.L)
        assert isinstance(vae, nn.Module)
        
        # Check architecture components exist
        assert hasattr(vae, 'enc_fc1')
        assert hasattr(vae, 'enc_mu')
        assert hasattr(vae, 'dec_out')
    
    def test_vae_forward_pass(self):
        """Test VAE forward pass with correct tensor shapes"""
        vae = VariationalAutoencoder(M=self.M, L=self.L)
        
        # Test input
        x = torch.randn(self.M)
        
        # Forward pass
        recon, mu, logvar, z = vae(x)
        
        # Check output shapes
        assert recon.shape == torch.Size([self.M])
        assert mu.shape == torch.Size([self.L])
        assert logvar.shape == torch.Size([self.L])
        assert z.shape == torch.Size([self.L])
    
    def test_vae_encoding(self):
        """Test VAE encoding specifically"""
        vae = VariationalAutoencoder(M=self.M, L=self.L)
        
        x = torch.randn(self.M)
        mu, logvar = vae.encode(x)
        
        assert mu.shape == torch.Size([self.L])
        assert logvar.shape == torch.Size([self.L])
        
        # Test reparameterization
        z = vae.reparameterize(mu, logvar)
        assert z.shape == torch.Size([self.L])
    
    def test_vae_batched_input(self):
        """Test VAE with batched input"""
        vae = VariationalAutoencoder(M=self.M, L=self.L)
        
        batch_size = 5
        x_batch = torch.randn(batch_size, self.M)
        
        recon, mu, logvar, z = vae(x_batch)
        
        assert recon.shape == torch.Size([batch_size, self.M])
        assert mu.shape == torch.Size([batch_size, self.L])
        assert z.shape == torch.Size([batch_size, self.L])


class TestGraphEmbed:
    """Test Graph Embedding layer with L=82 constraint (CORRECT!)"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.n_qubits = 10  # Maximum supported
        self.L_base = 64   # Base latent dimension
        self.meta_dim = 18 # Metadata dimension (n_qubits + 3 + M_evo)
        self.L_total = self.L_base + self.meta_dim  # 82 total - CORRECT!
        
        # Create chain graph connectivity
        edges = [(i, i+1) for i in range(self.n_qubits-1)] + [(i+1, i) for i in range(self.n_qubits-1)]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.ones(len(edges), 1, dtype=torch.float32) * 0.1
    
    def test_graph_embed_creation(self):
        """Test GraphEmbed creation with correct L=82"""
        graph_embed = GraphEmbed(
            n_qubits=self.n_qubits,
            L=self.L_total,  # L=82 is CORRECT for internal usage!
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            K=2
        )
        
        assert graph_embed.n_qubits == self.n_qubits
        assert graph_embed.L == self.L_total
        assert graph_embed.K == 2
    
    def test_graph_embed_forward(self):
        """Test GraphEmbed forward pass with L=82"""
        graph_embed = GraphEmbed(
            n_qubits=self.n_qubits,
            L=self.L_total,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            K=2
        )
        
        # Input must be L=82 dimensional
        z_input = torch.randn(self.L_total)
        z_output = graph_embed(z_input)
        
        assert z_output.shape == torch.Size([self.L_total])
    
    def test_graph_embed_batched(self):
        """Test GraphEmbed with batched input"""
        graph_embed = GraphEmbed(
            n_qubits=self.n_qubits,
            L=self.L_total,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            K=2
        )
        
        batch_size = 3
        z_batch = torch.randn(batch_size, self.L_total)
        z_output = graph_embed(z_batch)
        
        assert z_output.shape == torch.Size([batch_size, self.L_total])
    
    def test_10_qubit_system_only(self):
        """Test GraphEmbed works with 10-qubit system only (FIXED!)"""
        # 🔧 FIX: Only test 10-qubit system
        graph_embed = GraphEmbed(
            n_qubits=10,  # Only 10 qubits supported
            L=self.L_total,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            K=2
        )
        
        z_input = torch.randn(self.L_total)
        z_output = graph_embed(z_input)
        
        assert z_output.shape == torch.Size([self.L_total])


class TestTemporalContextualAggregator:
    """Test Temporal aggregation with L=82 (CORRECT!)"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.L = 82  # L=82 is CORRECT for internal usage!
        self.T = 10  # Temporal window
    
    def test_temporal_aggregator_creation(self):
        """Test TemporalContextualAggregator creation"""
        temp_agg = TemporalContextualAggregator(L=self.L, T=self.T)
        
        assert temp_agg.L == self.L
        assert temp_agg.T == self.T
    
    def test_temporal_aggregator_forward(self):
        """Test temporal aggregation forward pass"""
        temp_agg = TemporalContextualAggregator(L=self.L, T=self.T)
        
        # Create temporal buffer [T, L]
        buffer = torch.randn(self.T, self.L)
        context = temp_agg(buffer)
        
        assert context.shape == torch.Size([self.L])
    
    def test_temporal_aggregator_batched(self):
        """Test temporal aggregation with batched input"""
        temp_agg = TemporalContextualAggregator(L=self.L, T=self.T)
        
        batch_size = 2
        buffer_batch = torch.randn(batch_size, self.T, self.L)
        context_batch = temp_agg(buffer_batch)
        
        assert context_batch.shape == torch.Size([batch_size, self.L])
    
    def test_shorter_sequences(self):
        """Test with sequences shorter than T"""
        temp_agg = TemporalContextualAggregator(L=self.L, T=self.T)
        
        # Shorter sequence
        short_buffer = torch.randn(5, self.L)  # Only 5 time steps
        context = temp_agg(short_buffer)
        
        assert context.shape == torch.Size([self.L])


class TestPolicyValueHead:
    """Test Policy-Value head with L=82 (CORRECT!)"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.L = 82  # L=82 is CORRECT!
        self.A = 150  # Action space (10 qubits * 3 bases * 5 times)
    
    def test_policy_head_creation(self):
        """Test PolicyValueHead creation"""
        policy_head = PolicyValueHead(L=self.L, A=self.A)
        
        assert policy_head.L == self.L
        assert policy_head.A == self.A
    
    def test_policy_head_forward(self):
        """Test policy head forward pass"""
        policy_head = PolicyValueHead(L=self.L, A=self.A)
        
        context = torch.randn(self.L)
        dist, value = policy_head(context)
        
        # Check distribution
        assert hasattr(dist, 'sample')
        assert hasattr(dist, 'log_prob')
        
        # Check value
        assert value.shape == torch.Size([])
        
    def test_action_sampling(self):
        """Test action sampling from policy"""
        policy_head = PolicyValueHead(L=self.L, A=self.A)
        
        context = torch.randn(self.L)
        dist, value = policy_head(context)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        assert 0 <= action.item() < self.A
        assert isinstance(log_prob.item(), float)
    
    def test_policy_head_batched(self):
        """Test policy head with batched input"""
        policy_head = PolicyValueHead(L=self.L, A=self.A)
        
        batch_size = 3
        context_batch = torch.randn(batch_size, self.L)
        dist, value_batch = policy_head(context_batch)
        
        assert value_batch.shape == torch.Size([batch_size])


class TestFixedSymQNetWithEstimator:
    """Test complete SymQNet with correct constraints"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create mock VAE
        self.vae = VariationalAutoencoder(M=10, L=64)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        
        # Model parameters
        self.n_qubits = 10
        self.L = 64  # Base latent dimension (passed to SymQNet constructor)
        self.T = 10
        self.M_evo = 5
        self.A = self.n_qubits * 3 * self.M_evo
        
        # Graph connectivity
        edges = [(i, i+1) for i in range(self.n_qubits-1)] + [(i+1, i) for i in range(self.n_qubits-1)]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.ones(len(edges), 1, dtype=torch.float32) * 0.1
    
    def test_symqnet_creation(self):
        """Test SymQNet creation with correct parameters"""
        symqnet = FixedSymQNetWithEstimator(
            vae=self.vae,
            n_qubits=self.n_qubits,
            L=self.L,  # L=64 for constructor, internally becomes L+meta=82
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            T=self.T,
            A=self.A,
            M_evo=self.M_evo,
            K_gnn=2
        )
        
        assert symqnet.n_qubits == self.n_qubits
        assert symqnet.L == self.L
        assert symqnet.A == self.A
    
    def test_symqnet_forward_pass(self):
        """Test SymQNet forward pass with correct dimensions"""
        symqnet = FixedSymQNetWithEstimator(
            vae=self.vae,
            n_qubits=self.n_qubits,
            L=self.L,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            T=self.T,
            A=self.A,
            M_evo=self.M_evo,
            K_gnn=2
        )
        
        # Create inputs
        obs = torch.randn(10)  # Measurement observation
        metadata = torch.zeros(18)  # n_qubits + 3 + M_evo = 10 + 3 + 5
        
        # Forward pass
        dist, value, theta_hat = symqnet(obs, metadata)
        
        # Check outputs
        assert hasattr(dist, 'sample')
        assert value.shape == torch.Size([])
        assert theta_hat.shape == torch.Size([19])  # 2*n_qubits - 1 = 19
    
    def test_symqnet_buffer_management(self):
        """Test SymQNet ring buffer functionality"""
        symqnet = FixedSymQNetWithEstimator(
            vae=self.vae,
            n_qubits=self.n_qubits,
            L=self.L,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            T=self.T,
            A=self.A,
            M_evo=self.M_evo,
            K_gnn=2
        )
        
        # Reset buffer
        symqnet.reset_buffer()
        assert len(symqnet.zG_history) == 0
        
        # Run multiple steps
        for step in range(15):  # More than T=10
            obs = torch.randn(10)
            metadata = torch.zeros(18)
            
            dist, value, theta_hat = symqnet(obs, metadata)
            
            # Buffer should not exceed T
            assert len(symqnet.zG_history) <= self.T
        
        # Final buffer size should be T
        assert len(symqnet.zG_history) == self.T
    
    def test_parameter_estimation_shape(self):
        """Test parameter estimation outputs correct shape"""
        symqnet = FixedSymQNetWithEstimator(
            vae=self.vae,
            n_qubits=self.n_qubits,
            L=self.L,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            T=self.T,
            A=self.A,
            M_evo=self.M_evo,
            K_gnn=2
        )
        
        obs = torch.randn(10)
        metadata = torch.zeros(18)
        
        _, _, theta_hat = symqnet(obs, metadata)
        
        # Should estimate 19 parameters for 10-qubit system
        # 9 coupling parameters (J) + 10 field parameters (h)
        assert theta_hat.shape == torch.Size([19])


class TestSpinChainEnv:
    """Test SpinChain environment for 10-qubit systems ONLY"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.device = torch.device('cpu')
    
    def test_env_creation_10_qubits(self):
        """Test environment creation with 10 qubits"""
        env = SpinChainEnv(N=10, M_evo=5, T=8, device=self.device)
        
        assert env.N == 10
        assert env.M_evo == 5
        assert env.T == 8
    
    def test_env_reset(self):
        """Test environment reset"""
        env = SpinChainEnv(N=10, M_evo=5, T=8, device=self.device)
        
        obs = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert np.all(np.abs(obs) <= 1.0)  # Valid expectation values
    
    def test_env_step(self):
        """Test environment step"""
        env = SpinChainEnv(N=10, M_evo=5, T=8, device=self.device)
        
        env.reset()
        
        # Test valid action
        action = 42  # Some valid action index
        obs, reward, done, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check info structure
        required_keys = ['J_true', 'h_true', 'qubit_idx', 'basis_idx', 'time_idx']
        for key in required_keys:
            assert key in info
    
    def test_env_action_space(self):
        """Test environment action space matches expectations"""
        env = SpinChainEnv(N=10, M_evo=5, T=8, device=self.device)
        
        # Action space should be N * 3 * M_evo = 10 * 3 * 5 = 150
        expected_action_space = 10 * 3 * 5
        assert env.action_space.n == expected_action_space


class TestUtilityFunctions:
    """Test utility functions - FIXED to avoid 4-qubit violations"""
    
    def test_get_pauli_matrices(self):
        """Test Pauli matrix generation"""
        X, Y, Z, I = get_pauli_matrices()
        
        # Check shapes
        assert X.shape == (2, 2)
        assert Y.shape == (2, 2)
        assert Z.shape == (2, 2)
        assert I.shape == (2, 2)
        
        # Check types
        assert X.dtype == complex
        assert Y.dtype == complex
        assert Z.dtype == complex
        assert I.dtype == complex
    
    def test_kron_n(self):
        """Test kronecker product utility"""
        X, Y, Z, I = get_pauli_matrices()
        
        # 🔧 FIX: Test on 10-qubit system (not 4!)
        full_op = kron_n(X, 10, 0)  # X on qubit 0 of 10-qubit system
        
        expected_dim = 2**10
        assert full_op.shape == (expected_dim, expected_dim)
    
    def test_generate_measurement_pair(self):
        """Test measurement pair generation with 10 qubits"""
        # 🔧 FIX: Use 10 qubits to match constraint
        n_qubits = 10
        
        m_noisy, m_ideal = generate_measurement_pair(n_qubits)
        
        expected_length = n_qubits * 3  # 3 Pauli ops per qubit
        assert len(m_noisy) == expected_length
        assert len(m_ideal) == expected_length
        
        # Check value ranges
        assert np.all(np.abs(m_noisy) <= 1.0)
        assert np.all(np.abs(m_ideal) <= 1.0)


class TestMeasurementDataset:
    """Test measurement dataset for VAE training"""
    
    def test_dataset_creation(self):
        """Test dataset creation with 10 qubits"""
        # 🔧 FIX: Use 10 qubits
        n_qubits = 10
        num_samples = 100
        
        dataset = MeasurementDataset(n_qubits, num_samples)
        
        assert len(dataset) == num_samples
        
        # Test sample access
        m_noisy, m_ideal = dataset[0]
        
        expected_length = n_qubits * 3
        assert len(m_noisy) == expected_length
        assert len(m_ideal) == expected_length
    
    def test_dataset_iteration(self):
        """Test dataset iteration with 10 qubits"""
        # 🔧 FIX: Use 10 qubits
        n_qubits = 10
        num_samples = 10
        
        dataset = MeasurementDataset(n_qubits, num_samples)
        
        count = 0
        for m_noisy, m_ideal in dataset:
            count += 1
            assert isinstance(m_noisy, np.ndarray)
            assert isinstance(m_ideal, np.ndarray)
        
        assert count == num_samples


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
