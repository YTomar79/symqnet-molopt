"""
Exact architectures used for training SymQNet and VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.distributions import Categorical, Normal
from collections import deque
import gymnasium as gym
from gym import spaces
import random
from dataclasses import dataclass

def get_pauli_matrices():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    return X, Y, Z, I

def kron_n(operator, n_qubits, target_qubit):
    X, Y, Z, I = get_pauli_matrices()
    ops = [I] * n_qubits
    ops[target_qubit] = operator
    full_op = ops[0]
    for op in ops[1:]:
        full_op = np.kron(full_op, op)
    return full_op

def depolarizing_channel_kraus(p):
    X, Y, Z, I = get_pauli_matrices()
    k0 = np.sqrt(1 - p) * I
    k1 = np.sqrt(p / 3) * X
    k2 = np.sqrt(p / 3) * Y
    k3 = np.sqrt(p / 3) * Z
    return [k0, k1, k2, k3]

def amplitude_damping_kraus(gamma):
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [E0, E1]

def dephasing_kraus(lmbda):
    X, Y, Z, I = get_pauli_matrices()
    E0 = np.sqrt(1 - lmbda) * I
    E1 = np.sqrt(lmbda) * Z
    return [E0, E1]

def apply_single_qubit_channel(rho, n_qubits, qubit_idx, kraus_list):
    dim = 2 ** n_qubits
    new_rho = np.zeros((dim, dim), dtype=complex)
    for K in kraus_list:
        full_K = kron_n(K, n_qubits, qubit_idx)
        new_rho += full_K @ rho @ full_K.conj().T
    return new_rho

def apply_noise(rho, n_qubits, p_dep, gamma, lmbda):
    noisy_rho = rho.copy()
    for q in range(n_qubits):
        kraus_dep = depolarizing_channel_kraus(p_dep)
        noisy_rho = apply_single_qubit_channel(noisy_rho, n_qubits, q, kraus_dep)
        kraus_ad = amplitude_damping_kraus(gamma)
        noisy_rho = apply_single_qubit_channel(noisy_rho, n_qubits, q, kraus_ad)
        kraus_dp = dephasing_kraus(lmbda)
        noisy_rho = apply_single_qubit_channel(noisy_rho, n_qubits, q, kraus_dp)
    return noisy_rho

def compute_expectations(rho, n_qubits):
    X, Y, Z, I = get_pauli_matrices()
    expectations = []
    for q in range(n_qubits):
        X_q = kron_n(X, n_qubits, q)
        Y_q = kron_n(Y, n_qubits, q)
        Z_q = kron_n(Z, n_qubits, q)
        exp_X = np.real(np.trace(rho @ X_q))
        exp_Y = np.real(np.trace(rho @ Y_q))
        exp_Z = np.real(np.trace(rho @ Z_q))
        expectations.extend([exp_X, exp_Y, exp_Z])
    return np.array(expectations)

def shot_noise_sampling(expectations, shots=512):
    noisy_meas = np.zeros_like(expectations)
    for idx, exp_val in enumerate(expectations):
        p_plus = (1 + exp_val) / 2
        samples = np.where(np.random.rand(shots) < p_plus, 1, -1)
        noisy_meas[idx] = np.mean(samples)
    return noisy_meas

def generate_random_pure_state(n_qubits):
    dim = 2 ** n_qubits
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state /= np.linalg.norm(state)
    rho = np.outer(state, state.conj())
    return rho

def generate_measurement_pair(n_qubits, p_dep=0.02, gamma=0.01, lmbda=0.01, shots=512):
    rho_ideal = generate_random_pure_state(n_qubits)
    m_ideal = compute_expectations(rho_ideal, n_qubits)
    rho_noisy = apply_noise(rho_ideal, n_qubits, p_dep, gamma, lmbda)
    exp_noisy = compute_expectations(rho_noisy, n_qubits)
    m_noisy = shot_noise_sampling(exp_noisy, shots)
    return m_noisy, m_ideal

 
# DATASET FOR VAE PRETRAINING - EXACT  FROM THE CODE
 

class MeasurementDataset(Dataset):
    def __init__(self, n_qubits, num_samples):
        self.n_qubits = n_qubits
        self.num_samples = num_samples
        self.data = []
        for _ in range(num_samples):
            m_noisy, m_ideal = generate_measurement_pair(n_qubits)
            self.data.append((m_noisy.astype(np.float32), m_ideal.astype(np.float32)))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]



class VariationalAutoencoder(nn.Module):
    def __init__(self, M: int, L: int, hidden: int = 128):
        """
        Args:
          M (int): input / output dimension (e.g. # of measurements per step)
          L (int): latent dimension
          hidden (int): width of hidden layers
        """
        super().__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(M, hidden)
        self.enc_fc2 = nn.Linear(hidden, hidden)
        self.enc_mu  = nn.Linear(hidden, L)
        self.enc_logsigma = nn.Linear(hidden, L)

        # Decoder
        self.dec_fc1 = nn.Linear(L, hidden)
        self.dec_fc2 = nn.Linear(hidden, hidden)
        self.dec_out = nn.Linear(hidden, M)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        mu       = self.enc_mu(h)
        logvar   = self.enc_logsigma(h)
        logvar   = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

 
# GRAPH EMBED - EXACT  FROM THE CODE
 

class GraphEmbed(nn.Module):
    def __init__(self,
                 n_qubits: int,
                 L: int,
                 edge_index: torch.LongTensor,
                 edge_attr: torch.FloatTensor,
                 K: int = 2,
                 use_global_node: bool = False):
        """
        Vectorized Graph-Structured Embedding (Block 2) for SymQNet.
        """
        super().__init__()
        self.n_qubits       = n_qubits
        self.L              = L
        self.K              = K
        self.use_global_node= use_global_node
        self.total_nodes    = n_qubits + 1 if use_global_node else n_qubits

        # register buffers (will move with .to(device))
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr",  edge_attr)

        # Edge‐MLPs φₑ^(k): R^{2L+1}→R^L
        self.phi_e_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*L + 1, L),
                nn.ReLU(),
                nn.Linear(L, L),
            )
            for _ in range(K)
        ])

        # Node‐MLPs φₙ^(k): R^{2L}→R^L
        self.phi_n_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*L, L),
                nn.ReLU(),
                nn.LayerNorm(L),
                nn.Dropout(p=0.1),
                nn.Linear(L, L),
            )
            for _ in range(K)
        ])

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        """
        z0: [L] or [B, L]  (latent from Block 1)
        returns z_G: [L] or [B, L]
        """
        # ---- batch handling ----
        if z0.dim() == 1:
            z = z0.unsqueeze(0)
            squeeze = True
        else:
            z = z0
            squeeze = False
        B = z.size(0)
        device = z.device

        # ---- initialize h^(0) ----
        if self.use_global_node:
            h = torch.zeros(B, self.total_nodes, self.L, device=device)
            h[:, 1:(self.n_qubits+1), :] = z.unsqueeze(1).expand(-1, self.n_qubits, -1)
        else:
            h = z.unsqueeze(1).expand(-1, self.n_qubits, -1)

        # unpack edges
        src, tgt = self.edge_index
        E = src.size(0)
        e = self.edge_attr.view(1, E, 1).expand(B, E, 1)

        # ---- K layers of message passing ----
        for k in range(self.K):
            # 1) compute messages on all edges at once
            hi = h[:, src, :]
            hj = h[:, tgt, :]
            inp_e = torch.cat([hi, hj, e], dim=-1)
            m = self.phi_e_layers[k](inp_e)

            # 2) aggregate per‐node via scatter_add
            m_agg = torch.zeros_like(h)
            idx = src.view(1, E, 1).expand(B, E, self.L)
            m_agg.scatter_add_(dim=1, index=idx, src=m)

            # 3) node update (vectorized over all nodes)
            inp_n = torch.cat([h, m_agg], dim=-1)
            h = self.phi_n_layers[k](inp_n) + h

        # ---- global readout: mean over real qubit nodes ----
        if self.use_global_node:
            real = h[:, 1:(self.n_qubits+1), :]
        else:
            real = h

        z_G = real.mean(dim=1)
        return z_G.squeeze(0) if squeeze else z_G

 
# TEMPORAL CONTEXTUAL AGGREGATOR - EXACT  FROM THE CODE
 

class TemporalContextualAggregator(nn.Module):
    def __init__(self, L: int, T: int = 4, num_heads: int = 2, dropout: float = 0.1, n_layers: int = 2):
        """
        Block 3: Transformer-based Temporal & Contextual Feature Aggregator.

        - Uses a causal TransformerEncoder so the agent can attend to informative past measurements.
        - Keeps the "embed_dim divisible by num_heads" safety fix via an attn_dim projection.
        """
        super().__init__()
        self.L = int(L)
        self.T = int(T)
        self.num_heads = int(num_heads)

        # ---- Attention dimension fix ----
        self.attn_dim = ((self.L + self.num_heads - 1) // self.num_heads) * self.num_heads

        if self.attn_dim != self.L:
            self.pre = nn.Linear(self.L, self.attn_dim, bias=False)
            self.post = nn.Linear(self.attn_dim, self.L, bias=False)
        else:
            self.pre = nn.Identity()
            self.post = nn.Identity()

        # Learnable positional embedding (fixed horizon T)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.T, self.attn_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.attn_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.attn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        self.out = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.LayerNorm(self.L),
            nn.Dropout(dropout),
        )

        # Init linears
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, zG_buffer: torch.Tensor) -> torch.Tensor:
        """
        zG_buffer: [T, L] or [B, T, L]
        returns:   [L] or [B, L]
        """
        if zG_buffer.dim() == 2:
            x = zG_buffer.unsqueeze(0)  # [1, T, L]
            squeeze = True
        elif zG_buffer.dim() == 3:
            x = zG_buffer               # [B, T, L]
            squeeze = False
        else:
            raise ValueError("zG_buffer must be [T,L] or [B,T,L]")

        B, T_in, L_in = x.shape
        if L_in != self.L:
            raise ValueError(f"Expected last dim L={self.L}, got {L_in}")

        # Ensure length exactly self.T (pad left with zeros or take last T)
        if T_in < self.T:
            pad = torch.zeros(B, self.T - T_in, self.L, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
        elif T_in > self.T:
            x = x[:, -self.T:, :]

        # Project to attn_dim + add pos emb
        h = self.pre(x)  # [B, T, attn_dim]
        h = h + self.pos_emb[:, : h.size(1), :]

        # Causal mask (prevent attending to future)
        T_use = h.size(1)
        mask = torch.triu(torch.full((T_use, T_use), float("-inf"), device=h.device), diagonal=1)

        h = self.encoder(h, mask=mask)   # [B, T, attn_dim]
        o_t = h[:, -1, :]                # [B, attn_dim]
        o_t = self.post(o_t)             # [B, L]
        c_t = self.out(o_t)              # [B, L]

        return c_t.squeeze(0) if squeeze else c_t

 
# POLICY VALUE HEAD - EXACT  FROM THE CODE
 

class PolicyValueHead(nn.Module):
    def __init__(self, L: int, A: int = None, D: int = None):
        """
        Block 4: Policy & Value head.
        """
        super().__init__()
        assert (A is None) ^ (D is None), "Specify exactly one of A or D"
        self.L, self.A, self.D = L, A, D
        self.K = 2 * L

        # shared trunk
        self.shared_fc   = nn.Linear(L, self.K)
        self.shared_ln   = nn.LayerNorm(self.K)
        self.shared_drop = nn.Dropout(p=0.1)

        # policy head
        if A is not None:
            self.policy_fc = nn.Linear(self.K, A)
        else:
            self.mu_head      = nn.Linear(self.K, D)
            self.logsigma_head= nn.Linear(self.K, D)
            nn.init.constant_(self.logsigma_head.bias, -1.0)

        # value head
        self.value_fc = nn.Linear(self.K, 1)

        # Xavier init for all linears
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def shared_steps(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.shared_fc(x))
        x = self.shared_ln(x)
        return self.shared_drop(x)

    def forward(self, c_t: torch.Tensor):
        """
        c_t: [L] or [B, L]
        Returns:
          dist: Categorical (if A) or Normal (if D)
          V: scalar tensor or [B] tensor
        """
        # ensure batch dim
        single = (c_t.dim() == 1)
        x = c_t.unsqueeze(0) if single else c_t

        h = self.shared_steps(x)
        V = self.value_fc(h).squeeze(-1)

        # build distribution
        if self.A is not None:
            logits = self.policy_fc(h)
            dist   = Categorical(logits=logits)
        else:
            mu        = self.mu_head(h)
            log_sigma = self.logsigma_head(h).clamp(-20.0, 2.0)
            sigma     = log_sigma.exp()
            dist      = Normal(mu, sigma)

        # un-batch if needed
        if single:
            return dist, V.squeeze(0)
        return dist, V

    def get_action(self, c_t: torch.Tensor):
        """
        Samples an action and returns (action, logp, value).
        """
        dist, V = self(c_t)
        a  = dist.sample()
        logp = dist.log_prob(a)
        if self.D is not None:
            logp = logp.sum(-1)
        return a, logp, V

    def evaluate_actions(self, c_t: torch.Tensor, actions: torch.Tensor):
        """
        Computes log-prob and entropy of given actions under current policy.
        """
        dist, V = self(c_t)
        logp    = dist.log_prob(actions)
        if self.D is not None:
            logp = logp.sum(-1)
        ent = dist.entropy()
        if self.D is not None:
            ent = ent.sum(-1)
        return logp, ent, V 


@dataclass(frozen=True)
class MetadataLayout:
    n_qubits: int
    M_evo: int
    theta_dim: int
    base: int
    theta_slot0: int
    cov_slot0: int
    cov_feat_dim: int
    fisher_slot0: int
    fisher_feat_dim: int
    shots_slot: int
    meta_dim: int

    @classmethod
    def from_problem(cls, n_qubits: int, M_evo: int) -> "MetadataLayout":
        base = n_qubits + 3 + M_evo
        theta_dim = 2 * n_qubits - 1
        shots_slot = base
        theta_slot0 = shots_slot + 1
        cov_slot0 = theta_slot0 + theta_dim
        cov_feat_dim = theta_dim + 8
        fisher_slot0 = cov_slot0 + cov_feat_dim
        fisher_feat_dim = theta_dim
        meta_dim = fisher_slot0 + fisher_feat_dim
        return cls(
            n_qubits=n_qubits,
            M_evo=M_evo,
            theta_dim=theta_dim,
            base=base,
            theta_slot0=theta_slot0,
            cov_slot0=cov_slot0,
            cov_feat_dim=cov_feat_dim,
            fisher_slot0=fisher_slot0,
            fisher_feat_dim=fisher_feat_dim,
            shots_slot=shots_slot,
            meta_dim=meta_dim,
        )

    def empty(self, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(self.meta_dim, device=device, dtype=dtype)

class FixedSymQNetWithEstimator(nn.Module):
    """SymQNet that should be able to properly integrate all 4 blocks with metadata"""

    def __init__(self, vae, n_qubits, L, edge_index, edge_attr, T, A, M_evo, K_gnn=2, meta_dim=None):
        super().__init__()
        self.vae = vae
        self.n_qubits = n_qubits
        self.L = L
        self.T = T
        self.A = A
        self.M_evo = M_evo

        # Metadata dimensions
        self.metadata_layout = MetadataLayout.from_problem(n_qubits, M_evo)
        if meta_dim is None:
            meta_dim = self.metadata_layout.meta_dim
        self.meta_dim = meta_dim
        self.theta_dim = self.metadata_layout.theta_dim
        self.theta_slot0 = self.metadata_layout.theta_slot0
        self.cov_slot0 = self.metadata_layout.cov_slot0
        self.cov_feat_dim = self.metadata_layout.cov_feat_dim
        self.fisher_slot0 = self.metadata_layout.fisher_slot0
        self.fisher_feat_dim = self.metadata_layout.fisher_feat_dim
        self.shots_slot = self.metadata_layout.shots_slot

        # Block 1: Graph embedding (operates on latent + metadata)
        self.graph_embed = GraphEmbed(
            n_qubits=n_qubits,
            L=L + self.meta_dim,
            edge_index=edge_index,
            edge_attr=edge_attr,
            K=K_gnn,
            use_global_node=False
        )

        # Block 2: Temporal aggregation
        self.temp_agg = TemporalContextualAggregator(L + self.meta_dim, T)

        # Block 3: Policy-Value head
        self.policy_value = PolicyValueHead(L + self.meta_dim, A)

        # Ring buffer for temporal context
        self.zG_history = []
        self.current_metadata = None

    def reset_buffer(self):
        """Reset temporal buffer"""
        self.zG_history = []
        self.current_metadata = None

    def forward(self, obs, metadata, deterministic_inference: bool = False):
        """Forward pass that properly uses metadata throughout"""
        if metadata.dim() == 1:
            expected_dim = self.meta_dim
            if metadata.shape[0] != expected_dim:
                raise ValueError(
                    f"Metadata dimension mismatch: expected {expected_dim}, got {metadata.shape[0]}"
                )
        elif metadata.dim() == 2:
            expected_dim = self.meta_dim
            if metadata.shape[1] != expected_dim:
                raise ValueError(
                    f"Metadata dimension mismatch: expected {expected_dim}, got {metadata.shape[1]}"
                )
        else:
            raise ValueError(
                f"Metadata must be 1D or 2D, got shape {tuple(metadata.shape)}"
            )
        # Store current metadata
        self.current_metadata = metadata

        # Block 1: VAE encoding with metadata
        with torch.no_grad():
            mu_z, logvar_z = self.vae.encode(obs)
            z = self.vae.reparameterize(mu_z, logvar_z)
        z_with_meta = torch.cat([z, metadata], dim=-1)

        # Block 2: Graph embedding
        zG = self.graph_embed(z_with_meta)

        # Block 3: Update ring buffer & temporal aggregation
        self.zG_history.append(zG)
        if len(self.zG_history) > self.T:
            self.zG_history.pop(0)

        # Pad buffer if needed
        buf = self.zG_history[:]
        while len(buf) < self.T:
            buf.insert(0, torch.zeros_like(zG))

        buf_tensor = torch.stack(buf, dim=0)  # [T, L + meta_dim]
        c_t = self.temp_agg(buf_tensor)

        # Block 4a: Policy and Value
        dist, V = self.policy_value(c_t)

        return dist, V

 
# SPINCHAIN ENVIRONMENT - EXACT  FROM THE CODE
 

class SpinChainEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self,
                 N=10,
                 M_evo=5,
                 T=8,
                 noise_prob=0.02,
                 seed=None,
                 device=torch.device("cpu"),
                 resample_each_reset=True,
                 resample_every=1,
                 J_range=(0.5, 1.5),
                 h_range=(0.5, 1.5)):
        super().__init__()
        self.N         = N
        self.M_evo     = M_evo
        self.T         = T
        self.noise_prob= noise_prob
        self.step_count= 0
        self.device    = device

        self.resample_each_reset = resample_each_reset
        self.resample_every = int(resample_every)
        self.J_range = J_range
        self.h_range = h_range
        self._episode_counter = 0

        # Discrete evolution times
        self.times = np.linspace(0.1, 1.0, M_evo)

        # Pauli & identity on single qubit
        self.Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128, device=device)
        self.X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128, device=device)
        self.H = (1/np.sqrt(2)) * torch.tensor([[1,1],[1,-1]], dtype=torch.complex128, device=device)
        self.Sdg = torch.tensor([[1,0],[0,-1j]], dtype=torch.complex128, device=device)
        self.I = torch.eye(2, dtype=torch.complex128, device=device)

        # Precompute single-qubit readout rotations
        self.UX_list = []
        self.UY_list = []
        for q in range(N):
            UX = torch.eye(1, dtype=torch.complex128, device=device)
            UY = torch.eye(1, dtype=torch.complex128, device=device)
            for i in range(N):
                if i == q:
                    UX = torch.kron(UX, self.H)
                    UY = torch.kron(UY, self.Sdg @ self.H)
                else:
                    UX = torch.kron(UX, self.I)
                    UY = torch.kron(UY, self.I)
            self.UX_list.append(UX)
            self.UY_list.append(UY)

        # Initial state |0…0>
        dim = 2**N
        psi0 = torch.zeros((dim,1), dtype=torch.complex128, device=device)
        psi0[0,0] = 1.0 + 0j
        self.psi0 = psi0

        # Gym spaces
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(N,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N * 3 * M_evo)

        # Seed RNGs
        if seed is not None:
            self.seed(seed)

        self._resample_task()
        self.prev_mse = None

    def seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        return [seed]

    def _build_hamiltonian(self, J, h):
        N = self.N
        H = torch.zeros((2**N, 2**N), dtype=torch.complex128, device=self.device)
        # ZZ couplings
        for i in range(N-1):
            ops = [self.I]*N
            ops[i]   = self.Z
            ops[i+1] = self.Z
            term = ops[0]
            for o in ops[1:]:
                term = torch.kron(term, o)
            H = H + J[i] * term
        # X fields
        for i in range(N):
            ops = [self.I]*N
            ops[i] = self.X
            term = ops[0]
            for o in ops[1:]:
                term = torch.kron(term, o)
            H = H + h[i] * term
        return H

    def _resample_task(self):
        self.J_true = np.random.uniform(*self.J_range, size=(self.N - 1,))
        self.h_true = np.random.uniform(*self.h_range, size=(self.N,))
        self.H_true = self._build_hamiltonian(self.J_true, self.h_true).to(self.device)

        self.U_list = [
            torch.matrix_exp(-1j * self.H_true * tau).to(self.device)
            for tau in self.times
        ]

    def reset(self):
        self.step_count = 0
        self.prev_mse   = None

        self._episode_counter += 1
        if self.resample_each_reset and ((self._episode_counter - 1) % self.resample_every == 0):
            self._resample_task()

        return np.zeros(self.N, dtype=np.float32)

    def _measure(self, psi, basis, qubit_idx):
        # rotate into measurement basis
        if basis == 'Z':
            psi_rot = psi
        elif basis == 'X':
            psi_rot = self.UX_list[qubit_idx] @ psi
        elif basis == 'Y':
            psi_rot = self.UY_list[qubit_idx] @ psi
        else:
            raise ValueError("Invalid basis")

        probs = (psi_rot.abs()**2).flatten().real.cpu().numpy()
        idx   = np.random.choice(len(probs), p=probs)
        bits = np.array([(idx >> (self.N - 1 - i)) & 1 for i in range(self.N)], dtype=np.float32)
        bits = 2 * bits - 1.0
        if qubit_idx is not None:
            mask = np.zeros_like(bits)
            mask[qubit_idx] = bits[qubit_idx]
            bits = mask
        # add noise flips
        flips = np.random.rand(self.N) < self.noise_prob
        bits[flips] *= -1.0
        return bits

    def step(self, action):
        # decode scalar action
        a        = int(action)
        time_idx = a % self.M_evo
        a      //= self.M_evo
        basis_idx= a % 3
        qubit_idx= a // 3

        basis = ['X','Y','Z'][basis_idx]
        U     = self.U_list[time_idx]
        psi_t = U @ self.psi0

        obs = self._measure(psi_t, basis=basis, qubit_idx=qubit_idx)
        reward = 0.0
        self.step_count += 1
        done = (self.step_count >= self.T)

        info = {
            'J_true': self.J_true.copy(),
            'h_true': self.h_true.copy(),
            'qubit_idx': qubit_idx,
            'basis_idx': basis_idx,
            'time_idx': time_idx
        }
        return obs, reward, done, info

 
# KL DIVERGENCE UTILITY -  FROM  THE CODE
 

def kl_divergence(mu, log_sigma):
    """
    Compute KL divergence between N(mu, sigma^2) and N(0, I).
    """
    sigma2 = torch.exp(log_sigma)
    kl = 0.5 * torch.sum(sigma2 + mu**2 - 1 - log_sigma)
    return kl

 
# HELPER FUNCTIONS FOR TRAINING -  FROM THE CODE
 

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation (GAE)."""
    vals_1d = values.squeeze(-1)
    T = len(rewards)
    device = vals_1d.device

    vals_ext = torch.cat([vals_1d, torch.zeros(1, device=device)])
    advs = torch.zeros(T, device=device)
    last = 0.0

    for t in reversed(range(T)):
        nonterm = 1.0 if t < T - 1 else 0.0
        delta = rewards[t] + gamma * vals_ext[t + 1] * nonterm - vals_ext[t]
        last = delta + gamma * lam * nonterm * last
        advs[t] = last

    returns = advs + vals_1d
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    return returns, advs
