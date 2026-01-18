import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Reuse noise and measurement generation from previous steps
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

# ----------------------------
# Dataset for pretraining VAE
# ----------------------------
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


def kl_divergence(mu, log_sigma):
    """
    Compute KL divergence between N(mu, sigma^2) and N(0, I).
    KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
    Inputs:
      mu, log_sigma: [B, L]
    Returns:
      KL: scalar sum over batch (not mean)
    """
    sigma2 = torch.exp(log_sigma)
    kl = 0.5 * torch.sum(sigma2 + mu**2 - 1 - log_sigma)
    return kl

# --------------------------
# Pretraining the VAE Model
# --------------------------
def pretrain_vae(n_qubits, L, num_samples=50000, batch_size=128,
                 lr=1e-3, num_epochs=100, sigma_gauss=0.01,
                 mask_prob=0.1, beta=1e-3):
    """
    Pretrain the Variational Autoencoder on simulated measurement pairs.
    """
    M = 3 * n_qubits  # input dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset and dataloaders
    dataset = MeasurementDataset(n_qubits, num_samples)
    val_split = int(0.9 * num_samples)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [val_split, num_samples - val_split])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, loss function
    model = VariationalAutoencoder(M, L).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss(reduction='sum')  # sum over batch for reconstruction

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for m_noisy_batch, m_ideal_batch in train_loader:
            m_noisy_batch = m_noisy_batch.to(device)
            m_ideal_batch = m_ideal_batch.to(device)

            # Add Gaussian noise
            noise = torch.randn_like(m_noisy_batch) * sigma_gauss
            m_noisy_noisy = m_noisy_batch + noise

            # Random masking
            mask = (torch.rand_like(m_noisy_noisy) > mask_prob).float()
            m_corrupted = m_noisy_noisy * mask

            # Forward pass through VAE
            # Forward pass through VAE (correct return order: recon, mu, logsigma, z)
            m_recon, mu_z, log_sigma_z, z0 = model(m_corrupted)

            # Compute losses
            loss_recon = mse_loss(m_recon, m_ideal_batch)
            loss_kl    = kl_divergence(mu_z, log_sigma_z)
            loss       = loss_recon + beta * loss_kl


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for m_noisy_batch, m_ideal_batch in val_loader:
                m_noisy_batch = m_noisy_batch.to(device)
                m_ideal_batch = m_ideal_batch.to(device)

                # No Gaussian noise or masking during validation
                m_recon, mu_z, log_sigma_z, z0 = model(m_noisy_batch)
                loss_recon = mse_loss(m_recon, m_ideal_batch)
                loss_kl    = kl_divergence(mu_z, log_sigma_z)
                loss       = loss_recon + beta * loss_kl


                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_vae.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load("best_vae.pth"))

    # Freeze encoder weights (mu and logsigma heads and shared encoder layers)
    for param in model.enc_fc1.parameters():
        param.requires_grad = False
    for param in model.enc_fc2.parameters():
        param.requires_grad = False
    for param in model.enc_mu.parameters():
        param.requires_grad = False
    for param in model.enc_logsigma.parameters():
        param.requires_grad = False

    return model.to(device)

# ----------------------------
# Example Usage of Pretraining
# ----------------------------'

'''
if __name__ == "__main__":
    n_qubits = 3
    latent_dim = 64
    pretrained_vae = pretrain_vae(
        n_qubits=10,
        L=latent_dim,
        num_samples=50000,
        batch_size=64,
        lr=1e-3,
        num_epochs=100,
        sigma_gauss=0.01,
        mask_prob=0.1,
        beta=1e-3
    )
    print("VAE pretraining complete. Encoder weights are frozen.")
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphEmbed(nn.Module):
    def __init__(self,
                 n_qubits: int,
                 L: int,
                 edge_index: torch.LongTensor,   # [2, E]
                 edge_attr: torch.FloatTensor,   # [E, 1]
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
        self.register_buffer("edge_index", edge_index)  # [2, E]
        self.register_buffer("edge_attr",  edge_attr)   # [E, 1]

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
            z = z0.unsqueeze(0)    # [1, L]
            squeeze = True
        else:
            z = z0                 # [B, L]
            squeeze = False
        B = z.size(0)
        device = z.device

        # ---- initialize h^(0) ----
        if self.use_global_node:
            # h: [B, total_nodes, L], real nodes 1..N get z
            h = torch.zeros(B, self.total_nodes, self.L, device=device)
            h[:, 1 : (self.n_qubits+1), :] = z.unsqueeze(1).expand(-1, self.n_qubits, -1)
        else:
            # h: [B, N, L], each of the N nodes identical = z
            h = z.unsqueeze(1).expand(-1, self.n_qubits, -1)

        # unpack edges
        src, tgt = self.edge_index     # each [E]
        E = src.size(0)
        # prep edge_attr for batch: [B, E, 1]
        e = self.edge_attr.view(1, E, 1).expand(B, E, 1)

        # ---- K layers of message passing ----
        for k in range(self.K):
            # 1) compute messages on all edges at once
            hi = h[:, src, :]                          # [B, E, L]
            hj = h[:, tgt, :]                          # [B, E, L]
            inp_e = torch.cat([hi, hj, e], dim=-1)     # [B, E, 2L+1]
            m     = self.phi_e_layers[k](inp_e)        # [B, E, L]

            # 2) aggregate per‐node via scatter_add
            m_agg = torch.zeros_like(h)                # [B, total_nodes, L]
            idx   = src.view(1, E, 1).expand(B, E, self.L)
            m_agg.scatter_add_(dim=1, index=idx, src=m)

            # 3) node update (vectorized over all nodes)
            inp_n = torch.cat([h, m_agg], dim=-1)      # [B, total_nodes, 2L]
            h     = self.phi_n_layers[k](inp_n) + h    # residual

        # ---- global readout: mean over real qubit nodes ----
        if self.use_global_node:
            real = h[:, 1 : (self.n_qubits+1), :]      # [B, N, L]
        else:
            real = h                                  # [B, N, L]

        z_G = real.mean(dim=1)                        # [B, L]
        return z_G.squeeze(0) if squeeze else z_G

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalContextualAggregator(nn.Module):
    def __init__(self, L: int, T: int = 4, num_heads: int = 2, dropout: float = 0.1):
        """
        Block 3: Temporal & Contextual Feature Aggregator for SymQNet.

        Args:
          L (int): latent dimension
          T (int): expected window size (for positional‐info or asserts only)
          num_heads (int): number of attention heads
        """
        super().__init__()
        self.L = L
        self.T = T

        # Causal TCN layers
        self.conv1 = nn.Conv1d(L, L, kernel_size=2, dilation=1, padding=0)
        self.ln1   = nn.LayerNorm(L)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(L, L, kernel_size=2, dilation=2, padding=0)
        self.ln2   = nn.LayerNorm(L)
        self.drop2 = nn.Dropout(dropout)

        # Multi‐head self‐attention
        # batch_first=True so input is [B, T, L]
        self.attn = nn.MultiheadAttention(embed_dim=L,
                                          num_heads=num_heads,
                                          batch_first=True,
                                          dropout=dropout)

        # Final projection
        self.out = nn.Sequential(
            nn.Linear(L, L),
            nn.LayerNorm(L),
            nn.Dropout(dropout),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, zG_buffer: torch.Tensor) -> torch.Tensor:
        """
        Args:
          zG_buffer: [T, L] or [B, T, L] tensor of past graph embeddings.

        Returns:
          c_t: [L] or [B, L] contextual embedding.
        """
        # -- ensure batch dimension --
        if zG_buffer.dim() == 2:
            x = zG_buffer.unsqueeze(0)     # [1, T, L]
            squeeze = True
        elif zG_buffer.dim() == 3:
            x = zG_buffer                # [B, T, L]
            squeeze = False
        else:
            raise ValueError("zG_buffer must be [T,L] or [B,T,L]")

        B, T, L = x.shape
        assert L == self.L, f"Expected L={self.L}, got {L}"

        # -- TCN Layer 1 (dilation=1, causal) --
        # conv1d expects [B, channels=L, seq_len=T]
        x1 = x.transpose(1, 2)                # [B, L, T]
        x1 = F.pad(x1, (1, 0))                # pad left=1 for kernel_size=2
        x1 = self.conv1(x1)                   # [B, L, T]
        x1 = F.relu(x1).transpose(1, 2)       # [B, T, L]
        x1 = self.ln1(x1)
        x1 = self.drop1(x1)

        # -- TCN Layer 2 (dilation=2, causal) --
        x2 = x1.transpose(1, 2)               # [B, L, T]
        x2 = F.pad(x2, (2, 0))                # pad left=2
        x2 = self.conv2(x2)                   # [B, L, T]
        x2 = F.relu(x2).transpose(1, 2)       # [B, T, L]
        x2 = self.ln2(x2)
        x2 = self.drop2(x2)

        # Residual skip
        U = x1 + x2                           # [B, T, L]

        # -- Multi-Head Self-Attention Over Time --
        # attn: query/key/value = U
        # returns attn_output [B, T, L]
        O, _ = self.attn(U, U, U)             # batch_first=True

        # take the last time step
        o_t = O[:, -1, :]                     # [B, L]

        # final projection
        c_t = self.out(o_t)                   # [B, L]

        return c_t.squeeze(0) if squeeze else c_t

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class PolicyValueHead(nn.Module):
    def __init__(self, L: int, A: int = None, D: int = None):
        """
        Block 4: Policy & Value head.

        Args:
          L (int): input dimension of c_t
          A (int, optional): number of discrete actions
          D (int, optional): dimension of continuous actions
        Exactly one of A or D must be specified.
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
            # init logsigma bias so sigma≈0.37 initially
            nn.init.constant_(self.logsigma_head.bias, -1.0)

        # value head
        self.value_fc = nn.Linear(self.K, 1)

        # Xavier init for all linears
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def shared_steps(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] or [L]
        returns: [B, K] or [1, K]
        """
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
        x = c_t.unsqueeze(0) if single else c_t  # [1,L] or [B,L]

        h = self.shared_steps(x)                  # [1,K] or [B,K]
        V = self.value_fc(h).squeeze(-1)          # [] or [B]

        # build distribution
        if self.A is not None:
            logits = self.policy_fc(h)            # [1,A] or [B,A]
            dist   = Categorical(logits=logits)
        else:
            mu        = self.mu_head(h)           # [1,D] or [B,D]
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
        c_t: [L] or [B,L]
        """
        dist, V = self(c_t)
        a  = dist.sample()                    # [] or [B,...]
        logp = dist.log_prob(a)               # match action-shape
        # if continuous, sum over action dims
        if self.D is not None:
            logp = logp.sum(-1)
        return a, logp, V

    def evaluate_actions(self, c_t: torch.Tensor, actions: torch.Tensor):
        """
        Computes log-prob and entropy of given actions under current policy.
        Returns (logp, entropy, value).
        """
        dist, V = self(c_t)
        logp    = dist.log_prob(actions)
        if self.D is not None:
            logp = logp.sum(-1)
        ent = dist.entropy()
        if self.D is not None:
            ent = ent.sum(-1)
        return logp, ent, V


import torch.nn.functional as F

import torch, torch.nn as nn
from collections import deque

class SymQNet(nn.Module):
    def __init__(self,
                 vae,
                 n_qubits, L,
                 edge_index, edge_attr,
                 T, A=None, D=None,
                 K_gnn=2, use_global_node=False):
        super().__init__()
        assert (A is None) ^ (D is None)

        # -- Block 1 (VAE) --
        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # -- Block 2,3,4 --
        self.graph_embed   = GraphEmbed(n_qubits, L, edge_index, edge_attr, K_gnn, use_global_node)
        self.temp_agg      = TemporalContextualAggregator(L, T)
        self.policy_value  = PolicyValueHead(L, A=A, D=D)

        # -- Ring buffer for z_G --
        self.T = T
        self.L = L
        self.zG_history = deque(maxlen=T)
        # buffer for zero-padding
        self.register_buffer("_zero_pad", torch.zeros(T, L))

    def reset_buffer(self):
        self.zG_history.clear()

    def forward(self, m_t: torch.Tensor):
        # Block 1: encode (no_grad)
        with torch.no_grad():
            mu_z, log_sigma_z = self.vae.encode(m_t.unsqueeze(0))
            z0 = self.vae.reparameterize(mu_z, log_sigma_z).squeeze(0)

        # Block 2: graph embed
        z_G = self.graph_embed(z0)

        # update buffer
        self.zG_history.append(z_G)
        # build T×L tensor
        hist = list(self.zG_history)
        if len(hist) < self.T:
            pad = self._zero_pad[: self.T - len(hist)]
            buf = torch.cat([pad, torch.stack(hist)], dim=0)
        else:
            buf = torch.stack(hist)   # [T, L]

        # Block 3: temporal aggregator
        c_t = self.temp_agg(buf)      # [L]

        # Block 4: policy/value
        dist, V = self.policy_value(c_t)
        return dist, V




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gym
from gym import spaces

# ----------------------------------------------------
# 1) SpinChainEnv: Gym-Compatible, Precomputed Unitaries,
#    Single-Qubit Measurements, Proper Seeding, and Metadata Return
# ----------------------------------------------------
class SpinChainEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self,
                 N=10,
                 M_evo=5,
                 T=8,
                 noise_prob=0.02,
                 seed=None,
                 device=torch.device("cpu"),
                 # FIX: resampling controls
                 resample_each_reset=True,
                 resample_every=1,          # set >1 if matrix_exp is too expensive
                 J_range=(0.5, 1.5),
                 h_range=(0.5, 1.5)):
        super().__init__()
        self.N         = N
        self.M_evo     = M_evo
        self.T         = T
        self.noise_prob= noise_prob
        self.step_count= 0
        self.device    = device

        # FIX: task sampling config
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

        # Precompute single-qubit readout rotations (independent of J/h)
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

        # Seed RNGs _after_ spaces are defined
        if seed is not None:
            self.seed(seed)

        # FIX: build the first task (J/h/H/U_list)
        self._resample_task()

        # For reward‐shaping later
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

    # FIX: task resampling helper
    def _resample_task(self):
        self.J_true = np.random.uniform(*self.J_range, size=(self.N - 1,))
        self.h_true = np.random.uniform(*self.h_range, size=(self.N,))
        self.H_true = self._build_hamiltonian(self.J_true, self.h_true).to(self.device)

        # Precompute all evolution unitaries U(τ) = e^{-i H τ}
        self.U_list = [
            torch.matrix_exp(-1j * self.H_true * tau).to(self.device)
            for tau in self.times
        ]

    def reset(self):
        self.step_count = 0
        self.prev_mse   = None

        # FIX: resample Hamiltonian across episodes (optionally throttled)
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

        # mask out other qubits if needed
        if qubit_idx is not None:
            mask = np.zeros_like(bits)
            mask[qubit_idx] = bits[qubit_idx]
            bits = mask

        # add noise flips
        flips = np.random.rand(self.N) < self.noise_prob
        bits[flips] *= -1.0
        return bits

    def step(self, action):
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


# --------------------------------------
# VAE Pre‐Training for M = N  (with stability fixes)
# --------------------------------------
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

# Hyper‐parameters
n_qubits   = 10
M          = n_qubits          # input/output dimension of VAE
L          = 64                # latent dimension
batch_size = 32
epochs     = 300
beta       = 5e-3              # KL weight
n_samples  = 15000             # dataset size
lr_vae     = 3e-4
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################################################
# 0) Define the VAE (with logvar clamping)
################################################################################
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, M: int, L: int, hidden: int = 128):
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

        # Initialize all linears
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        """
        Returns (mu, logvar) of shape [B, L].
        We then clamp logvar to keep it in a stable range.
        """
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        mu       = self.enc_mu(h)
        logvar   = self.enc_logsigma(h)
        # Clamp logvar to avoid numerical overflow when exponentiating
        logvar   = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)      # safe because logvar is clamped
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)           # clamped logvar here
        z           = self.reparameterize(mu, logvar)
        recon       = self.decode(z)
        return recon, mu, logvar, z

'''
################################################################################
# 1) Instantiate environment for data sampling
################################################################################
# (Make sure `SpinChainEnv` is in your Python path.)
env = SpinChainEnv(
    N=n_qubits,
    M_evo=5,
    T=10,                 # T here doesn’t matter much for VAE data collection
    noise_prob=0.02,
    device=device
)

################################################################################
# 2) Collect raw measurement vectors to build a dataset
################################################################################
data = []
for _ in tqdm(range(n_samples), desc="Sampling data"):
    m = env.reset()       # returns a numpy array of shape [n_qubits]
    data.append(m)
data = torch.tensor(data, dtype=torch.float32)  # shape: [n_samples, n_qubits]

################################################################################
# 3) Wrap in a DataLoader
################################################################################
dataset = TensorDataset(data)
loader  = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

################################################################################
# 4) Build the VAE and optimizer
################################################################################
vae = VariationalAutoencoder(M=n_qubits, L=L).to(device)
opt = torch.optim.Adam(vae.parameters(), lr=lr_vae)

################################################################################
# 5) Training loop, with occasional sanity checks
################################################################################
for epoch in range(1, epochs + 1):
    total_recon, total_kl = 0.0, 0.0
    for (batch_data,) in loader:
        batch_data = batch_data.to(device)           # [B, M]
        recon, mu, logvar, z = vae(batch_data)       # recon: [B, M]

        # Reconstruction loss (MSE per sample)
        recon_loss = F.mse_loss(recon, batch_data, reduction='sum') / batch_data.size(0)
        # KL divergence (closed‐form for Gaussian)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_data.size(0)

        loss = recon_loss + beta * kl_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_recon += recon_loss.item() * batch_data.size(0)
        total_kl    += kl_loss.item()   * batch_data.size(0)

    avg_recon = total_recon / n_samples
    avg_kl    = total_kl    / n_samples
    print(f"Epoch {epoch:02d} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    # Every 10 epochs, print out a few mu/logvar/z statistics to ensure no NaNs/infs
    if epoch % 10 == 0:
        with torch.no_grad():
            # Sample one batch from loader for diagnostics
            sample_batch, = next(iter(loader))
            sample_batch = sample_batch.to(device)
            _, mu_samp, logvar_samp, z_samp = vae(sample_batch)
            mu_mean    = mu_samp.mean(dim=0).cpu().numpy()
            logvar_mean= logvar_samp.mean(dim=0).cpu().numpy()
            z_std      = z_samp.std(dim=0).cpu().numpy()
            print("\n  [Diagnostic] mu mean (first 5 dims):", np.round(mu_mean[:5], 4))
            print("  [Diagnostic] logvar mean (first 5 dims):", np.round(logvar_mean[:5], 4))
            print("  [Diagnostic] z_std (first 5 dims):   ", np.round(z_std[:5], 4), "\n")

################################################################################
# 6) Save the stabilized VAE checkpoint
################################################################################
torch.save(vae.state_dict(), "vae_M10_f.pth")
print("Saved VAE checkpoint to vae_M10_fixed.pth.")

################################################################################
# 7) Final sanity check on the full dataset
################################################################################
with torch.no_grad():
    # We'll run the full `data` tensor (all n_samples) through the encoder and compute z.std()
    full_data = data.to(device)         # [n_samples, n_qubits]
    # Compute mu/logvar/z in batches if memory is tight:
    loader_full = DataLoader(full_data, batch_size=1024, shuffle=False)
    all_z = []

    for batch_full in loader_full:
        _, mu_full, logvar_full, z_full = vae(batch_full)
        all_z.append(z_full.cpu())
    all_z = torch.cat(all_z, dim=0)     # [n_samples, L]
    overall_z_std = all_z.std(dim=0)     # [L]
    print("Final Z std (first 10 dims):", overall_z_std[:10].numpy())
'''

# -----------------------------
# FIXED Colab Training Cell for SymQNet
# -----------------------------
import torch, random, numpy as np
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# 1) Helpers
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    """
    vals_1d = values.squeeze(-1)
    T = len(rewards)
    device = vals_1d.device

    # Append a zero at the end for bootstrap
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



# 3) FIXED: SymQNet that properly uses metadata throughout
class FixedSymQNetWithEstimator(nn.Module):
    """Fixed SymQNet that properly integrates all 4 blocks with metadata"""

    def __init__(self, vae, n_qubits, L, edge_index, edge_attr, T, A, M_evo, K_gnn=2):
        super().__init__()
        self.vae = vae
        self.n_qubits = n_qubits
        self.L = L
        self.T = T
        self.A = A
        self.M_evo = M_evo

        # Metadata dimensions
        self.meta_dim = n_qubits + 3 + M_evo  # qubit + basis + time

        # Block 1: Graph embedding (operates on latent + metadata)
        # FIXED: Use correct GraphEmbed signature
        self.graph_embed = GraphEmbed(
            n_qubits=n_qubits,           # Correct parameter
            L=L + self.meta_dim,         # Enhanced dimension with metadata
            edge_index=edge_index,       # Correct parameter
            edge_attr=edge_attr,         # Correct parameter
            K=K_gnn,                     # Correct parameter name
            use_global_node=False        # Add this parameter
        )

        # Block 2: Temporal aggregation
        self.temp_agg = TemporalContextualAggregator(L + self.meta_dim, T)

        # Block 3: Policy-Value head
        self.policy_value = PolicyValueHead(L + self.meta_dim, A)

        # Block 4: Parameter estimator
        self.estimator = nn.Sequential(
            nn.Linear(L + self.meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_qubits - 1)  # J + h parameters
        )

        # Ring buffer for temporal context
        self.zG_history = []

        # Current step metadata (for proper integration)
        self.current_metadata = None

    def reset_buffer(self):
        """Reset temporal buffer"""
        self.zG_history = []
        self.current_metadata = None

    def forward(self, obs, metadata):
        """FIXED: Forward pass that properly uses metadata throughout"""
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

        # Block 4b: Parameter estimation
        theta_hat = self.estimator(c_t)

        return dist, V, theta_hat

# 4) Hyper-parameters
n_qubits    = 10
L           = 64
T           = 10
episodes    = 2_500_000
max_steps   = 150
gamma       = 0.99
lam         = 0.95
initial_lr  = 1e-4
ent_coef    = 0.02
val_coef    = 0.05
est_coef    = 0.4      # ADDED: Estimation loss coefficient
clip_grad   = 0.5     # ADDED: Gradient clipping
seed        = 777
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(seed)

# 5) Environment
env = SpinChainEnv(
    N=n_qubits,
    M_evo=5,
    T=T,
    noise_prob=0.02,
    seed=seed,
    device=device
)
env.seed(seed)


# 6) CORRECTED: Use original VAE, add metadata after encoding
vae = VariationalAutoencoder(M=n_qubits, L=L).to(device)
vae.load_state_dict(torch.load("vae_M10_f.pth", map_location=device))
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# 7) Graph connectivity - FIXED: Ensure proper tensor format
edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=device) * 0.1

# 8) FIXED: Agent with proper metadata integration
A = n_qubits * 3 * env.M_evo

# Create agent with properly formatted edge tensors
agent = FixedSymQNetWithEstimator(
    vae=vae,
    n_qubits=n_qubits,
    L=L,                  # FIX: use VAE latent dim (64)
    edge_index=edge_index,
    edge_attr=edge_attr,
    T=T,
    A=A,
    M_evo=env.M_evo,
    K_gnn=2
).to(device)



optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, agent.parameters()),
    lr=initial_lr
)

def lr_lambda(current_episode: int):
    frac = current_episode / episodes
    return max(1.0 - frac, 0.0)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# 9) TensorBoard
writer = SummaryWriter(log_dir="FIXED_SYMQNET_GO_")

# 10) FIXED: Training loop with proper metadata tracking
print(f"Training FIXED SymQNet on {device} for {episodes} episodes...")

# Validation tracking
best_performance = float('inf')
validation_freq = 100

for ep in range(1, episodes + 1):
    obs = env.reset()
    agent.reset_buffer()

    init_mse = None
    logp_buf, val_buf, rew_buf, ent_buf, est_loss_buf = [], [], [], [], []

    # FIXED: Track metadata from previous step's info
    prev_info = None

    # --- drop-in: replace your entire inner loop with this ---
    prev_info = None
    pending = None  # (logp, V, ent) for the action whose reward we haven't assigned yet
    init_mse = None
    prev_mse = None

    for step in range(max_steps):
        obs_tensor = torch.from_numpy(obs).float().to(device)

        # metadata describes the action that produced *this* obs
        if prev_info is not None:
            qi = prev_info['qubit_idx']
            bi = prev_info['basis_idx']
            ti = prev_info['time_idx']
        else:
            qi = bi = ti = 0

        metadata = torch.zeros(agent.meta_dim, device=device)
        metadata[qi] = 1.0
        metadata[n_qubits + bi] = 1.0
        metadata[n_qubits + 3 + ti] = 1.0

        dist, V, theta_hat = agent(obs_tensor, metadata)

        # True params are constant within an episode
        true_theta = np.concatenate([env.J_true, env.h_true])
        pred_theta = theta_hat.detach().cpu().numpy()
        curr_mse = float(((pred_theta - true_theta) ** 2).mean())

        if init_mse is None:
            init_mse = curr_mse
        if prev_mse is None:
            prev_mse = curr_mse

        # Pay out reward for the *previous* action now that we’ve seen its resulting obs
        if pending is not None:
            prev_logp, prev_V, prev_ent = pending

            improvement = (prev_mse - curr_mse) / (init_mse + 1e-8)
            r = np.tanh(improvement * 10.0) - 0.01

            logp_buf.append(prev_logp)
            val_buf.append(prev_V)
            ent_buf.append(prev_ent)
            rew_buf.append(r)

            est_loss_buf.append(
                F.mse_loss(theta_hat, torch.from_numpy(true_theta).float().to(device))
            )

        # Choose the next action and hold onto it until we observe the next state
        a = dist.sample().item()
        logp = dist.log_prob(torch.tensor(a, device=device))
        ent = dist.entropy()
        pending = (logp, V, ent)

        obs2, _, done, info = env.step(a)

        # advance env
        prev_info = info
        obs = obs2
        prev_mse = curr_mse

        if done:
            # Flush terminal reward for the last pending action using the final post-action estimate
            obs_tensor = torch.from_numpy(obs).float().to(device)
            qi = prev_info['qubit_idx']
            bi = prev_info['basis_idx']
            ti = prev_info['time_idx']

            metadata = torch.zeros(agent.meta_dim, device=device)
            metadata[qi] = 1.0
            metadata[n_qubits + bi] = 1.0
            metadata[n_qubits + 3 + ti] = 1.0

            _, _, theta_hat_T = agent(obs_tensor, metadata)
            pred_theta_T = theta_hat_T.detach().cpu().numpy()
            final_mse = float(((pred_theta_T - true_theta) ** 2).mean())

            improvement_ratio = (init_mse - final_mse) / (init_mse + 1e-8)
            rT = 5.0 * np.tanh(improvement_ratio)

            last_logp, last_V, last_ent = pending
            logp_buf.append(last_logp)
            val_buf.append(last_V)
            ent_buf.append(last_ent)
            rew_buf.append(rT)

            est_loss_buf.append(
                F.mse_loss(theta_hat_T, torch.from_numpy(true_theta).float().to(device))
            )

            curr_mse = final_mse  # so your logging reflects the real final estimate
            break


    # Rest of the training loop remains the same...


    # FIXED: Compute losses with estimation auxiliary loss
    vals = torch.stack(val_buf)
    logps = torch.stack(logp_buf)
    ents = torch.stack(ent_buf)
    est_losses = torch.stack(est_loss_buf)

    returns, advs = compute_gae(rew_buf, vals, gamma, lam)

    # FIXED: Multi-objective loss
    policy_loss = -(logps * advs.detach()).mean()
    value_loss = (returns - vals).pow(2).mean()
    entropy_loss = -ents.mean()
    estimation_loss = est_losses.mean()  # Auxiliary estimation loss

    total_loss = (100000000 * 0.35 * policy_loss +
                  0.2 * value_loss +
                  0.1 * entropy_loss +
                  0.35 * estimation_loss)

    optimizer.zero_grad()
    total_loss.backward()

    # FIXED: Gradient clipping
    torch.nn.utils.clip_grad_norm_(agent.parameters(), clip_grad)

    optimizer.step()
    scheduler.step()

    # Enhanced logging
    total_r = sum(rew_buf)
    current_lr = optimizer.param_groups[0]['lr']

    writer.add_scalar("Reward/episode", total_r, ep)
    writer.add_scalar("Loss/policy", policy_loss.item(), ep)
    writer.add_scalar("Loss/value", value_loss.item(), ep)
    writer.add_scalar("Loss/entropy", -entropy_loss.item(), ep)
    writer.add_scalar("Loss/estimation", estimation_loss.item(), ep)
    writer.add_scalar("Loss/total", total_loss.item(), ep)
    writer.add_scalar("LearningRate", current_lr, ep)
    writer.add_scalar("MSE/final", curr_mse, ep)
    writer.add_scalar("MSE/improvement", init_mse - curr_mse, ep)

    if ep % 10 == 0 or ep == 1:
        print(f"Ep {ep:06d} | R:{total_r:.4f} | MSE:{curr_mse:.4f} | "
              f"P/L:{policy_loss:.4f} | V/L:{value_loss:.4f} | "
              f"Est/L:{estimation_loss:.4f} | LR:{current_lr:.2e}")

    # FIXED: Validation and best model tracking
    if ep % validation_freq == 0:
        if curr_mse < best_performance:
            best_performance = curr_mse
            torch.save({
                'episode': ep,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'performance': curr_mse,
                'reward': total_r
            }, "BEST_SYMQNET_MODEL.pth")
            print(f"→ New best model saved! MSE: {curr_mse:.6f}")

    # Regular checkpoints
    if ep % 10 == 0:
        ckpt = {
            'episode': ep,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'performance': curr_mse,
            'reward': total_r
        }
        torch.save(ckpt, f"symqnet_fixed_ep{ep:07d}.pth")
        print(f"→ Checkpoint saved: ep {ep}")

# 11) Final save
torch.save({
    'model_state_dict': agent.state_dict(),
    'final_performance': curr_mse,
    'episodes_trained': episodes
}, "FINAL_FIXED_SYMQNET.pth")

print("✅ FIXED training completed!")
print(f"Best performance achieved: {best_performance:.6f}")
writer.close()

