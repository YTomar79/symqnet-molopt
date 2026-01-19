import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Faster matmuls on Ampere+ (safe for this workload)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

@torch.no_grad()
def mean_outcome_from_state_precomp(
    psi: torch.Tensor,          # [..., dim] complex
    flip_idx: torch.Tensor,     # [dim] int64
    sign01: torch.Tensor,       # [dim] float (|0>->-1, |1>->+1)
    phase: torch.Tensor,        # [dim] float (|0>:+1, |1>:-1)
    basis_idx: int,             # 0=X, 1=Y, 2=Z
) -> torch.Tensor:
    """
    Returns mean outcome in [-1,1] under YOUR convention:
      computational |1> -> +1, |0> -> -1
    which corresponds to -<Pauli> for X/Y/Z in the usual physics convention.
    """
    psi_flip = psi[..., flip_idx]

    if basis_idx == 2:  # Z
        probs = (psi.abs() ** 2).to(sign01.dtype)
        return (probs * sign01).sum(dim=-1)

    if basis_idx == 0:  # X
        ex = (psi.conj() * psi_flip).sum(dim=-1).real
        return -ex

    if basis_idx == 1:  # Y
        i_unit = torch.tensor(1j, device=psi.device, dtype=psi.dtype)
        ey = (psi.conj() * (i_unit * phase) * psi_flip).sum(dim=-1).real
        return -ey

    raise ValueError("basis_idx must be 0(X), 1(Y), or 2(Z)")

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
    p_plus = 0.5 * (1.0 + expectations)
    p_plus = np.clip(p_plus, 0.0, 1.0)
    n_plus = np.random.binomial(shots, p_plus)
    return (2.0 * n_plus - shots) / shots

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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        m_noisy, m_ideal = generate_measurement_pair(self.n_qubits)
        return m_noisy.astype(np.float32), m_ideal.astype(np.float32)


import math

def covariance_to_features(cov: torch.Tensor, max_eigs: int = 8):
    """
    Converts covariance matrix to compact belief features.

    Returns:
      feat = [ log_diag , top_eigenvalues ]
    """
    diag = torch.log(torch.diag(cov) + 1e-8)

    # Low-rank uncertainty structure
    eigvals = torch.linalg.eigvalsh(cov)
    topk = eigvals[-max_eigs:]

    return torch.cat([diag, topk], dim=0)


def gaussian_entropy_from_cov(cov: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    """
    Differential entropy of N(·; 0, cov) in nats:
      H = 0.5 * ( d*(1 + ln(2π)) + ln(det(cov)) )

    Returns a scalar tensor (nats). Uses slogdet for stability.
    """
    d = int(cov.shape[0])
    I = torch.eye(d, device=cov.device, dtype=cov.dtype)

    C = cov + eps * I
    sign, logdet = torch.linalg.slogdet(C)

    # If numerical issues: add more jitter once
    if (sign <= 0).any():
        C = cov + 1e-6 * I
        sign, logdet = torch.linalg.slogdet(C)

    return 0.5 * (d * (1.0 + math.log(2.0 * math.pi)) + logdet)



class SMCParticleFilter:
    """
    Sequential Monte Carlo filter over theta = [J_0..J_{N-2}, h_0..h_{N-1}] (dim = 2N-1).

    Update uses an approximate likelihood for the observed mean outcome y in [-1,1]:
      y ~ Normal(E(theta,a), (1 - E^2)/shots)
    where E(theta,a) is the predicted mean outcome under the particle's Hamiltonian
    and the chosen (qubit, basis, time_idx).
    """

    def __init__(
        self,
        env,
        n_particles: int = 48,
        ess_frac: float = 0.5,
        roughen_frac: float = 0.02,
        device=None,
    ):
        self.env = env
        self.N = int(env.N)
        self.theta_dim = 2 * self.N - 1
        self.dim = 2 ** self.N
        self.P = int(n_particles)
        self.ess_threshold = float(ess_frac) * self.P
        self.roughen_frac = float(roughen_frac)
        self.device = device if device is not None else env.device

        # Prior bounds (vectorized)
        J_lo, J_hi = env.J_range
        h_lo, h_hi = env.h_range
        self.prior_low  = torch.tensor([J_lo] * (self.N - 1) + [h_lo] * self.N, device=self.device, dtype=torch.float32)
        self.prior_high = torch.tensor([J_hi] * (self.N - 1) + [h_hi] * self.N, device=self.device, dtype=torch.float32)

        # Precompute operator terms so building H(theta) is cheap:
        # H(theta) = sum_i J_i ZZ_i + sum_i h_i X_i
        self.ZZ_terms = []
        for i in range(self.N - 1):
            ops = [env.I] * self.N
            ops[i] = env.Z
            ops[i + 1] = env.Z
            term = ops[0]
            for o in ops[1:]:
                term = torch.kron(term, o)
            self.ZZ_terms.append(term)

        self.X_terms = []
        for i in range(self.N):
            ops = [env.I] * self.N
            ops[i] = env.X
            term = ops[0]
            for o in ops[1:]:
                term = torch.kron(term, o)
            self.X_terms.append(term)

        # --- Vectorization helpers ---
        # Stack all Hamiltonian basis terms once:
        # terms_stack: [T, dim, dim] where T = (N-1) + N = 2N-1 = theta_dim
        # Use complex64 for the heavy SMC simulation to cut memory ~2x
        self.sim_dtype = torch.complex64

        terms = [t.to(self.device) for t in (self.ZZ_terms + self.X_terms)]
        self.terms_stack = torch.stack([t.to(self.sim_dtype) for t in terms], dim=0)  # [T,dim,dim]
        self.T = self.terms_stack.shape[0]

        # Much smaller default chunk; matrix_exp peak is huge
        self.sim_chunk_size = int(getattr(self, "sim_chunk_size", 8))


        # Cache for predicted means for a given (qubit,basis,time) within current particle set
        # key: (qubit_idx, basis_idx, time_idx) -> E_batch [P]
        self._cache = {}

        # Precompute bit masks to compute p_plus quickly
        # Precompute per-qubit bitflip indices + sign/phase vectors (used for fast X/Y/Z means)
        idxs = torch.arange(self.dim, device=self.device, dtype=torch.int64)
        self._flip_idx = []
        self._sign01   = []
        self._phase    = []
        for q in range(self.N):
            bitpos = self.N - 1 - q
            flip = idxs ^ (1 << bitpos)
            bit  = ((idxs >> bitpos) & 1).to(torch.float32)
            self._flip_idx.append(flip)
            self._sign01.append(2.0 * bit - 1.0)
            self._phase.append(1.0 - 2.0 * bit)


        self.reset()

    @torch.no_grad()
    def _mean_outcome_from_state_batch(self, psi: torch.Tensor, qubit_idx: int, basis_idx: int) -> torch.Tensor:
        """
        psi: [B, dim] complex
        returns: [B] float (mean outcome in [-1,1] under your convention)
        """
        q = int(qubit_idx)
        flip = self._flip_idx[q]
        sign01 = self._sign01[q].to(device=psi.device, dtype=psi.real.dtype)
        phase  = self._phase[q].to(device=psi.device, dtype=psi.real.dtype)
        return mean_outcome_from_state_precomp(psi, flip, sign01, phase, int(basis_idx))

    def reset(self):
        # Sample prior particles uniformly
        u = torch.rand(self.P, self.theta_dim, device=self.device)
        self.particles = self.prior_low + u * (self.prior_high - self.prior_low)
        self.w = torch.full((self.P,), 1.0 / self.P, device=self.device, dtype=torch.float32)
        self._cache.clear()

    @torch.no_grad()
    def posterior_mean_and_covariance(self):
        """
        Returns:
          mean: [theta_dim]
          cov:  [theta_dim, theta_dim]
        """
        w = self.w
        parts = self.particles

        mean = (w.unsqueeze(-1) * parts).sum(dim=0)  # [theta_dim]
        centered = parts - mean  # [P,theta_dim]

        # Weighted covariance (vectorized):
        # cov = sum_p w_p * outer(centered_p, centered_p)
        cov = torch.einsum("p,pi,pj->ij", w, centered, centered).to(torch.float32)

        cov = cov + 1e-6 * torch.eye(self.theta_dim, device=self.device, dtype=torch.float32)
        return mean.float(), cov

    @torch.no_grad()
    def _build_H(self, theta: torch.Tensor):
        # theta: [theta_dim]
        J = theta[: self.N - 1]
        h = theta[self.N - 1 :]
        H = torch.zeros((self.dim, self.dim), device=self.device, dtype=torch.complex128)
        for i in range(self.N - 1):
            H = H + (J[i].to(torch.complex128) * self.ZZ_terms[i].to(torch.complex128))
        for i in range(self.N):
            H = H + (h[i].to(torch.complex128) * self.X_terms[i].to(torch.complex128))
        return H

    @torch.no_grad()
    def _predict_mean_outcome_batch(self, thetas: torch.Tensor, qubit_idx: int, basis_idx: int, time_idx: int):
        """
        Memory-safe batched Hamiltonian simulation.

        thetas:  [P, theta_dim]
        returns: [P] float32
        """
        P_total = int(thetas.shape[0])
        tau = float(self.env.times[int(time_idx)])

        # ---- pick chunk size (auto shrink on small GPUs) ----
        chunk_size = int(getattr(self, "sim_chunk_size", 8))

        if self.device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
            except TypeError:
                free_bytes, _ = torch.cuda.mem_get_info()

            bytes_per_complex = 8 if self.sim_dtype == torch.complex64 else 16
            mat_bytes = int(self.dim * self.dim * bytes_per_complex)

            # Very conservative multiplier for H, U, and matrix_exp workspace
            est_bytes_per_particle = 10 * mat_bytes

            # use at most ~40% of currently free memory for this chunk
            max_chunk = max(1, int((0.40 * free_bytes) // est_bytes_per_particle))
            chunk_size = max(1, min(chunk_size, max_chunk, P_total))
        else:
            chunk_size = max(1, min(chunk_size, P_total))

        p_readout = float(self.env.noise_prob)
        out = []

        for start in range(0, P_total, chunk_size):
            end = min(start + chunk_size, P_total)
            batch_thetas = thetas[start:end]                 # [B,theta_dim]
            coeffs = batch_thetas.to(self.sim_dtype)         # [B,T]

            # H: [B,dim,dim]
            H = torch.einsum("bt,tdk->bdk", coeffs, self.terms_stack)

            # U = exp(-i H tau): [B,dim,dim]
            U = torch.matrix_exp((-1j) * H * tau)

            # psi(t) = U @ |0...0>  == first column of U
            psi = U[..., :, 0]                                # [B,dim]

            # Fast mean outcome in desired basis (no rotations, no probs, no masks)
            E = self._mean_outcome_from_state_batch(psi, int(qubit_idx), int(basis_idx))

            # Readout flip noise scales mean by (1 - 2p)
            E = E * (1.0 - 2.0 * p_readout)

            out.append(E.to(torch.float32))

            del H, U, psi, E

        return torch.cat(out, dim=0)




    @torch.no_grad()
    def _predict_mean_outcome(self, theta: torch.Tensor, qubit_idx: int, basis_idx: int, time_idx: int):
        # Keep signature; use the batch path with a single element
        E1 = self._predict_mean_outcome_batch(theta.unsqueeze(0), qubit_idx, basis_idx, time_idx)
        return E1[0]

    @torch.no_grad()
    def update(self, obs: torch.Tensor, info: dict, env=None):
        """
        obs:  [N] float tensor with only obs[qubit_idx] nonzero (mean outcome)
        info: from env.step(...) containing qubit_idx, basis_idx, time_idx, shots
        """
        qubit_idx = int(info["qubit_idx"])
        basis_idx = int(info["basis_idx"])
        time_idx  = int(info["time_idx"])
        shots     = int(info.get("shots", getattr(self.env, "current_shots", 1)))

        y = obs[qubit_idx].float().clamp(-1.0, 1.0)

        # Vectorized E for all particles (with per-action cache)
        key = (qubit_idx, basis_idx, time_idx)
        if key in self._cache:
            E = self._cache[key]  # [P]
        else:
            E = self._predict_mean_outcome_batch(self.particles, qubit_idx, basis_idx, time_idx)  # [P]
            self._cache[key] = E

        # Approx variance of sample mean of +/-1 draws
        var = (1.0 - E * E).clamp_min(1e-6) / max(1, shots)  # [P]

        # log-likelihood for each particle
        ll = -0.5 * ((y - E) ** 2) / var - 0.5 * torch.log(var)  # [P]

        # Weight update (stable)
        logw = ll - ll.max()
        w_new = self.w * torch.exp(logw)
        w_new = w_new / (w_new.sum() + 1e-12)
        self.w = w_new

        # Resample if ESS low
        ess = 1.0 / (self.w.pow(2).sum() + 1e-12)
        if float(ess.item()) < self.ess_threshold:
            self._systematic_resample()
            self._roughen()
            self._cache.clear()

        return self.posterior_mean_and_covariance()

    @torch.no_grad()
    def _systematic_resample(self):
        positions = (torch.rand((), device=self.device) + torch.arange(self.P, device=self.device)) / self.P
        cdf = torch.cumsum(self.w, dim=0)
        idx = torch.searchsorted(cdf, positions).clamp(0, self.P - 1)
        self.particles = self.particles[idx]
        self.w = torch.full((self.P,), 1.0 / self.P, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _roughen(self):
        span = (self.prior_high - self.prior_low)
        noise = torch.randn_like(self.particles) * (self.roughen_frac * span)
        self.particles = (self.particles + noise).clamp(self.prior_low, self.prior_high)

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
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )


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
            self.pre  = nn.Linear(self.L, self.attn_dim, bias=False)
            self.post = nn.Linear(self.attn_dim, self.L, bias=False)
        else:
            self.pre  = nn.Identity()
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
             h_range=(0.5, 1.5),
             # NEW: shot control
             default_shots=128,
             shots_set=(32, 64, 128, 256, 512),
             sample_shots_each_step=False):

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

        # NEW: shots config
        self.default_shots = int(default_shots)
        self.shots_set = tuple(shots_set) if shots_set is not None else None
        self.sample_shots_each_step = bool(sample_shots_each_step)
        self.shots_max = max(self.shots_set) if self.shots_set is not None else self.default_shots
        self.current_shots = self.default_shots

        # Discrete evolution times
        self.times = np.linspace(0.1, 1.0, M_evo)

        # Pauli & identity on single qubit
        cdtype = torch.complex64
        fdtype = torch.float32

        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=cdtype, device=device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=cdtype, device=device)

        inv_sqrt2 = torch.tensor(1.0 / np.sqrt(2.0), dtype=fdtype, device=device)
        self.H = inv_sqrt2 * torch.tensor([[1, 1], [1, -1]], dtype=cdtype, device=device)

        self.Sdg = torch.tensor([[1, 0], [0, -1j]], dtype=cdtype, device=device)
        self.I = torch.eye(2, dtype=cdtype, device=device)


        # Precompute single-qubit readout rotations (independent of J/h)
        # NOTE: No dense UX/UY rotation matrices.
        # We compute X/Y/Z outcomes directly from the statevector using bit tricks.
        self.UX_list = None
        self.UY_list = None



        # Initial state |0…0>
        dim = 2**N
        psi0 = torch.zeros((dim, 1), dtype=cdtype, device=device)
        psi0[0, 0] = 1.0 + 0j
        self.psi0 = psi0

        # Precompute per-qubit bitflip indices + sign/phase vectors (GPU-friendly)
        idxs = torch.arange(dim, device=device, dtype=torch.int64)
        self._flip_idx = []
        self._sign01   = []   # |0>->-1, |1>->+1
        self._phase    = []   # |0>:+1, |1>:-1
        for q in range(N):
            bitpos = N - 1 - q
            flip = idxs ^ (1 << bitpos)
            bit  = ((idxs >> bitpos) & 1).to(fdtype)
            self._flip_idx.append(flip)
            self._sign01.append(2.0 * bit - 1.0)
            self._phase.append(1.0 - 2.0 * bit)


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
        cdtype = self.I.dtype  # keep consistent with env matrices

        # prevent dtype promotion (float64 -> complex128)
        J = torch.as_tensor(J, device=self.device, dtype=torch.float32)
        h = torch.as_tensor(h, device=self.device, dtype=torch.float32)

        H = torch.zeros((2**N, 2**N), dtype=cdtype, device=self.device)

        # ZZ couplings
        for i in range(N - 1):
            ops = [self.I] * N
            ops[i] = self.Z
            ops[i + 1] = self.Z
            term = ops[0]
            for o in ops[1:]:
                term = torch.kron(term, o)
            H = H + J[i] * term

        # X fields
        for i in range(N):
            ops = [self.I] * N
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
        self.H_true = self._build_hamiltonian(self.J_true, self.h_true)

        i_unit = torch.tensor(1j, device=self.device, dtype=self.I.dtype)  # complex64

        # Store only psi(t) = exp(-i H t) |0...0>, not the full U(t)
        self.psi_list = []
        for tau in self.times:
            tau_t = torch.tensor(float(tau), device=self.device, dtype=torch.float32)
            U = torch.matrix_exp((-i_unit) * self.H_true * tau_t)
            self.psi_list.append(U[:, 0:1].contiguous())  # first column == U @ |0...0>
            del U

        self.U_list = None  # not needed anymore



    def reset(self):
        self.step_count = 0
        self.prev_mse   = None

        # FIX: resample Hamiltonian across episodes (optionally throttled)
        self._episode_counter += 1
        if self.resample_each_reset and ((self._episode_counter - 1) % self.resample_every == 0):
            self._resample_task()

        # NEW: choose shots for this episode (unless sampling each step)
        if (self.shots_set is not None) and (not self.sample_shots_each_step):
            self.current_shots = int(np.random.choice(self.shots_set))
        else:
            self.current_shots = int(self.default_shots)

        return np.zeros(self.N, dtype=np.float32)


    def _measure(self, psi, basis, qubit_idx, shots: int = 1):
        # psi: [dim,1] or [dim]
        if psi.dim() == 2:
            psi = psi.squeeze(-1)
        psi = psi.to(device=self.device, dtype=self.I.dtype)

        basis_idx = {"X": 0, "Y": 1, "Z": 2}[basis]
        q = int(qubit_idx)

        flip  = self._flip_idx[q]
        sign01 = self._sign01[q].to(device=psi.device, dtype=psi.real.dtype)
        phase  = self._phase[q].to(device=psi.device, dtype=psi.real.dtype)

        # Ideal mean outcome (your convention)
        E = mean_outcome_from_state_precomp(psi, flip, sign01, phase, basis_idx)

        # Readout flip noise: +/-1 outcome flips with prob p -> scales mean by (1 - 2p)
        E = E * (1.0 - 2.0 * float(self.noise_prob))

        # Sample "shots" times (GPU) and return mean in [-1, 1]
        # Sample "shots" times and return the empirical mean in [-1, 1]
        # This matches the SMC update model y ~ Normal(E, (1-E^2)/shots).
        shots = int(max(1, shots))

        if shots == 1:
            mean_outcome = E
        else:
            p_plus = (0.5 * (1.0 + E)).clamp(0.0, 1.0)
            # Binomial draw for number of +1 outcomes
            n_plus = torch.distributions.Binomial(total_count=shots, probs=p_plus).sample()
            mean_outcome = (2.0 * n_plus - shots) / shots



        obs = np.zeros(self.N, dtype=np.float32)
        obs[q] = float(mean_outcome.item())
        return obs



    def step(self, action):
        a        = int(action)
        time_idx = a % self.M_evo
        a      //= self.M_evo
        basis_idx= a % 3
        qubit_idx= a // 3

        basis = ['X','Y','Z'][basis_idx]
        psi_t = self.psi_list[time_idx]   # already exp(-iHt)|0...0>


        # NEW: optionally resample shots each step
        if self.sample_shots_each_step and (self.shots_set is not None):
            self.current_shots = int(np.random.choice(self.shots_set))

        obs = self._measure(psi_t, basis=basis, qubit_idx=qubit_idx, shots=self.current_shots)

        reward = 0.0
        self.step_count += 1
        done = (self.step_count >= self.T)

        info = {
            'J_true': self.J_true.copy(),
            'h_true': self.h_true.copy(),
            'qubit_idx': qubit_idx,
            'basis_idx': basis_idx,
            'time_idx': time_idx,
            # NEW:
            'shots': int(self.current_shots),
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
n_qubits   = 5
M          = n_qubits          # input/output dimension of VAE
L          = 16                # latent dimension
batch_size = 32
epochs     = 300
beta       = 3e-3              # KL weight
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

def compute_gae_dones(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    GAE for rollouts that may contain multiple episodes.

    rewards:     list/1D tensor length T
    values:      1D tensor length T (V(s_t) under old policy)
    dones:       1D tensor length T with 1.0 where episode ended at t
    last_value:  scalar tensor V(s_{T}) for bootstrap (ignored when dones[-1]=1)
    """
    if not torch.is_tensor(rewards):
        rewards = torch.tensor(rewards, device=values.device, dtype=values.dtype)
    if not torch.is_tensor(dones):
        dones = torch.tensor(dones, device=values.device, dtype=values.dtype)

    T = rewards.shape[0]
    adv = torch.zeros((), device=values.device, dtype=values.dtype)
    advs = torch.zeros(T, device=values.device, dtype=values.dtype)

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        adv = delta + gamma * lam * nonterminal * adv
        advs[t] = adv

    returns = advs + values
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
        # Metadata dimensions (+ parameter feedback)
        # Metadata dimensions
        # base: qubit one-hot (N) + basis one-hot (3) + time one-hot (M_evo)
        base = n_qubits + 3 + M_evo

        # Belief-state feedback dims
        self.theta_dim = 2 * n_qubits - 1

        # Slots (action one-hots, shots, posterior mean/cov/fisher)
        self.shots_slot = base
        self.theta_slot0 = self.shots_slot + 1
        self.cov_slot0 = self.theta_slot0 + self.theta_dim

        self.cov_feat_dim = self.theta_dim + 8        # diag + top-8 eigs
        self.fisher_slot0 = self.cov_slot0 + self.cov_feat_dim
        self.fisher_feat_dim = self.theta_dim         # fisher feature length

        self.meta_dim = self.fisher_slot0 + self.fisher_feat_dim





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
        # Block 4: Parameter estimator (mu + logvar => enables Fisher/CRB-style feedback)



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
            z = mu_z  # deterministic encoding is important for PPO stability / consistent re-eval
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
        # Policy and Value only (belief comes from SMC via metadata)
        dist, V = self.policy_value(c_t)
        return dist, V


    def forward_window(self, z_with_meta_window: torch.Tensor):
        """
        Stateless forward used for PPO minibatching.

        z_with_meta_window: [T, D] or [B, T, D], where D = L + meta_dim.
        Returns: dist, V, theta_mu, theta_logvar (batched if B>1).
        """
        if z_with_meta_window.dim() == 2:
            x = z_with_meta_window.unsqueeze(0)
            squeeze = True
        elif z_with_meta_window.dim() == 3:
            x = z_with_meta_window
            squeeze = False
        else:
            raise ValueError("z_with_meta_window must be [T,D] or [B,T,D]")

        B, T, D = x.shape

        # Graph embed each step, vectorized over (B*T)
        zG = self.graph_embed(x.reshape(B * T, D)).reshape(B, T, D)

        # Temporal aggregation
        c_t = self.temp_agg(zG)

        # Policy/value + estimator
        dist, V = self.policy_value(c_t)

        if squeeze:
            return dist, V.squeeze(0)
        return dist, V




# 4) Hyper-parameters
n_qubits    = 5
L           = 16

episodes    = 2_500_000

theta_dim   = 2 * n_qubits - 1
T_env       = min(150, max(30, int(4 * theta_dim)))   # n_qubits=10 -> 76
T           = min(20, T_env)                           # model history window (keep compute reasonable)
max_steps   = T_env                                    # actual episode horizon

gamma       = 0.99
lam         = 0.95
initial_lr  = 1e-4
ent_coef    = 0.02
val_coef    = 0.05
est_coef    = 0.4      # ADDED: Estimation loss coefficient
clip_grad   = 0.5     # ADDED: Gradient clipping
seed        = 777
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_eps    = 0.2

set_seed(seed)

# 5) Environment
env = SpinChainEnv(
    N=n_qubits,
    M_evo=5,
    T=T_env,
    noise_prob=0.02,
    seed=seed,
    device=device,

    # throttle expensive Hamiltonian resampling / expm
    resample_each_reset=True,
    resample_every=50,  # tune: 10..200 depending on speed/variance tradeoff

    default_shots=128,
    shots_set=(32, 64, 128, 256, 512),
    sample_shots_each_step=False,  # per-episode shot count
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
writer = SummaryWriter(log_dir="_FIXED_SYMQNET_FINAL")

# 10) FIXED: Training loop with proper metadata tracking
# 10) PPO v2: batched rollouts + multi-epoch minibatch updates
print(f"Training PPO v2 SymQNet on {device} for {episodes} PPO updates...")

# IMPORTANT for PPO ratio stability with dropout modules (Transformer encoder uses Dropout)
agent.eval()

# PPO rollout/update hyper-params
rollout_steps   = 64          # typical PPO batch; try 1024 if env is slow
ppo_epochs      = 2
num_minibatches = 2
minibatch_size  = rollout_steps // num_minibatches
assert rollout_steps % num_minibatches == 0

# slots for metadata composition
shots_slot   = agent.shots_slot
theta_slot0  = agent.theta_slot0
fisher_slot0 = agent.fisher_slot0

best_performance = float("inf")
validation_freq  = 25  # measured in PPO updates

# carry env state across rollouts
obs = env.reset()
agent.reset_buffer()
prev_info = None
pending = None

# per-episode shaping state
init_mse = None
prev_mse = None

# belief-state feedback (1-step lag)
prev_theta_feat  = torch.zeros(agent.theta_dim, device=device)
prev_fisher_feat = torch.zeros(agent.theta_dim, device=device)

prev_cov_feat = torch.zeros(agent.cov_feat_dim, device=device)


# SMC posterior over theta (J,h)
smc = SMCParticleFilter(env, n_particles=48, ess_frac=0.6, roughen_frac=0.03, device=device)

# Force tiny chunks on 8GB GPUs (start with 4; if still OOM, use 2)
smc.sim_chunk_size = 4

smc.reset()



# true theta for current task
true_theta = np.concatenate([env.J_true, env.h_true])
true_theta_t = torch.from_numpy(true_theta).float().to(device)

for update in range(1, episodes + 1):
    # rollout storage
    obs_buf, meta_buf, act_buf = [], [], []
    logp_buf, val_buf, rew_buf = [], [], []
    info_gain_buf = []

    done_buf, theta_true_buf   = [], []

    ep_return = 0.0
    ep_returns = []
    ep_final_mses = []

    last_value = torch.zeros((), device=device)

    while len(rew_buf) < rollout_steps:
        obs_tensor = torch.from_numpy(obs).float().to(device)

        # metadata describes the action that produced *this* obs
                # --- Build metadata for the action that produced *this* obs + update belief once ---
        info_gain_prev_action = 0.0
        info_gain_log = 0.0

        # default: all-zero action metadata (first step in episode)
        metadata = torch.zeros(agent.meta_dim, device=device)

        if prev_info is not None:
            qi = int(prev_info["qubit_idx"])
            bi = int(prev_info["basis_idx"])
            ti = int(prev_info["time_idx"])
            shots_val = int(prev_info.get("shots", getattr(env, "current_shots", 1)))
            shots_max = float(getattr(env, "shots_max", max(2, shots_val)))

            # PRIOR (before conditioning on obs produced by prev_info)
            _, cov_prior = smc.posterior_mean_and_covariance()
            H_prior = gaussian_entropy_from_cov(cov_prior)

            # POSTERIOR (after conditioning)
            theta_mean, theta_cov = smc.update(obs_tensor, prev_info, env)
            H_post = gaussian_entropy_from_cov(theta_cov)

            ig = torch.clamp(H_prior - H_post, min=0.0)  # nats
            info_gain_prev_action = float(ig.item())
            info_gain_log = info_gain_prev_action

            # Update belief feedback features for the policy
            prev_theta_feat = theta_mean.detach()
            prev_cov_feat = covariance_to_features(theta_cov).detach()

            # Action metadata (one-hot + shots)
            metadata[qi] = 1.0
            metadata[n_qubits + bi] = 1.0
            metadata[n_qubits + 3 + ti] = 1.0
            metadata[shots_slot] = float(np.log2(max(1, shots_val)) / np.log2(max(2.0, shots_max)))

        # Belief feedback always present (zeros on first step)
        metadata[theta_slot0 : theta_slot0 + agent.theta_dim] = prev_theta_feat
        metadata[agent.cov_slot0 : agent.cov_slot0 + agent.cov_feat_dim] = prev_cov_feat
        metadata[fisher_slot0 : fisher_slot0 + agent.theta_dim] = prev_fisher_feat




        # 2) Compute current MSE for shaping (using posterior mean)
        pred_theta = prev_theta_feat.detach().cpu().numpy()
        curr_mse = float(((pred_theta - true_theta) ** 2).mean())
        if init_mse is None:
            init_mse = curr_mse
        if prev_mse is None:
            prev_mse = curr_mse

        # 3) Policy/value under old policy (no grad)
        with torch.no_grad():
            dist, V = agent(obs_tensor, metadata)


        # pay out reward for previous action now that we see its outcome
        if pending is not None:
            # InfoGain reward (nats). Optional tiny step-cost if you still want it:
            r = float(info_gain_prev_action)  # - 0.01
            info_gain_buf.append(info_gain_log)



            obs_buf.append(pending["obs"])
            meta_buf.append(pending["meta"])
            act_buf.append(pending["action"])
            logp_buf.append(pending["logp_old"])
            val_buf.append(pending["V_old"])
            rew_buf.append(r)
            done_buf.append(0.0)
            theta_true_buf.append(pending["true_theta"])

            ep_return += r
            pending = None

            # advance prev_mse after paying reward
            prev_mse = curr_mse

            # if we just filled the rollout, bootstrap from V(current obs)
            if len(rew_buf) >= rollout_steps:
                last_value = V.detach().squeeze()
                break

        # choose next action
        a_t = dist.sample()
        logp_t = dist.log_prob(a_t)

        pending = {
            "obs": obs_tensor.detach(),
            "meta": metadata.detach(),
            "action": a_t.detach(),
            "logp_old": logp_t.detach(),
            "V_old": V.detach().squeeze(),
            "true_theta": true_theta_t.detach(),
        }

        # step env
        obs2, _, done, info = env.step(int(a_t.item()))
        prev_info = info
        obs = obs2

        # if episode ended, assign terminal reward for pending now
        if done:
            obs_T = torch.from_numpy(obs).float().to(device)

            qi = int(prev_info["qubit_idx"])
            bi = int(prev_info["basis_idx"])
            ti = int(prev_info["time_idx"])
            shots_val = int(prev_info.get("shots", getattr(env, "current_shots", 1)))
            shots_max = float(getattr(env, "shots_max", max(1, shots_val)))

            meta_T = torch.zeros(agent.meta_dim, device=device)
            meta_T[qi] = 1.0
            meta_T[n_qubits + bi] = 1.0
            meta_T[n_qubits + 3 + ti] = 1.0
            meta_T[shots_slot] = float(np.log2(max(1, shots_val)) / np.log2(max(2.0, shots_max)))

            # belief feedback (match non-terminal metadata layout)
            meta_T[theta_slot0 : theta_slot0 + agent.theta_dim] = prev_theta_feat
            meta_T[agent.cov_slot0 : agent.cov_slot0 + agent.cov_feat_dim] = prev_cov_feat
            meta_T[fisher_slot0 : fisher_slot0 + agent.theta_dim] = prev_fisher_feat


            # Final posterior update for the terminal observation + terminal InfoGain reward
            with torch.no_grad():
                # prior entropy before conditioning on terminal obs
                _, cov_prior_T = smc.posterior_mean_and_covariance()
                H_prior_T = gaussian_entropy_from_cov(cov_prior_T)

                # update on terminal obs
                theta_mean_T, theta_cov_T = smc.update(obs_T, prev_info, env)

                H_post_T = gaussian_entropy_from_cov(theta_cov_T)

            # terminal InfoGain in nats (clamped)
            # terminal InfoGain in nats (clamped)
            rT = float(torch.clamp(H_prior_T - H_post_T, min=0.0).item())

            # NEW: include terminal IG in TB stats
            info_gain_buf.append(rT)

            # (optional) keep your final MSE logging if you want it
            pred_theta_T = theta_mean_T.detach().cpu().numpy()
            final_mse = float(((pred_theta_T - true_theta) ** 2).mean())


            obs_buf.append(pending["obs"])
            meta_buf.append(pending["meta"])
            act_buf.append(pending["action"])
            logp_buf.append(pending["logp_old"])
            val_buf.append(pending["V_old"])
            rew_buf.append(rT)
            done_buf.append(1.0)
            theta_true_buf.append(pending["true_theta"])


            ep_return += rT
            ep_returns.append(ep_return)
            ep_final_mses.append(final_mse)

            pending = None

            # terminal => no bootstrap
            last_value = torch.zeros((), device=device)

            # reset episode state
            obs = env.reset()
            agent.reset_buffer()
            prev_info = None

            smc.reset()

            init_mse = None
            prev_mse = None
            ep_return = 0.0

            prev_theta_feat  = torch.zeros(agent.theta_dim, device=device)
            prev_fisher_feat = torch.zeros(agent.theta_dim, device=device)
            prev_cov_feat = torch.zeros(agent.cov_feat_dim, device=device)


            true_theta = np.concatenate([env.J_true, env.h_true])
            true_theta_t = torch.from_numpy(true_theta).float().to(device)

    # --- convert rollout to tensors ---
    obs_b    = torch.stack(obs_buf).to(device)
    meta_b   = torch.stack(meta_buf).to(device)
    act_b    = torch.stack(act_buf).to(device)
    old_logp = torch.stack(logp_buf).to(device)
    old_val  = torch.stack(val_buf).to(device)
    dones_t  = torch.tensor(done_buf, device=device, dtype=old_val.dtype)
    true_th  = torch.stack(theta_true_buf).to(device)

    # GAE (multi-episode safe)
    returns, advs = compute_gae_dones(rew_buf, old_val, dones_t, last_value.detach(), gamma, lam)
    returns = returns.detach()
    advs    = advs.detach()

    # deterministic latents (VAE frozen)
    with torch.no_grad():
        mu_z, _ = agent.vae.encode(obs_b)
        z = mu_z
    z_with_meta = torch.cat([z, meta_b], dim=-1)  # [N, D]

    # build causal windows per timestep (respect episode boundaries)
    T_win = agent.T
    N, D = z_with_meta.shape
    windows = torch.zeros(N, T_win, D, device=device, dtype=z_with_meta.dtype)

    ep_start = 0
    for t in range(N):
        if t > 0 and float(dones_t[t - 1].item()) == 1.0:
            ep_start = t
        s = max(ep_start, t - T_win + 1)
        seq = z_with_meta[s : t + 1]
        windows[t, -seq.shape[0] :, :] = seq

    # --- PPO update: multi-epoch minibatch ---
    # --- PPO update: multi-epoch minibatch ---
    policy_losses, value_losses, ent_loss_vals, ent_vals, est_losses, total_losses = [], [], [], [], [], []



    for epoch in range(ppo_epochs):
        idx = torch.randperm(N, device=device)
        for start in range(0, N, minibatch_size):
            mb = idx[start : start + minibatch_size]

            win_mb       = windows[mb]
            act_mb       = act_b[mb]
            old_logp_mb  = old_logp[mb]
            old_val_mb   = old_val[mb]
            ret_mb       = returns[mb]
            adv_mb       = advs[mb]
            th_mb        = true_th[mb]

            dist, V = agent.forward_window(win_mb)

            # --- PPO losses MUST be computed fresh per minibatch ---
            act_mb = act_mb.long()  # Categorical expects Long actions

            new_logp = dist.log_prob(act_mb)     # [B]
            entropy  = dist.entropy()            # [B]

            ratio = torch.exp(new_logp - old_logp_mb)  # [B]
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -(torch.min(surr1, surr2)).mean()

            value_loss = F.mse_loss(V, ret_mb)   # scalar

            # maximize entropy => subtract it in the loss
            entropy_mean = entropy.mean()
            entropy_loss = -entropy_mean
            

            # (optional placeholder so your existing logging doesn't crash)
            est_loss = torch.zeros((), device=device)

            total_loss = policy_loss + val_coef * value_loss + ent_coef * entropy_loss + est_coef * est_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), clip_grad)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            ent_loss_vals.append(float(entropy_loss.detach().item()))
            ent_vals.append(float(entropy_mean.detach().item()))
            est_losses.append(float(est_loss.detach().item()))
            total_losses.append(float(total_loss.detach().item()))




    scheduler.step()

    # --- logging ---
    total_r = float(sum(rew_buf))
    current_lr = optimizer.param_groups[0]["lr"]
    avg_final_mse = float(np.mean(ep_final_mses)) if len(ep_final_mses) else float("nan")

    writer.add_scalar("Update/return_sum", total_r, update)
    writer.add_scalar("Reward/info_gain_mean", float(np.mean(info_gain_buf)), update)
    writer.add_scalar("Reward/info_gain_sum", float(np.sum(info_gain_buf)), update)

    writer.add_scalar("Update/episodes_in_rollout", len(ep_final_mses), update)
    writer.add_scalar("Loss/policy", float(np.mean(policy_losses)), update)
    writer.add_scalar("Loss/value", float(np.mean(value_losses)), update)
    writer.add_scalar("Loss/entropy_loss", float(np.mean(ent_loss_vals)), update)
    writer.add_scalar("Policy/entropy", float(np.mean(ent_vals)), update)
    writer.add_scalar("Loss/estimation", float(np.mean(est_losses)), update)
    writer.add_scalar("Loss/total", float(np.mean(total_losses)), update)

    writer.add_scalar("LearningRate", current_lr, update)
    writer.add_scalar("MSE/final_mean", avg_final_mse, update)

    if update % 5 == 0 or update == 1:
        print(
            f"Upd {update:06d} | steps:{N} | eps:{len(ep_final_mses)} | "
            f"Rsum:{total_r:.3f} | MSEmean:{avg_final_mse:.6f} | LR:{current_lr:.2e}"
        )

    # checkpoint on mean final MSE
    if update % validation_freq == 0 and len(ep_final_mses):
        if avg_final_mse < best_performance:
            best_performance = avg_final_mse
            torch.save(
                {
                    "checkpoint_format": "symqnet-ppo-v2",
                    "checkpoint_version": 1,
                    "update": update,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "performance": best_performance,
                    "rollout_steps": int(rollout_steps),
                    "ppo_epochs": int(ppo_epochs),
                    "minibatch_size": int(minibatch_size),
                    "include_shots": True,
                    "meta_dim": agent.meta_dim,
                    "shots_encoding": {"type": "log2_norm", "shots_max": int(getattr(env, "shots_max", 1))},
                    "n_qubits": int(n_qubits),
                    "M_evo": int(env.M_evo),
                },
                "BEST_SYMQNET_MODEL_PPOV2.pth",
            )
            print(f"→ New best model saved! mean final MSE: {best_performance:.6f}")

print("✅ PPO v2 training completed!")
print(f"Best mean-final-MSE achieved: {best_performance:.6f}")
writer.close()
