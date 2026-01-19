"""
Sequential Monte Carlo (SMC) belief update utilities.
"""

from __future__ import annotations

import math
import torch


@torch.no_grad()
def mean_outcome_from_state_precomp(
    psi: torch.Tensor,
    flip_idx: torch.Tensor,
    sign01: torch.Tensor,
    phase: torch.Tensor,
    basis_idx: int,
) -> torch.Tensor:
    """
    Returns mean outcome in [-1, 1] under convention:
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


def covariance_to_features(cov: torch.Tensor, max_eigs: int = 8) -> torch.Tensor:
    """
    Converts covariance matrix to compact belief features.

    Returns:
      feat = [ log_diag , top_eigenvalues ]
    """
    diag = torch.log(torch.diag(cov) + 1e-8)
    eigvals = torch.linalg.eigvalsh(cov)
    topk = eigvals[-max_eigs:]
    return torch.cat([diag, topk], dim=0)


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

        J_lo, J_hi = env.J_range
        h_lo, h_hi = env.h_range
        self.prior_low = torch.tensor(
            [J_lo] * (self.N - 1) + [h_lo] * self.N,
            device=self.device,
            dtype=torch.float32,
        )
        self.prior_high = torch.tensor(
            [J_hi] * (self.N - 1) + [h_hi] * self.N,
            device=self.device,
            dtype=torch.float32,
        )

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

        self.sim_dtype = torch.complex64
        terms = [t.to(self.device) for t in (self.ZZ_terms + self.X_terms)]
        self.terms_stack = torch.stack([t.to(self.sim_dtype) for t in terms], dim=0)
        self.T = self.terms_stack.shape[0]

        self.sim_chunk_size = int(getattr(self, "sim_chunk_size", 8))

        self._cache = {}

        idxs = torch.arange(self.dim, device=self.device, dtype=torch.int64)
        self._flip_idx = []
        self._sign01 = []
        self._phase = []
        for q in range(self.N):
            bitpos = self.N - 1 - q
            flip = idxs ^ (1 << bitpos)
            bit = ((idxs >> bitpos) & 1).to(torch.float32)
            self._flip_idx.append(flip)
            self._sign01.append(2.0 * bit - 1.0)
            self._phase.append(1.0 - 2.0 * bit)

        self.reset()

    @torch.no_grad()
    def _mean_outcome_from_state_batch(
        self, psi: torch.Tensor, qubit_idx: int, basis_idx: int
    ) -> torch.Tensor:
        q = int(qubit_idx)
        flip = self._flip_idx[q]
        sign01 = self._sign01[q].to(device=psi.device, dtype=psi.real.dtype)
        phase = self._phase[q].to(device=psi.device, dtype=psi.real.dtype)
        return mean_outcome_from_state_precomp(psi, flip, sign01, phase, int(basis_idx))

    def reset(self) -> None:
        u = torch.rand(self.P, self.theta_dim, device=self.device)
        self.particles = self.prior_low + u * (self.prior_high - self.prior_low)
        self.w = torch.full((self.P,), 1.0 / self.P, device=self.device, dtype=torch.float32)
        self._cache.clear()

    @torch.no_grad()
    def posterior_mean_and_covariance(self):
        w = self.w
        parts = self.particles
        mean = (w.unsqueeze(-1) * parts).sum(dim=0)
        centered = parts - mean
        cov = torch.einsum("p,pi,pj->ij", w, centered, centered).to(torch.float32)
        cov = cov + 1e-6 * torch.eye(self.theta_dim, device=self.device, dtype=torch.float32)
        return mean.float(), cov

    @torch.no_grad()
    def _predict_mean_outcome_batch(
        self, thetas: torch.Tensor, qubit_idx: int, basis_idx: int, time_idx: int
    ):
        P_total = int(thetas.shape[0])
        tau = float(self.env.times[int(time_idx)])

        chunk_size = int(getattr(self, "sim_chunk_size", 8))
        if self.device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
            except TypeError:
                free_bytes, _ = torch.cuda.mem_get_info()

            bytes_per_complex = 8 if self.sim_dtype == torch.complex64 else 16
            mat_bytes = int(self.dim * self.dim * bytes_per_complex)
            est_bytes_per_particle = 10 * mat_bytes
            max_chunk = max(1, int((0.40 * free_bytes) // est_bytes_per_particle))
            chunk_size = max(1, min(chunk_size, max_chunk, P_total))
        else:
            chunk_size = max(1, min(chunk_size, P_total))

        p_readout = float(self.env.noise_prob)
        out = []

        for start in range(0, P_total, chunk_size):
            end = min(start + chunk_size, P_total)
            batch_thetas = thetas[start:end]
            coeffs = batch_thetas.to(self.sim_dtype)
            H = torch.einsum("bt,tdk->bdk", coeffs, self.terms_stack)
            U = torch.matrix_exp((-1j) * H * tau)
            psi = U[..., :, 0]
            E = self._mean_outcome_from_state_batch(psi, int(qubit_idx), int(basis_idx))
            E = E * (1.0 - 2.0 * p_readout)
            out.append(E.to(torch.float32))
            del H, U, psi, E

        return torch.cat(out, dim=0)

    @torch.no_grad()
    def update(self, obs: torch.Tensor, info: dict):
        qubit_idx = int(info["qubit_idx"])
        basis_idx = int(info["basis_idx"])
        time_idx = int(info["time_idx"])
        shots = int(info.get("shots", getattr(self.env, "current_shots", 1)))

        y = obs[qubit_idx].float().clamp(-1.0, 1.0)

        key = (qubit_idx, basis_idx, time_idx)
        if key in self._cache:
            E = self._cache[key]
        else:
            E = self._predict_mean_outcome_batch(self.particles, qubit_idx, basis_idx, time_idx)
            self._cache[key] = E

        var = (1.0 - E * E).clamp_min(1e-6) / max(1, shots)
        ll = -0.5 * ((y - E) ** 2) / var - 0.5 * torch.log(var)

        logw = ll - ll.max()
        w_new = self.w * torch.exp(logw)
        w_new = w_new / (w_new.sum() + 1e-12)
        self.w = w_new

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
