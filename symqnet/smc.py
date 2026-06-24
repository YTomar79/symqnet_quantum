from __future__ import annotations

from dataclasses import dataclass

import torch

from .backends import mps_local_expectation_batch
from .env import SpinChainEnv
from .math_utils import mean_outcome_from_state


@dataclass
class Posterior:
    mean: torch.Tensor
    cov: torch.Tensor


class SMCParticleFilter:
    """Sequential Monte Carlo posterior over theta = [J_0..J_N-2, h_0..h_N-1]."""

    def __init__(
        self,
        env: SpinChainEnv,
        n_particles: int = 48,
        ess_frac: float = 0.6,
        roughen_frac: float = 0.03,
        device: torch.device | str | None = None,
    ):
        self.env = env
        self.N = int(env.N)
        self.theta_dim = 2 * self.N - 1
        self.dim = 2**self.N
        self.P = int(n_particles)
        self.ess_threshold = float(ess_frac) * self.P
        self.roughen_frac = float(roughen_frac)
        self.device = torch.device(device) if device is not None else env.device
        self.sim_dtype = torch.complex64
        self.sim_chunk_size = 8
        self.simulator_backend = getattr(env, "simulator_backend", "statevector")

        j_lo, j_hi = env.J_range
        h_lo, h_hi = env.h_range
        self.prior_low = torch.tensor([j_lo] * (self.N - 1) + [h_lo] * self.N, device=self.device)
        self.prior_high = torch.tensor([j_hi] * (self.N - 1) + [h_hi] * self.N, device=self.device)
        self._flip_idx: list[torch.Tensor] = []
        self._sign01: list[torch.Tensor] = []
        self._phase: list[torch.Tensor] = []
        self.terms_stack: torch.Tensor | None = None
        if self.simulator_backend == "statevector":
            self.terms_stack = self._build_terms().to(self.sim_dtype)
            idxs = torch.arange(self.dim, device=self.device, dtype=torch.int64)
            for q in range(self.N):
                bitpos = self.N - 1 - q
                bit = ((idxs >> bitpos) & 1).to(torch.float32)
                self._flip_idx.append(idxs ^ (1 << bitpos))
                self._sign01.append(2.0 * bit - 1.0)
                self._phase.append(1.0 - 2.0 * bit)

        self._cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self.reset()

    def _kron_ops(self, ops: list[torch.Tensor]) -> torch.Tensor:
        out = ops[0]
        for item in ops[1:]:
            out = torch.kron(out, item)
        return out

    def _build_terms(self) -> torch.Tensor:
        terms = []
        if getattr(self.env, "hamiltonian", "tfim") == "tfim":
            for i in range(self.N - 1):
                ops = [self.env.I] * self.N
                ops[i] = self.env.Z
                ops[i + 1] = self.env.Z
                terms.append(self._kron_ops(ops))
            for i in range(self.N):
                ops = [self.env.I] * self.N
                ops[i] = self.env.X
                terms.append(self._kron_ops(ops))
        else:
            for i in range(self.N - 1):
                coupling = torch.zeros((self.dim, self.dim), dtype=self.env.I.dtype, device=self.env.device)
                for op in (self.env.X, self.env.Y, self.env.Z):
                    ops = [self.env.I] * self.N
                    ops[i] = op
                    ops[i + 1] = op
                    coupling = coupling + self._kron_ops(ops)
                terms.append(coupling)
            for i in range(self.N):
                ops = [self.env.I] * self.N
                ops[i] = self.env.Z
                terms.append(self._kron_ops(ops))
        return torch.stack(terms, dim=0)

    def clone(self, *, share_cache: bool = False) -> "SMCParticleFilter":
        other = object.__new__(type(self))
        other.__dict__ = self.__dict__.copy()
        other.particles = self.particles.clone()
        other.w = self.w.clone()
        other._cache = self._cache if share_cache else {}
        return other

    @torch.no_grad()
    def reset(self) -> None:
        u = torch.rand(self.P, self.theta_dim, device=self.device)
        self.particles = self.prior_low + u * (self.prior_high - self.prior_low)
        self.w = torch.full((self.P,), 1.0 / self.P, device=self.device)
        self._cache.clear()

    @torch.no_grad()
    def posterior(self) -> Posterior:
        mean = (self.w[:, None] * self.particles).sum(dim=0)
        centered = self.particles - mean
        cov = torch.einsum("p,pi,pj->ij", self.w, centered, centered)
        cov = cov + 1e-6 * torch.eye(self.theta_dim, device=self.device)
        return Posterior(mean.float(), cov.float())

    @torch.no_grad()
    def _predict_mean_outcome_batch(
        self,
        thetas: torch.Tensor,
        qubit_idx: int,
        basis_idx: int,
        time_idx: int,
    ) -> torch.Tensor:
        tau = float(self.env.times[int(time_idx)])
        if self.simulator_backend == "mps_tebd":
            E = mps_local_expectation_batch(
                self.N,
                thetas,
                int(qubit_idx),
                int(basis_idx),
                tau,
                bond_dim=int(getattr(self.env, "mps_bond_dim", 32)),
                trotter_steps=int(getattr(self.env, "mps_trotter_steps", 8)),
            )
            return self.env.apply_noise_to_expectation(E, tau).float()
        if self.terms_stack is None:
            raise RuntimeError("statevector SMC backend is not initialized")
        out = []
        chunk_size = max(1, min(int(self.sim_chunk_size), int(thetas.shape[0])))
        for start in range(0, int(thetas.shape[0]), chunk_size):
            batch = thetas[start : start + chunk_size].to(self.sim_dtype)
            H = torch.einsum("bt,tdk->bdk", batch, self.terms_stack)
            U = torch.matrix_exp((-1j) * H * tau)
            psi = U[..., :, 0]
            E = mean_outcome_from_state(
                psi,
                self._flip_idx[qubit_idx],
                self._sign01[qubit_idx].to(dtype=psi.real.dtype),
                self._phase[qubit_idx].to(dtype=psi.real.dtype),
                basis_idx,
            )
            out.append(self.env.apply_noise_to_expectation(E, tau).float())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def predicted_means(self, qubit_idx: int, basis_idx: int, time_idx: int) -> torch.Tensor:
        key = (int(qubit_idx), int(basis_idx), int(time_idx))
        if key not in self._cache:
            self._cache[key] = self._predict_mean_outcome_batch(self.particles, *key)
        return self._cache[key]

    @torch.no_grad()
    def update(self, obs: torch.Tensor, info: dict[str, object]) -> Posterior:
        qubit_idx = int(info["qubit_idx"])
        basis_idx = int(info["basis_idx"])
        time_idx = int(info["time_idx"])
        shots = max(1, int(info.get("shots", self.env.current_shots)))
        y = obs[qubit_idx].float().clamp(-1.0, 1.0)

        E = self.predicted_means(qubit_idx, basis_idx, time_idx)
        var = (1.0 - E * E).clamp_min(1e-6) / shots
        ll = -0.5 * ((y - E) ** 2) / var - 0.5 * torch.log(var)
        logw = ll - ll.max()
        self.w = self.w * torch.exp(logw)
        self.w = self.w / (self.w.sum() + 1e-12)

        ess = 1.0 / (self.w.pow(2).sum() + 1e-12)
        if float(ess.item()) < self.ess_threshold:
            self._systematic_resample()
            self._roughen()
            self._cache.clear()
        return self.posterior()

    @torch.no_grad()
    def update_pseudo_observation(
        self,
        y: torch.Tensor,
        qubit_idx: int,
        basis_idx: int,
        time_idx: int,
        shots: int,
    ) -> Posterior:
        obs = torch.zeros(self.N, device=self.device)
        obs[int(qubit_idx)] = y
        return self.update(
            obs,
            {
                "qubit_idx": int(qubit_idx),
                "basis_idx": int(basis_idx),
                "time_idx": int(time_idx),
                "shots": int(shots),
            },
        )

    @torch.no_grad()
    def _systematic_resample(self) -> None:
        positions = (torch.rand((), device=self.device) + torch.arange(self.P, device=self.device)) / self.P
        cdf = torch.cumsum(self.w, dim=0)
        idx = torch.searchsorted(cdf, positions).clamp(0, self.P - 1)
        self.particles = self.particles[idx]
        self.w = torch.full((self.P,), 1.0 / self.P, device=self.device)

    @torch.no_grad()
    def _roughen(self) -> None:
        span = self.prior_high - self.prior_low
        noise = torch.randn_like(self.particles) * (self.roughen_frac * span)
        self.particles = (self.particles + noise).clamp(self.prior_low, self.prior_high)
