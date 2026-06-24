from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch


def validate_backend(name: str, hamiltonian: str) -> str:
    backend = str(name).lower()
    if backend not in {"statevector", "mps_tebd"}:
        raise ValueError(f"Unknown simulator backend: {name}")
    if backend == "mps_tebd" and str(hamiltonian).lower() != "tfim":
        raise ValueError("mps_tebd backend currently supports TFIM only")
    return backend


@dataclass(frozen=True)
class MpsTebdSettings:
    bond_dim: int = 32
    trotter_steps: int = 8


class TFIMMpsTebdSimulator:
    """Small self-contained MPS/TEBD helper for TFIM local observables.

    The dense statevector path remains the authoritative backend for N<=7
    validation. This helper avoids dense Hamiltonian construction for larger-N
    scaling runs while preserving the codebase's measurement convention:
    basis 0/2 returns negative standard Pauli X/Z expectation, while basis
    1 follows the standard-Y sign convention used by mean_outcome_from_state.
    """

    def __init__(
        self,
        n_qubits: int,
        j_values: np.ndarray | torch.Tensor,
        h_values: np.ndarray | torch.Tensor,
        *,
        bond_dim: int = 32,
        trotter_steps: int = 8,
    ):
        self.n_qubits = int(n_qubits)
        self.j = np.asarray(j_values, dtype=np.float64).reshape(self.n_qubits - 1)
        self.h = np.asarray(h_values, dtype=np.float64).reshape(self.n_qubits)
        self.bond_dim = max(1, int(bond_dim))
        self.trotter_steps = max(1, int(trotter_steps))
        self._svd_fallbacks = 0

    def local_expectation(self, qubit_idx: int, basis_idx: int, tau: float) -> float:
        mps = self._initial_mps()
        self._evolve(mps, float(tau))
        state = self._to_state(mps)
        return float(self._local_expectation_from_state(state, int(qubit_idx), int(basis_idx)))

    def local_expectations(self, times: np.ndarray) -> np.ndarray:
        out = np.empty((len(times), self.n_qubits, 3), dtype=np.float32)
        for time_idx, tau in enumerate(times):
            mps = self._initial_mps()
            self._evolve(mps, float(tau))
            state = self._to_state(mps)
            for q in range(self.n_qubits):
                for b in range(3):
                    out[time_idx, q, b] = self._local_expectation_from_state(state, q, b)
        return out

    def _initial_mps(self) -> list[np.ndarray]:
        tensors = []
        for _ in range(self.n_qubits):
            tensor = np.zeros((1, 2, 1), dtype=np.complex128)
            tensor[0, 0, 0] = 1.0
            tensors.append(tensor)
        return tensors

    def _evolve(self, mps: list[np.ndarray], tau: float) -> None:
        if tau == 0.0:
            return
        dt = tau / self.trotter_steps
        for _ in range(self.trotter_steps):
            self._apply_x_layer(mps, 0.5 * dt)
            for bond in range(self.n_qubits - 1):
                self._apply_zz_gate(mps, bond, dt)
            self._apply_x_layer(mps, 0.5 * dt)

    def _apply_x_layer(self, mps: list[np.ndarray], dt: float) -> None:
        for site, field in enumerate(self.h):
            c = np.cos(field * dt)
            s = np.sin(field * dt)
            gate = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
            mps[site] = np.einsum("ab,lbr->lar", gate, mps[site], optimize=True)

    def _apply_zz_gate(self, mps: list[np.ndarray], site: int, dt: float) -> None:
        z = np.array([1.0, -1.0], dtype=np.float64)
        phase = np.exp(-1j * self.j[site] * np.outer(z, z) * dt)
        gate = np.zeros((2, 2, 2, 2), dtype=np.complex128)
        for a in range(2):
            for b in range(2):
                gate[a, b, a, b] = phase[a, b]
        self._apply_two_site_gate(mps, site, gate)

    def _apply_two_site_gate(self, mps: list[np.ndarray], site: int, gate: np.ndarray) -> None:
        left = mps[site]
        right = mps[site + 1]
        theta = np.einsum("lam,mbr->labr", left, right, optimize=True)
        theta = np.einsum("abij,lijr->labr", gate, theta, optimize=True)
        ldim, _, _, rdim = theta.shape
        matrix = theta.reshape(ldim * 2, 2 * rdim)
        u, s, vh = self._stable_svd(matrix, site)
        keep = min(self.bond_dim, s.shape[0])
        u = u[:, :keep]
        s = s[:keep]
        vh = vh[:keep, :]
        norm = np.linalg.norm(s)
        if norm > 0.0:
            s = s / norm
        mps[site] = u.reshape(ldim, 2, keep)
        mps[site + 1] = (s[:, None] * vh).reshape(keep, 2, rdim)

    def _stable_svd(self, matrix: np.ndarray, site: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = np.asarray(matrix, dtype=np.complex128)
        if not np.all(np.isfinite(matrix)):
            matrix = np.nan_to_num(matrix, copy=True)
        try:
            return np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError as original_error:
            last_error: Exception = original_error
            try:
                from scipy.linalg import svd as scipy_svd

                try:
                    return scipy_svd(matrix, full_matrices=False, lapack_driver="gesvd", check_finite=False)
                except Exception as exc:  # pragma: no cover - rare LAPACK fallback path
                    last_error = exc
            except Exception as exc:  # pragma: no cover - scipy might not be present
                last_error = exc

            rows, cols = matrix.shape
            diag_n = min(rows, cols)
            for eps in (1e-12, 1e-10, 1e-8, 1e-6):
                regularized = matrix.copy()
                idx = np.arange(diag_n)
                regularized[idx, idx] += eps
                try:
                    self._svd_fallbacks += 1
                    if self._svd_fallbacks <= 5:
                        warnings.warn(
                            f"MPS TEBD SVD needed diagonal regularization at site={site} eps={eps:g}; "
                            "continuing with a tiny perturbation.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    return np.linalg.svd(regularized, full_matrices=False)
                except np.linalg.LinAlgError as exc:
                    last_error = exc

            raise np.linalg.LinAlgError(f"SVD did not converge after fallback attempts at site={site}") from last_error

    def _to_state(self, mps: list[np.ndarray]) -> np.ndarray:
        state = mps[0]
        for tensor in mps[1:]:
            state = np.tensordot(state, tensor, axes=([-1], [0]))
        state = np.squeeze(state, axis=(0, -1)).reshape(-1)
        norm = np.linalg.norm(state)
        return state / norm if norm > 0.0 else state

    def _local_expectation_from_state(self, state: np.ndarray, qubit_idx: int, basis_idx: int) -> float:
        shaped = state.reshape((2,) * self.n_qubits)
        if basis_idx == 0:
            op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        elif basis_idx == 1:
            op = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        elif basis_idx == 2:
            op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        else:
            raise ValueError("basis_idx must be 0 (X), 1 (Y), or 2 (Z)")
        moved = np.moveaxis(shaped, qubit_idx, 0).reshape(2, -1)
        standard = np.einsum("ai,ab,bi->", moved.conj(), op, moved, optimize=True).real
        sign = 1.0 if basis_idx == 1 else -1.0
        return float(np.clip(sign * standard, -1.0, 1.0))


def mps_local_expectation_batch(
    n_qubits: int,
    particles: torch.Tensor,
    qubit_idx: int,
    basis_idx: int,
    tau: float,
    *,
    bond_dim: int,
    trotter_steps: int,
) -> torch.Tensor:
    values = []
    particles_cpu = particles.detach().cpu().numpy()
    for theta in particles_cpu:
        sim = TFIMMpsTebdSimulator(
            n_qubits,
            theta[: n_qubits - 1],
            theta[n_qubits - 1 :],
            bond_dim=bond_dim,
            trotter_steps=trotter_steps,
        )
        values.append(sim.local_expectation(qubit_idx, basis_idx, tau))
    return torch.tensor(values, device=particles.device, dtype=torch.float32)
