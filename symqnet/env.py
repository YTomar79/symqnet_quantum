from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .backends import TFIMMpsTebdSimulator, validate_backend
from .math_utils import mean_outcome_from_state


@dataclass(frozen=True)
class StepInfo:
    j_true: np.ndarray
    h_true: np.ndarray
    qubit_idx: int
    basis_idx: int
    time_idx: int
    shots: int


class SpinChainEnv:
    """Dense statevector simulator for 1D TFIM Hamiltonian-learning episodes."""

    def __init__(
        self,
        n_qubits: int = 5,
        m_evo: int = 5,
        horizon: int = 36,
        hamiltonian: str = "tfim",
        noise_prob: float = 0.02,
        noise_model: str = "readout_flip",
        readout_p01: float | None = None,
        readout_p10: float | None = None,
        t1_us: float = 100.0,
        t2_us: float = 70.0,
        time_scale_us: float = 1.0,
        seed: int | None = None,
        device: torch.device | str = "cpu",
        j_range: tuple[float, float] = (0.5, 1.5),
        h_range: tuple[float, float] = (0.5, 1.5),
        default_shots: int = 128,
        shots_set: tuple[int, ...] | None = (32, 64, 128, 256, 512),
        sample_shots_each_step: bool = False,
        simulator_backend: str = "statevector",
        mps_bond_dim: int = 32,
        mps_trotter_steps: int = 8,
    ):
        self.N = int(n_qubits)
        self.M_evo = int(m_evo)
        self.T = int(horizon)
        self.hamiltonian = str(hamiltonian).lower()
        if self.hamiltonian not in {"tfim", "xxz", "heisenberg"}:
            raise ValueError(f"Unknown Hamiltonian family: {hamiltonian}")
        self.noise_prob = float(noise_prob)
        self.noise_model = str(noise_model)
        if self.noise_model not in {"readout_flip", "native_decoherence"}:
            raise ValueError(f"Unknown noise model: {noise_model}")
        self.readout_p01 = float(noise_prob if readout_p01 is None else readout_p01)
        self.readout_p10 = float(noise_prob if readout_p10 is None else readout_p10)
        self.t1_us = float(t1_us)
        self.t2_us = float(t2_us)
        self.time_scale_us = float(time_scale_us)
        self.device = torch.device(device)
        self.J_range = tuple(j_range)
        self.h_range = tuple(h_range)
        self.default_shots = int(default_shots)
        self.shots_set = tuple(shots_set) if shots_set is not None else None
        self.sample_shots_each_step = bool(sample_shots_each_step)
        self.simulator_backend = validate_backend(simulator_backend, self.hamiltonian)
        self.mps_bond_dim = int(mps_bond_dim)
        self.mps_trotter_steps = int(mps_trotter_steps)
        self.shots_max = max(self.shots_set) if self.shots_set else self.default_shots
        self.current_shots = self.default_shots
        self.times = np.linspace(0.1, 1.0, self.M_evo)
        self.n_actions = self.N * 3 * self.M_evo
        self.step_count = 0

        cdtype = torch.complex64
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=cdtype, device=self.device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=cdtype, device=self.device)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=cdtype, device=self.device)
        self.I = torch.eye(2, dtype=cdtype, device=self.device)

        self.dim = 2**self.N
        idxs = torch.arange(self.dim, device=self.device, dtype=torch.int64)
        self._flip_idx: list[torch.Tensor] = []
        self._sign01: list[torch.Tensor] = []
        self._phase: list[torch.Tensor] = []
        for q in range(self.N):
            bitpos = self.N - 1 - q
            bit = ((idxs >> bitpos) & 1).to(torch.float32)
            self._flip_idx.append(idxs ^ (1 << bitpos))
            self._sign01.append(2.0 * bit - 1.0)
            self._phase.append(1.0 - 2.0 * bit)

        if seed is not None:
            self.seed(seed)
        self._resample_task()

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _single_site_term(self, op: torch.Tensor, qubit: int) -> torch.Tensor:
        ops = [self.I] * self.N
        ops[qubit] = op
        out = ops[0]
        for item in ops[1:]:
            out = torch.kron(out, item)
        return out

    def _zz_term(self, qubit: int) -> torch.Tensor:
        return self._two_site_term(self.Z, self.Z, qubit)

    def _two_site_term(self, op_left: torch.Tensor, op_right: torch.Tensor, qubit: int) -> torch.Tensor:
        ops = [self.I] * self.N
        ops[qubit] = op_left
        ops[qubit + 1] = op_right
        out = ops[0]
        for item in ops[1:]:
            out = torch.kron(out, item)
        return out

    def _build_hamiltonian(self, j_values: np.ndarray, h_values: np.ndarray) -> torch.Tensor:
        hamiltonian = torch.zeros((self.dim, self.dim), dtype=self.I.dtype, device=self.device)
        if self.hamiltonian == "tfim":
            for i, value in enumerate(j_values):
                hamiltonian = hamiltonian + float(value) * self._zz_term(i)
            for i, value in enumerate(h_values):
                hamiltonian = hamiltonian + float(value) * self._single_site_term(self.X, i)
        else:
            for i, value in enumerate(j_values):
                coupling = (
                    self._two_site_term(self.X, self.X, i)
                    + self._two_site_term(self.Y, self.Y, i)
                    + self._two_site_term(self.Z, self.Z, i)
                )
                hamiltonian = hamiltonian + float(value) * coupling
            for i, value in enumerate(h_values):
                hamiltonian = hamiltonian + float(value) * self._single_site_term(self.Z, i)
        return hamiltonian

    def _resample_task(self) -> None:
        self.J_true = np.random.uniform(*self.J_range, size=(self.N - 1,))
        self.h_true = np.random.uniform(*self.h_range, size=(self.N,))
        self._set_task_arrays(self.J_true, self.h_true)

    def _set_task_arrays(self, j_values: np.ndarray, h_values: np.ndarray) -> None:
        self.J_true = np.asarray(j_values, dtype=np.float32).copy()
        self.h_true = np.asarray(h_values, dtype=np.float32).copy()
        if self.J_true.shape != (self.N - 1,):
            raise ValueError(f"J task shape must be {(self.N - 1,)}, got {self.J_true.shape}")
        if self.h_true.shape != (self.N,):
            raise ValueError(f"h task shape must be {(self.N,)}, got {self.h_true.shape}")
        self.psi_list = []
        self.H_true = None
        self._mps_observables = None
        if self.simulator_backend == "statevector":
            self.H_true = self._build_hamiltonian(self.J_true, self.h_true)
            i_unit = torch.tensor(1j, device=self.device, dtype=self.I.dtype)
            for tau in self.times:
                U = torch.matrix_exp((-i_unit) * self.H_true * float(tau))
                self.psi_list.append(U[:, 0].contiguous())
        else:
            simulator = TFIMMpsTebdSimulator(
                self.N,
                self.J_true,
                self.h_true,
                bond_dim=self.mps_bond_dim,
                trotter_steps=self.mps_trotter_steps,
            )
            self._mps_observables = simulator.local_expectations(self.times)

    def set_task(self, j_values: np.ndarray, h_values: np.ndarray) -> None:
        self._set_task_arrays(j_values, h_values)

    def true_theta(self) -> np.ndarray:
        return np.concatenate([self.J_true, self.h_true]).astype(np.float32)

    def reset(self, *, resample: bool = True) -> np.ndarray:
        self.step_count = 0
        if resample:
            self._resample_task()
        if self.shots_set and not self.sample_shots_each_step:
            self.current_shots = int(np.random.choice(self.shots_set))
        else:
            self.current_shots = self.default_shots
        return np.zeros(self.N, dtype=np.float32)

    def decode_action(self, action: int) -> tuple[int, int, int]:
        a = int(action)
        time_idx = a % self.M_evo
        a //= self.M_evo
        basis_idx = a % 3
        qubit_idx = a // 3
        return qubit_idx, basis_idx, time_idx

    def apply_noise_to_expectation(self, expectation: torch.Tensor, tau: float | None = None) -> torch.Tensor:
        if self.noise_model == "readout_flip":
            return expectation * (1.0 - 2.0 * self.noise_prob)
        tau_us = float(tau or 0.0) * self.time_scale_us
        t1 = max(self.t1_us, 1e-9)
        t2 = max(self.t2_us, 1e-9)
        relaxation = float(np.exp(-tau_us / t1))
        dephasing = float(np.exp(-tau_us / t2))
        attenuated = expectation * min(relaxation, dephasing)
        return (1.0 - self.readout_p01 - self.readout_p10) * attenuated + (self.readout_p10 - self.readout_p01)

    def _measure(self, psi: torch.Tensor | None, qubit_idx: int, basis_idx: int, time_idx: int, shots: int) -> np.ndarray:
        if self.simulator_backend == "mps_tebd":
            if self._mps_observables is None:
                raise RuntimeError("MPS observables are not initialized")
            E = torch.tensor(
                float(self._mps_observables[int(time_idx), int(qubit_idx), int(basis_idx)]),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            if psi is None:
                raise RuntimeError("statevector measurement requires a statevector")
            E = mean_outcome_from_state(
                psi,
                self._flip_idx[qubit_idx],
                self._sign01[qubit_idx].to(dtype=psi.real.dtype),
                self._phase[qubit_idx].to(dtype=psi.real.dtype),
                basis_idx,
            )
        E = self.apply_noise_to_expectation(E, self.times[int(time_idx)])
        shots = max(1, int(shots))
        if shots == 1:
            mean_outcome = E
        else:
            p_plus = (0.5 * (1.0 + E)).clamp(0.0, 1.0)
            n_plus = torch.distributions.Binomial(total_count=shots, probs=p_plus).sample()
            mean_outcome = (2.0 * n_plus - shots) / shots
        obs = np.zeros(self.N, dtype=np.float32)
        obs[qubit_idx] = float(mean_outcome.item())
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        qubit_idx, basis_idx, time_idx = self.decode_action(action)
        if self.sample_shots_each_step and self.shots_set:
            self.current_shots = int(np.random.choice(self.shots_set))
        psi = self.psi_list[time_idx] if self.simulator_backend == "statevector" else None
        obs = self._measure(psi, qubit_idx, basis_idx, time_idx, self.current_shots)
        self.step_count += 1
        done = self.step_count >= self.T
        info = {
            "J_true": self.J_true.copy(),
            "h_true": self.h_true.copy(),
            "qubit_idx": qubit_idx,
            "basis_idx": basis_idx,
            "time_idx": time_idx,
            "shots": int(self.current_shots),
        }
        return obs, 0.0, done, info
