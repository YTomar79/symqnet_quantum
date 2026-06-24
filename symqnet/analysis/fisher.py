from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ..config import load_config
from ..env import SpinChainEnv


def _kron_ops(ops: list[torch.Tensor]) -> torch.Tensor:
    out = ops[0]
    for item in ops[1:]:
        out = torch.kron(out, item)
    return out


def hamiltonian_terms(env: SpinChainEnv) -> torch.Tensor:
    terms = []
    for i in range(env.N - 1):
        ops = [env.I] * env.N
        ops[i] = env.Z
        ops[i + 1] = env.Z
        terms.append(_kron_ops(ops))
    for i in range(env.N):
        ops = [env.I] * env.N
        ops[i] = env.X
        terms.append(_kron_ops(ops))
    return torch.stack(terms, dim=0)


def differentiable_mean_outcome(env: SpinChainEnv, psi: torch.Tensor, qubit_idx: int, basis_idx: int, tau: float) -> torch.Tensor:
    flip = env._flip_idx[qubit_idx]
    sign01 = env._sign01[qubit_idx].to(device=psi.device, dtype=psi.real.dtype)
    phase = env._phase[qubit_idx].to(device=psi.device, dtype=psi.real.dtype)
    psi_flip = psi[flip]
    if basis_idx == 2:
        E = ((psi.abs() ** 2).to(sign01.dtype) * sign01).sum()
    elif basis_idx == 0:
        E = -(psi.conj() * psi_flip).sum().real
    elif basis_idx == 1:
        E = -(psi.conj() * (torch.tensor(1j, device=psi.device, dtype=psi.dtype) * phase) * psi_flip).sum().real
    else:
        raise ValueError("basis_idx must be 0, 1, or 2")
    return env.apply_noise_to_expectation(E, tau)


def classical_fisher_for_actions(
    env: SpinChainEnv,
    theta: torch.Tensor,
    actions: list[int],
    shots: int,
) -> torch.Tensor:
    terms = hamiltonian_terms(env).to(theta.device)
    theta = theta.detach().clone().to(torch.float32).requires_grad_(True)
    fim = torch.zeros(theta.numel(), theta.numel(), device=theta.device, dtype=torch.float32)
    for action in actions:
        q, b, t = env.decode_action(action)
        tau = float(env.times[t])
        H = torch.einsum("k,kij->ij", theta.to(terms.dtype), terms)
        U = torch.matrix_exp((-1j) * H * tau)
        psi = U[:, 0]
        E = differentiable_mean_outcome(env, psi, q, b, tau)
        grad = torch.autograd.grad(E, theta, retain_graph=False, create_graph=False)[0]
        var = ((1.0 - E.detach() * E.detach()).clamp_min(1e-6) / max(1, shots)).float()
        fim = fim + torch.outer(grad, grad) / var
    return fim


def crlb_theta_mse(fim: torch.Tensor, ridge: float = 1e-8) -> float:
    eye = torch.eye(fim.shape[0], device=fim.device, dtype=fim.dtype)
    cov_lb = torch.linalg.pinv(fim + ridge * eye)
    return float(torch.trace(cov_lb).item() / fim.shape[0])


def fixed_protocol_actions(env: SpinChainEnv) -> list[int]:
    return [(q * 3 + b) * env.M_evo for b in range(3) for q in range(env.N)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a CRLB theta-MSE reference for a fixed protocol.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    env = SpinChainEnv(
        n_qubits=cfg.env.n_qubits,
        m_evo=cfg.env.m_evo,
        horizon=cfg.env.horizon,
        hamiltonian=cfg.env.hamiltonian,
        noise_prob=cfg.env.noise_prob,
        noise_model=cfg.env.noise_model,
        readout_p01=cfg.env.readout_p01,
        readout_p10=cfg.env.readout_p10,
        t1_us=cfg.env.t1_us,
        t2_us=cfg.env.t2_us,
        time_scale_us=cfg.env.time_scale_us,
        seed=cfg.seed,
        device=device,
        j_range=cfg.env.j_range,
        h_range=cfg.env.h_range,
        default_shots=args.shots,
        shots_set=None,
    )
    theta = torch.from_numpy(env.true_theta()).to(device)
    actions = (fixed_protocol_actions(env) * ((env.T // max(1, 3 * env.N)) + 1))[: env.T]
    fim = classical_fisher_for_actions(env, theta, actions, args.shots)
    row = {"protocol": "fixed", "shots": args.shots, "crlb_theta_mse": crlb_theta_mse(fim)}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"saved {out}")
    else:
        print(row)


if __name__ == "__main__":
    main()
