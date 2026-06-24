from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from symqnet.backends import TFIMMpsTebdSimulator
from symqnet.env import SpinChainEnv
from symqnet.math_utils import mean_outcome_from_state


@torch.no_grad()
def dense_observables(env: SpinChainEnv) -> np.ndarray:
    out = np.empty((env.M_evo, env.N, 3), dtype=np.float32)
    for time_idx, psi in enumerate(env.psi_list):
        for q in range(env.N):
            for b in range(3):
                value = mean_outcome_from_state(
                    psi,
                    env._flip_idx[q],
                    env._sign01[q].to(dtype=psi.real.dtype),
                    env._phase[q].to(dtype=psi.real.dtype),
                    b,
                )
                out[time_idx, q, b] = float(value.item())
    return out


def validate_mps(
    n_values: list[int],
    *,
    m_evo: int = 5,
    tasks: int = 3,
    seed: int = 20260516,
    bond_dim: int = 32,
    trotter_steps: int = 8,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = []
    errors = []
    for n_qubits in n_values:
        env = SpinChainEnv(
            n_qubits=n_qubits,
            m_evo=m_evo,
            horizon=2,
            noise_prob=0.0,
            seed=seed + n_qubits,
            shots_set=None,
            simulator_backend="statevector",
        )
        for task_idx in range(tasks):
            j = rng.uniform(0.5, 1.5, size=n_qubits - 1).astype(np.float32)
            h = rng.uniform(0.5, 1.5, size=n_qubits).astype(np.float32)
            env.set_task(j, h)
            dense = dense_observables(env)
            mps = TFIMMpsTebdSimulator(
                n_qubits,
                j,
                h,
                bond_dim=bond_dim,
                trotter_steps=trotter_steps,
            ).local_expectations(env.times)
            err = np.abs(dense - mps)
            row = {
                "n_qubits": int(n_qubits),
                "task_idx": int(task_idx),
                "mean_abs_error": float(err.mean()),
                "max_abs_error": float(err.max()),
            }
            rows.append(row)
            errors.append(row["mean_abs_error"])
    max_mean = float(max(errors)) if errors else float("inf")
    return {
        "n_values": [int(n) for n in n_values],
        "tasks": int(tasks),
        "m_evo": int(m_evo),
        "bond_dim": int(bond_dim),
        "trotter_steps": int(trotter_steps),
        "mean_abs_error": float(np.mean(errors)) if errors else float("inf"),
        "max_mean_abs_error": max_mean,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MPS/TEBD local observables against dense statevector TFIM.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--m-evo", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--bond-dim", type=int, default=32)
    parser.add_argument("--trotter-steps", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = validate_mps(
        args.n_values,
        m_evo=args.m_evo,
        tasks=args.tasks,
        seed=args.seed,
        bond_dim=args.bond_dim,
        trotter_steps=args.trotter_steps,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
