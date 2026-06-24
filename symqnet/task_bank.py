from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def sample_task_bank(
    path: str | Path,
    count: int,
    n_qubits: int,
    j_range: tuple[float, float],
    h_range: tuple[float, float],
    seed: int,
    overwrite: bool = False,
) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        return path
    rng = np.random.default_rng(int(seed))
    j = rng.uniform(*j_range, size=(int(count), int(n_qubits) - 1)).astype(np.float32)
    h = rng.uniform(*h_range, size=(int(count), int(n_qubits))).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        J=j,
        h=h,
        n_qubits=np.array(int(n_qubits), dtype=np.int64),
        seed=np.array(int(seed), dtype=np.int64),
        j_range=np.array(j_range, dtype=np.float32),
        h_range=np.array(h_range, dtype=np.float32),
    )
    return path


def load_task_bank(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path)) as data:
        j = np.asarray(data["J"], dtype=np.float32)
        h = np.asarray(data["h"], dtype=np.float32)
    if j.ndim != 2 or h.ndim != 2 or j.shape[0] != h.shape[0] or h.shape[1] != j.shape[1] + 1:
        raise ValueError(f"Invalid task bank shapes: J={j.shape}, h={h.shape}")
    return j, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-sample fixed Hamiltonian task banks for paired evaluation.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--n-qubits", type=int, default=5)
    parser.add_argument("--j-range", type=float, nargs=2, default=(0.5, 1.5))
    parser.add_argument("--h-range", type=float, nargs=2, default=(0.5, 1.5))
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = sample_task_bank(args.out, args.count, args.n_qubits, tuple(args.j_range), tuple(args.h_range), args.seed, args.overwrite)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
