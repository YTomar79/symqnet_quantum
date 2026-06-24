from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "symqnet_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "symqnet_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from symqnet.baselines import make_baseline
from symqnet.env import SpinChainEnv
from symqnet.metadata import build_metadata
from symqnet.models.agent import SymQNetAgent
from symqnet.models.vae import VariationalAutoencoder
from symqnet.smc import SMCParticleFilter


def _time_policy(policy, obs: np.ndarray, smc: SMCParticleFilter, trials: int) -> tuple[float, float]:
    values = []
    for _ in range(trials):
        smc._cache.clear()
        start = time.perf_counter()
        policy.select_action(obs, smc)
        values.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(values)), float(np.percentile(values, 95))


@torch.no_grad()
def _time_agent(agent: SymQNetAgent, env: SpinChainEnv, smc: SMCParticleFilter, obs: np.ndarray, trials: int) -> tuple[float, float]:
    values = []
    posterior = smc.posterior()
    metadata = build_metadata(
        env.N,
        env.M_evo,
        agent.theta_dim,
        agent.cov_feat_dim,
        agent.use_smc_feedback,
        agent.belief_mode,
        env.device,
        None,
        posterior,
        env.shots_max,
    )
    obs_t = torch.from_numpy(obs).float().to(env.device)
    for _ in range(trials):
        agent.reset_buffer()
        start = time.perf_counter()
        dist, _ = agent(obs_t, metadata)
        int(torch.argmax(dist.probs, dim=-1).item())
        values.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(values)), float(np.percentile(values, 95))


def run_scaling(n_values: list[int], m_values: list[int], trials: int, methods: list[str], device: torch.device):
    rows = []
    for n_qubits in n_values:
        for m_evo in m_values:
            env = SpinChainEnv(n_qubits=n_qubits, m_evo=m_evo, horizon=4, seed=777, device=device, shots_set=None)
            smc = SMCParticleFilter(env, n_particles=256, device=device)
            obs = env.reset()
            smc.reset()
            n_actions = env.n_actions
            vae = VariationalAutoencoder(env.N, latent_dim=env.N).to(device)
            agent = SymQNetAgent(
                vae,
                env.N,
                latent_dim=env.N,
                history=4,
                n_actions=n_actions,
                m_evo=env.M_evo,
                gnn_layers=1,
                temporal="last",
                use_vae=False,
                device=device,
            ).to(device)
            for method in methods:
                if method == "symqnet":
                    mean_ms, p95_ms = _time_agent(agent, env, smc, obs, trials)
                else:
                    policy = make_baseline(method, env.N, env.M_evo, n_actions, seed=777)
                    mean_ms, p95_ms = _time_policy(policy, obs, smc, trials)
                rows.append(
                    {
                        "n_qubits": n_qubits,
                        "m_evo": m_evo,
                        "n_actions": n_actions,
                        "method": method,
                        "trials": trials,
                        "decision_ms_mean": mean_ms,
                        "decision_ms_p95": p95_ms,
                    }
                )
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(rows: list[dict[str, object]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        points = sorted((int(row["n_actions"]), float(row["decision_ms_mean"])) for row in rows if row["method"] == method)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=method.replace("_", " "))
    ax.set_yscale("log")
    ax.set_xlabel("Action-space size")
    ax.set_ylabel("Mean decision time (ms)")
    ax.set_title("Online decision latency scaling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure decision-time scaling across action-space sizes.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--m-values", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=["symqnet", "smc_adaptive", "bald_1step", "bald_2step"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-svg", required=True)
    args = parser.parse_args()

    rows = run_scaling(args.n_values, args.m_values, args.trials, args.methods, torch.device(args.device))
    write_csv(rows, Path(args.out_csv))
    write_svg(rows, Path(args.out_svg))
    print(f"saved {args.out_csv}")
    print(f"saved {args.out_svg}")


if __name__ == "__main__":
    main()
