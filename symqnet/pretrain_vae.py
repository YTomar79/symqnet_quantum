from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config
from .env import SpinChainEnv
from .math_utils import set_seed
from .models.vae import VariationalAutoencoder, vae_loss
from .provenance import hardware_label, stable_config_hash


@torch.no_grad()
def collect_observations(env: SpinChainEnv, samples: int, cache_path: Path | None = None) -> torch.Tensor:
    rows = []
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if cached.ndim == 2 and cached.shape[1] == env.N:
            rows = [torch.from_numpy(row.astype(np.float32, copy=False)).float() for row in cached[:samples]]
            print(f"resumed {len(rows)}/{samples} cached observations from {cache_path}", flush=True)
            if len(rows) >= samples:
                return torch.stack(rows[:samples], dim=0)
    obs = env.reset()
    progress_interval = max(1, int(samples) // 20)
    cache_interval = max(progress_interval, 1)
    for i in range(len(rows), samples):
        if i == 0 or i % progress_interval == 0:
            print(f"collect_observations {i}/{samples}", flush=True)
        if i % max(1, env.T) == 0:
            obs = env.reset()
        action = int(torch.randint(env.n_actions, ()).item())
        obs, _, done, _ = env.step(action)
        rows.append(torch.from_numpy(obs).float())
        if done:
            obs = env.reset()
        if cache_path is not None and (len(rows) % cache_interval == 0 or len(rows) == samples):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, torch.stack(rows, dim=0).cpu().numpy())
            print(f"cached {len(rows)}/{samples} observations -> {cache_path}", flush=True)
    print(f"collect_observations {samples}/{samples}", flush=True)
    return torch.stack(rows, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the SymQNet observation VAE.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--samples", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=3e-3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--observation-cache", default=None, help="Optional .npy cache for resumable VAE observation collection.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    set_seed(cfg.seed)
    started_at = time.perf_counter()
    config_hash = stable_config_hash(cfg)
    print(
        "==> Pretraining VAE "
        f"config={args.config} n_qubits={cfg.env.n_qubits} backend={cfg.env.simulator_backend} "
        f"samples={args.samples} epochs={args.epochs} batch_size={args.batch_size}",
        flush=True,
    )
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
        default_shots=cfg.env.default_shots,
        shots_set=cfg.env.shots_set,
        sample_shots_each_step=cfg.env.sample_shots_each_step,
        simulator_backend=cfg.env.simulator_backend,
        mps_bond_dim=cfg.env.mps_bond_dim,
        mps_trotter_steps=cfg.env.mps_trotter_steps,
    )

    print("==> Collecting VAE observations", flush=True)
    cache_path = Path(args.observation_cache) if args.observation_cache else Path(cfg.model.vae_checkpoint).with_suffix(".observations.npy")
    data = collect_observations(env, args.samples, cache_path).to(device)
    print(f"==> Finished observation collection: shape={tuple(data.shape)}", flush=True)
    loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=True)
    model = VariationalAutoencoder(cfg.env.n_qubits, cfg.model.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_history = []
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for (batch,) in loader:
            recon, mu, logvar, _ = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch.shape[0]
        epoch_loss = total / len(data)
        loss_history.append(float(epoch_loss))
        print(f"epoch={epoch:03d} loss={epoch_loss:.6f}")

    path = Path(cfg.model.vae_checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"input_dim": cfg.env.n_qubits, "latent_dim": cfg.model.latent_dim},
            "pretrain_metadata": {
                "config_path": str(args.config),
                "config_hash": config_hash,
                "seed": int(cfg.seed),
                "device": str(device),
                "hardware": hardware_label(device),
                "samples": int(args.samples),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "beta": float(args.beta),
                "learning_rate": float(args.lr),
                "loss_history": loss_history,
                "final_loss": loss_history[-1] if loss_history else "",
                "train_wallclock_sec": float(time.perf_counter() - started_at),
            },
        },
        path,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
