from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from .baselines import make_baseline
from .config import load_config
from .env import SpinChainEnv
from .metadata import build_metadata
from .models.agent import SymQNetAgent
from .smc import SMCParticleFilter
from .train_ppo import load_vae


def _device(cfg) -> torch.device:
    return torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))


def _make_env(cfg, device: torch.device) -> SpinChainEnv:
    return SpinChainEnv(
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


def _make_agent(cfg, env: SpinChainEnv, device: torch.device) -> SymQNetAgent:
    vae = load_vae(cfg.model.vae_checkpoint, cfg.env.n_qubits, cfg.model.latent_dim, device, allow_random=not cfg.model.use_vae)
    return SymQNetAgent(
        vae,
        cfg.env.n_qubits,
        cfg.model.latent_dim,
        cfg.model.history,
        env.n_actions,
        cfg.env.m_evo,
        cfg.model.gnn_layers,
        cfg.model.graph,
        cfg.model.temporal,
        cfg.model.use_smc_feedback,
        cfg.model.belief_mode,
        cfg.model.use_vae,
        device,
    ).to(device)


@torch.no_grad()
def collect_windows(cfg, agent: SymQNetAgent, env: SpinChainEnv, episodes: int, policy_name: str, device: torch.device):
    smc = SMCParticleFilter(env, cfg.smc.particles, cfg.smc.ess_frac, cfg.smc.roughen_frac, device)
    smc.sim_chunk_size = cfg.smc.sim_chunk_size
    policy = make_baseline(policy_name, env.N, env.M_evo, env.n_actions, cfg.seed)
    windows = []
    actions = []
    for _ in range(episodes):
        obs = env.reset()
        smc.reset()
        policy.reset()
        agent.reset_buffer()
        posterior = smc.posterior()
        prev_info = None
        z_window = []
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).float().to(device)
            metadata = build_metadata(
                env.N,
                env.M_evo,
                agent.theta_dim,
                agent.cov_feat_dim,
                agent.use_smc_feedback,
                agent.belief_mode,
                device,
                prev_info,
                posterior,
                env.shots_max,
            )
            z_window.append(agent.encode_observation(obs_t, metadata).detach())
            action = policy.select_action(obs, smc)
            padded = torch.zeros(agent.history, z_window[-1].numel(), device=device)
            seq = torch.stack(z_window[-agent.history :], dim=0)
            padded[-seq.shape[0] :, :] = seq
            windows.append(padded)
            actions.append(int(action))
            obs, _, done, info = env.step(action)
            posterior = smc.update(torch.from_numpy(obs).float().to(device), info)
            prev_info = info
    return torch.stack(windows), torch.tensor(actions, device=device, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone SymQNet from a BALD-style policy.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--policy", default="bald_2step")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _device(cfg)
    started = time.perf_counter()
    env = _make_env(cfg, device)
    agent = _make_agent(cfg, env, device)
    windows, actions = collect_windows(cfg, agent, env, args.episodes, args.policy, device)
    optimizer = torch.optim.Adam((p for p in agent.parameters() if p.requires_grad), lr=cfg.ppo.learning_rate)
    for epoch in range(1, args.epochs + 1):
        dist, _ = agent.forward_window(windows)
        loss = -dist.log_prob(actions).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.clip_grad)
        optimizer.step()
        print(f"bc_epoch={epoch:03d} loss={float(loss.item()):.6f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "behavior_clone_metadata": {
                "config": args.config,
                "episodes": int(args.episodes),
                "epochs": int(args.epochs),
                "policy": args.policy,
                "train_wallclock_sec": float(time.perf_counter() - started),
            },
        },
        out,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
