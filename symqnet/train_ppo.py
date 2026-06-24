from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config, save_config
from .env import SpinChainEnv
from .math_utils import gaussian_entropy_from_cov, set_seed
from .metadata import build_metadata
from .models.agent import SymQNetAgent
from .models.vae import VariationalAutoencoder
from .provenance import hardware_label, stable_config_hash
from .smc import SMCParticleFilter
from .task_bank import load_task_bank, sample_task_bank


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, gamma: float, lam: float):
    adv = torch.zeros((), device=values.device)
    advs = torch.zeros_like(rewards)
    for t in reversed(range(rewards.numel())):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == rewards.numel() - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        adv = delta + gamma * lam * nonterminal * adv
        advs[t] = adv
    returns = advs + values
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    return returns.detach(), advs.detach()


@torch.no_grad()
def bootstrap_value(agent: SymQNetAgent, obs: np.ndarray, metadata: torch.Tensor, device: torch.device) -> torch.Tensor:
    history = list(agent._history)
    obs_t = torch.from_numpy(obs).float().to(device)
    _, value = agent(obs_t, metadata)
    agent._history = history
    return value.detach()


def load_vae(
    path: str,
    input_dim: int,
    latent_dim: int,
    device: torch.device,
    *,
    allow_random: bool = False,
) -> VariationalAutoencoder:
    vae = VariationalAutoencoder(input_dim, latent_dim).to(device)
    checkpoint = Path(path)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device)
        state = payload.get("model_state_dict", payload)
        vae.load_state_dict(state)
    else:
        if not allow_random:
            raise FileNotFoundError(
                f"VAE checkpoint {checkpoint} is required because model.use_vae=true. "
                "Run symqnet.pretrain_vae first, or pass --allow-random-vae only for smoke/debug runs."
            )
        print(f"warning: {checkpoint} not found; using a randomly initialized frozen VAE")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def _make_env_and_smc(cfg, seed: int, device: torch.device, shots_set=None) -> tuple[SpinChainEnv, SMCParticleFilter]:
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
        seed=seed,
        device=device,
        j_range=cfg.env.j_range,
        h_range=cfg.env.h_range,
        default_shots=cfg.env.default_shots,
        shots_set=shots_set,
        sample_shots_each_step=False,
        simulator_backend=cfg.env.simulator_backend,
        mps_bond_dim=cfg.env.mps_bond_dim,
        mps_trotter_steps=cfg.env.mps_trotter_steps,
    )
    smc = SMCParticleFilter(env, cfg.smc.particles, cfg.smc.ess_frac, cfg.smc.roughen_frac, device)
    smc.sim_chunk_size = cfg.smc.sim_chunk_size
    return env, smc


def _rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def prepare_validation_task_bank(path: Path, cfg, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sample_task_bank(path, count, cfg.env.n_qubits, cfg.env.j_range, cfg.env.h_range, seed)
    j, h = load_task_bank(path)
    if j.shape[0] < count:
        raise ValueError(f"Validation task bank {path} has {j.shape[0]} tasks but {count} are required")
    return j[:count], h[:count]


@torch.no_grad()
def evaluate_validation_mse(
    agent: SymQNetAgent,
    cfg,
    device: torch.device,
    tasks: tuple[np.ndarray, np.ndarray],
    validation_seed: int,
) -> float:
    saved_history = list(agent._history)
    rng = _rng_state(device)
    try:
        set_seed(validation_seed)
        env, smc = _make_env_and_smc(cfg, validation_seed, device, shots_set=None)
        mses = []
        for task_idx in range(tasks[0].shape[0]):
            env.set_task(tasks[0][task_idx], tasks[1][task_idx])
            obs = env.reset(resample=False)
            smc.reset()
            agent.reset_buffer()
            posterior = smc.posterior()
            true_theta = torch.from_numpy(env.true_theta()).float().to(device)
            prev_info = None
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
                dist, _ = agent(obs_t, metadata)
                action = int(torch.argmax(dist.probs, dim=-1).item())
                obs, _, done, info = env.step(action)
                posterior = smc.update(torch.from_numpy(obs).float().to(device), info)
                prev_info = info
            mses.append(float(torch.mean((posterior.mean - true_theta) ** 2).item()))
        return float(np.mean(mses)) if mses else float("inf")
    finally:
        agent._history = saved_history
        _restore_rng_state(rng)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SymQNet PPO.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-random-vae", action="store_true", help="Debug only: allow missing VAE checkpoints.")
    parser.add_argument("--validation-task-bank", default=None)
    parser.add_argument("--validation-episodes", type=int, default=16)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-seed", type=int, default=20260516)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.updates is not None:
        cfg.ppo.updates = args.updates
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    set_seed(cfg.seed)
    config_hash = stable_config_hash(cfg)
    hardware = hardware_label(device)
    train_start = time.perf_counter()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.json")

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
    smc = SMCParticleFilter(env, cfg.smc.particles, cfg.smc.ess_frac, cfg.smc.roughen_frac, device)
    smc.sim_chunk_size = cfg.smc.sim_chunk_size
    vae = load_vae(
        cfg.model.vae_checkpoint,
        cfg.env.n_qubits,
        cfg.model.latent_dim,
        device,
        allow_random=(args.allow_random_vae or not cfg.model.use_vae),
    )
    agent = SymQNetAgent(
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
    if cfg.ppo.behavior_clone_checkpoint:
        payload = torch.load(cfg.ppo.behavior_clone_checkpoint, map_location=device)
        state = payload.get("model_state_dict", payload)
        agent.load_state_dict(state, strict=False)
    optimizer = torch.optim.Adam((p for p in agent.parameters() if p.requires_grad), lr=cfg.ppo.learning_rate)
    if cfg.ppo.reward_mode not in {"info_gain", "oracle_mse_delta"}:
        raise ValueError(f"Unknown PPO reward_mode: {cfg.ppo.reward_mode}")

    obs = env.reset()
    smc.reset()
    posterior = smc.posterior()
    prev_info = None
    agent.reset_buffer()
    true_theta = torch.from_numpy(env.true_theta()).float().to(device)
    best_mse = float("inf")
    best_validation_mse = float("inf")
    best_path = out_dir / "best_agent.pt"
    validation_history: list[dict[str, float | int]] = []
    validation_episodes = max(0, int(args.validation_episodes))
    validation_interval = max(1, int(args.validation_interval))
    validation_task_bank = Path(args.validation_task_bank) if args.validation_task_bank else out_dir / "validation_task_bank.npz"
    validation_tasks = None
    if validation_episodes > 0:
        validation_tasks = prepare_validation_task_bank(validation_task_bank, cfg, validation_episodes, args.validation_seed)

    update_log_interval = max(1, int(os.environ.get("SYMQNET_TRAIN_UPDATE_LOG_INTERVAL", "1")))
    rollout_log_interval = max(0, int(os.environ.get("SYMQNET_ROLLOUT_LOG_INTERVAL", str(max(1, cfg.ppo.rollout_steps // 4)))))
    print(
        "==> Training SymQNet "
        f"config={args.config} seed={cfg.seed} updates={cfg.ppo.updates} rollout_steps={cfg.ppo.rollout_steps} "
        f"output_dir={out_dir} update_log_interval={update_log_interval} rollout_log_interval={rollout_log_interval}",
        flush=True,
    )

    for update in range(1, cfg.ppo.updates + 1):
        log_update = update == 1 or update % update_log_interval == 0 or update == cfg.ppo.updates
        if log_update:
            print(f"==> update={update:05d}/{cfg.ppo.updates:05d} collecting_rollout", flush=True)
        obs_buf, meta_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []
        final_mses = []
        info_gains = []
        last_value = torch.zeros((), device=device)

        while len(rew_buf) < cfg.ppo.rollout_steps:
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

            with torch.no_grad():
                dist, value = agent(obs_t, metadata)

            action = dist.sample()
            obs_next, _, done, info = env.step(int(action.item()))
            obs_next_t = torch.from_numpy(obs_next).float().to(device)
            h_prior = gaussian_entropy_from_cov(posterior.cov)
            mse_prior = torch.mean((posterior.mean - true_theta) ** 2)
            posterior = smc.update(obs_next_t, info)
            if cfg.ppo.reward_mode == "oracle_mse_delta":
                reward = float((mse_prior - torch.mean((posterior.mean - true_theta) ** 2)).item())
            else:
                reward = float(torch.clamp(h_prior - gaussian_entropy_from_cov(posterior.cov), min=0.0).item())
            info_gains.append(reward)
            obs_buf.append(obs_t.detach())
            meta_buf.append(metadata.detach())
            act_buf.append(action.detach())
            logp_buf.append(dist.log_prob(action).detach())
            val_buf.append(value.detach())
            rew_buf.append(reward)
            done_buf.append(1.0 if done else 0.0)
            if log_update and rollout_log_interval > 0 and (
                len(rew_buf) == 1 or len(rew_buf) % rollout_log_interval == 0 or len(rew_buf) >= cfg.ppo.rollout_steps
            ):
                current_mse = float(torch.mean((posterior.mean - true_theta) ** 2).item())
                print(
                    f"    update={update:05d} rollout={len(rew_buf)}/{cfg.ppo.rollout_steps} "
                    f"reward={reward:.6g} mse={current_mse:.6g} done={done}",
                    flush=True,
                )

            if done:
                final_mses.append(float(torch.mean((posterior.mean - true_theta) ** 2).item()))

                obs = env.reset()
                smc.reset()
                posterior = smc.posterior()
                prev_info = None
                agent.reset_buffer()
                true_theta = torch.from_numpy(env.true_theta()).float().to(device)
                last_value = torch.zeros((), device=device)
            else:
                obs = obs_next
                prev_info = info
                next_metadata = build_metadata(
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
                last_value = bootstrap_value(agent, obs, next_metadata, device)

        obs_b = torch.stack(obs_buf)
        meta_b = torch.stack(meta_buf)
        act_b = torch.stack(act_buf).long()
        old_logp = torch.stack(logp_buf)
        old_val = torch.stack(val_buf)
        rewards = torch.tensor(rew_buf, device=device, dtype=torch.float32)
        dones = torch.tensor(done_buf, device=device, dtype=torch.float32)
        returns, advs = compute_gae(rewards, old_val, dones, last_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)

        with torch.no_grad():
            z_meta = agent.encode_observation(obs_b, meta_b)
        windows = torch.zeros(z_meta.shape[0], agent.history, z_meta.shape[1], device=device)
        ep_start = 0
        for t in range(z_meta.shape[0]):
            if t > 0 and dones[t - 1] == 1:
                ep_start = t
            seq = z_meta[max(ep_start, t - agent.history + 1) : t + 1]
            windows[t, -seq.shape[0] :, :] = seq

        losses = []
        minibatch_size = cfg.ppo.rollout_steps // cfg.ppo.minibatches
        for _ in range(cfg.ppo.ppo_epochs):
            order = torch.randperm(windows.shape[0], device=device)
            for start in range(0, windows.shape[0], minibatch_size):
                idx = order[start : start + minibatch_size]
                dist, value = agent.forward_window(windows[idx])
                new_logp = dist.log_prob(act_b[idx])
                ratio = torch.exp(new_logp - old_logp[idx])
                surr1 = ratio * advs[idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.ppo.clip_eps, 1.0 + cfg.ppo.clip_eps) * advs[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, returns[idx])
                entropy_loss = -dist.entropy().mean()
                loss = policy_loss + cfg.ppo.value_coef * value_loss + cfg.ppo.entropy_coef * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.clip_grad)
                optimizer.step()
                losses.append(float(loss.item()))

        mean_mse = float(np.mean(final_mses)) if final_mses else float("nan")
        validation_mse = float("nan")
        should_validate = validation_tasks is not None and (update == 1 or update == cfg.ppo.updates or update % validation_interval == 0)
        if should_validate:
            validation_mse = evaluate_validation_mse(agent, cfg, device, validation_tasks, args.validation_seed)
            validation_history.append({"update": int(update), "validation_mse": float(validation_mse)})
        if should_validate and validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_mse = validation_mse
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "config": cfg,
                    "train_seed": int(cfg.seed),
                    "config_hash": config_hash,
                    "device": str(device),
                    "hardware": hardware,
                    "best_mse": best_mse,
                    "best_validation_mse": best_validation_mse,
                    "checkpoint_selection_metric": "heldout_validation_mse",
                    "validation_task_bank_path": str(validation_task_bank),
                    "validation_episodes": validation_episodes,
                    "validation_interval": validation_interval,
                    "validation_seed": int(args.validation_seed),
                    "train_wallclock_sec": float(time.perf_counter() - train_start),
                    "reward_mode": cfg.ppo.reward_mode,
                    "behavior_clone_checkpoint": cfg.ppo.behavior_clone_checkpoint,
                },
                best_path,
            )
        if update == 1 or update % 10 == 0:
            val_text = f" val_mse={validation_mse:.6f}" if should_validate else ""
            print(
                f"update={update:05d} loss={np.mean(losses):.4f} "
                f"ig={np.mean(info_gains):.4f} final_mse={mean_mse:.6f}{val_text} best={best_mse:.6f}"
            )

    train_wallclock_sec = float(time.perf_counter() - train_start)
    payload_meta = {
        "train_seed": int(cfg.seed),
        "config_hash": config_hash,
        "device": str(device),
        "hardware": hardware,
        "best_mse": best_mse,
        "best_validation_mse": best_validation_mse,
        "checkpoint_selection_metric": "heldout_validation_mse" if validation_tasks is not None else "",
        "validation_task_bank_path": str(validation_task_bank) if validation_tasks is not None else "",
        "validation_episodes": validation_episodes,
        "validation_interval": validation_interval,
        "validation_seed": int(args.validation_seed),
        "train_wallclock_sec": train_wallclock_sec,
        "reward_mode": cfg.ppo.reward_mode,
        "behavior_clone_checkpoint": cfg.ppo.behavior_clone_checkpoint,
        "validation_history": validation_history,
    }
    torch.save({"model_state_dict": agent.state_dict(), "config": cfg, **payload_meta}, out_dir / "last_agent.pt")
    if best_path.exists():
        try:
            best_payload = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            best_payload = torch.load(best_path, map_location=device)
        best_payload.update(payload_meta)
        torch.save(best_payload, best_path)
    metrics_path = out_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(payload_meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out_dir / 'last_agent.pt'}")
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
