from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from .analysis.fisher import classical_fisher_for_actions, crlb_theta_mse
from .baselines import KNOWN_BASELINES, make_baseline
from .config import load_config
from .env import SpinChainEnv
from .evaluation import run_episode_with_action_policy, run_episode_with_agent, summarize
from .math_utils import set_seed
from .models.agent import SymQNetAgent
from .models.vae import VariationalAutoencoder
from .provenance import hardware_label, stable_config_hash
from .smc import SMCParticleFilter
from .task_bank import load_task_bank, sample_task_bank


def _progress_interval(total: int, env_name: str, default_divisor: int = 20) -> int:
    raw = os.environ.get(env_name)
    if raw is not None and raw.strip() != "":
        return max(0, int(raw))
    return max(1, int(total) // max(1, default_divisor))


def _should_log_episode(index: int, total: int, interval: int) -> bool:
    return interval > 0 and (index == 0 or (index + 1) % interval == 0 or index + 1 == total)


def _load_torch(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_env_and_smc(cfg, seed: int, shots: int, device: torch.device) -> tuple[SpinChainEnv, SMCParticleFilter]:
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
        default_shots=shots,
        shots_set=None,
        sample_shots_each_step=False,
        simulator_backend=cfg.env.simulator_backend,
        mps_bond_dim=cfg.env.mps_bond_dim,
        mps_trotter_steps=cfg.env.mps_trotter_steps,
    )
    smc = SMCParticleFilter(env, cfg.smc.particles, cfg.smc.ess_frac, cfg.smc.roughen_frac, device)
    smc.sim_chunk_size = cfg.smc.sim_chunk_size
    return env, smc


def load_agent(cfg, checkpoint_path: str, device: torch.device) -> tuple[SymQNetAgent, dict[str, object]]:
    vae = VariationalAutoencoder(cfg.env.n_qubits, cfg.model.latent_dim).to(device)
    agent = SymQNetAgent(
        vae,
        cfg.env.n_qubits,
        cfg.model.latent_dim,
        cfg.model.history,
        cfg.env.n_qubits * 3 * cfg.env.m_evo,
        cfg.env.m_evo,
        cfg.model.gnn_layers,
        cfg.model.graph,
        cfg.model.temporal,
        cfg.model.use_smc_feedback,
        cfg.model.belief_mode,
        cfg.model.use_vae,
        device,
    ).to(device)
    payload = _load_torch(Path(checkpoint_path), device)
    state = payload.get("model_state_dict", payload)
    agent.load_state_dict(state)
    agent.eval()
    metadata = payload if isinstance(payload, dict) else {}
    return agent, metadata


def add_crlb(env: SpinChainEnv, result) -> None:
    if not result.actions or result.true_theta is None:
        return
    theta = torch.tensor(result.true_theta, device=env.device, dtype=torch.float32)
    fim = classical_fisher_for_actions(env, theta, result.actions, result.shots)
    result.crlb_theta_mse = crlb_theta_mse(fim)


def provenance_fields(
    cfg,
    device: torch.device,
    checkpoint_path: str | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
    train_seed: int | str | None = None,
    task_bank_path: str | None = None,
) -> dict[str, object]:
    checkpoint_metadata = checkpoint_metadata or {}
    resolved_train_seed = train_seed
    if resolved_train_seed is None:
        resolved_train_seed = checkpoint_metadata.get("train_seed", "")
    return {
        "train_seed": resolved_train_seed if resolved_train_seed is not None else "",
        "checkpoint_path": checkpoint_path or "",
        "checkpoint_config_hash": checkpoint_metadata.get("config_hash", ""),
        "checkpoint_selection_metric": checkpoint_metadata.get("checkpoint_selection_metric", ""),
        "best_validation_mse": checkpoint_metadata.get("best_validation_mse", ""),
        "validation_task_bank_path": checkpoint_metadata.get("validation_task_bank_path", ""),
        "config_hash": stable_config_hash(cfg),
        "device": str(device),
        "hardware": hardware_label(device),
        "train_wallclock_sec": checkpoint_metadata.get("train_wallclock_sec", ""),
        "reward_mode": checkpoint_metadata.get("reward_mode", ""),
        "behavior_clone_checkpoint": checkpoint_metadata.get("behavior_clone_checkpoint", ""),
        "task_bank_path": task_bank_path or "",
        "policy_params": json.dumps(
            {
                "name": "symqnet",
                "action_selection": "argmax",
                "use_vae": bool(cfg.model.use_vae),
                "graph": cfg.model.graph,
                "temporal": cfg.model.temporal,
                "use_smc_feedback": bool(cfg.model.use_smc_feedback),
                "belief_mode": cfg.model.belief_mode,
            },
            sort_keys=True,
        ),
    }


def episode_row(method: str, shots: int, idx: int, result, provenance: dict[str, object]) -> dict[str, object]:
    return {
        "method": method,
        "shots": shots,
        "episode_idx": idx,
        "seed": result.seed,
        "eval_seed": result.eval_seed if result.eval_seed is not None else result.seed,
        "task_id": result.task_id if result.task_id is not None else "",
        "final_mse": result.final_mse,
        "total_info_gain": result.total_info_gain,
        "decision_ms_mean": result.decision_ms_mean,
        "decision_ms_p95": result.decision_ms_p95,
        "smc_update_ms_mean": result.smc_update_ms_mean,
        "smc_update_ms_p95": result.smc_update_ms_p95,
        "step_total_ms_mean": result.step_total_ms_mean,
        "step_total_ms_p95": result.step_total_ms_p95,
        "crlb_theta_mse": result.crlb_theta_mse if result.crlb_theta_mse is not None else "",
        "mse_crlb_ratio": result.final_mse / result.crlb_theta_mse if result.crlb_theta_mse not in {None, 0.0} else "",
        "actions": " ".join(str(a) for a in (result.actions or [])),
        "true_theta": " ".join(f"{x:.8g}" for x in (result.true_theta or [])),
        **provenance,
    }


def prepare_task_bank(path: str | None, cfg, episodes: int):
    if path is None:
        return None
    needed = len(cfg.eval.seeds) * int(episodes)
    bank_path = Path(path)
    if not bank_path.exists():
        sample_task_bank(
            bank_path,
            needed,
            cfg.env.n_qubits,
            cfg.env.j_range,
            cfg.env.h_range,
            cfg.seed,
        )
        print(f"created task bank {bank_path} with {needed} tasks")
    j, h = load_task_bank(bank_path)
    if j.shape[0] < needed:
        raise ValueError(f"Task bank {bank_path} has {j.shape[0]} tasks but evaluation needs {needed}")
    if j.shape[1] != cfg.env.n_qubits - 1 or h.shape[1] != cfg.env.n_qubits:
        raise ValueError(f"Task bank {bank_path} does not match n_qubits={cfg.env.n_qubits}: J={j.shape}, h={h.shape}")
    return j, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SymQNet baselines and trained agents.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--agent-checkpoint", default=None)
    parser.add_argument("--agent-name", default="symqnet")
    parser.add_argument("--train-seed", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--episodes-out", default=None)
    parser.add_argument("--with-crlb", action="store_true")
    parser.add_argument("--task-bank", default=None, help="Path to a .npz bank of fixed (J, h) tasks. Created if missing.")
    parser.add_argument("--baselines", nargs="+", default=None, help="Optional baseline list to evaluate.")
    parser.add_argument("--shot-budgets", nargs="+", type=int, default=None, help="Optional subset of configured shot budgets to evaluate.")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=None, help="Optional subset of evaluation seeds.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg.eval.episodes = args.episodes
    if args.eval_seeds is not None:
        cfg.eval.seeds = tuple(int(seed) for seed in args.eval_seeds)
    if args.baselines is not None:
        unknown = sorted(set(args.baselines) - set(KNOWN_BASELINES))
        if unknown:
            raise ValueError(f"Unknown baseline(s): {unknown}. Known baselines: {sorted(KNOWN_BASELINES)}")
        cfg.eval.baselines = tuple(args.baselines)
    if args.shot_budgets is not None:
        unknown_shots = sorted(set(args.shot_budgets) - set(cfg.eval.shot_budgets))
        if unknown_shots:
            raise ValueError(f"Unknown shot budget(s) for {args.config}: {unknown_shots}")
        cfg.eval.shot_budgets = tuple(args.shot_budgets)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    out_path = Path(args.out or Path(cfg.output_dir) / "eval.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    episode_rows = []
    agent = None
    checkpoint_metadata: dict[str, object] = {}
    if args.agent_checkpoint:
        agent, checkpoint_metadata = load_agent(cfg, args.agent_checkpoint, device)
    task_bank = prepare_task_bank(args.task_bank, cfg, cfg.eval.episodes)
    baseline_provenance = provenance_fields(cfg, device, task_bank_path=args.task_bank)
    agent_provenance = provenance_fields(cfg, device, args.agent_checkpoint, checkpoint_metadata, args.train_seed, args.task_bank)
    eval_seed_field = " ".join(str(seed) for seed in cfg.eval.seeds)
    episode_log_interval = _progress_interval(cfg.eval.episodes, "SYMQNET_EPISODE_LOG_INTERVAL")
    step_log_interval = _progress_interval(cfg.env.horizon, "SYMQNET_STEP_LOG_INTERVAL", default_divisor=6)
    print(
        f"==> Eval progress settings episode_interval={episode_log_interval} step_interval={step_log_interval}",
        flush=True,
    )

    for shots in cfg.eval.shot_budgets:
        if not args.skip_baselines:
            for baseline_name in cfg.eval.baselines:
                print(f"==> Evaluating baseline={baseline_name} shots={shots} episodes={cfg.eval.episodes}", flush=True)
                all_results = []
                baseline_policy_metadata = ""
                for seed_idx, seed in enumerate(cfg.eval.seeds):
                    print(f"    baseline={baseline_name} shots={shots} eval_seed={seed}", flush=True)
                    set_seed(seed)
                    env, smc = build_env_and_smc(cfg, seed, shots, device)
                    policy = make_baseline(baseline_name, env.N, env.M_evo, env.n_actions, seed)
                    baseline_policy_metadata = json.dumps(policy.metadata(), sort_keys=True)
                    for episode_idx in range(cfg.eval.episodes):
                        log_episode = _should_log_episode(episode_idx, cfg.eval.episodes, episode_log_interval)
                        task_id = seed_idx * cfg.eval.episodes + episode_idx
                        task = (task_bank[0][task_id], task_bank[1][task_id]) if task_bank is not None else None
                        if log_episode:
                            print(
                                f"    baseline={baseline_name} shots={shots} eval_seed={seed} "
                                f"episode={episode_idx + 1}/{cfg.eval.episodes} task_id={task_id}",
                                flush=True,
                            )
                        result = run_episode_with_action_policy(
                            env,
                            smc,
                            policy,
                            seed,
                            task,
                            task_id if task is not None else None,
                            progress_label=(
                                f"baseline={baseline_name} shots={shots} seed={seed} episode={episode_idx + 1}/{cfg.eval.episodes}"
                                if log_episode
                                else None
                            ),
                            progress_interval_steps=step_log_interval,
                        )
                        if args.with_crlb:
                            if log_episode:
                                print(
                                    f"      baseline={baseline_name} shots={shots} seed={seed} "
                                    f"episode={episode_idx + 1}/{cfg.eval.episodes} computing_crlb",
                                    flush=True,
                                )
                            add_crlb(env, result)
                            if log_episode:
                                print(
                                    f"      baseline={baseline_name} shots={shots} seed={seed} "
                                    f"episode={episode_idx + 1}/{cfg.eval.episodes} crlb_done",
                                    flush=True,
                                )
                        all_results.append(result)
                summary = summarize(all_results, cfg.eval.bootstrap_samples)
                row = {
                    "method": baseline_name,
                    "shots": shots,
                    "eval_seed": eval_seed_field,
                    **{**baseline_provenance, "policy_params": baseline_policy_metadata},
                    **summary,
                }
                rows.append(row)
                per_policy_provenance = {**baseline_provenance, "policy_params": baseline_policy_metadata}
                for idx, result in enumerate(all_results):
                    episode_rows.append(episode_row(baseline_name, shots, idx, result, per_policy_provenance))
                print(f"<== Finished baseline={baseline_name} shots={shots}: mse_mean={row.get('mse_mean')}", flush=True)
                print(row, flush=True)

        if agent is not None:
            print(f"==> Evaluating agent={args.agent_name} shots={shots} episodes={cfg.eval.episodes}", flush=True)
            all_results = []
            for seed_idx, seed in enumerate(cfg.eval.seeds):
                print(f"    agent={args.agent_name} shots={shots} eval_seed={seed}", flush=True)
                set_seed(seed)
                env, smc = build_env_and_smc(cfg, seed, shots, device)
                for episode_idx in range(cfg.eval.episodes):
                    log_episode = _should_log_episode(episode_idx, cfg.eval.episodes, episode_log_interval)
                    task_id = seed_idx * cfg.eval.episodes + episode_idx
                    task = (task_bank[0][task_id], task_bank[1][task_id]) if task_bank is not None else None
                    if log_episode:
                        print(
                            f"    agent={args.agent_name} shots={shots} eval_seed={seed} "
                            f"episode={episode_idx + 1}/{cfg.eval.episodes} task_id={task_id}",
                            flush=True,
                        )
                    result = run_episode_with_agent(
                        env,
                        smc,
                        agent,
                        seed,
                        task,
                        task_id if task is not None else None,
                        progress_label=(
                            f"agent={args.agent_name} shots={shots} seed={seed} episode={episode_idx + 1}/{cfg.eval.episodes}"
                            if log_episode
                            else None
                        ),
                        progress_interval_steps=step_log_interval,
                    )
                    if args.with_crlb:
                        if log_episode:
                            print(
                                f"      agent={args.agent_name} shots={shots} seed={seed} "
                                f"episode={episode_idx + 1}/{cfg.eval.episodes} computing_crlb",
                                flush=True,
                            )
                            add_crlb(env, result)
                            if log_episode:
                                print(
                                    f"      agent={args.agent_name} shots={shots} seed={seed} "
                                    f"episode={episode_idx + 1}/{cfg.eval.episodes} crlb_done",
                                    flush=True,
                                )
                    all_results.append(result)
            summary = summarize(all_results, cfg.eval.bootstrap_samples)
            row = {
                "method": args.agent_name,
                "shots": shots,
                "eval_seed": eval_seed_field,
                **agent_provenance,
                **summary,
            }
            rows.append(row)
            for idx, result in enumerate(all_results):
                episode_rows.append(episode_row(args.agent_name, shots, idx, result, agent_provenance))
            print(f"<== Finished agent={args.agent_name} shots={shots}: mse_mean={row.get('mse_mean')}", flush=True)
            print(row, flush=True)

    if not rows:
        raise SystemExit("No methods were evaluated. Provide baselines or --agent-checkpoint.")

    write_header = True
    mode = "w"
    if args.append and out_path.exists() and out_path.stat().st_size > 0:
        write_header = False
        mode = "a"

    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out_path}")

    if args.episodes_out:
        episode_path = Path(args.episodes_out)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        episode_write_header = True
        episode_mode = "w"
        if args.append and episode_path.exists() and episode_path.stat().st_size > 0:
            episode_write_header = False
            episode_mode = "a"
        with episode_path.open(episode_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            if episode_write_header:
                writer.writeheader()
            writer.writerows(episode_rows)
        print(f"saved {episode_path}")


if __name__ == "__main__":
    main()
