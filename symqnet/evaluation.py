from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from .baselines import ActionPolicy
from .env import SpinChainEnv
from .math_utils import gaussian_entropy_from_cov
from .metadata import build_metadata
from .models.agent import SymQNetAgent
from .smc import SMCParticleFilter


@dataclass
class EpisodeResult:
    final_mse: float
    total_info_gain: float
    steps: int
    shots: int
    seed: int | None = None
    eval_seed: int | None = None
    decision_ms_mean: float = 0.0
    decision_ms_p95: float = 0.0
    smc_update_ms_mean: float = 0.0
    smc_update_ms_p95: float = 0.0
    step_total_ms_mean: float = 0.0
    step_total_ms_p95: float = 0.0
    actions: list[int] | None = None
    true_theta: list[float] | None = None
    crlb_theta_mse: float | None = None
    task_id: int | None = None


@torch.no_grad()
def run_episode_with_action_policy(
    env: SpinChainEnv,
    smc: SMCParticleFilter,
    policy: ActionPolicy,
    seed: int | None = None,
    task: tuple[np.ndarray, np.ndarray] | None = None,
    task_id: int | None = None,
    progress_label: str | None = None,
    progress_interval_steps: int = 0,
) -> EpisodeResult:
    if task is not None:
        env.set_task(task[0], task[1])
    obs = env.reset(resample=task is None)
    smc.reset()
    policy.reset()
    posterior = smc.posterior()
    true_theta = torch.from_numpy(env.true_theta()).float().to(smc.device)
    total_ig = 0.0
    decision_ms = []
    smc_update_ms = []
    step_total_ms = []
    actions = []
    done = False
    steps = 0
    if progress_label:
        print(f"      {progress_label} start horizon={env.T}", flush=True)
    while not done:
        step_start = time.perf_counter()
        start = time.perf_counter()
        action = policy.select_action(obs, smc)
        decision_ms.append((time.perf_counter() - start) * 1000.0)
        actions.append(int(action))
        obs, _, done, info = env.step(action)
        obs_t = torch.from_numpy(obs).float().to(smc.device)
        h_prior = gaussian_entropy_from_cov(posterior.cov)
        smc_start = time.perf_counter()
        posterior = smc.update(obs_t, info)
        smc_update_ms.append((time.perf_counter() - smc_start) * 1000.0)
        total_ig += float(torch.clamp(h_prior - gaussian_entropy_from_cov(posterior.cov), min=0.0).item())
        step_total_ms.append((time.perf_counter() - step_start) * 1000.0)
        steps += 1
        if progress_label and progress_interval_steps > 0 and (steps == 1 or steps % progress_interval_steps == 0 or done):
            print(
                f"      {progress_label} step={steps}/{env.T} action={int(action)} "
                f"decision_ms={decision_ms[-1]:.1f} smc_ms={smc_update_ms[-1]:.1f} "
                f"step_ms={step_total_ms[-1]:.1f}",
                flush=True,
            )
    mse = torch.mean((posterior.mean - true_theta) ** 2).item()
    if progress_label:
        print(f"      {progress_label} done mse={float(mse):.6g} info_gain={total_ig:.6g}", flush=True)
    return EpisodeResult(
        float(mse),
        float(total_ig),
        steps,
        int(env.current_shots),
        seed,
        seed,
        float(np.mean(decision_ms)) if decision_ms else 0.0,
        float(np.percentile(decision_ms, 95)) if decision_ms else 0.0,
        float(np.mean(smc_update_ms)) if smc_update_ms else 0.0,
        float(np.percentile(smc_update_ms, 95)) if smc_update_ms else 0.0,
        float(np.mean(step_total_ms)) if step_total_ms else 0.0,
        float(np.percentile(step_total_ms, 95)) if step_total_ms else 0.0,
        actions,
        [float(x) for x in true_theta.detach().cpu().tolist()],
        task_id=task_id,
    )


@torch.no_grad()
def run_episode_with_agent(
    env: SpinChainEnv,
    smc: SMCParticleFilter,
    agent: SymQNetAgent,
    seed: int | None = None,
    task: tuple[np.ndarray, np.ndarray] | None = None,
    task_id: int | None = None,
    progress_label: str | None = None,
    progress_interval_steps: int = 0,
) -> EpisodeResult:
    if task is not None:
        env.set_task(task[0], task[1])
    obs = env.reset(resample=task is None)
    smc.reset()
    agent.reset_buffer()
    posterior = smc.posterior()
    true_theta = torch.from_numpy(env.true_theta()).float().to(env.device)
    prev_info = None
    total_ig = 0.0
    decision_ms = []
    smc_update_ms = []
    step_total_ms = []
    actions = []
    done = False
    steps = 0
    if progress_label:
        print(f"      {progress_label} start horizon={env.T}", flush=True)

    while not done:
        step_start = time.perf_counter()
        obs_t = torch.from_numpy(obs).float().to(env.device)
        metadata = build_metadata(
            env.N,
            env.M_evo,
            agent.theta_dim,
            agent.cov_feat_dim,
            agent.use_smc_feedback,
            agent.belief_mode,
            env.device,
            prev_info,
            posterior,
            env.shots_max,
        )
        start = time.perf_counter()
        dist, _ = agent(obs_t, metadata)
        action = int(torch.argmax(dist.probs, dim=-1).item())
        decision_ms.append((time.perf_counter() - start) * 1000.0)
        actions.append(int(action))
        obs, _, done, info = env.step(action)

        obs_next = torch.from_numpy(obs).float().to(env.device)
        h_prior = gaussian_entropy_from_cov(posterior.cov)
        smc_start = time.perf_counter()
        posterior = smc.update(obs_next, info)
        smc_update_ms.append((time.perf_counter() - smc_start) * 1000.0)
        total_ig += float(torch.clamp(h_prior - gaussian_entropy_from_cov(posterior.cov), min=0.0).item())
        prev_info = info
        step_total_ms.append((time.perf_counter() - step_start) * 1000.0)
        steps += 1
        if progress_label and progress_interval_steps > 0 and (steps == 1 or steps % progress_interval_steps == 0 or done):
            print(
                f"      {progress_label} step={steps}/{env.T} action={int(action)} "
                f"decision_ms={decision_ms[-1]:.1f} smc_ms={smc_update_ms[-1]:.1f} "
                f"step_ms={step_total_ms[-1]:.1f}",
                flush=True,
            )

    mse = torch.mean((posterior.mean - true_theta) ** 2).item()
    if progress_label:
        print(f"      {progress_label} done mse={float(mse):.6g} info_gain={total_ig:.6g}", flush=True)
    return EpisodeResult(
        float(mse),
        float(total_ig),
        steps,
        int(env.current_shots),
        seed,
        seed,
        float(np.mean(decision_ms)) if decision_ms else 0.0,
        float(np.percentile(decision_ms, 95)) if decision_ms else 0.0,
        float(np.mean(smc_update_ms)) if smc_update_ms else 0.0,
        float(np.percentile(smc_update_ms, 95)) if smc_update_ms else 0.0,
        float(np.mean(step_total_ms)) if step_total_ms else 0.0,
        float(np.percentile(step_total_ms, 95)) if step_total_ms else 0.0,
        actions,
        [float(x) for x in true_theta.detach().cpu().tolist()],
        task_id=task_id,
    )


def _bootstrap_ci(values: np.ndarray, samples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if values.size <= 1 or samples <= 0:
        value = float(values.mean()) if values.size else float("nan")
        return value, value
    rng = np.random.default_rng(20260516)
    boot = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        boot[i] = rng.choice(values, size=values.size, replace=True).mean()
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def summarize(results: list[EpisodeResult], bootstrap_samples: int = 1000) -> dict[str, float | str]:
    mses = np.array([r.final_mse for r in results], dtype=np.float64)
    info_gain = np.array([r.total_info_gain for r in results], dtype=np.float64)
    decision_mean = np.array([r.decision_ms_mean for r in results], dtype=np.float64)
    decision_p95 = np.array([r.decision_ms_p95 for r in results], dtype=np.float64)
    smc_update_mean = np.array([r.smc_update_ms_mean for r in results], dtype=np.float64)
    smc_update_p95 = np.array([r.smc_update_ms_p95 for r in results], dtype=np.float64)
    step_total_mean = np.array([r.step_total_ms_mean for r in results], dtype=np.float64)
    step_total_p95 = np.array([r.step_total_ms_p95 for r in results], dtype=np.float64)
    crlbs = np.array([r.crlb_theta_mse for r in results if r.crlb_theta_mse is not None], dtype=np.float64)
    mse_crlb = np.array([r.final_mse / r.crlb_theta_mse for r in results if r.crlb_theta_mse not in {None, 0.0}], dtype=np.float64)
    ci_lo, ci_hi = _bootstrap_ci(mses, bootstrap_samples)
    seed_groups: dict[int, list[float]] = {}
    for result in results:
        if result.eval_seed is not None:
            seed_groups.setdefault(int(result.eval_seed), []).append(result.final_mse)
        elif result.seed is not None:
            seed_groups.setdefault(int(result.seed), []).append(result.final_mse)
    per_seed = ";".join(f"{seed}:{np.mean(vals):.8g}" for seed, vals in sorted(seed_groups.items()))
    return {
        "episodes": float(len(results)),
        "mse_mean": float(mses.mean()),
        "mse_mean_ci95_lo": ci_lo,
        "mse_mean_ci95_hi": ci_hi,
        "mse_median": float(np.median(mses)),
        "mse_iqr": float(np.percentile(mses, 75) - np.percentile(mses, 25)),
        "info_gain_mean": float(info_gain.mean()),
        "decision_ms_mean": float(decision_mean.mean()),
        "decision_ms_p95": float(decision_p95.mean()),
        "smc_update_ms_mean": float(smc_update_mean.mean()),
        "smc_update_ms_p95": float(smc_update_p95.mean()),
        "step_total_ms_mean": float(step_total_mean.mean()),
        "step_total_ms_p95": float(step_total_p95.mean()),
        "crlb_theta_mse_mean": float(crlbs.mean()) if crlbs.size else "",
        "mse_crlb_ratio_mean": float(mse_crlb.mean()) if mse_crlb.size else "",
        "per_seed_mse_mean": per_seed,
    }
