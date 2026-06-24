from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class EnvConfig:
    n_qubits: int = 5
    m_evo: int = 5
    horizon: int = 36
    hamiltonian: str = "tfim"
    noise_prob: float = 0.02
    noise_model: str = "readout_flip"
    readout_p01: float = 0.02
    readout_p10: float = 0.02
    t1_us: float = 100.0
    t2_us: float = 70.0
    time_scale_us: float = 1.0
    default_shots: int = 128
    shots_set: tuple[int, ...] = (32, 64, 128, 256, 512)
    sample_shots_each_step: bool = False
    j_range: tuple[float, float] = (0.5, 1.5)
    h_range: tuple[float, float] = (0.5, 1.5)
    simulator_backend: str = "statevector"
    mps_bond_dim: int = 32
    mps_trotter_steps: int = 8


@dataclass
class ModelConfig:
    latent_dim: int = 16
    use_vae: bool = True
    history: int = 20
    gnn_layers: int = 2
    temporal: str = "transformer"
    graph: str = "chain"
    use_smc_feedback: bool = True
    belief_mode: str = "both"
    vae_checkpoint: str = "artifacts/vae_n5_l16.pt"


@dataclass
class SMCConfig:
    particles: int = 256
    ess_frac: float = 0.6
    roughen_frac: float = 0.03
    sim_chunk_size: int = 4


@dataclass
class PPOConfig:
    updates: int = 2500
    rollout_steps: int = 64
    ppo_epochs: int = 2
    minibatches: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 1e-4
    entropy_coef: float = 0.02
    value_coef: float = 0.05
    clip_eps: float = 0.2
    clip_grad: float = 0.5
    reward_mode: str = "info_gain"
    behavior_clone_checkpoint: str = ""


@dataclass
class EvalConfig:
    episodes: int = 200
    seeds: tuple[int, ...] = (777, 778, 779, 780, 781)
    shot_budgets: tuple[int, ...] = (32, 64, 128, 256, 512)
    baselines: tuple[str, ...] = ("random", "fixed_optimized", "fisher_greedy_fast", "bald_2step_fast")
    bootstrap_samples: int = 1000


@dataclass
class ExperimentConfig:
    seed: int = 777
    device: str = "auto"
    output_dir: str = "runs/symqnet"
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    smc: SMCConfig = field(default_factory=SMCConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _coerce_value(current: Any, incoming: Any) -> Any:
    if is_dataclass(current):
        return _merge_dataclass(current, incoming)
    if isinstance(current, tuple) and isinstance(incoming, list):
        return tuple(incoming)
    return incoming


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    valid = {f.name for f in fields(instance)}
    unknown = sorted(set(updates) - valid)
    if unknown:
        raise KeyError(f"Unknown config key(s) for {type(instance).__name__}: {unknown}")
    for key, value in updates.items():
        setattr(instance, key, _coerce_value(getattr(instance, key), value))
    return instance


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    base_path = data.pop("extends", None)
    if base_path is not None:
        base = Path(base_path)
        if not base.is_absolute():
            base = path.parent / base
        cfg = load_config(base)
    return _merge_dataclass(cfg, data)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
