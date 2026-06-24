from __future__ import annotations

import csv
from pathlib import Path

import torch

from symqnet.analysis.mps_validation import validate_mps
from symqnet.analysis.total_time import total_time_rows
from symqnet.config import load_config
from symqnet.env import SpinChainEnv
from symqnet.smc import SMCParticleFilter


def _write_csv(path: Path, fieldnames: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def test_default_config_uses_statevector_backend() -> None:
    cfg = load_config("configs/default.json")

    assert cfg.env.simulator_backend == "statevector"
    assert cfg.env.mps_bond_dim == 32
    assert cfg.env.mps_trotter_steps == 8


def test_mps_observables_match_statevector_small_chain() -> None:
    report = validate_mps([4], m_evo=2, tasks=1, bond_dim=32, trotter_steps=32)

    assert report["max_mean_abs_error"] <= 2e-2


def test_mps_smc_cache_clears_after_resampling_and_roughening() -> None:
    env = SpinChainEnv(
        n_qubits=4,
        m_evo=2,
        horizon=2,
        seed=777,
        shots_set=None,
        simulator_backend="mps_tebd",
        mps_trotter_steps=4,
    )
    smc = SMCParticleFilter(env, n_particles=8, ess_frac=1.1)

    smc.predicted_means(0, 0, 0)
    assert smc._cache
    smc.update(torch.zeros(env.N), {"qubit_idx": 0, "basis_idx": 0, "time_idx": 0, "shots": 128})

    assert smc._cache == {}


def test_dad_transformer_config_is_neural_bed_baseline() -> None:
    cfg = load_config("configs/dad_transformer.json")

    assert cfg.model.use_vae is False
    assert cfg.model.graph == "none"
    assert cfg.model.temporal == "transformer"
    assert cfg.model.use_smc_feedback is False
    assert cfg.model.belief_mode == "none"


def test_total_time_rows_use_quantum_and_classical_terms(tmp_path: Path) -> None:
    summary = tmp_path / "shot_budget.csv"
    _write_csv(
        summary,
        ["method", "shots", "decision_ms_mean", "smc_update_ms_mean"],
        [["symqnet", 128, 2.0, 3.0]],
    )

    rows = total_time_rows(summary, horizon=36, shot_times_us=[1000.0])

    assert rows[0]["quantum_ms_per_step"] == 128.0
    assert rows[0]["classical_ms_per_step"] == 5.0
    assert rows[0]["total_episode_time_ms"] == 36 * 133.0
