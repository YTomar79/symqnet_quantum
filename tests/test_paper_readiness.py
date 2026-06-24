from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from symqnet.analysis.baseline_params import baseline_param_rows
from symqnet.analysis.claim_gate import claim_report
from symqnet.analysis.complexity import complexity_rows
from symqnet.analysis.compact_table import compact_rows
from symqnet.analysis.paired_main import comparison_rows, load_episode_rows
from symqnet.analysis.paper_readiness import DEFAULT_REQUIRED_FILES, readiness_errors
from symqnet.analysis.reward_objective import reward_objective_rows
from symqnet.analysis.seed_stability import seed_stability_rows
from symqnet.analysis.stats import p_adjust_bh, p_adjust_holm
from symqnet.analysis.validate_results import validation_report
from symqnet.baselines import make_baseline
from symqnet.env import SpinChainEnv
from symqnet.manifest import _project_root, anonymize_manifest
from symqnet.metadata import build_metadata
from symqnet.models.agent import SymQNetAgent
from symqnet.models.vae import VariationalAutoencoder
from symqnet.paper_cpu_cluster import build_workflow
from symqnet.smc import SMCParticleFilter


def _write_csv(path: Path, fieldnames: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def test_cpu_cluster_workflow_uses_per_seed_outputs(tmp_path: Path) -> None:
    workflow = build_workflow(
        config="configs/default.json",
        run_root=tmp_path / "paper",
        episodes=2,
        updates=3,
        seeds=[777, 778],
        with_crlb=True,
        comparisons=["bald_2step", "bald_1step", "smc_adaptive"],
        n_qubits=5,
        m_evo=5,
    )

    assert len(workflow["train"]) == 2
    assert len(workflow["eval"]) == 2
    assert any("main_result_table.tex" in cmd for command in workflow["postprocess"] for cmd in command)
    assert any("seed_stability_table.tex" in cmd for command in workflow["postprocess"] for cmd in command)
    eval_outputs = [" ".join(command) for command in workflow["eval"]]
    assert "symqnet_seed_777/eval.csv" in eval_outputs[0]
    assert "symqnet_seed_778/eval.csv" in eval_outputs[1]


def test_paired_stats_average_train_seeds_and_emit_effect_sizes(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    _write_csv(
        path,
        ["method", "shots", "eval_seed", "task_id", "final_mse", "decision_ms_mean"],
        [
            ["bald_2step", 128, 777, 0, 0.10, 20.0],
            ["bald_2step", 128, 777, 1, 0.20, 22.0],
            ["symqnet", 128, 777, 0, 0.12, 1.0],
            ["symqnet", 128, 777, 0, 0.14, 1.2],
            ["symqnet", 128, 777, 1, 0.18, 0.8],
            ["symqnet", 128, 777, 1, 0.22, 1.0],
        ],
    )

    rows = comparison_rows(load_episode_rows(path), "symqnet", ["bald_2step"])

    assert len(rows) == 1
    row = rows[0]
    assert row["paired_tasks"] == 2
    assert abs(row["reference_mse_mean"] - 0.165) < 1e-12
    assert abs(row["baseline_mse_mean"] - 0.15) < 1e-12
    assert "mse_ratio_ci95_lo" in row
    assert "latency_speedup_ci95_hi" in row
    assert "wilcoxon_p_bh" in row
    assert "wilcoxon_p_holm" in row
    assert "cliffs_delta" in row
    assert "rank_biserial" in row
    assert row["latency_speedup_baseline_over_reference"] > 20.0


def test_p_adjustments_are_deterministic() -> None:
    values = [0.01, 0.04, 0.03]
    assert p_adjust_bh(values) == pytest.approx([0.03, 0.04, 0.04])
    assert p_adjust_holm(values) == pytest.approx([0.03, 0.06, 0.06])


def test_bald_primary_is_strong_and_fast_variant_is_pruned() -> None:
    strong = make_baseline("bald_2step", n_qubits=5, m_evo=5, n_actions=75, seed=777)
    fast = make_baseline("bald_2step_fast", n_qubits=5, m_evo=5, n_actions=75, seed=777)

    assert strong.metadata()["predictive_samples"] == 5
    assert strong.metadata()["top_k"] == 10
    assert fast.metadata()["predictive_samples"] == 1
    assert fast.metadata()["top_k"] == 2
    assert fast.metadata()["max_actions"] == 3
    assert fast.metadata()["future_max_actions"] == 2


def test_validation_requires_matched_tasks_and_provenance(tmp_path: Path) -> None:
    summary = tmp_path / "shot_budget.csv"
    episodes = tmp_path / "episodes.csv"
    _write_csv(
        summary,
        [
            "method",
            "shots",
            "eval_seed",
            "config_hash",
            "hardware",
            "task_bank_path",
            "train_seed",
            "checkpoint_path",
            "checkpoint_config_hash",
            "checkpoint_selection_metric",
            "best_validation_mse",
            "validation_task_bank_path",
            "train_wallclock_sec",
            "policy_params",
            "decision_ms_mean",
            "smc_update_ms_mean",
            "step_total_ms_mean",
        ],
        [
            [
                "bald_2step",
                128,
                "777",
                "abc",
                "cpu:arm64",
                "tasks.npz",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                '{"depth": 2}',
                20.0,
                3.0,
                25.0,
            ],
            [
                "symqnet",
                128,
                "777",
                "abc",
                "cpu:arm64",
                "tasks.npz",
                777,
                "best.pt",
                "def",
                "heldout_validation_mse",
                0.11,
                "validation.npz",
                12.0,
                '{"name": "symqnet"}',
                1.0,
                3.0,
                5.0,
            ],
        ],
    )
    _write_csv(
        episodes,
        ["method", "shots", "eval_seed", "task_id", "final_mse"],
        [
            ["bald_2step", 128, 777, 0, 0.1],
            ["bald_2step", 128, 777, 1, 0.2],
            ["symqnet", 128, 777, 0, 0.11],
            ["symqnet", 128, 777, 1, 0.18],
        ],
    )

    report = validation_report(summary, episodes, require_agent_checkpoints=True)

    assert report["ok"] is True
    assert report["errors"] == []


def test_compact_table_refuses_smoke_sized_rows(tmp_path: Path) -> None:
    paired = tmp_path / "paired_main.csv"
    _write_csv(
        paired,
        [
            "shots",
            "reference",
            "baseline",
            "paired_tasks",
            "mse_ratio_reference_over_baseline",
            "mse_ratio_ci95_lo",
            "mse_ratio_ci95_hi",
            "latency_speedup_baseline_over_reference",
            "latency_speedup_ci95_lo",
            "latency_speedup_ci95_hi",
        ],
        [[128, "symqnet", "bald_2step", 1, 1.05, 1.0, 1.1, 25.0, 20.0, 30.0]],
    )

    with pytest.raises(ValueError, match="smoke-sized"):
        compact_rows(paired, [128], ["bald_2step"], min_paired_tasks=100)


def test_baseline_param_export_includes_bald_and_environment_parameters(tmp_path: Path) -> None:
    summary = tmp_path / "shot_budget.csv"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"smc": {"particles": 96}, "env": {"n_qubits": 5, "m_evo": 5, "horizon": 36}}), encoding="utf-8")
    _write_csv(
        summary,
        ["method", "shots", "episodes", "policy_params"],
        [
            [
                "bald_2step",
                128,
                500,
                '{"name": "bald_nstep", "depth": 2, "top_k": 10, "predictive_samples": 5}',
            ],
            ["symqnet", 128, 500, '{"name": "symqnet", "graph": "chain", "temporal": "transformer"}'],
        ],
    )

    rows = {row["method"]: row for row in baseline_param_rows(summary, config)}

    assert rows["bald_2step"]["depth"] == 2
    assert rows["bald_2step"]["top_k"] == 10
    assert rows["bald_2step"]["predictive_samples"] == 5
    assert rows["bald_2step"]["smc_particles"] == 96
    assert rows["bald_2step"]["n_actions"] == 75


def test_agent_smc_and_metadata_initialize_for_scaling_sizes() -> None:
    device = torch.device("cpu")
    for n_qubits in [4, 5, 6, 7]:
        env = SpinChainEnv(n_qubits=n_qubits, m_evo=2, horizon=2, seed=777, device=device, shots_set=None)
        smc = SMCParticleFilter(env, n_particles=8, device=device)
        agent = SymQNetAgent(
            VariationalAutoencoder(env.N, latent_dim=4),
            env.N,
            latent_dim=4,
            history=2,
            n_actions=env.n_actions,
            m_evo=env.M_evo,
            use_vae=False,
            device=device,
        )
        metadata = build_metadata(
            env.N,
            env.M_evo,
            agent.theta_dim,
            agent.cov_feat_dim,
            agent.use_smc_feedback,
            agent.belief_mode,
            device,
            None,
            smc.posterior(),
            env.shots_max,
        )
        dist, _ = agent(torch.zeros(env.N), metadata)
        assert dist.probs.numel() == env.n_actions


def test_native_noise_transforms_expectations() -> None:
    env = SpinChainEnv(n_qubits=3, m_evo=2, horizon=2, seed=777, shots_set=None, noise_prob=0.0)
    value = torch.tensor(0.5)
    assert env.apply_noise_to_expectation(value, 0.1).item() == pytest.approx(0.5)

    noisy = SpinChainEnv(
        n_qubits=3,
        m_evo=2,
        horizon=2,
        seed=777,
        shots_set=None,
        noise_model="native_decoherence",
        readout_p01=0.1,
        readout_p10=0.2,
        t1_us=1e12,
        t2_us=1e12,
    )
    assert noisy.apply_noise_to_expectation(value, 0.1).item() == pytest.approx(0.45, abs=1e-6)


def test_seed_stability_and_reward_objective_tables_use_episode_rows(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.csv"
    _write_csv(
        episodes,
        ["method", "shots", "train_seed", "final_mse", "total_info_gain"],
        [
            ["symqnet", 128, 777, 0.10, 2.0],
            ["symqnet", 128, 777, 0.12, 1.8],
            ["symqnet", 128, 778, 0.20, 1.2],
            ["symqnet", 128, 778, 0.22, 1.0],
        ],
    )

    stability = seed_stability_rows(episodes)
    reward_rows = reward_objective_rows(episodes, ["symqnet"])

    assert stability[0]["train_seeds"] == 2
    assert stability[0]["seed_mean_mse_mean"] == pytest.approx(0.16)
    assert reward_rows[0]["episodes"] == 4
    assert reward_rows[0]["pearson_info_gain_vs_mse"] < 0.0


def test_paper_readiness_accepts_full_metadata_and_rejects_smoke(tmp_path: Path) -> None:
    run_root = tmp_path / "main_result"
    run_root.mkdir()
    config = tmp_path / "config.json"
    vae = tmp_path / "vae.pt"
    torch.save({"model_state_dict": {}, "pretrain_metadata": {"samples": 10}}, vae)
    config.write_text(json.dumps({"model": {"vae_checkpoint": str(vae)}}), encoding="utf-8")
    for name in DEFAULT_REQUIRED_FILES:
        (run_root / name).write_text("ok\n", encoding="utf-8")
    (run_root / "validation_report.json").write_text(json.dumps({"ok": True, "errors": []}), encoding="utf-8")
    _write_csv(
        run_root / "paired_main.csv",
        [
            "shots",
            "reference",
            "baseline",
            "paired_tasks",
            "wilcoxon_p",
            "wilcoxon_p_bh",
            "wilcoxon_p_holm",
            "cliffs_delta",
            "rank_biserial",
        ],
        [[128, "symqnet", "bald_2step", 500, 0.01, 0.02, 0.03, -0.2, -0.4]],
    )
    _write_csv(
        run_root / "shot_budget.csv",
        ["method", "shots", "episodes"],
        [
            ["random", 128, 500],
            ["fixed", 128, 500],
            ["bald_2step", 128, 500],
            ["symqnet", 128, 500],
        ],
    )
    _write_csv(
        run_root / "reward_objective.csv",
        ["shots", "method", "episodes", "pearson_info_gain_vs_mse", "spearman_info_gain_vs_mse"],
        [[128, "symqnet", 500, -0.4, -0.3]],
    )

    errors = readiness_errors(
        run_root=run_root,
        config_path=config,
        expected_methods=["random", "fixed", "bald_2step", "symqnet"],
        expected_shots=[128],
        min_summary_episodes=100,
        required_files=DEFAULT_REQUIRED_FILES,
        extra_required_files=[],
    )
    assert errors == []

    _write_csv(run_root / "shot_budget.csv", ["method", "shots", "episodes"], [["symqnet", 128, 1]])
    errors = readiness_errors(
        run_root=run_root,
        config_path=config,
        expected_methods=["symqnet"],
        expected_shots=[128],
        min_summary_episodes=100,
        required_files=["main_result_table.tex"],
        extra_required_files=[],
    )
    assert any("smoke-sized" in error for error in errors)


def test_paper_readiness_rejects_weak_reward_objective_alignment(tmp_path: Path) -> None:
    run_root = tmp_path / "main_result"
    run_root.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": {"use_vae": False}}), encoding="utf-8")
    (run_root / "validation_report.json").write_text(json.dumps({"ok": True, "errors": []}), encoding="utf-8")
    _write_csv(run_root / "shot_budget.csv", ["method", "shots", "episodes"], [["symqnet", 128, 500]])
    _write_csv(
        run_root / "paired_main.csv",
        [
            "shots",
            "reference",
            "baseline",
            "paired_tasks",
            "wilcoxon_p",
            "wilcoxon_p_bh",
            "wilcoxon_p_holm",
            "cliffs_delta",
            "rank_biserial",
        ],
        [[128, "symqnet", "random", 500, 0.01, 0.02, 0.03, -0.2, -0.4]],
    )
    _write_csv(
        run_root / "reward_objective.csv",
        ["shots", "method", "episodes", "pearson_info_gain_vs_mse", "spearman_info_gain_vs_mse"],
        [[128, "symqnet", 500, -0.2, 0.01]],
    )

    errors = readiness_errors(
        run_root=run_root,
        config_path=config,
        expected_methods=["symqnet"],
        expected_shots=[128],
        min_summary_episodes=100,
        required_files=[],
        extra_required_files=[],
    )

    assert any("not directionally useful" in error for error in errors)


def test_claim_gate_accepts_flat_reference_latency(tmp_path: Path) -> None:
    scaling = tmp_path / "scaling.csv"
    _write_csv(
        scaling,
        ["n_qubits", "method", "decision_ms_mean", "mse_mean"],
        [
            [4, "symqnet", 1.0, 0.10],
            [7, "symqnet", 1.1, 0.12],
            [4, "bald_2step_fast", 10.0, 0.09],
            [7, "bald_2step_fast", 100.0, 0.10],
        ],
    )

    report = claim_report(scaling, mse_ratio_max=1.25)

    assert report["scaling_claim_ok"] is True


def test_complexity_rows_capture_action_and_exponential_terms() -> None:
    rows = complexity_rows([4, 5], m_evo=5, particles=256, top_k=10, predictive_samples=5)

    assert rows[0]["actions"] == 60
    assert rows[1]["actions"] == 75
    assert rows[1]["hilbert_dim"] == 32
    assert rows[1]["bald_2step_relative_units"] > rows[0]["bald_2step_relative_units"]


def test_manifest_anonymization_strips_local_paths() -> None:
    root = _project_root()
    home = Path.home()
    payload = {
        "python": "/opt/example/.venv/bin/python",
        "path": str(root / "runs" / "main_result" / "shot_budget.csv"),
        "home": str(home / ".cache" / "something"),
    }

    scrubbed = anonymize_manifest(payload)

    assert scrubbed["python"] == "python"
    assert "<PROJECT_ROOT>" in scrubbed["path"]
    assert str(root) not in scrubbed["path"]
    assert "<HOME>" in scrubbed["home"]
    assert str(home) not in scrubbed["home"]
