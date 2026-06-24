from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from symqnet.analysis.paired_main import comparison_rows, load_episode_rows
from symqnet.config import load_config
from symqnet.cross_eval import main as cross_eval_main
from symqnet.eval import episode_row, provenance_fields
from symqnet.train_ppo import load_vae


class DummyResult:
    seed = 777
    eval_seed = 777
    task_id = 3
    final_mse = 0.12
    total_info_gain = 1.5
    decision_ms_mean = 0.4
    decision_ms_p95 = 0.8
    smc_update_ms_mean = 1.4
    smc_update_ms_p95 = 1.8
    step_total_ms_mean = 2.4
    step_total_ms_p95 = 2.8
    crlb_theta_mse = 0.06
    actions = [0, 1, 2]
    true_theta = [0.5, 1.5]


def test_episode_rows_include_reproducibility_provenance() -> None:
    cfg = load_config("configs/smoke.json")
    provenance = provenance_fields(
        cfg,
        torch.device("cpu"),
        checkpoint_path="runs/main_result/symqnet_seed_777/best_agent.pt",
        checkpoint_metadata={"train_seed": 777, "config_hash": "abc123", "train_wallclock_sec": 9.5},
    )
    row = episode_row("symqnet", 16, 0, DummyResult(), provenance)

    assert row["train_seed"] == 777
    assert row["eval_seed"] == 777
    assert row["task_id"] == 3
    assert row["checkpoint_path"].endswith("best_agent.pt")
    assert row["checkpoint_config_hash"] == "abc123"
    assert row["config_hash"]
    assert row["device"] == "cpu"
    assert row["hardware"].startswith("cpu:")
    assert row["train_wallclock_sec"] == 9.5
    assert row["smc_update_ms_mean"] == 1.4
    assert row["step_total_ms_p95"] == 2.8
    assert "symqnet" in row["policy_params"]


def test_missing_vae_checkpoint_fails_without_debug_escape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pretrain_vae"):
        load_vae(str(tmp_path / "missing.pt"), input_dim=3, latent_dim=4, device=torch.device("cpu"))


def test_paired_main_uses_matched_tasks_and_averages_train_seeds(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    fieldnames = ["method", "shots", "eval_seed", "task_id", "final_mse", "decision_ms_mean"]
    rows = [
        ["bald_2step", 128, 777, 0, 0.10, 20.0],
        ["bald_2step", 128, 777, 1, 0.20, 22.0],
        ["symqnet", 128, 777, 0, 0.11, 1.0],
        ["symqnet", 128, 777, 0, 0.13, 1.2],
        ["symqnet", 128, 777, 1, 0.18, 0.8],
        ["symqnet", 128, 777, 1, 0.22, 1.0],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    out = comparison_rows(load_episode_rows(path), "symqnet", ["bald_2step"])

    assert len(out) == 1
    assert out[0]["paired_tasks"] == 2
    assert abs(out[0]["reference_mse_mean"] - 0.16) < 1e-12
    assert abs(out[0]["baseline_mse_mean"] - 0.15) < 1e-12
    assert "wilcoxon_p_bh" in out[0]
    assert "cliffs_delta" in out[0]
    assert out[0]["latency_speedup_baseline_over_reference"] > 20.0


def test_cross_eval_entrypoint_does_not_call_training() -> None:
    source = Path(cross_eval_main.__code__.co_filename).read_text(encoding="utf-8")
    assert "train_ppo" not in source


def test_scaling_runner_dry_run_prints_per_n_commands() -> None:
    result = subprocess.run(
        ["bash", "scripts/run_scaling.sh"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "N_VALUES": "4",
            "EPISODES": "1",
            "UPDATES": "1",
            "SEEDS": "777",
            "JOBS": "1",
        },
    )

    assert "configs/scaling/n4.json" in result.stdout
    assert "symqnet.paper_cpu_cluster" in result.stdout
    assert "symqnet.analysis.claim_gate" in result.stdout


def test_xxz_transfer_script_references_cross_eval() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_xxz_transfer.sh").read_text(encoding="utf-8")

    assert "configs/transfer_xxz.json" in source
    assert "symqnet.cross_eval" in source
    assert "symqnet.train_ppo" not in source


def test_training_checkpoint_records_heldout_validation_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "agent"
    val_bank = tmp_path / "validation_task_bank.npz"
    cmd = [
        sys.executable,
        "-m",
        "symqnet.train_ppo",
        "--config",
        "configs/smoke.json",
        "--updates",
        "1",
        "--seed",
        "777",
        "--output-dir",
        str(out_dir),
        "--validation-task-bank",
        str(val_bank),
        "--validation-episodes",
        "1",
        "--validation-interval",
        "1",
        "--allow-random-vae",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    payload = torch.load(out_dir / "best_agent.pt", map_location="cpu", weights_only=False)

    assert payload["checkpoint_selection_metric"] == "heldout_validation_mse"
    assert payload["validation_task_bank_path"] == str(val_bank)
    assert payload["validation_episodes"] == 1
    assert payload["best_validation_mse"] == payload["best_mse"]
    assert payload["reward_mode"] == "info_gain"


def test_oracle_mse_delta_reward_mode_records_metadata(tmp_path: Path) -> None:
    config = tmp_path / "oracle_reward.json"
    smoke = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"
    config.write_text(
        json.dumps({"extends": str(smoke), "ppo": {"reward_mode": "oracle_mse_delta"}}) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "agent"
    cmd = [
        sys.executable,
        "-m",
        "symqnet.train_ppo",
        "--config",
        str(config),
        "--updates",
        "1",
        "--seed",
        "777",
        "--output-dir",
        str(out_dir),
        "--validation-episodes",
        "1",
        "--validation-interval",
        "1",
        "--allow-random-vae",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    payload = torch.load(out_dir / "best_agent.pt", map_location="cpu", weights_only=False)

    assert payload["reward_mode"] == "oracle_mse_delta"
