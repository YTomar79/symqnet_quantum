from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import load_config
from .manifest import build_manifest, write_manifest
from .task_bank import sample_task_bank


def _bool_flag(value: str | int | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _python() -> str:
    return sys.executable


def baseline_command(config: str, episodes: int, out_csv: Path, episodes_out: Path, task_bank: Path, with_crlb: bool) -> list[str]:
    cmd = [
        _python(),
        "-m",
        "symqnet.eval",
        "--config",
        config,
        "--episodes",
        str(episodes),
        "--out",
        str(out_csv),
        "--episodes-out",
        str(episodes_out),
        "--task-bank",
        str(task_bank),
    ]
    if with_crlb:
        cmd.append("--with-crlb")
    return cmd


def train_command(
    config: str,
    updates: int,
    seed: int,
    seed_dir: Path,
    validation_task_bank: Path,
    validation_episodes: int,
    validation_interval: int,
    allow_random_vae: bool = False,
) -> list[str]:
    cmd = [
        _python(),
        "-m",
        "symqnet.train_ppo",
        "--config",
        config,
        "--updates",
        str(updates),
        "--seed",
        str(seed),
        "--output-dir",
        str(seed_dir),
        "--validation-task-bank",
        str(validation_task_bank),
        "--validation-episodes",
        str(validation_episodes),
        "--validation-interval",
        str(validation_interval),
    ]
    if allow_random_vae:
        cmd.append("--allow-random-vae")
    return cmd


def eval_agent_command(
    config: str,
    episodes: int,
    seed: int,
    seed_dir: Path,
    out_csv: Path,
    episodes_out: Path,
    task_bank: Path,
    with_crlb: bool,
    agent_name: str = "symqnet",
) -> list[str]:
    cmd = [
        _python(),
        "-m",
        "symqnet.eval",
        "--config",
        config,
        "--episodes",
        str(episodes),
        "--agent-checkpoint",
        str(seed_dir / "best_agent.pt"),
        "--agent-name",
        agent_name,
        "--train-seed",
        str(seed),
        "--skip-baselines",
        "--out",
        str(out_csv),
        "--episodes-out",
        str(episodes_out),
        "--task-bank",
        str(task_bank),
    ]
    if with_crlb:
        cmd.append("--with-crlb")
    return cmd


def postprocess_commands(
    run_root: Path,
    config: str,
    n_qubits: int,
    m_evo: int,
    comparisons: list[str],
    compact_min_paired_tasks: int,
    compact_table_shots: list[int],
) -> list[list[str]]:
    shot_csv = run_root / "shot_budget.csv"
    episodes_csv = run_root / "episodes.csv"
    commands = [
        [_python(), "-m", "symqnet.plot_shot_budget", "--csv", str(shot_csv), "--out", str(run_root / "shot_budget.svg")],
        [_python(), "-m", "symqnet.analysis.tables", "--csv", str(shot_csv), "--out", str(run_root / "wallclock_mse_crlb_table.tex")],
        [
            _python(),
            "-m",
            "symqnet.analysis.reward_scatter",
            "--episodes-csv",
            str(episodes_csv),
            "--out",
            str(run_root / "reward_vs_objective.svg"),
        ],
        [
            _python(),
            "-m",
            "symqnet.analysis.paired_main",
            "--episodes-csv",
            str(episodes_csv),
            "--reference",
            "symqnet",
            "--baselines",
            *comparisons,
            "--out-csv",
            str(run_root / "paired_main.csv"),
            "--out-tex",
            str(run_root / "paired_main_table.tex"),
            "--acceptance-tex",
            str(run_root / "main_result_table.tex"),
        ],
        [
            _python(),
            "-m",
            "symqnet.analysis.seed_stability",
            "--episodes-csv",
            str(episodes_csv),
            "--method",
            "symqnet",
            "--out-csv",
            str(run_root / "seed_stability.csv"),
            "--out-tex",
            str(run_root / "seed_stability_table.tex"),
        ],
        [
            _python(),
            "-m",
            "symqnet.analysis.reward_objective",
            "--episodes-csv",
            str(episodes_csv),
            "--methods",
            "symqnet",
            "--out-csv",
            str(run_root / "reward_objective.csv"),
            "--out-tex",
            str(run_root / "reward_objective_table.tex"),
        ],
        [
            _python(),
            "-m",
            "symqnet.analysis.baseline_params",
            "--summary-csv",
            str(shot_csv),
            "--config",
            config,
            "--out-csv",
            str(run_root / "baseline_params.csv"),
            "--out-tex",
            str(run_root / "baseline_params_table.tex"),
        ],
        [
            _python(),
            "-m",
            "symqnet.analysis.total_time",
            "--summary-csv",
            str(shot_csv),
            "--config",
            config,
            "--out",
            str(run_root / "total_time.csv"),
        ],
        [_python(), "-m", "symqnet.analysis.paper_figures", "latency", "--csv", str(shot_csv), "--out", str(run_root / "latency.svg")],
        [_python(), "-m", "symqnet.analysis.paper_figures", "pareto", "--csv", str(shot_csv), "--out", str(run_root / "mse_latency_pareto.svg")],
        [
            _python(),
            "-m",
            "symqnet.analysis.paper_figures",
            "action-heatmap",
            "--episodes-csv",
            str(episodes_csv),
            "--out",
            str(run_root / "action_heatmap_symqnet.svg"),
            "--n-qubits",
            str(n_qubits),
            "--m-evo",
            str(m_evo),
            "--method",
            "symqnet",
            "--shot",
            "128",
        ],
        [
            _python(),
            "-m",
            "symqnet.analysis.validate_results",
            "--summary-csv",
            str(shot_csv),
            "--episodes-csv",
            str(episodes_csv),
            "--require-agent-checkpoints",
            "--out",
            str(run_root / "validation_report.json"),
        ],
    ]
    if compact_min_paired_tasks > 0:
        commands.insert(
            4,
            [
                _python(),
                "-m",
                "symqnet.analysis.compact_table",
                "--paired-csv",
                str(run_root / "paired_main.csv"),
                "--shots",
                *[str(shot) for shot in compact_table_shots],
                "--baselines",
                *comparisons,
                "--min-paired-tasks",
                str(compact_min_paired_tasks),
                "--out-csv",
                str(run_root / "compact_main_table.csv"),
                "--out-tex",
                str(run_root / "compact_main_table.tex"),
            ],
        )
    return commands


def build_workflow(
    *,
    config: str,
    run_root: Path,
    episodes: int,
    updates: int,
    seeds: list[int],
    with_crlb: bool,
    comparisons: list[str],
    n_qubits: int,
    m_evo: int,
    validation_episodes: int = 32,
    validation_interval: int = 25,
    compact_table_shots: list[int] | None = None,
    extra_agents: list[tuple[str, str]] | None = None,
    allow_random_vae: bool = False,
) -> dict[str, object]:
    extra_agents = extra_agents or []
    task_bank = run_root / "task_bank.npz"
    validation_task_bank = run_root / "validation_task_bank.npz"
    baseline_csv = run_root / "baselines.csv"
    baseline_episodes = run_root / "baselines_episodes.csv"
    train = [
        train_command(config, updates, seed, run_root / f"symqnet_seed_{seed}", validation_task_bank, validation_episodes, validation_interval, allow_random_vae)
        for seed in seeds
    ]
    evals = [
        eval_agent_command(
            config,
            episodes,
            seed,
            run_root / f"symqnet_seed_{seed}",
            run_root / f"symqnet_seed_{seed}" / "eval.csv",
            run_root / f"symqnet_seed_{seed}" / "episodes.csv",
            task_bank,
            with_crlb,
            "symqnet",
        )
        for seed in seeds
    ]
    extra_train: list[list[str]] = []
    extra_evals: list[list[str]] = []
    summary_inputs = [baseline_csv, *[run_root / f"symqnet_seed_{seed}" / "eval.csv" for seed in seeds]]
    episode_inputs = [baseline_episodes, *[run_root / f"symqnet_seed_{seed}" / "episodes.csv" for seed in seeds]]
    for agent_name, agent_config in extra_agents:
        for seed in seeds:
            seed_dir = run_root / f"{agent_name}_seed_{seed}"
            extra_train.append(train_command(agent_config, updates, seed, seed_dir, validation_task_bank, validation_episodes, validation_interval, allow_random_vae))
            extra_evals.append(
                eval_agent_command(
                    agent_config,
                    episodes,
                    seed,
                    seed_dir,
                    seed_dir / "eval.csv",
                    seed_dir / "episodes.csv",
                    task_bank,
                    with_crlb,
                    agent_name,
                )
            )
            summary_inputs.append(seed_dir / "eval.csv")
            episode_inputs.append(seed_dir / "episodes.csv")
    compact_min_paired_tasks = 100 if episodes >= 100 else 0
    post = postprocess_commands(run_root, config, n_qubits, m_evo, comparisons, compact_min_paired_tasks, compact_table_shots or [128, 512])
    return {
        "task_bank": task_bank,
        "validation_task_bank": validation_task_bank,
        "baseline": baseline_command(config, episodes, baseline_csv, baseline_episodes, task_bank, with_crlb),
        "train": [*train, *extra_train],
        "eval": [*evals, *extra_evals],
        "postprocess": post,
        "summary_inputs": summary_inputs,
        "episode_inputs": episode_inputs,
        "all_commands": [baseline_command(config, episodes, baseline_csv, baseline_episodes, task_bank, with_crlb), *train, *extra_train, *evals, *extra_evals, *post],
    }


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _run_parallel(commands: list[list[str]], jobs: int, env: dict[str, str]) -> None:
    if not commands:
        return
    if jobs <= 1:
        for cmd in commands:
            _run(cmd, env)
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run, cmd, env) for cmd in commands]
        for future in as_completed(futures):
            future.result()


def merge_csv(inputs: list[Path], out: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"CSV schema mismatch in {path}: {reader.fieldnames} != {fieldnames}")
            rows.extend(reader)
    if fieldnames is None:
        raise ValueError("No CSV inputs to merge.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged {len(rows)} rows -> {out}")


def _subprocess_env(jobs: int) -> dict[str, str]:
    env = os.environ.copy()
    if "OMP_NUM_THREADS" not in env:
        env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 1) // max(1, jobs)))
    if "MKL_NUM_THREADS" not in env:
        env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-parallel paper experiment runner for SymQNet.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--run-root", default="runs/main_result")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[777, 778, 779, 780, 781])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--with-crlb", default="0")
    parser.add_argument("--main-comparisons", nargs="+", default=["bald_2step_fast", "fisher_greedy_fast", "fixed_optimized", "random"])
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--compact-table-shots", nargs="+", type=int, default=[128, 512])
    parser.add_argument(
        "--extra-agent",
        action="append",
        default=[],
        help="Additional learned agent as name=config_path, trained/evaluated on the shared task bank.",
    )
    parser.add_argument("--anonymize-manifest", action="store_true")
    parser.add_argument("--allow-random-vae", action="store_true", help="Debug/smoke only: pass through to training commands.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    run_root = Path(args.run_root)
    cfg = load_config(args.config)
    extra_agents: list[tuple[str, str]] = []
    for item in args.extra_agent:
        if "=" not in item:
            raise ValueError(f"--extra-agent must be name=config_path, got {item!r}")
        name, path = item.split("=", 1)
        extra_agents.append((name, path))
    with_crlb = _bool_flag(args.with_crlb)
    workflow = build_workflow(
        config=args.config,
        run_root=run_root,
        episodes=args.episodes,
        updates=args.updates,
        seeds=args.seeds,
        with_crlb=with_crlb,
        comparisons=args.main_comparisons,
        n_qubits=cfg.env.n_qubits,
        m_evo=cfg.env.m_evo,
        validation_episodes=args.validation_episodes,
        validation_interval=args.validation_interval,
        compact_table_shots=args.compact_table_shots,
        extra_agents=extra_agents,
        allow_random_vae=args.allow_random_vae,
    )
    commands = workflow["all_commands"]
    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    run_root.mkdir(parents=True, exist_ok=True)
    env = _subprocess_env(args.jobs)
    _run(workflow["baseline"], env)
    sample_task_bank(
        workflow["validation_task_bank"],
        args.validation_episodes,
        cfg.env.n_qubits,
        cfg.env.j_range,
        cfg.env.h_range,
        20260516,
    )
    _run_parallel(workflow["train"], args.jobs, env)
    _run_parallel(workflow["eval"], args.jobs, env)

    merge_csv(workflow["summary_inputs"], run_root / "shot_budget.csv")
    merge_csv(workflow["episode_inputs"], run_root / "episodes.csv")

    for cmd in workflow["postprocess"]:
        _run(cmd, env)

    manifest = build_manifest(
        run_root=run_root,
        config=cfg,
        args={**vars(args), "with_crlb": with_crlb},
        commands=commands,
        task_bank=workflow["task_bank"],
        started_at=started_at,
        files=[args.config, cfg.model.vae_checkpoint, "scripts/run_paper_cpu_cluster.sh", "symqnet/paper_cpu_cluster.py"],
        outputs=[run_root / "shot_budget.csv", run_root / "episodes.csv", run_root / "paired_main.csv", run_root / "main_result_table.tex"],
        anonymize=args.anonymize_manifest,
    )
    write_manifest(run_root / "manifest.json", manifest)
    print(f"saved {run_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
