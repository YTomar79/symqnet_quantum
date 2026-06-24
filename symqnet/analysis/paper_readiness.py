from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from symqnet.config import load_config


DEFAULT_REQUIRED_FILES = [
    "main_result_table.tex",
    "compact_main_table.tex",
    "seed_stability_table.tex",
    "reward_objective.csv",
    "reward_objective_table.tex",
    "baseline_params.csv",
    "baseline_params_table.tex",
    "total_time.csv",
    "paired_main.csv",
    "mse_latency_pareto.svg",
    "latency.svg",
    "manifest.json",
    "validation_report.json",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: str, label: str) -> tuple[float | None, str | None]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f"{label} is not numeric: {value!r}"
    if not math.isfinite(parsed):
        return None, f"{label} is not finite: {value!r}"
    return parsed, None


def _check_vae_checkpoint(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        return [f"missing VAE checkpoint: {path}"]
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return [f"could not load VAE checkpoint {path}: {exc}"]
    metadata = payload.get("pretrain_metadata", {}) if isinstance(payload, dict) else {}
    if not metadata:
        errors.append(f"VAE checkpoint lacks pretrain_metadata: {path}")
    return errors


def _check_reward_objective(
    run_root: Path,
    expected_shots: list[int],
    min_episodes: int,
    method: str = "symqnet",
    max_allowed_correlation: float = -0.05,
) -> list[str]:
    path = run_root / "reward_objective.csv"
    if not path.exists():
        return [f"missing reward/objective diagnostic: {path}"]
    rows = [row for row in _read_csv(path) if row.get("method") == method]
    by_shot = {int(float(row["shots"])): row for row in rows}
    errors: list[str] = []
    missing = sorted(set(expected_shots) - set(by_shot))
    if missing:
        errors.append(f"reward/objective diagnostic missing {method} shots: {missing}")
    for shot, row in sorted(by_shot.items()):
        episodes = int(float(row.get("episodes", 0) or 0))
        if episodes < min_episodes:
            errors.append(f"reward/objective row is smoke-sized: method={method} shots={shot} episodes={episodes} < {min_episodes}")
        for key in ("pearson_info_gain_vs_mse", "spearman_info_gain_vs_mse"):
            value, err = _finite_float(row.get(key, ""), f"reward/objective {method} shot={shot} {key}")
            if err is not None:
                errors.append(err)
                continue
            if value is not None and value > max_allowed_correlation:
                errors.append(
                    f"reward/objective {method} shot={shot} {key}={value:.4g} is not directionally useful; "
                    f"expected <= {max_allowed_correlation:.4g} so higher information gain predicts lower final MSE"
                )
    return errors


def _check_paired_stats(run_root: Path) -> list[str]:
    path = run_root / "paired_main.csv"
    if not path.exists():
        return [f"missing paired statistics: {path}"]
    rows = _read_csv(path)
    if not rows:
        return [f"paired statistics are empty: {path}"]
    required = {"wilcoxon_p", "wilcoxon_p_bh", "wilcoxon_p_holm", "cliffs_delta", "rank_biserial"}
    missing = sorted(required - set(rows[0]))
    return [f"paired statistics missing columns: {missing}"] if missing else []


def readiness_errors(
    *,
    run_root: Path,
    config_path: Path,
    expected_methods: list[str],
    expected_shots: list[int],
    min_summary_episodes: int,
    required_files: list[str],
    extra_required_files: list[str],
    reward_objective_max_correlation: float = -0.05,
) -> list[str]:
    errors: list[str] = []
    summary_csv = run_root / "shot_budget.csv"
    validation_json = run_root / "validation_report.json"
    if not summary_csv.exists():
        errors.append(f"missing summary CSV: {summary_csv}")
    if not validation_json.exists():
        errors.append(f"missing validation report: {validation_json}")

    for name in required_files:
        path = run_root / name
        if not path.exists():
            errors.append(f"missing required paper artifact: {path}")
    for name in extra_required_files:
        path = Path(name)
        if not path.exists():
            errors.append(f"missing required cross-experiment artifact: {path}")

    if validation_json.exists():
        try:
            validation = _load_json(validation_json)
            if validation.get("ok") is not True:
                errors.append(f"validation_report.json is not ok: {validation.get('errors', [])}")
        except Exception as exc:
            errors.append(f"could not parse validation report: {exc}")

    if summary_csv.exists():
        rows = _read_csv(summary_csv)
        methods = {row["method"] for row in rows}
        shots = {int(float(row["shots"])) for row in rows}
        missing_methods = sorted(set(expected_methods) - methods)
        missing_shots = sorted(set(expected_shots) - shots)
        if missing_methods:
            errors.append(f"missing expected methods: {missing_methods}")
        if missing_shots:
            errors.append(f"missing expected shot budgets: {missing_shots}")
        for row in rows:
            episodes = float(row.get("episodes", 0) or 0)
            if episodes < min_summary_episodes:
                errors.append(
                    f"summary row is smoke-sized: method={row.get('method')} shots={row.get('shots')} "
                    f"episodes={episodes:g} < {min_summary_episodes}"
                )

    cfg = load_config(config_path)
    if cfg.model.use_vae:
        errors.extend(_check_vae_checkpoint(Path(cfg.model.vae_checkpoint)))
    errors.extend(_check_reward_objective(run_root, expected_shots, min_summary_episodes, max_allowed_correlation=reward_objective_max_correlation))
    errors.extend(_check_paired_stats(run_root))
    return errors


def write_report(errors: list[str], out: Path | None) -> dict[str, object]:
    report = {"ok": not errors, "errors": errors}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"saved {out}")
    else:
        print(text)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast readiness check for QCRL paper artifacts.")
    parser.add_argument("--run-root", default="runs/main_result")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--expected-methods", nargs="+", default=["random", "fixed_optimized", "fisher_greedy_fast", "bald_2step_fast", "dad_transformer", "symqnet"])
    parser.add_argument("--expected-shots", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--min-summary-episodes", type=int, default=100)
    parser.add_argument("--required-files", nargs="+", default=DEFAULT_REQUIRED_FILES)
    parser.add_argument("--extra-required-files", nargs="*", default=[])
    parser.add_argument(
        "--reward-objective-max-correlation",
        type=float,
        default=-0.05,
        help="Maximum allowed Pearson/Spearman correlation between total information gain and final MSE.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    errors = readiness_errors(
        run_root=Path(args.run_root),
        config_path=Path(args.config),
        expected_methods=args.expected_methods,
        expected_shots=args.expected_shots,
        min_summary_episodes=args.min_summary_episodes,
        required_files=args.required_files,
        extra_required_files=args.extra_required_files,
        reward_objective_max_correlation=args.reward_objective_max_correlation,
    )
    write_report(errors, Path(args.out) if args.out else None)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
