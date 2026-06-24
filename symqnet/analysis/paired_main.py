from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .stats import cliffs_delta, p_adjust_bh, p_adjust_holm, rank_biserial_paired, wilcoxon_signed_rank


def _task_key(row: dict[str, str]) -> tuple[int, int, int]:
    eval_seed = row.get("eval_seed") or row.get("seed") or "-1"
    task_id = row.get("task_id") or row.get("episode_idx")
    return int(float(row["shots"])), int(float(eval_seed)), int(float(task_id))


def load_episode_rows(path: Path):
    grouped = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[(row["method"], _task_key(row))].append(
                {
                    "final_mse": float(row["final_mse"]),
                    "decision_ms_mean": float(row["decision_ms_mean"]),
                }
            )

    out = {}
    for key, values in grouped.items():
        out[key] = {
            "final_mse": float(np.mean([v["final_mse"] for v in values])),
            "decision_ms_mean": float(np.mean([v["decision_ms_mean"] for v in values])),
        }
    return out


def _bootstrap_ci(values: np.ndarray, fn, samples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.shape[0] <= 1 or samples <= 0:
        value = float(fn(values))
        return value, value
    rng = np.random.default_rng(20260516)
    boot = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        idx = rng.integers(0, values.shape[0], size=values.shape[0])
        boot[i] = float(fn(values[idx]))
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def comparison_rows(grouped, reference: str, baselines: list[str]):
    shots_values = sorted({key[1][0] for key in grouped if key[0] == reference})
    rows = []
    for shots in shots_values:
        ref_items = {
            task_key: value
            for (method, task_key), value in grouped.items()
            if method == reference and task_key[0] == shots
        }
        for baseline in baselines:
            ref_mse, base_mse, ref_ms, base_ms = [], [], [], []
            for task_key, ref_value in sorted(ref_items.items()):
                base_value = grouped.get((baseline, task_key))
                if base_value is None:
                    continue
                ref_mse.append(ref_value["final_mse"])
                base_mse.append(base_value["final_mse"])
                ref_ms.append(ref_value["decision_ms_mean"])
                base_ms.append(base_value["decision_ms_mean"])
            if not ref_mse:
                continue
            w, p, n = wilcoxon_signed_rank(ref_mse, base_mse)
            ref_arr = np.asarray(ref_mse, dtype=np.float64)
            base_arr = np.asarray(base_mse, dtype=np.float64)
            ref_ms_arr = np.asarray(ref_ms, dtype=np.float64)
            base_ms_arr = np.asarray(base_ms, dtype=np.float64)
            paired = np.column_stack([ref_arr, base_arr, ref_ms_arr, base_ms_arr])

            def mse_ratio(sample: np.ndarray) -> float:
                return float(np.mean(sample[:, 0]) / max(np.mean(sample[:, 1]), 1e-12))

            def latency_speedup(sample: np.ndarray) -> float:
                return float(np.mean(sample[:, 3]) / max(np.mean(sample[:, 2]), 1e-12))

            ratio_ci_lo, ratio_ci_hi = _bootstrap_ci(paired, mse_ratio)
            speed_ci_lo, speed_ci_hi = _bootstrap_ci(paired, latency_speedup)
            rows.append(
                {
                    "shots": shots,
                    "reference": reference,
                    "baseline": baseline,
                    "paired_tasks": len(ref_mse),
                    "wilcoxon_nonzero_pairs": n,
                    "reference_mse_mean": float(np.mean(ref_arr)),
                    "reference_mse_median": float(np.median(ref_arr)),
                    "reference_mse_iqr": float(np.percentile(ref_arr, 75) - np.percentile(ref_arr, 25)),
                    "baseline_mse_mean": float(np.mean(base_arr)),
                    "baseline_mse_median": float(np.median(base_arr)),
                    "baseline_mse_iqr": float(np.percentile(base_arr, 75) - np.percentile(base_arr, 25)),
                    "mse_delta_reference_minus_baseline": float(np.mean(ref_arr - base_arr)),
                    "mse_ratio_reference_over_baseline": mse_ratio(paired),
                    "mse_ratio_ci95_lo": ratio_ci_lo,
                    "mse_ratio_ci95_hi": ratio_ci_hi,
                    "reference_decision_ms_mean": float(np.mean(ref_ms_arr)),
                    "baseline_decision_ms_mean": float(np.mean(base_ms_arr)),
                    "latency_speedup_baseline_over_reference": latency_speedup(paired),
                    "latency_speedup_ci95_lo": speed_ci_lo,
                    "latency_speedup_ci95_hi": speed_ci_hi,
                    "wilcoxon_w": w,
                    "wilcoxon_p": p,
                    "wilcoxon_p_bh": p,
                    "wilcoxon_p_holm": p,
                    "cliffs_delta": cliffs_delta(ref_mse, base_mse),
                    "rank_biserial": rank_biserial_paired(ref_mse, base_mse),
                }
            )
    p_values = [float(row["wilcoxon_p"]) for row in rows]
    for row, p_bh, p_holm in zip(rows, p_adjust_bh(p_values), p_adjust_holm(p_values)):
        row["wilcoxon_p_bh"] = p_bh
        row["wilcoxon_p_holm"] = p_holm
    return rows


def write_csv(rows, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows, out: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Shots & Comparison & $n$ & MSE ratio & $\Delta$MSE & Speedup & $p_{\mathrm{BH}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        ref_name = str(row["reference"]).replace("_", r"\_")
        base_name = str(row["baseline"]).replace("_", r"\_")
        comp = f"{ref_name} / {base_name}"
        lines.append(
            f"{row['shots']} & {comp} & {row['paired_tasks']} & "
            f"{row['mse_ratio_reference_over_baseline']:.3g} & "
            f"{row['mse_delta_reference_minus_baseline']:.3g} & "
            f"{row['latency_speedup_baseline_over_reference']:.3g} & "
            f"{row['wilcoxon_p_bh']:.3g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_acceptance_latex(rows, out: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Shots & Baseline & MSE ratio [95\% CI] & Speedup [95\% CI] & $\Delta$MSE & $p_{\mathrm{BH}}$ \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=lambda item: (int(item["shots"]), str(item["baseline"]))):
        baseline = str(row["baseline"]).replace("_", r"\_")
        ratio = f"{row['mse_ratio_reference_over_baseline']:.3g} [{row['mse_ratio_ci95_lo']:.3g}, {row['mse_ratio_ci95_hi']:.3g}]"
        speed = f"{row['latency_speedup_baseline_over_reference']:.3g} [{row['latency_speedup_ci95_lo']:.3g}, {row['latency_speedup_ci95_hi']:.3g}]"
        lines.append(
            f"{row['shots']} & {baseline} & {ratio} & {speed} & "
            f"{row['mse_delta_reference_minus_baseline']:.3g} & {row['wilcoxon_p_bh']:.3g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired SymQNet-vs-baseline statistics from per-episode rows.")
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument("--reference", default="symqnet")
    parser.add_argument("--baselines", nargs="+", default=["bald_2step_fast", "fisher_greedy_fast", "fixed_optimized", "random"])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-tex", default=None)
    parser.add_argument("--acceptance-tex", default=None)
    args = parser.parse_args()

    rows = comparison_rows(load_episode_rows(Path(args.episodes_csv)), args.reference, args.baselines)
    if not rows:
        raise SystemExit("No paired rows found for requested comparisons.")
    write_csv(rows, Path(args.out_csv))
    print(f"saved {args.out_csv}")
    if args.out_tex:
        write_latex(rows, Path(args.out_tex))
        print(f"saved {args.out_tex}")
    if args.acceptance_tex:
        write_acceptance_latex(rows, Path(args.acceptance_tex))
        print(f"saved {args.acceptance_tex}")


if __name__ == "__main__":
    main()
