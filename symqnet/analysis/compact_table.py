from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compact_rows(
    paired_csv: Path,
    shots: list[int],
    baselines: list[str] | None = None,
    min_paired_tasks: int = 100,
) -> list[dict[str, object]]:
    wanted_shots = {int(item) for item in shots}
    wanted_baselines = set(baselines or [])
    rows = []
    for row in _read_csv(paired_csv):
        shot = int(float(row["shots"]))
        baseline = row["baseline"]
        if shot not in wanted_shots:
            continue
        if wanted_baselines and baseline not in wanted_baselines:
            continue
        paired_tasks = int(float(row["paired_tasks"]))
        if paired_tasks < min_paired_tasks:
            raise ValueError(
                f"Refusing to build compact paper table from smoke-sized result: "
                f"shot={shot}, baseline={baseline}, paired_tasks={paired_tasks} < {min_paired_tasks}"
            )
        rows.append(
            {
                "shots": shot,
                "baseline": baseline,
                "paired_tasks": paired_tasks,
                "mse_ratio": float(row["mse_ratio_reference_over_baseline"]),
                "mse_ratio_ci95_lo": float(row["mse_ratio_ci95_lo"]),
                "mse_ratio_ci95_hi": float(row["mse_ratio_ci95_hi"]),
                "latency_speedup": float(row["latency_speedup_baseline_over_reference"]),
                "latency_speedup_ci95_lo": float(row["latency_speedup_ci95_lo"]),
                "latency_speedup_ci95_hi": float(row["latency_speedup_ci95_hi"]),
                "wilcoxon_p_bh": float(row.get("wilcoxon_p_bh", row.get("wilcoxon_p", "nan"))),
                "cliffs_delta": float(row.get("cliffs_delta", "nan")),
            }
        )
    if not rows:
        raise ValueError(f"No paired rows found for shots={sorted(wanted_shots)} and baselines={sorted(wanted_baselines)}")
    return sorted(rows, key=lambda item: (int(item["shots"]), str(item["baseline"])))


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Shots & Baseline & $n$ & MSE ratio [95\% CI] & Speedup [95\% CI] & $p_{\mathrm{BH}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        baseline = str(row["baseline"]).replace("_", r"\_")
        ratio = f"{row['mse_ratio']:.3g} [{row['mse_ratio_ci95_lo']:.3g}, {row['mse_ratio_ci95_hi']:.3g}]"
        speed = f"{row['latency_speedup']:.3g} [{row['latency_speedup_ci95_lo']:.3g}, {row['latency_speedup_ci95_hi']:.3g}]"
        lines.append(f"{row['shots']} & {baseline} & {row['paired_tasks']} & {ratio} & {speed} & {row['wilcoxon_p_bh']:.3g} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact camera-ready main-result table.")
    parser.add_argument("--paired-csv", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--baselines", nargs="+", default=["bald_2step_fast", "fisher_greedy_fast", "fixed_optimized", "random"])
    parser.add_argument("--min-paired-tasks", type=int, default=100)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-tex", required=True)
    args = parser.parse_args()

    rows = compact_rows(Path(args.paired_csv), args.shots, args.baselines, args.min_paired_tasks)
    if args.out_csv:
        write_csv(rows, Path(args.out_csv))
        print(f"saved {args.out_csv}")
    write_latex(rows, Path(args.out_tex))
    print(f"saved {args.out_tex}")


if __name__ == "__main__":
    main()
