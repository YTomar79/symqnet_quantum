from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ablation_rows(summary_csv: Path, reference: str = "full", shot: int = 128) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in _read_csv(summary_csv):
        if int(float(row["shots"])) != int(shot):
            continue
        grouped[row["method"]].append(float(row["mse_mean"]))
    if reference not in grouped:
        raise ValueError(f"Reference method {reference!r} not found at shot={shot}.")
    ref_mean = float(np.mean(grouped[reference]))
    rows = []
    for method, values in sorted(grouped.items()):
        mse = float(np.mean(values))
        rows.append(
            {
                "shot": shot,
                "method": method,
                "mse_mean": mse,
                "delta_mse": mse - ref_mean,
                "delta_pct": 100.0 * (mse - ref_mean) / max(ref_mean, 1e-12),
            }
        )
    return rows


def write_latex(rows: list[dict[str, object]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & MSE & $\Delta$MSE & $\Delta$\% \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        lines.append(f"{method} & {row['mse_mean']:.3g} & {row['delta_mse']:.3g} & {row['delta_pct']:.3g} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact paper ablation table at one shot budget.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default="full")
    parser.add_argument("--shot", type=int, default=128)
    args = parser.parse_args()
    rows = ablation_rows(Path(args.csv), args.reference, args.shot)
    write_latex(rows, Path(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
