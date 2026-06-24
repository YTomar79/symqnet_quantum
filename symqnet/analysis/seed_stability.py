from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_stability_rows(episodes_csv: Path, method: str = "symqnet") -> list[dict[str, object]]:
    by_shot_seed: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in _read_csv(episodes_csv):
        if row["method"] != method:
            continue
        train_seed = row.get("train_seed", "")
        if train_seed in {"", "None"}:
            continue
        key = (int(float(row["shots"])), int(float(train_seed)))
        by_shot_seed[key].append(float(row["final_mse"]))

    by_shot: dict[int, list[float]] = defaultdict(list)
    for (shot, _seed), values in by_shot_seed.items():
        by_shot[shot].append(float(np.mean(values)))

    rows = []
    for shot, seed_means in sorted(by_shot.items()):
        values = np.asarray(seed_means, dtype=np.float64)
        mean = float(values.mean())
        rows.append(
            {
                "shots": shot,
                "method": method,
                "train_seeds": int(values.size),
                "seed_mean_mse_mean": mean,
                "seed_mean_mse_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "seed_mean_mse_min": float(values.min()),
                "seed_mean_mse_max": float(values.max()),
                "seed_mean_mse_cv": float(values.std(ddof=1) / max(mean, 1e-12)) if values.size > 1 else 0.0,
            }
        )
    if not rows:
        raise ValueError(f"No train-seed rows found for method={method!r} in {episodes_csv}")
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Shots & Train seeds & Mean MSE & Seed std & Range \\",
        r"\midrule",
    ]
    for row in rows:
        value_range = f"{row['seed_mean_mse_min']:.3g}-{row['seed_mean_mse_max']:.3g}"
        lines.append(
            f"{row['shots']} & {row['train_seeds']} & {row['seed_mean_mse_mean']:.3g} & "
            f"{row['seed_mean_mse_std']:.3g} & {value_range} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SymQNet train-seed stability from per-episode rows.")
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument("--method", default="symqnet")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-tex", required=True)
    args = parser.parse_args()

    rows = seed_stability_rows(Path(args.episodes_csv), args.method)
    if args.out_csv:
        write_csv(rows, Path(args.out_csv))
        print(f"saved {args.out_csv}")
    write_latex(rows, Path(args.out_tex))
    print(f"saved {args.out_tex}")


if __name__ == "__main__":
    main()
