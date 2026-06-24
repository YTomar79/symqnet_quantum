from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and np.isclose(values[order[j]], values[order[i]]):
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def reward_objective_rows(episodes_csv: Path, methods: list[str] | None = None) -> list[dict[str, object]]:
    wanted = set(methods or [])
    grouped: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    for row in _read_csv(episodes_csv):
        method = row["method"]
        if wanted and method not in wanted:
            continue
        if row.get("total_info_gain", "") in {"", "None"}:
            continue
        grouped[(int(float(row["shots"])), method)].append((float(row["total_info_gain"]), float(row["final_mse"])))

    rows = []
    for (shot, method), pairs in sorted(grouped.items()):
        data = np.asarray(pairs, dtype=np.float64)
        info_gain = data[:, 0]
        mse = data[:, 1]
        rows.append(
            {
                "shots": shot,
                "method": method,
                "episodes": int(data.shape[0]),
                "info_gain_mean": float(info_gain.mean()),
                "final_mse_mean": float(mse.mean()),
                "pearson_info_gain_vs_mse": _pearson(info_gain, mse),
                "spearman_info_gain_vs_mse": _pearson(_rankdata(info_gain), _rankdata(mse)),
            }
        )
    if not rows:
        raise ValueError(f"No reward/objective rows found in {episodes_csv}")
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Shots & Method & $n$ & Pearson & Spearman \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        lines.append(
            f"{row['shots']} & {method} & {row['episodes']} & "
            f"{row['pearson_info_gain_vs_mse']:.3g} & {row['spearman_info_gain_vs_mse']:.3g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerically relate entropy-reduction reward to final MSE objective.")
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument("--methods", nargs="+", default=["symqnet"])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-tex", default=None)
    args = parser.parse_args()

    rows = reward_objective_rows(Path(args.episodes_csv), args.methods)
    write_csv(rows, Path(args.out_csv))
    print(f"saved {args.out_csv}")
    if args.out_tex:
        write_latex(rows, Path(args.out_tex))
        print(f"saved {args.out_tex}")


if __name__ == "__main__":
    main()
