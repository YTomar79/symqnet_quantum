from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except Exception:  # pragma: no cover - scipy is a declared dependency, fallback is defensive.
    scipy_wilcoxon = None


def _rank_abs(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(abs(v) for v in values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and math.isclose(indexed[j][1], indexed[i][1]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, float, int]:
    diffs = [a - b for a, b in zip(x, y) if not math.isclose(a, b, abs_tol=1e-15)]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0
    if scipy_wilcoxon is not None:
        try:
            result = scipy_wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
            return float(result.statistic), float(result.pvalue), n
        except ValueError:
            return 0.0, 1.0, 0
    ranks = _rank_abs(diffs)
    w_pos = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0)
    w_neg = sum(rank for rank, diff in zip(ranks, diffs) if diff < 0)
    w = min(w_pos, w_neg)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w - mean) / math.sqrt(var) if var > 0 else 0.0
    p_two_sided = math.erfc(abs(z) / math.sqrt(2.0))
    return float(w), float(p_two_sided), n


def p_adjust_bh(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    ordered = sorted(enumerate(p_values), key=lambda item: item[1], reverse=True)
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_end, (idx, p_value) in enumerate(ordered):
        rank = m - rank_from_end
        running = min(running, float(p_value) * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def p_adjust_holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * m
    running = 0.0
    for rank, (idx, p_value) in enumerate(ordered, start=1):
        running = max(running, float(p_value) * (m - rank + 1))
        adjusted[idx] = min(1.0, running)
    return adjusted


def cliffs_delta(x: list[float], y: list[float]) -> float:
    if not x or not y:
        return float("nan")
    greater = 0
    lesser = 0
    for a in x:
        for b in y:
            if a > b:
                greater += 1
            elif a < b:
                lesser += 1
    return float((greater - lesser) / (len(x) * len(y)))


def rank_biserial_paired(x: list[float], y: list[float]) -> float:
    diffs = [a - b for a, b in zip(x, y) if not math.isclose(a, b, abs_tol=1e-15)]
    if not diffs:
        return 0.0
    ranks = _rank_abs(diffs)
    w_pos = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0)
    w_neg = sum(rank for rank, diff in zip(ranks, diffs) if diff < 0)
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom else 0.0


def load_episode_rows(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "method": row["method"],
                    "shots": int(float(row["shots"])),
                    "seed": int(float(row["seed"])) if row["seed"] not in {"", "None"} else -1,
                    "episode_idx": int(float(row["episode_idx"])),
                    "task_id": int(float(row["task_id"])) if row.get("task_id", "") not in {"", "None"} else None,
                    "final_mse": float(row["final_mse"]),
                }
            )
    return rows


def make_table(rows, reference: str) -> str:
    by_cell = defaultdict(dict)
    for row in rows:
        paired_idx = row["task_id"] if row.get("task_id") is not None else row["episode_idx"]
        key = (row["shots"], row["seed"], paired_idx)
        by_cell[(row["method"], key)] = row["final_mse"]

    methods = sorted({row["method"] for row in rows if row["method"] != reference})
    shots_values = sorted({row["shots"] for row in rows})
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Shots & Method & $n$ & $W$ & $p$ \\",
        r"\midrule",
    ]
    for shots in shots_values:
        ref_items = {
            key: value
            for (method, key), value in by_cell.items()
            if method == reference and key[0] == shots
        }
        for method in methods:
            x = []
            y = []
            for key, ref_value in sorted(ref_items.items()):
                value = by_cell.get((method, key))
                if value is not None:
                    x.append(ref_value)
                    y.append(value)
            if not x:
                continue
            w, p, n = wilcoxon_signed_rank(x, y)
            lines.append(f"{shots} & {method.replace('_', r'\\_')} & {n} & {w:.1f} & {p:.3g} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paired Wilcoxon LaTeX table from per-episode eval CSV.")
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument("--reference", default="full")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    table = make_table(load_episode_rows(Path(args.episodes_csv)), args.reference)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"saved {out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
