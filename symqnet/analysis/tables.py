from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _fmt(value: str | float, digits: int = 3) -> str:
    if value == "" or value is None:
        return "--"
    return f"{float(value):.{digits}g}"


def load_summary(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(float(row["shots"])))].append(row)
    out = []
    for (method, shots), items in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        def mean_field(name: str):
            vals = [float(item[name]) for item in items if item.get(name, "") not in {"", None}]
            return sum(vals) / len(vals) if vals else ""

        out.append(
            {
                "method": method,
                "shots": shots,
                "mse_mean": mean_field("mse_mean"),
                "decision_ms_mean": mean_field("decision_ms_mean"),
                "decision_ms_p95": mean_field("decision_ms_p95"),
                "mse_crlb_ratio_mean": mean_field("mse_crlb_ratio_mean"),
            }
        )
    return out


def latex_table(rows) -> str:
    has_crlb = any(row.get("mse_crlb_ratio_mean", "") not in {"", None} for row in rows)
    lines = [r"\begin{tabular}{llrrr}" if has_crlb else r"\begin{tabular}{llrr}", r"\toprule"]
    if has_crlb:
        lines.append(r"Shots & Method & MSE & Decision ms & MSE/CRLB \\")
    else:
        lines.append(r"Shots & Method & MSE & Decision ms \\")
    lines.append(r"\midrule")
    for row in rows:
        method = row["method"].replace("_", r"\_")
        base = f"{row['shots']} & {method} & {_fmt(row['mse_mean'])} & {_fmt(row['decision_ms_mean'])}"
        if has_crlb:
            base += f" & {_fmt(row['mse_crlb_ratio_mean'])}"
        lines.append(base + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit wall-clock/MSE LaTeX table from eval CSV, including MSE/CRLB only when present.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    table = latex_table(aggregate(load_summary(Path(args.csv))))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"saved {out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
