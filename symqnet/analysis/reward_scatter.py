from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = {
    "random": "#6b7280",
    "fixed": "#16a34a",
    "fixed_optimized": "#16a34a",
    "fisher_greedy": "#0f766e",
    "fisher_greedy_fast": "#0f766e",
    "bald_2step": "#dc2626",
    "bald_2step_fast": "#dc2626",
    "symqnet": "#2563eb",
}


def load_points(path: Path):
    points = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mse = float(row["final_mse"])
            if mse <= 0:
                continue
            points.append((row["method"], float(row["total_info_gain"]), 1.0 / mse))
    return points


def write_svg(points, out: Path) -> None:
    width, height = 760, 520
    left, right, top, bottom = 78, 34, 42, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [p[1] for p in points] or [0.0, 1.0]
    ys = [p[2] for p in points] or [0.0, 1.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)
    x_min -= x_pad
    x_max += x_pad
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad

    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}.tick{font-size:12px;fill:#4b5563}.label{font-size:14px;fill:#374151}.title{font-size:20px;font-weight:700}.grid{stroke:#e5e7eb}.axis{stroke:#111827;stroke-width:1.2}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="28">Reward sanity check</text>',
    ]
    for i in range(6):
        x = left + i * plot_w / 5
        y = top + i * plot_h / 5
        x_val = x_min + i * (x_max - x_min) / 5
        y_val = y_max - i * (y_max - y_min) / 5
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>')
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle">{x_val:.2g}</text>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{y_val:.2g}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="label" x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle">Total information-gain reward</text>')
    parts.append(f'<text class="label" transform="translate(22 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">1 / final theta-MSE</text>')

    for method, x, y in points:
        color = COLORS.get(method, "#9333ea")
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.2" fill="{color}" opacity="0.62"/>')
    parts.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot total info-gain reward vs inverse final MSE.")
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_svg(load_points(Path(args.episodes_csv)), Path(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
