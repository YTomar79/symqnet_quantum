from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4b5563",
    "#be123c",
]


def _nice_name(name: str) -> str:
    return name.replace("_", " ")


def read_rows(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    grouped: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            shots = int(float(row["shots"]))
            grouped[(method, shots)].append(
                {
                    "mse_median": float(row["mse_median"]),
                    "mse_iqr": float(row.get("mse_iqr", 0.0)),
                    "mse_mean": float(row.get("mse_mean", row["mse_median"])),
                }
            )

    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for (method, shots), values in grouped.items():
        out[method][shots] = {
            key: sum(v[key] for v in values) / len(values)
            for key in ("mse_median", "mse_iqr", "mse_mean")
        }
    return dict(out)


def write_svg(series: dict[str, dict[int, dict[str, float]]], out_path: Path, title: str) -> None:
    width, height = 980, 620
    left, right, top, bottom = 92, 260, 56, 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    shots = sorted({shot for points in series.values() for shot in points})
    values = []
    for points in series.values():
        for item in points.values():
            values.extend([item["mse_median"] - 0.5 * item["mse_iqr"], item["mse_median"] + 0.5 * item["mse_iqr"]])
    y_min = max(0.0, min(values) if values else 0.0)
    y_max = max(values) if values else 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad

    x_logs = [math.log2(x) for x in shots]
    x_min, x_max = min(x_logs), max(x_logs)
    if math.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5

    def sx(shot: int) -> float:
        return left + (math.log2(shot) - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}",
        ".axis{stroke:#111827;stroke-width:1.2}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".label{font-size:14px;fill:#374151}",
        ".tick{font-size:12px;fill:#4b5563}",
        ".title{font-size:22px;font-weight:700}",
        ".legend{font-size:13px;fill:#111827}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="34">{title}</text>',
    ]

    y_ticks = 5
    for i in range(y_ticks + 1):
        value = y_min + (y_max - y_min) * i / y_ticks
        y = sy(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.3g}</text>')

    for shot in shots:
        x = sx(shot)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{shot}</text>')

    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="label" x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle">Shots per measurement</text>')
    parts.append(f'<text class="label" transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Median final theta-MSE</text>')

    for idx, (method, points) in enumerate(series.items()):
        color = PALETTE[idx % len(PALETTE)]
        ordered = [(shot, points[shot]) for shot in shots if shot in points]
        if not ordered:
            continue
        upper = [(sx(s), sy(v["mse_median"] + 0.5 * v["mse_iqr"])) for s, v in ordered]
        lower = [(sx(s), sy(max(0.0, v["mse_median"] - 0.5 * v["mse_iqr"]))) for s, v in reversed(ordered)]
        band = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
        line = " ".join(f"{sx(s):.2f},{sy(v['mse_median']):.2f}" for s, v in ordered)
        parts.append(f'<polygon points="{band}" fill="{color}" opacity="0.13"/>')
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>')
        for shot, item in ordered:
            parts.append(f'<circle cx="{sx(shot):.2f}" cy="{sy(item["mse_median"]):.2f}" r="4" fill="{color}"/>')

        lx = left + plot_w + 34
        ly = top + 24 + 24 * idx
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<text class="legend" x="{lx + 32}" y="{ly + 4}">{_nice_name(method)}</text>')

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the shot-budget SVG figure from an eval CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default="Shot budget vs final theta-MSE")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".svg")
    series = read_rows(csv_path)
    write_svg(series, out_path, args.title)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
