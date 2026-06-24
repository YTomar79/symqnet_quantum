from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "symqnet_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "symqnet_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _nice(name: str) -> str:
    return name.replace("_", " ")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean_by(rows: list[dict[str, str]], keys: tuple[str, ...], value: str):
    grouped = defaultdict(list)
    for row in rows:
        if row.get(value, "") == "":
            continue
        grouped[tuple(row[k] for k in keys)].append(float(row[value]))
    return {key: float(np.mean(vals)) for key, vals in grouped.items()}


def latency(summary_csv: Path, out: Path, shot: int | None = None) -> None:
    rows = _read_csv(summary_csv)
    if shot is not None:
        rows = [row for row in rows if int(float(row["shots"])) == shot]
    values = _mean_by(rows, ("method",), "decision_ms_mean")
    methods = sorted(values, key=values.get, reverse=True)
    labels = [method[0] for method in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh([_nice(m) for m in labels], [values[m] for m in methods], color="#2563eb")
    ax.set_xlabel("Mean online decision time (ms)")
    ax.set_xscale("log")
    title = "Online decision latency" if shot is None else f"Online decision latency, {shot} shots"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def pareto(summary_csv: Path, out: Path) -> None:
    rows = _read_csv(summary_csv)
    mse = _mean_by(rows, ("method", "shots"), "mse_mean")
    latency_ms = _mean_by(rows, ("method", "shots"), "decision_ms_mean")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for (method, shots), mse_value in sorted(mse.items()):
        ms = latency_ms.get((method, shots))
        if ms is None:
            continue
        ax.scatter(ms, mse_value, s=42)
        ax.annotate(f"{_nice(method)} {shots}", (ms, mse_value), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean online decision time (ms)")
    ax.set_ylabel("Mean final theta-MSE")
    ax.set_title("Accuracy-latency Pareto view")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def scaling(summary_csv: Path, out: Path) -> None:
    rows = _read_csv(summary_csv)
    methods = sorted({row["method"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for method in methods:
        points_latency = sorted(
            (int(float(row["n_qubits"])), float(row["decision_ms_mean"]))
            for row in rows
            if row["method"] == method and row.get("decision_ms_mean", "") != ""
        )
        points_mse = sorted(
            (int(float(row["n_qubits"])), float(row["mse_mean"]))
            for row in rows
            if row["method"] == method and row.get("mse_mean", "") != ""
        )
        if points_latency:
            axes[0].plot([p[0] for p in points_latency], [p[1] for p in points_latency], marker="o", label=_nice(method))
        if points_mse:
            axes[1].plot([p[0] for p in points_mse], [p[1] for p in points_mse], marker="o", label=_nice(method))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Qubits N")
    axes[0].set_ylabel("Mean decision time (ms)")
    axes[0].set_title("Decision latency scaling")
    axes[1].set_xlabel("Qubits N")
    axes[1].set_ylabel("Mean final theta-MSE")
    axes[1].set_title("Accuracy under scaling")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def ablation_delta(summary_csv: Path, out: Path, reference: str = "full", shot: int = 128) -> None:
    rows = [row for row in _read_csv(summary_csv) if int(float(row["shots"])) == shot]
    values = _mean_by(rows, ("method",), "mse_mean")
    reference_key = (reference,)
    if reference_key not in values:
        raise SystemExit(f"Reference method {reference!r} not found at shot={shot}.")
    ref = values[reference_key]
    methods = [m for m in sorted(values) if m != reference_key]
    deltas = [100.0 * (values[m] - ref) / ref for m in methods]
    colors = ["#dc2626" if d > 0 else "#16a34a" for d in deltas]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.barh([_nice(m[0]) for m in methods], deltas, color=colors)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel(f"Mean MSE delta vs {_nice(reference)} (%)")
    ax.set_title(f"Ablation impact at {shot} shots")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def action_heatmap(episodes_csv: Path, out: Path, n_qubits: int, m_evo: int, method: str, shot: int | None = None) -> None:
    counts = Counter()
    for row in _read_csv(episodes_csv):
        if row["method"] != method:
            continue
        if shot is not None and int(float(row["shots"])) != shot:
            continue
        for item in row.get("actions", "").split():
            action = int(item)
            time_idx = action % m_evo
            action //= m_evo
            _basis_idx = action % 3
            qubit_idx = action // 3
            counts[(qubit_idx, time_idx)] += 1
    matrix = np.zeros((n_qubits, m_evo), dtype=np.float64)
    for (q, t), count in counts.items():
        if 0 <= q < n_qubits and 0 <= t < m_evo:
            matrix[q, t] = count
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Action count")
    ax.set_xlabel("Evolution-time index")
    ax.set_ylabel("Qubit index")
    title = f"{_nice(method)} action heatmap" if shot is None else f"{_nice(method)} action heatmap, {shot} shots"
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-facing SVG figure utilities.")
    sub = parser.add_subparsers(dest="figure", required=True)

    p = sub.add_parser("latency")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shot", type=int, default=None)

    p = sub.add_parser("pareto")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("scaling")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("ablation-delta")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default="full")
    p.add_argument("--shot", type=int, default=128)

    p = sub.add_parser("action-heatmap")
    p.add_argument("--episodes-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-qubits", type=int, required=True)
    p.add_argument("--m-evo", type=int, required=True)
    p.add_argument("--method", default="symqnet")
    p.add_argument("--shot", type=int, default=None)

    args = parser.parse_args()
    if args.figure == "latency":
        latency(Path(args.csv), Path(args.out), args.shot)
    elif args.figure == "pareto":
        pareto(Path(args.csv), Path(args.out))
    elif args.figure == "scaling":
        scaling(Path(args.csv), Path(args.out))
    elif args.figure == "ablation-delta":
        ablation_delta(Path(args.csv), Path(args.out), args.reference, args.shot)
    elif args.figure == "action-heatmap":
        action_heatmap(Path(args.episodes_csv), Path(args.out), args.n_qubits, args.m_evo, args.method, args.shot)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
