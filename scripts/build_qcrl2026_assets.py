from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "paper" / "qcrl2026_assets"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
DATA_DIR = OUT / "data"


LABEL = {
    "symqnet": "SymQNet",
    "dad_transformer": "DAD-style Transformer",
    "random": "Random",
    "fixed": "Fixed schedule",
    "fixed_optimized": "Optimized fixed",
    "fisher_greedy_fast": "Fisher-info (bounded)",
    "bald_2step_fast": "2-step BALD (bounded)",
    "full": "Full SymQNet",
    "no_vae": "No VAE",
    "no_graph": "No graph encoder",
    "mlp_only": "MLP, no transformer",
}


COLORS = {
    "symqnet": "#0072B2",
    "dad_transformer": "#009E73",
    "random": "#999999",
    "fixed": "#666666",
    "fixed_optimized": "#CC79A7",
    "fisher_greedy_fast": "#D55E00",
    "bald_2step_fast": "#E69F00",
    "full": "#0072B2",
    "no_vae": "#56B4E9",
    "no_graph": "#D55E00",
    "mlp_only": "#CC79A7",
}


MARKERS = {
    "symqnet": "o",
    "dad_transformer": "s",
    "random": "D",
    "fixed": "v",
    "fixed_optimized": "P",
    "fisher_greedy_fast": "^",
    "bald_2step_fast": "X",
}


def setup() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for path in (FIG_DIR, TABLE_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "method_label_map.json").write_text(json.dumps(LABEL, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "figure.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def label(method: str) -> str:
    return LABEL.get(method, method.replace("_", " "))


def _ci95(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(clean.mean())
    if len(clean) <= 1:
        return mean, mean, mean
    sem = float(clean.std(ddof=1) / math.sqrt(len(clean)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _fmt_num(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}g}"


def _fmt_fixed(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _bold(value: object) -> str:
    return rf"\textbf{{{value}}}"


def _fmt_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0 or value < 1e-300:
        return "$<10^{-300}$"
    if abs(value) < 0.001:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${float(mantissa):.2g}\times10^{{{int(exponent)}}}$"
    return _fmt_num(value, 2)


def _fmt_mean_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if abs(hi - lo) < 1e-12:
        return _fmt_num(mean, digits)
    return f"{_fmt_num(mean, digits)} [{_fmt_num(lo, digits)}, {_fmt_num(hi, digits)}]"


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    saved = []
    for ext, dpi in (("svg", None), ("pdf", None), ("png", 300)):
        path = FIG_DIR / f"{stem}.{ext}"
        kwargs = {"dpi": dpi} if dpi else {}
        fig.savefig(path, **kwargs)
        saved.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return saved


FLOW_BOX_FONTSIZE = 7.4
FLOW_ARROW_FONTSIZE = 7.4
FLOW_PANEL_FONTSIZE = 9.2
FLOW_BOX_LINESPACING = 1.38


def _flow_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fontsize: float = FLOW_BOX_FONTSIZE,
    weight: str = "normal",
    linespacing: float = FLOW_BOX_LINESPACING,
) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="black", linewidth=1.25))
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        fontfamily="sans-serif",
        linespacing=linespacing,
    )


def _flow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label_text: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10.5,
        linewidth=1.25,
        color="black",
        connectionstyle=connectionstyle,
        shrinkA=0.5,
        shrinkB=0.5,
    )
    ax.add_patch(arrow)
    if label_text:
        ax.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label_text,
            ha="center",
            va="center",
            fontsize=FLOW_ARROW_FONTSIZE,
            fontfamily="sans-serif",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )


def build_mdp_method_flow() -> list[str]:
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
        }
    ):
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.34))
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        ax = axes[0]
        ax.text(
            0.0,
            0.98,
            "(a) Belief-state MDP loop",
            ha="left",
            va="top",
            fontsize=FLOW_PANEL_FONTSIZE,
            fontweight="bold",
            fontfamily="sans-serif",
        )
        _flow_box(ax, 0.03, 0.285, 0.30, 0.30, "Belief state\n$s_t=(b_t,h_t,G,t)$")
        _flow_box(ax, 0.43, 0.56, 0.24, 0.28, "Policy\n$\\pi_\\phi(a_t|s_t)$")
        _flow_box(ax, 0.75, 0.56, 0.22, 0.28, "Experiment\n$a_t=(i,\\sigma,\\tau)$")
        _flow_box(ax, 0.75, 0.035, 0.22, 0.28, "Outcome\n$y_t$")
        _flow_box(ax, 0.43, 0.035, 0.24, 0.28, "SMC update\n$b_{t+1}$")
        _flow_arrow(ax, (0.33, 0.435), (0.43, 0.70), "$s_t$", (-0.01, 0.06))
        _flow_arrow(ax, (0.67, 0.70), (0.75, 0.70), "$a_t$", (0.0, 0.055))
        _flow_arrow(ax, (0.86, 0.56), (0.86, 0.315), "$p(y_t\\,|\\,a_t,\\,\\theta)$", (0.1, 0.02))
        _flow_arrow(ax, (0.75, 0.175), (0.67, 0.175))
        _flow_arrow(
            ax,
            (0.43, 0.175),
            (0.29, 0.285),
            "$s_{t+1}, r_t$",
            (-0.02, -0.067),
            connectionstyle="arc3,rad=0.12",
        )

        ax = axes[1]
        ax.text(
            0.0,
            0.98,
            "(b) Offline training and evaluation",
            ha="left",
            va="top",
            fontsize=FLOW_PANEL_FONTSIZE,
            fontweight="bold",
            fontfamily="sans-serif",
        )
        _flow_box(ax, 0.03, 0.495, 0.22, 0.26, "TFIM tasks\n$\\theta\\sim p(\\theta)$")
        _flow_box(ax, 0.31, 0.495, 0.22, 0.26, "Simulated\nepisodes")
        _flow_box(ax, 0.59, 0.495, 0.24, 0.26, "PPO update\ninfo gain")
        _flow_box(ax, 0.49, 0.055, 0.24, 0.26, "Trained\npolicy")
        _flow_box(ax, 0.77, 0.055, 0.22, 0.26, "Eval.\nMSE, latency")
        _flow_arrow(ax, (0.25, 0.625), (0.31, 0.625))
        _flow_arrow(ax, (0.53, 0.625), (0.59, 0.625))
        _flow_arrow(ax, (0.71, 0.495), (0.71, 0.315), "select", (0.074, 0.01))
        _flow_arrow(ax, (0.73, 0.185), (0.77, 0.185))
        _flow_arrow(
            ax,
            (0.49, 0.185),
            (0.42, 0.495),
            "policy",
            (-0.08, -0.065),
            connectionstyle="arc3,rad=-0.25",
        )
        fig.subplots_adjust(left=0.015, right=0.99, top=0.985, bottom=0.03, hspace=0.23)
        return save_figure(fig, "fig0_mdp_method_flow")


def write_latex(rows: list[list[str]], headers: list[str], out: Path, align: str | None = None) -> None:
    align = align or ("l" * len(headers))
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def copy_core_inputs() -> list[str]:
    copied = []
    for src in [
        RUNS / "main_result" / "paired_main.csv",
        RUNS / "main_result" / "shot_budget.csv",
        RUNS / "main_result" / "paper_readiness_report.json",
        RUNS / "scaling" / "scaling_summary.csv",
        RUNS / "scaling" / "claim_gate.json",
        RUNS / "scaling" / "mps_validation.json",
        RUNS / "scaling" / "complexity.csv",
        RUNS / "ablations_paper" / "paired_ablations.csv",
        RUNS / "ablations_paper" / "shot_budget.csv",
        RUNS / "noisy_native" / "paired_main.csv",
        RUNS / "ood_wide" / "paired_main.csv",
        RUNS / "reward_mse_delta" / "paired_main.csv",
    ]:
        if src.exists():
            dst = DATA_DIR / f"{src.parent.name}_{src.name}"
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(ROOT)))
    return copied


def build_main_pareto() -> list[str]:
    df = pd.read_csv(RUNS / "main_result" / "shot_budget.csv")
    selected = [
        "symqnet",
        "dad_transformer",
        "fisher_greedy_fast",
        "bald_2step_fast",
        "fixed_optimized",
        "fixed",
        "random",
    ]
    shots = [128, 512]
    plot_raw = df[df["method"].isin(selected) & df["shots"].isin(shots)].copy()
    rows = []
    for (shot, method), group in plot_raw.groupby(["shots", "method"], sort=True):
        mse_mean, mse_lo, mse_hi = _ci95(group["mse_mean"])
        lat_mean, lat_lo, lat_hi = _ci95(group["decision_ms_mean"])
        rows.append(
            {
                "shots": int(shot),
                "method": method,
                "method_label": label(method),
                "mse_mean": mse_mean,
                "mse_ci95_lo": mse_lo,
                "mse_ci95_hi": mse_hi,
                "decision_ms_mean": lat_mean,
                "decision_ms_ci95_lo": lat_lo,
                "decision_ms_ci95_hi": lat_hi,
                "n_rows": len(group),
            }
        )
    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(DATA_DIR / "fig_main_pareto_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharey=True)
    for ax, shot in zip(axes, shots):
        sub = plot_df[plot_df["shots"] == shot]
        for _, row in sub.iterrows():
            method = row["method"]
            x = row["decision_ms_mean"]
            y = row["mse_mean"]
            xerr = np.array(
                [
                    [max(x - row["decision_ms_ci95_lo"], 0.0)],
                    [max(row["decision_ms_ci95_hi"] - x, 0.0)],
                ]
            )
            yerr = np.array(
                [
                    [max(y - row["mse_ci95_lo"], 0.0)],
                    [max(row["mse_ci95_hi"] - y, 0.0)],
                ]
            )
            ax.errorbar(
                [x],
                [y],
                xerr=xerr,
                yerr=yerr,
                fmt=MARKERS.get(method, "o"),
                markersize=6.0 if method == "symqnet" else 5.2,
                color=COLORS.get(method, "#333333"),
                markeredgecolor="black" if method == "symqnet" else "white",
                markeredgewidth=0.45,
                elinewidth=0.7,
                capsize=2.0,
                zorder=3,
                label=label(method),
            )
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.08, linewidth=0.4)
        ax.set_xlabel("Policy decision time per step (ms)")
        ax.set_title(f"N=5, {shot} shots")
        ax.text(
            0.04,
            0.95,
            "lower is better",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="#444444",
        )
    axes[0].set_ylabel(r"Final parameter MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.30, wspace=0.12)
    return save_figure(fig, "fig1_main_pareto")


def build_scaling_figure() -> list[str]:
    df = pd.read_csv(RUNS / "scaling" / "scaling_summary.csv")
    selected = ["fixed", "dad_transformer", "symqnet", "fisher_greedy_fast", "bald_2step_fast"]
    df = df[df["method"].isin(selected)].copy()

    rows = []
    for (n_qubits, method), group in df.groupby(["n_qubits", "method"], sort=True):
        mse_mean, mse_lo, mse_hi = _ci95(group["mse_mean"])
        lat_mean, lat_lo, lat_hi = _ci95(group["decision_ms_mean"])
        rows.append(
            {
                "n_qubits": int(n_qubits),
                "method": method,
                "method_label": label(method),
                "mse_mean": mse_mean,
                "mse_ci95_lo": mse_lo,
                "mse_ci95_hi": mse_hi,
                "decision_ms_mean": lat_mean,
                "decision_ms_ci95_lo": lat_lo,
                "decision_ms_ci95_hi": lat_hi,
                "n_rows": len(group),
            }
        )
    agg = pd.DataFrame(rows).sort_values(["method", "n_qubits"])
    agg.to_csv(DATA_DIR / "fig_scaling_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    for method in selected:
        sub = agg[agg["method"] == method].sort_values("n_qubits")
        if sub.empty:
            continue
        x = sub["n_qubits"].to_numpy()
        y_lat = sub["decision_ms_mean"].to_numpy()
        y_mse = sub["mse_mean"].to_numpy()
        lat_yerr = np.vstack(
            [
                y_lat - sub["decision_ms_ci95_lo"].to_numpy(),
                sub["decision_ms_ci95_hi"].to_numpy() - y_lat,
            ]
        )
        mse_yerr = np.vstack(
            [
                y_mse - sub["mse_ci95_lo"].to_numpy(),
                sub["mse_ci95_hi"].to_numpy() - y_mse,
            ]
        )
        style = {
            "color": COLORS[method],
            "marker": MARKERS.get(method, "o"),
            "linewidth": 1.4 if method == "symqnet" else 1.1,
            "markersize": 4.2 if method == "symqnet" else 3.8,
            "label": label(method),
        }
        axes[0].errorbar(x, y_lat, yerr=lat_yerr, capsize=2.2, **style)
        axes[1].errorbar(x, y_mse, yerr=mse_yerr, capsize=2.2, **style)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of qubits")
    axes[0].set_ylabel("Decision time per step (ms)")
    axes[0].set_title("Online decision cost")
    axes[1].set_xlabel("Number of qubits")
    axes[1].set_ylabel("Final parameter MSE")
    axes[1].set_title("Accuracy under scaling")
    for ax in axes:
        ax.set_xticks([8, 10, 12])
        ax.grid(True, alpha=0.25, linewidth=0.6)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(wspace=0.28, bottom=0.25)
    return save_figure(fig, "fig2_scaling_latency_mse")


def build_combined_evidence_figure() -> list[str]:
    scaling = pd.read_csv(DATA_DIR / "fig_scaling_source.csv")
    pareto = pd.read_csv(DATA_DIR / "fig_main_pareto_source.csv")
    scaling_methods = ["fixed", "dad_transformer", "symqnet", "fisher_greedy_fast", "bald_2step_fast"]
    pareto_methods = [
        "symqnet",
        "dad_transformer",
        "fisher_greedy_fast",
        "bald_2step_fast",
        "fixed_optimized",
        "fixed",
        "random",
    ]

    fig = plt.figure(figsize=(7.35, 2.45))
    gs = fig.add_gridspec(1, 4, wspace=0.42)
    ax_latency = fig.add_subplot(gs[0, 0])
    ax_mse = fig.add_subplot(gs[0, 1])
    ax_p128 = fig.add_subplot(gs[0, 2])
    ax_p512 = fig.add_subplot(gs[0, 3], sharey=ax_p128)

    for method in scaling_methods:
        sub = scaling[scaling["method"] == method].sort_values("n_qubits")
        if sub.empty:
            continue
        x = sub["n_qubits"].to_numpy()
        y_lat = sub["decision_ms_mean"].to_numpy()
        y_mse = sub["mse_mean"].to_numpy()
        style = {
            "color": COLORS[method],
            "marker": MARKERS.get(method, "o"),
            "linewidth": 1.4 if method == "symqnet" else 1.12,
            "markersize": 4.2 if method == "symqnet" else 3.7,
            "markeredgecolor": "#222222",
            "markeredgewidth": 0.35,
            "label": label(method),
        }
        lat_yerr = np.vstack(
            [
                np.maximum(y_lat - sub["decision_ms_ci95_lo"].to_numpy(), 0.0),
                np.maximum(sub["decision_ms_ci95_hi"].to_numpy() - y_lat, 0.0),
            ]
        )
        mse_yerr = np.vstack(
            [
                np.maximum(y_mse - sub["mse_ci95_lo"].to_numpy(), 0.0),
                np.maximum(sub["mse_ci95_hi"].to_numpy() - y_mse, 0.0),
            ]
        )
        ax_latency.errorbar(x, y_lat, yerr=lat_yerr, capsize=1.9, elinewidth=0.75, **style)
        ax_mse.errorbar(x, y_mse, yerr=mse_yerr, capsize=1.9, elinewidth=0.75, **style)

    ax_latency.set_yscale("log")
    ax_latency.set_xlim(7.65, 12.35)
    ax_mse.set_xlim(7.65, 12.35)
    ax_mse.set_ylim(0.066, 0.111)
    ax_latency.set_xticks([8, 10, 12])
    ax_mse.set_xticks([8, 10, 12])
    ax_latency.set_xlabel("Qubits")
    ax_mse.set_xlabel("Qubits")
    ax_latency.set_ylabel("Decision ms/step")
    ax_mse.set_ylabel("Final MSE")
    ax_latency.set_title("(a) Latency")
    ax_mse.set_title("(b) Scaling MSE")
    for ax in (ax_latency, ax_mse):
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="both", which="major", pad=1.5)

    for ax, shot, title in ((ax_p128, 128, "(c) N=5, 128 shots"), (ax_p512, 512, "(d) N=5, 512 shots")):
        sub = pareto[pareto["shots"] == shot]
        for _, row in sub.iterrows():
            method = row["method"]
            if method not in pareto_methods:
                continue
            x = row["decision_ms_mean"]
            y = row["mse_mean"]
            xerr = np.array(
                [
                    [max(x - row["decision_ms_ci95_lo"], 0.0)],
                    [max(row["decision_ms_ci95_hi"] - x, 0.0)],
                ]
            )
            yerr = np.array(
                [
                    [max(y - row["mse_ci95_lo"], 0.0)],
                    [max(row["mse_ci95_hi"] - y, 0.0)],
                ]
            )
            ax.errorbar(
                [x],
                [y],
                xerr=xerr,
                yerr=yerr,
                fmt=MARKERS.get(method, "o"),
                markersize=5.35 if method == "symqnet" else 4.55,
                color=COLORS.get(method, "#333333"),
                markeredgecolor="#111111",
                markeredgewidth=0.55 if method == "symqnet" else 0.35,
                elinewidth=0.75,
                capsize=1.85,
                zorder=4 if method in {"symqnet", "dad_transformer"} else 3,
                label=label(method),
            )
        ax.set_xscale("log")
        ax.set_xlim(7e-4, 220)
        ax.set_ylim(0.024, 0.086)
        ax.set_title(title)
        ax.set_xlabel("Decision ms/step")
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.08, linewidth=0.4)
        ax.text(0.04, 0.94, "lower is better", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color="#444444")
        ax.tick_params(axis="both", which="major", pad=1.5)
    ax_p128.set_ylabel("Final MSE")
    ax_p512.tick_params(labelleft=False)

    handles, legend_labels = [], []
    for ax in (ax_latency, ax_p128):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        legend_labels.extend(l)
    by_label = dict(zip(legend_labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.058, right=0.995, bottom=0.35, top=0.82)
    return save_figure(fig, "fig1_combined_evidence")


def build_robustness_figure() -> list[str]:
    runs = [
        ("Main N=5", RUNS / "main_result" / "paired_main.csv", 128),
        ("Native noise", RUNS / "noisy_native" / "paired_main.csv", 128),
        ("OOD wide", RUNS / "ood_wide" / "paired_main.csv", 128),
        ("MSE-delta reward", RUNS / "reward_mse_delta" / "paired_main.csv", 128),
    ]
    baselines = ["bald_2step_fast", "fisher_greedy_fast", "dad_transformer"]
    rows = []
    for context, path, shot in runs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        sub = df[(df["shots"] == shot) & (df["baseline"].isin(baselines))].copy()
        for _, row in sub.iterrows():
            rows.append(
                {
                    "context": context,
                    "baseline": row["baseline"],
                    "baseline_label": label(row["baseline"]),
                    "mse_ratio_symqnet_over_baseline": row["mse_ratio_reference_over_baseline"],
                    "speedup_baseline_over_symqnet": row["latency_speedup_baseline_over_reference"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "fig_robustness_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharex=True)
    xlabels = list(dict.fromkeys(out["context"].tolist()))
    x = np.arange(len(xlabels))
    width = 0.22
    for i, baseline in enumerate(baselines):
        sub = out[out["baseline"] == baseline].set_index("context").reindex(xlabels)
        offset = (i - 1) * width
        axes[0].bar(
            x + offset,
            sub["mse_ratio_symqnet_over_baseline"],
            width=width,
            color=COLORS[baseline],
            label=label(baseline),
        )
        axes[1].bar(
            x + offset,
            sub["speedup_baseline_over_symqnet"],
            width=width,
            color=COLORS[baseline],
            label=label(baseline),
        )
    axes[0].axhline(1.0, color="#222222", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("MSE ratio")
    axes[0].set_title("SymQNet MSE / baseline")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Speedup")
    axes[1].set_title("Baseline latency / SymQNet")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(wspace=0.30, bottom=0.32)
    return save_figure(fig, "fig3_robustness_summary")


def build_ablation_figure() -> list[str]:
    df = pd.read_csv(RUNS / "ablations_paper" / "paired_ablations.csv")
    shot = 128
    selected = ["no_vae", "no_graph", "mlp_only"]
    sub = df[(df["shots"] == shot) & (df["baseline"].isin(selected))].copy()
    sub["variant_mse_delta_pct"] = 100.0 * (
        sub["baseline_mse_mean"] - sub["reference_mse_mean"]
    ) / sub["reference_mse_mean"]
    sub["baseline_label"] = sub["baseline"].map(label)
    sub.to_csv(DATA_DIR / "fig_ablation_source.csv", index=False)
    sub = sub.sort_values("variant_mse_delta_pct")

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.barh(
        sub["baseline_label"],
        sub["variant_mse_delta_pct"],
        color=[COLORS.get(m, "#999999") for m in sub["baseline"]],
    )
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Variant MSE vs full SymQNet (%)")
    ax.set_title("Ablation impact at 128 shots")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    return save_figure(fig, "fig4_ablation_impact")


def build_reward_diagnostic() -> list[str]:
    df = pd.read_csv(RUNS / "main_result" / "reward_objective.csv")
    df = df[df["method"] == "symqnet"].copy()
    df.to_csv(DATA_DIR / "fig_reward_diagnostic_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(df["shots"], df["pearson_info_gain_vs_mse"], marker="o", label="Pearson")
    ax.plot(df["shots"], df["spearman_info_gain_vs_mse"], marker="s", label="Spearman")
    ax.axhline(-0.05, color="#222222", linestyle="--", linewidth=0.8, label="desired direction")
    ax.axhline(0.0, color="#999999", linewidth=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Shots")
    ax.set_ylabel("Correlation with final MSE")
    ax.set_title("Reward-objective diagnostic")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    return save_figure(fig, "fig5_reward_diagnostic")


def build_tables() -> list[str]:
    written = []
    paired = pd.read_csv(RUNS / "main_result" / "paired_main.csv")
    main_selected = ["dad_transformer", "bald_2step_fast", "fisher_greedy_fast", "fixed_optimized"]
    main = paired[(paired["shots"].isin([128, 512])) & (paired["baseline"].isin(main_selected))].copy()
    main["Baseline"] = main["baseline"].map(label)
    main["MSE ratio"] = main.apply(
        lambda r: _fmt_mean_ci(
            r["mse_ratio_reference_over_baseline"],
            r["mse_ratio_ci95_lo"],
            r["mse_ratio_ci95_hi"],
        ),
        axis=1,
    )
    main["Speedup"] = main.apply(
        lambda r: _fmt_mean_ci(
            r["latency_speedup_baseline_over_reference"],
            r["latency_speedup_ci95_lo"],
            r["latency_speedup_ci95_hi"],
        ),
        axis=1,
    )
    main["$p_{\\mathrm{BH}}$"] = main["wilcoxon_p_bh"].map(_fmt_p_value)
    main_out = main[
        [
            "shots",
            "Baseline",
            "paired_tasks",
            "MSE ratio",
            "Speedup",
            "$p_{\\mathrm{BH}}$",
        ]
    ].rename(columns={"shots": "Shots", "paired_tasks": "Episodes"})
    main_out.to_csv(DATA_DIR / "table_main_comparison.csv", index=False)
    main_latex = main_out.astype(str).copy()
    bounded_rows = main["baseline"].isin(["fisher_greedy_fast", "bald_2step_fast"]).to_numpy()
    main_latex.loc[bounded_rows, "Speedup"] = main_latex.loc[bounded_rows, "Speedup"].map(_bold)
    write_latex(
        main_latex.values.tolist(),
        list(main_latex.columns),
        TABLE_DIR / "table_main_comparison.tex",
        align="llrlll",
    )
    written += [
        str((DATA_DIR / "table_main_comparison.csv").relative_to(ROOT)),
        str((TABLE_DIR / "table_main_comparison.tex").relative_to(ROOT)),
    ]

    scaling = pd.read_csv(DATA_DIR / "fig_scaling_source.csv")
    scaling_selected = ["symqnet", "dad_transformer", "bald_2step_fast", "fisher_greedy_fast", "fixed"]
    scaling = scaling[scaling["method"].isin(scaling_selected)].copy()
    scaling["Method"] = scaling["method"].map(label)
    scaling["MSE"] = scaling.apply(lambda r: _fmt_mean_ci(r["mse_mean"], r["mse_ci95_lo"], r["mse_ci95_hi"]), axis=1)
    scaling["Decision ms"] = scaling.apply(
        lambda r: _fmt_mean_ci(r["decision_ms_mean"], r["decision_ms_ci95_lo"], r["decision_ms_ci95_hi"]),
        axis=1,
    )
    scaling_out = scaling[["n_qubits", "Method", "MSE", "Decision ms"]].rename(columns={"n_qubits": "N"})
    scaling_out.to_csv(DATA_DIR / "table_scaling_summary.csv", index=False)
    write_latex(
        scaling_out.astype(str).values.tolist(),
        list(scaling_out.columns),
        TABLE_DIR / "table_scaling_summary.tex",
        align="rlll",
    )
    written += [
        str((DATA_DIR / "table_scaling_summary.csv").relative_to(ROOT)),
        str((TABLE_DIR / "table_scaling_summary.tex").relative_to(ROOT)),
    ]

    ab = pd.read_csv(DATA_DIR / "fig_ablation_source.csv")
    ab["Variant"] = ab["baseline"].map(label)
    ab["Variant MSE"] = ab["baseline_mse_mean"].map(lambda x: _fmt_fixed(x, 4))
    ab["Full MSE"] = ab["reference_mse_mean"].map(lambda x: _fmt_fixed(x, 4))
    ab["Delta"] = ab["variant_mse_delta_pct"].map(lambda x: f"{x:+.1f}\\%")
    ab_out = ab[["Variant", "Full MSE", "Variant MSE", "Delta"]]
    ab_out.to_csv(DATA_DIR / "table_ablation_summary.csv", index=False)
    ab_latex = ab_out.astype(str).copy()
    key_ablation = ab["baseline"].eq("no_graph").to_numpy()
    ab_latex.loc[key_ablation, "Delta"] = ab_latex.loc[key_ablation, "Delta"].map(_bold)
    write_latex(
        ab_latex.values.tolist(),
        list(ab_latex.columns),
        TABLE_DIR / "table_ablation_summary.tex",
        align="lccc",
    )
    written += [
        str((DATA_DIR / "table_ablation_summary.csv").relative_to(ROOT)),
        str((TABLE_DIR / "table_ablation_summary.tex").relative_to(ROOT)),
    ]

    with (RUNS / "scaling" / "claim_gate.json").open(encoding="utf-8") as f:
        claim = json.load(f)
    claim_rows = [
        ["Reference baseline", label(claim.get("baseline", ""))],
        ["SymQNet latency slope", _fmt_fixed(float(claim["reference_latency_log_slope"]), 3)],
        ["Baseline latency slope", _fmt_fixed(float(claim["baseline_latency_log_slope"]), 3)],
        [
            "Latency slope ratio",
            _bold(_fmt_fixed(float(claim["latency_slope_ratio_reference_over_baseline"]), 3)),
        ],
        ["Worst MSE ratio", _bold(_fmt_fixed(float(claim["worst_mse_ratio_reference_over_baseline"]), 3))],
    ]
    write_latex(claim_rows, ["Quantity", "Value"], TABLE_DIR / "table_claim_gate.tex", align="ll")
    (DATA_DIR / "table_claim_gate.json").write_text(json.dumps(claim, indent=2) + "\n", encoding="utf-8")
    written += [
        str((DATA_DIR / "table_claim_gate.json").relative_to(ROOT)),
        str((TABLE_DIR / "table_claim_gate.tex").relative_to(ROOT)),
    ]
    return written


def write_manifest(copied: list[str], figures: list[str], tables: list[str]) -> None:
    with (RUNS / "scaling" / "claim_gate.json").open(encoding="utf-8") as f:
        claim = json.load(f)
    with (RUNS / "main_result" / "paper_readiness_report.json").open(encoding="utf-8") as f:
        readiness = json.load(f)

    lines = [
        "# QCRL 2026 Asset Pack",
        "",
        "Generated from the manuscript experiment summaries for the completed QCRL 2026 study.",
        "",
        "## Recommended Main-Paper Assets",
        "",
        "- `figures/fig0_mdp_method_flow.*`: compact MDP and methodology flow diagram for the method section.",
        "- `figures/fig1_combined_evidence.*`: recommended main result figure. Combines N=8/10/12 scaling with the N=5 Pareto benchmark.",
        "- `figures/fig2_scaling_latency_mse.*`: standalone scaling figure for backup or slides.",
        "- `figures/fig1_main_pareto.*`: standalone N=5 Pareto figure for backup or slides.",
        "- `tables/table_main_comparison.tex`: compact paired comparison table for the N=5 benchmark.",
        "- `tables/table_claim_gate.tex`: tiny numeric scaling-summary table if the text needs exact latency-scaling numbers.",
        "",
        "## Appendix/Backup Assets",
        "",
        "- `figures/fig3_robustness_summary.*`: main/noisy/OOD/reward-context summary versus DAD, bounded BALD, and bounded Fisher-information search.",
        "- `figures/fig4_ablation_impact.*`: ablation summary at 128 shots.",
        "- `figures/fig5_reward_diagnostic.*`: diagnostic only. The reward-objective gate is not strong enough for a headline claim.",
        "- `tables/table_scaling_summary.tex`: full numeric scaling table.",
        "- `tables/table_ablation_summary.tex`: ablation numbers.",
        "",
        "## Claim Notes",
        "",
        f"- Scaling criterion satisfied: `{claim.get('strong_scaling_claim_ok')}`.",
        f"- Claim recommendation: {claim.get('claim_recommendation')}",
        f"- Reward-objective diagnostic passed: `{readiness.get('ok')}`.",
        "- The reported readiness warnings are reward/objective-correlation diagnostics, not missing-file or scaling-criterion failures. Frame information gain as the training objective and keep reward alignment as a limitation/diagnostic.",
        "- Exact full Fisher/BALD settings exposed the dominant computational bottleneck; figures and tables label the tractable baselines as bounded variants.",
        "",
        "## Generated Files",
        "",
        "### Figures",
        *[f"- `{item}`" for item in figures],
        "",
        "### Tables And Derived Data",
        *[f"- `{item}`" for item in tables],
        "",
        "### Copied Source Inputs",
        *[f"- `{item}`" for item in copied],
        "- `paper/qcrl2026_assets/data/method_label_map.json`",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    copied = copy_core_inputs()
    figures: list[str] = []
    figures.extend(build_mdp_method_flow())
    figures.extend(build_main_pareto())
    figures.extend(build_scaling_figure())
    figures.extend(build_combined_evidence_figure())
    figures.extend(build_robustness_figure())
    figures.extend(build_ablation_figure())
    figures.extend(build_reward_diagnostic())
    tables = build_tables()
    write_manifest(copied, figures, tables)
    print(f"Wrote QCRL asset pack to {OUT.relative_to(ROOT)}")
    print(f"Figures: {len(figures)} files")
    print(f"Tables/data: {len(tables)} files")


if __name__ == "__main__":
    main()
