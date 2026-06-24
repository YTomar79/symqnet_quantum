from __future__ import annotations

import argparse
import json
from pathlib import Path

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "symqnet_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "symqnet_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(path: Path) -> list[dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("validation_history", [])
    if not isinstance(history, list):
        raise ValueError(f"validation_history is not a list in {path}")
    return [row for row in history if isinstance(row, dict)]


def write_svg(history: list[dict[str, float]], out: Path) -> None:
    if not history:
        raise ValueError("No validation history to plot.")
    updates = [int(row["update"]) for row in history]
    values = [float(row["validation_mse"]) for row in history]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(updates, values, marker="o", linewidth=1.5)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Held-out validation theta-MSE")
    ax.set_title("SymQNet validation learning curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation MSE vs PPO update from train_metrics.json.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_svg(load_history(Path(args.metrics)), Path(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
