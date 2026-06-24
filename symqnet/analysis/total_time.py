from __future__ import annotations

import argparse
import csv
from pathlib import Path

from symqnet.config import load_config


def total_time_rows(summary_csv: Path, horizon: int, shot_times_us: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shots = float(row["shots"])
            decision_ms = float(row.get("decision_ms_mean", 0.0) or 0.0)
            smc_ms = float(row.get("smc_update_ms_mean", 0.0) or 0.0)
            for shot_time_us in shot_times_us:
                quantum_ms = shots * float(shot_time_us) / 1000.0
                rows.append(
                    {
                        "method": row["method"],
                        "shots": int(shots),
                        "horizon": int(horizon),
                        "shot_time_us": float(shot_time_us),
                        "decision_ms_mean": decision_ms,
                        "smc_update_ms_mean": smc_ms,
                        "quantum_ms_per_step": quantum_ms,
                        "classical_ms_per_step": decision_ms + smc_ms,
                        "total_episode_time_ms": int(horizon) * (quantum_ms + decision_ms + smc_ms),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute total episode time under simple quantum shot-time models.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--shot-times-us", nargs="+", type=float, default=[100.0, 1000.0, 10000.0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = total_time_rows(Path(args.summary_csv), int(cfg.env.horizon), args.shot_times_us)
    write_csv(rows, Path(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
