from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from symqnet.config import load_config


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_policy_params(value: str) -> dict[str, object]:
    if value in {"", "None"}:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {"raw_policy_params": value}
    return payload if isinstance(payload, dict) else {"raw_policy_params": value}


def baseline_param_rows(summary_csv: Path, config_path: Path) -> list[dict[str, object]]:
    cfg = load_config(config_path)
    seen: set[str] = set()
    rows = []
    for row in _read_csv(summary_csv):
        method = row["method"]
        if method in seen:
            continue
        seen.add(method)
        params = _parse_policy_params(row.get("policy_params", ""))
        rows.append(
            {
                "method": method,
                "hamiltonian": cfg.env.hamiltonian,
                "n_qubits": cfg.env.n_qubits,
                "m_evo": cfg.env.m_evo,
                "horizon": cfg.env.horizon,
                "n_actions": cfg.env.n_qubits * 3 * cfg.env.m_evo,
                "noise_prob": cfg.env.noise_prob,
                "noise_model": cfg.env.noise_model,
                "readout_p01": cfg.env.readout_p01,
                "readout_p10": cfg.env.readout_p10,
                "t1_us": cfg.env.t1_us,
                "t2_us": cfg.env.t2_us,
                "smc_particles": cfg.smc.particles,
                "predictive_samples": params.get("predictive_samples", ""),
                "depth": params.get("depth", ""),
                "top_k": params.get("top_k", ""),
                "objective": params.get("objective", ""),
                "non_adaptive": params.get("non_adaptive", ""),
                "basis_order": " ".join(str(item) for item in params.get("basis_order", []))
                if isinstance(params.get("basis_order", []), list)
                else params.get("basis_order", ""),
                "graph": params.get("graph", ""),
                "temporal": params.get("temporal", ""),
                "use_smc_feedback": params.get("use_smc_feedback", ""),
            }
        )
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")
    return sorted(rows, key=lambda item: str(item["method"]))


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Method & $N$ & $M$ & Actions & SMC $P$ & Samples & Depth/top-$k$ & Objective \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        depth_topk = ""
        if row["depth"] != "":
            depth_topk = f"{row['depth']}/{row['top_k']}"
        lines.append(
            f"{method} & {row['n_qubits']} & {row['m_evo']} & {row['n_actions']} & "
            f"{row['smc_particles']} & {row['predictive_samples']} & {depth_topk} & "
            f"{str(row['objective']).replace('_', r'\\_')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-facing baseline and policy parameters.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-tex", default=None)
    args = parser.parse_args()

    rows = baseline_param_rows(Path(args.summary_csv), Path(args.config))
    write_csv(rows, Path(args.out_csv))
    print(f"saved {args.out_csv}")
    if args.out_tex:
        write_latex(rows, Path(args.out_tex))
        print(f"saved {args.out_tex}")


if __name__ == "__main__":
    main()
