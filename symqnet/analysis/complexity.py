from __future__ import annotations

import argparse
import csv
from pathlib import Path


def complexity_rows(n_values: list[int], m_evo: int, particles: int, top_k: int, predictive_samples: int) -> list[dict[str, object]]:
    rows = []
    for n_qubits in n_values:
        actions = n_qubits * 3 * m_evo
        hilbert_dim = 2**n_qubits
        smc_predict_units = particles * hilbert_dim * n_qubits
        bald_units = actions * smc_predict_units * (1 + top_k * predictive_samples)
        symqnet_chain_units = n_qubits
        symqnet_transformer_units = n_qubits * n_qubits
        rows.append(
            {
                "n_qubits": n_qubits,
                "m_evo": m_evo,
                "actions": actions,
                "hilbert_dim": hilbert_dim,
                "smc_particles": particles,
                "top_k": top_k,
                "predictive_samples": predictive_samples,
                "smc_predict_units": smc_predict_units,
                "bald_2step_relative_units": bald_units,
                "symqnet_chain_relative_units": symqnet_chain_units,
                "symqnet_transformer_relative_units": symqnet_transformer_units,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$N$ & $|\mathcal{A}|$ & $2^N$ & BALD rel. units & SymQNet rel. units \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['n_qubits']} & {row['actions']} & {row['hilbert_dim']} & "
            f"{row['bald_2step_relative_units']:.3g} & {row['symqnet_transformer_relative_units']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit Prop. 1 complexity table inputs for SymQNet vs BALD.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--m-evo", type=int, default=5)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--predictive-samples", type=int, default=5)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-tex", required=True)
    args = parser.parse_args()

    rows = complexity_rows(args.n_values, args.m_evo, args.particles, args.top_k, args.predictive_samples)
    write_csv(rows, Path(args.out_csv))
    write_latex(rows, Path(args.out_tex))
    print(f"saved {args.out_csv}")
    print(f"saved {args.out_tex}")


if __name__ == "__main__":
    main()
