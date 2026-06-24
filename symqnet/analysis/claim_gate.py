from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _method_points(rows: list[dict[str, str]], method: str, value_key: str) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for row in rows:
        if row.get("method") != method or row.get(value_key, "") == "":
            continue
        points.append((float(row["n_qubits"]), float(row[value_key])))
    if not points:
        return np.asarray([]), np.asarray([])
    by_n: dict[float, list[float]] = {}
    for n_qubits, value in points:
        by_n.setdefault(n_qubits, []).append(value)
    xs = np.asarray(sorted(by_n), dtype=np.float64)
    ys = np.asarray([np.mean(by_n[x]) for x in xs], dtype=np.float64)
    return xs, ys


def _log_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    mask = (xs > 0) & (ys > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), deg=1)
    return float(slope)


def claim_report(
    scaling_csv: Path,
    reference: str = "symqnet",
    baseline: str = "bald_2step_fast",
    slope_ratio_max: float = 0.5,
    mse_ratio_max: float = 1.25,
    strong_mse_ratio_ci_hi_max: float = 1.35,
    strong_latency_speedup_ci_lo_min: float = 10.0,
    anchor_ns: tuple[int, ...] = (10, 12),
    mps_validation_json: Path | None = None,
    mps_error_max: float = 2e-2,
) -> dict[str, object]:
    rows = _read_csv(scaling_csv)
    ref_n, ref_latency = _method_points(rows, reference, "decision_ms_mean")
    base_n, base_latency = _method_points(rows, baseline, "decision_ms_mean")
    ref_mse_n, ref_mse = _method_points(rows, reference, "mse_mean")
    base_mse_n, base_mse = _method_points(rows, baseline, "mse_mean")
    ref_slope = _log_slope(ref_n, ref_latency)
    base_slope = _log_slope(base_n, base_latency)
    common = sorted(set(ref_mse_n.tolist()) & set(base_mse_n.tolist()))
    mse_ratios = []
    for n_qubits in common:
        ref_value = ref_mse[np.where(ref_mse_n == n_qubits)][0]
        base_value = base_mse[np.where(base_mse_n == n_qubits)][0]
        mse_ratios.append(float(ref_value / max(base_value, 1e-12)))
    worst_mse_ratio = max(mse_ratios) if mse_ratios else float("nan")
    slope_ratio = ref_slope / max(base_slope, 1e-12) if np.isfinite(ref_slope) and np.isfinite(base_slope) else float("nan")
    ok = bool(np.isfinite(slope_ratio) and slope_ratio <= slope_ratio_max and np.isfinite(worst_mse_ratio) and worst_mse_ratio <= mse_ratio_max)
    anchor_rows = []
    for n_qubits in anchor_ns:
        paired_path = scaling_csv.parent / f"n{int(n_qubits)}" / "paired_main.csv"
        if not paired_path.exists():
            continue
        for row in _read_csv(paired_path):
            if row.get("reference") != reference or row.get("baseline") != baseline:
                continue
            if int(float(row.get("shots", "nan"))) != 128:
                continue
            mse_ci_hi = float(row["mse_ratio_ci95_hi"])
            speed_ci_lo = float(row["latency_speedup_ci95_lo"])
            anchor_rows.append(
                {
                    "n_qubits": int(n_qubits),
                    "mse_ratio_ci95_hi": mse_ci_hi,
                    "latency_speedup_ci95_lo": speed_ci_lo,
                    "passes": bool(mse_ci_hi <= strong_mse_ratio_ci_hi_max and speed_ci_lo >= strong_latency_speedup_ci_lo_min),
                }
            )
    validation = {"present": False, "passes": False}
    if mps_validation_json is not None and mps_validation_json.exists():
        validation_payload = json.loads(mps_validation_json.read_text(encoding="utf-8"))
        mean_abs_error = float(validation_payload.get("max_mean_abs_error", validation_payload.get("mean_abs_error", float("inf"))))
        validation = {
            "present": True,
            "passes": bool(mean_abs_error <= mps_error_max),
            "max_mean_abs_error": mean_abs_error,
            "threshold": float(mps_error_max),
        }
    strong_ok = bool(any(row["passes"] for row in anchor_rows) and validation["passes"])
    recommendation = (
        "The strong action-scaling claim is supported by an N>=10 anchor and MPS validation."
        if strong_ok
        else (
            "The scaling diagnostic supports mentioning flatter decision-latency scaling, but keep the abstract Pareto/amortization-first unless a larger-N anchor passes the strong gate."
            if ok
            else "Do not use the strong scaling claim without qualification; fall back to the small-chain amortized-policy framing."
        )
    )
    return {
        "scaling_claim_ok": ok,
        "strong_scaling_claim_ok": strong_ok,
        "reference": reference,
        "baseline": baseline,
        "reference_latency_log_slope": ref_slope,
        "baseline_latency_log_slope": base_slope,
        "latency_slope_ratio_reference_over_baseline": slope_ratio,
        "worst_mse_ratio_reference_over_baseline": worst_mse_ratio,
        "slope_ratio_max": slope_ratio_max,
        "mse_ratio_max": mse_ratio_max,
        "strong_gate": {
            "anchor_ns": list(anchor_ns),
            "shot": 128,
            "mse_ratio_ci95_hi_max": strong_mse_ratio_ci_hi_max,
            "latency_speedup_ci95_lo_min": strong_latency_speedup_ci_lo_min,
            "anchor_rows": anchor_rows,
            "mps_validation": validation,
        },
        "claim_recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate the QCRL scaling claim from merged scaling results.")
    parser.add_argument("--scaling-csv", required=True)
    parser.add_argument("--reference", default="symqnet")
    parser.add_argument("--baseline", default="bald_2step_fast")
    parser.add_argument("--slope-ratio-max", type=float, default=0.5)
    parser.add_argument("--mse-ratio-max", type=float, default=1.25)
    parser.add_argument("--strong-mse-ratio-ci-hi-max", type=float, default=1.35)
    parser.add_argument("--strong-latency-speedup-ci-lo-min", type=float, default=10.0)
    parser.add_argument("--anchor-ns", nargs="+", type=int, default=[10, 12])
    parser.add_argument("--mps-validation-json", default=None)
    parser.add_argument("--mps-error-max", type=float, default=2e-2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = claim_report(
        Path(args.scaling_csv),
        args.reference,
        args.baseline,
        args.slope_ratio_max,
        args.mse_ratio_max,
        args.strong_mse_ratio_ci_hi_max,
        args.strong_latency_speedup_ci_lo_min,
        tuple(args.anchor_ns),
        Path(args.mps_validation_json) if args.mps_validation_json else None,
        args.mps_error_max,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {out}")
    if not report["scaling_claim_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
