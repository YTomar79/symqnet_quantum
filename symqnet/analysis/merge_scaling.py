from __future__ import annotations

import argparse
import csv
from pathlib import Path


def merge_scaling(run_roots: list[Path], out: Path) -> None:
    rows: list[dict[str, object]] = []
    fieldnames: list[str] | None = None
    for run_root in run_roots:
        csv_path = run_root / "shot_budget.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            local_fields = ["n_qubits", *(reader.fieldnames or [])]
            if fieldnames is None:
                fieldnames = local_fields
            for row in reader:
                tagged = {"n_qubits": row.get("n_qubits") or _infer_n(run_root), **row}
                rows.append(tagged)
    if not rows or fieldnames is None:
        raise ValueError("No scaling rows found.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


def _infer_n(path: Path) -> int:
    for part in reversed(path.parts):
        if part.lower().startswith("n") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"Could not infer n_qubits from {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-N scaling run summaries.")
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    merge_scaling([Path(item) for item in args.run_roots], Path(args.out))


if __name__ == "__main__":
    main()
