from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _present(value: object) -> bool:
    return value not in {"", None, "None"}


def validate_summary(rows: list[dict[str, str]], require_agent_checkpoints: bool) -> list[str]:
    errors: list[str] = []
    required = [
        "method",
        "shots",
        "eval_seed",
        "config_hash",
        "hardware",
        "task_bank_path",
        "policy_params",
        "decision_ms_mean",
        "smc_update_ms_mean",
        "step_total_ms_mean",
    ]
    for idx, row in enumerate(rows):
        for key in required:
            if not _present(row.get(key)):
                errors.append(f"summary row {idx} missing {key}")
        if require_agent_checkpoints and row.get("method") == "symqnet":
            for key in (
                "train_seed",
                "checkpoint_path",
                "checkpoint_config_hash",
                "checkpoint_selection_metric",
                "best_validation_mse",
                "validation_task_bank_path",
                "train_wallclock_sec",
            ):
                if not _present(row.get(key)):
                    errors.append(f"summary row {idx} symqnet missing {key}")
    return errors


def validate_episode_tasks(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    by_shot_method: dict[tuple[int, str], set[tuple[int, int]]] = defaultdict(set)
    for idx, row in enumerate(rows):
        if not _present(row.get("task_id")):
            errors.append(f"episode row {idx} missing task_id")
            continue
        if not _present(row.get("eval_seed")):
            errors.append(f"episode row {idx} missing eval_seed")
            continue
        shot = int(float(row["shots"]))
        method = row["method"]
        task_key = (int(float(row["eval_seed"])), int(float(row["task_id"])))
        by_shot_method[(shot, method)].add(task_key)

    by_shot: dict[int, dict[str, set[tuple[int, int]]]] = defaultdict(dict)
    for (shot, method), tasks in by_shot_method.items():
        by_shot[shot][method] = tasks
    for shot, methods in by_shot.items():
        if len(methods) <= 1:
            continue
        reference_method = sorted(methods)[0]
        reference_tasks = methods[reference_method]
        for method, tasks in sorted(methods.items()):
            if tasks != reference_tasks:
                missing = len(reference_tasks - tasks)
                extra = len(tasks - reference_tasks)
                errors.append(
                    f"shot {shot}: method {method} task set differs from {reference_method} "
                    f"(missing={missing}, extra={extra})"
                )
    return errors


def validation_report(summary_csv: Path, episodes_csv: Path, require_agent_checkpoints: bool) -> dict[str, object]:
    summary_rows = _read_csv(summary_csv)
    episode_rows = _read_csv(episodes_csv)
    errors = validate_summary(summary_rows, require_agent_checkpoints)
    errors.extend(validate_episode_tasks(episode_rows))
    methods = sorted({row["method"] for row in summary_rows})
    shots = sorted({int(float(row["shots"])) for row in summary_rows})
    return {
        "ok": not errors,
        "errors": errors,
        "summary_rows": len(summary_rows),
        "episode_rows": len(episode_rows),
        "methods": methods,
        "shots": shots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paper experiment CSV provenance and matched-task structure.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument("--require-agent-checkpoints", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = validation_report(Path(args.summary_csv), Path(args.episodes_csv), args.require_agent_checkpoints)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"saved {out}")
    else:
        print(text)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
