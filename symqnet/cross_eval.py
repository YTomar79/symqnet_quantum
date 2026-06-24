from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints under an alternate config without retraining.")
    parser.add_argument("--config", required=True, help="Evaluation config, e.g. configs/ood_wide.json.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Trained checkpoint to evaluate. Repeat per seed.")
    parser.add_argument("--agent-name", default="symqnet")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes-out", required=True)
    parser.add_argument("--task-bank", required=True)
    parser.add_argument("--with-crlb", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    episodes_path = Path(args.episodes_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    if episodes_path.exists():
        episodes_path.unlink()

    common = [
        sys.executable,
        "-m",
        "symqnet.eval",
        "--config",
        args.config,
        "--out",
        str(out_path),
        "--episodes-out",
        str(episodes_path),
        "--task-bank",
        args.task_bank,
    ]
    if args.episodes is not None:
        common.extend(["--episodes", str(args.episodes)])
    if args.with_crlb:
        common.append("--with-crlb")

    if args.include_baselines:
        subprocess.run(common, check=True)

    for idx, checkpoint in enumerate(args.checkpoint):
        cmd = [
            *common,
            "--agent-checkpoint",
            checkpoint,
            "--agent-name",
            args.agent_name,
            "--skip-baselines",
        ]
        if idx > 0 or args.include_baselines:
            cmd.append("--append")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
