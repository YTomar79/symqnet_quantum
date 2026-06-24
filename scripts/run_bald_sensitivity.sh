#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/bald_sensitivity.json}"
RUN_ROOT="${RUN_ROOT:-runs/bald_sensitivity}"
EPISODES="${EPISODES:-100}"
TASK_BANK="${TASK_BANK:-$RUN_ROOT/task_bank.npz}"
WITH_CRLB="${WITH_CRLB:-0}"

mkdir -p "$RUN_ROOT"

cmd=(.venv/bin/python -m symqnet.eval
  --config "$CONFIG" \
  --episodes "$EPISODES" \
  --out "$RUN_ROOT/shot_budget.csv" \
  --episodes-out "$RUN_ROOT/episodes.csv" \
  --task-bank "$TASK_BANK")

if [[ "$WITH_CRLB" == "1" ]]; then
  cmd+=(--with-crlb)
fi

"${cmd[@]}"

.venv/bin/python -m symqnet.plot_shot_budget \
  --csv "$RUN_ROOT/shot_budget.csv" \
  --out "$RUN_ROOT/shot_budget.svg" \
  --title "BALD sensitivity: strong vs pruned lookahead"

.venv/bin/python -m symqnet.analysis.paired_main \
  --episodes-csv "$RUN_ROOT/episodes.csv" \
  --reference bald_2step \
  --baselines bald_2step_fast \
  --out-csv "$RUN_ROOT/paired_main.csv" \
  --out-tex "$RUN_ROOT/paired_main_table.tex"
