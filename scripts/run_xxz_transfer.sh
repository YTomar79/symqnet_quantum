#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/transfer_xxz.json}"
RUN_ROOT="${RUN_ROOT:-runs/transfer_xxz}"
MAIN_ROOT="${MAIN_ROOT:-runs/main_result}"
EPISODES="${EPISODES:-300}"
SEEDS="${SEEDS:-777 778 779}"
TASK_BANK="${TASK_BANK:-$RUN_ROOT/task_bank.npz}"
WITH_CRLB="${WITH_CRLB:-0}"

checkpoints=()
for seed in $SEEDS; do
  checkpoints+=(--checkpoint "$MAIN_ROOT/symqnet_seed_$seed/best_agent.pt")
done

mkdir -p "$RUN_ROOT"

cmd=(.venv/bin/python -m symqnet.cross_eval
  --config "$CONFIG"
  "${checkpoints[@]}"
  --agent-name symqnet_xxz_transfer
  --episodes "$EPISODES"
  --out "$RUN_ROOT/shot_budget.csv"
  --episodes-out "$RUN_ROOT/episodes.csv"
  --task-bank "$TASK_BANK"
  --include-baselines)

if [[ "$WITH_CRLB" == "1" ]]; then
  cmd+=(--with-crlb)
fi

"${cmd[@]}"

.venv/bin/python -m symqnet.analysis.paired_main \
  --episodes-csv "$RUN_ROOT/episodes.csv" \
  --reference symqnet_xxz_transfer \
  --baselines bald_2step_fast fisher_greedy_fast fixed_optimized random \
  --out-csv "$RUN_ROOT/paired_main.csv" \
  --out-tex "$RUN_ROOT/paired_main_table.tex"
