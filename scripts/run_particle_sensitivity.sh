#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/particle_p96.json}"
RUN_ROOT="${RUN_ROOT:-runs/particle_p96}"
MAIN_ROOT="${MAIN_ROOT:-runs/main_result}"
EPISODES="${EPISODES:-100}"
SEEDS="${SEEDS:-777 778 779 780 781}"
TASK_BANK="${TASK_BANK:-$RUN_ROOT/task_bank.npz}"

checkpoints=()
for seed in $SEEDS; do
  checkpoints+=(--checkpoint "$MAIN_ROOT/symqnet_seed_$seed/best_agent.pt")
done

mkdir -p "$RUN_ROOT"

.venv/bin/python -m symqnet.cross_eval \
  --config "$CONFIG" \
  "${checkpoints[@]}" \
  --agent-name symqnet_p96 \
  --episodes "$EPISODES" \
  --out "$RUN_ROOT/shot_budget.csv" \
  --episodes-out "$RUN_ROOT/episodes.csv" \
  --task-bank "$TASK_BANK" \
  --include-baselines

.venv/bin/python -m symqnet.plot_shot_budget \
  --csv "$RUN_ROOT/shot_budget.csv" \
  --out "$RUN_ROOT/shot_budget.svg" \
  --title "Particle sensitivity: P=96 at 128 shots"

.venv/bin/python -m symqnet.analysis.paired_main \
  --episodes-csv "$RUN_ROOT/episodes.csv" \
  --reference symqnet_p96 \
  --baselines bald_2step_fast fisher_greedy_fast fixed_optimized random \
  --out-csv "$RUN_ROOT/paired_main.csv" \
  --out-tex "$RUN_ROOT/paired_main_table.tex"
