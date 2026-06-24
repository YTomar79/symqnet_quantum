#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.json}"
EPISODES="${EPISODES:-200}"
OUT="${OUT:-runs/shot_budget.csv}"
EPISODES_OUT="${EPISODES_OUT:-runs/shot_budget_episodes.csv}"

.venv/bin/python -m symqnet.eval \
  --config "$CONFIG" \
  --episodes "$EPISODES" \
  --out "$OUT" \
  --episodes-out "$EPISODES_OUT"

.venv/bin/python -m symqnet.plot_shot_budget \
  --csv "$OUT" \
  --out "${OUT%.csv}.svg"
