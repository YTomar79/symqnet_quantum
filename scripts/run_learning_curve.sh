#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.json}"
RUN_ROOT="${RUN_ROOT:-runs/learning_curve}"
SEED="${SEED:-777}"
UPDATES="${UPDATES:-2500}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-64}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-25}"
PRETRAIN_SAMPLES="${PRETRAIN_SAMPLES:-15000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"

if [[ "$CONFIG" == "configs/default.json" && ! -f artifacts/vae_n5_l16.pt ]]; then
  .venv/bin/python -m symqnet.pretrain_vae \
    --config "$CONFIG" \
    --samples "$PRETRAIN_SAMPLES" \
    --epochs "$PRETRAIN_EPOCHS"
fi

mkdir -p "$RUN_ROOT"

.venv/bin/python -m symqnet.train_ppo \
  --config "$CONFIG" \
  --updates "$UPDATES" \
  --seed "$SEED" \
  --output-dir "$RUN_ROOT/seed_$SEED" \
  --validation-task-bank "$RUN_ROOT/validation_task_bank.npz" \
  --validation-episodes "$VALIDATION_EPISODES" \
  --validation-interval "$VALIDATION_INTERVAL"

.venv/bin/python -m symqnet.analysis.learning_curve \
  --metrics "$RUN_ROOT/seed_$SEED/train_metrics.json" \
  --out "$RUN_ROOT/validation_mse.svg"

.venv/bin/python -m symqnet.analysis.complexity \
  --out-csv "$RUN_ROOT/complexity.csv" \
  --out-tex "$RUN_ROOT/complexity_table.tex"
