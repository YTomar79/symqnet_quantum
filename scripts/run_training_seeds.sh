#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.json}"
RUN_ROOT="${RUN_ROOT:-runs/training_seeds}"
UPDATES="${UPDATES:-2500}"
SEEDS="${SEEDS:-777 778 779}"

for seed in $SEEDS; do
  echo "==> Training seed $seed"
  .venv/bin/python -m symqnet.train_ppo \
    --config "$CONFIG" \
    --updates "$UPDATES" \
    --seed "$seed" \
    --output-dir "$RUN_ROOT/seed_$seed"
done
