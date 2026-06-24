#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT=configs/ablations_smoke \
RUN_ROOT=runs/ablations_smoke \
UPDATES=1 \
EPISODES=1 \
SEEDS=777 \
VALIDATION_EPISODES=1 \
VALIDATION_INTERVAL=1 \
ABLATION_DELTA_SHOT=16 \
bash scripts/run_ablations.sh
