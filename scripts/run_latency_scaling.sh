#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-runs/latency_scaling}"
TRIALS="${TRIALS:-3}"

mkdir -p "$OUT_DIR"

.venv/bin/python -m symqnet.analysis.latency_scaling \
  --trials "$TRIALS" \
  --out-csv "$OUT_DIR/latency_scaling.csv" \
  --out-svg "$OUT_DIR/latency_scaling.svg"
