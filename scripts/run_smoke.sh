#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m symqnet.eval --config configs/smoke.json
.venv/bin/python -m symqnet.train_ppo --config configs/smoke.json --updates 1
