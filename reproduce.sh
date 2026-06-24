#!/usr/bin/env bash
# End-to-end reproduction of the full experimental suite.
# Creates a virtual environment, runs the test suite, then executes the
# complete benchmark sweep that produces every result under runs/.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"$PYTHON" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests

# Main result, scaling, noisy, transfer, ablation, and reward-alignment stages.
EPISODES="${EPISODES:-500}" \
UPDATES="${UPDATES:-2500}" \
bash scripts/run_qcrl2026_repositioned.sh
