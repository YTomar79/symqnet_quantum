#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-500}"
UPDATES="${UPDATES:-2500}"
SEEDS="${SEEDS:-777 778 779 780 781}"
JOBS="${JOBS:-4}"
PRETRAIN_SAMPLES="${PRETRAIN_SAMPLES:-15000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
RUN_XXZ="${RUN_XXZ:-0}"
PAPER_READINESS="${PAPER_READINESS:-1}"

if [[ ! -f artifacts/vae_n5_l16.pt ]]; then
  echo "==> Pretraining N=5 VAE required by configs/default.json"
  .venv/bin/python -m symqnet.pretrain_vae \
    --config configs/default.json \
    --samples "$PRETRAIN_SAMPLES" \
    --epochs "$PRETRAIN_EPOCHS"
fi

echo "==> E1/E2: Pareto main result and wall-clock table"
EPISODES="$EPISODES" UPDATES="$UPDATES" SEEDS="$SEEDS" JOBS="$JOBS" bash scripts/run_paper_cpu_cluster.sh

echo "==> E2: trimmed paper ablations"
CONFIG_ROOT=configs/ablations_paper \
RUN_ROOT=runs/ablations_paper \
UPDATES="$UPDATES" \
EPISODES="$EPISODES" \
SEEDS="${ABLATION_SEEDS:-777 778 779}" \
bash scripts/run_ablations.sh

echo "==> E4: OOD wide-range evaluation without retraining"
ood_checkpoints=()
for seed in $SEEDS; do
  ood_checkpoints+=(--checkpoint "runs/main_result/symqnet_seed_$seed/best_agent.pt")
done
.venv/bin/python -m symqnet.cross_eval \
  --config configs/ood_wide.json \
  "${ood_checkpoints[@]}" \
  --agent-name symqnet \
  --episodes "$EPISODES" \
  --out runs/ood_wide/shot_budget.csv \
  --episodes-out runs/ood_wide/episodes.csv \
  --task-bank runs/ood_wide/task_bank.npz \
  --include-baselines

.venv/bin/python -m symqnet.plot_shot_budget \
  --csv runs/ood_wide/shot_budget.csv \
  --out runs/ood_wide/shot_budget.svg \
  --title "OOD wide-range evaluation: shot budget vs final theta-MSE"

.venv/bin/python -m symqnet.analysis.paired_main \
  --episodes-csv runs/ood_wide/episodes.csv \
  --reference symqnet \
  --baselines bald_2step_fast fisher_greedy_fast fixed_optimized random \
  --out-csv runs/ood_wide/paired_main.csv \
  --out-tex runs/ood_wide/paired_main_table.tex

if [[ "$RUN_XXZ" == "1" ]]; then
  echo "==> Optional: XXZ Hamiltonian sanity check"
  CONFIG=configs/xxz.json \
  RUN_ROOT=runs/xxz \
  UPDATES="$UPDATES" \
  EPISODES="$EPISODES" \
  WITH_CRLB=0 \
  bash scripts/run_main_result.sh
else
  echo "==> Skipping XXZ by default; keep the QCRL manuscript focused on TFIM unless this check is clean"
fi

echo "==> E5b: particle sensitivity at P=96"
EPISODES="${SENSITIVITY_EPISODES:-100}" SEEDS="$SEEDS" bash scripts/run_particle_sensitivity.sh

echo "==> E5c: stronger BALD sensitivity at 128 and 512 shots"
EPISODES="${SENSITIVITY_EPISODES:-100}" bash scripts/run_bald_sensitivity.sh

echo "==> Wilcoxon table for ablations"
.venv/bin/python -m symqnet.analysis.stats \
  --episodes-csv runs/ablations_paper/episodes.csv \
  --reference full \
  --out runs/ablations_paper/wilcoxon_table.tex

echo "==> E6: latency scaling"
TRIALS="${LATENCY_TRIALS:-3}" bash scripts/run_latency_scaling.sh

if [[ "$PAPER_READINESS" == "1" ]]; then
  echo "==> Final paper-readiness check"
  .venv/bin/python -m symqnet.analysis.paper_readiness \
    --run-root runs/main_result \
    --config configs/default.json \
    --extra-required-files \
      runs/ablations_paper/ablation_delta_128.tex \
      runs/particle_p96/paired_main.csv \
      runs/bald_sensitivity/paired_main.csv \
      runs/bald_sensitivity/paired_main_table.tex \
      runs/ood_wide/paired_main.csv \
    --out runs/main_result/paper_readiness_report.json
fi
