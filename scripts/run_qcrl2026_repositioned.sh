#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-500}"
UPDATES="${UPDATES:-2500}"
SEEDS="${SEEDS:-777 778 779 780 781}"
JOBS="${JOBS:-4}"
PRETRAIN_SAMPLES="${PRETRAIN_SAMPLES:-15000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
PAPER_READINESS="${PAPER_READINESS:-1}"
ANONYMIZE_MANIFEST="${ANONYMIZE_MANIFEST:-1}"
RUN_XXZ_TRANSFER="${RUN_XXZ_TRANSFER:-0}"

if [[ ! -f artifacts/vae_n5_l16.pt ]]; then
  .venv/bin/python -m symqnet.pretrain_vae \
    --config configs/default.json \
    --samples "$PRETRAIN_SAMPLES" \
    --epochs "$PRETRAIN_EPOCHS"
fi

echo "==> E1: N=5 Pareto benchmark with bounded BALD/Fisher, fixed-optimized, and P=256"
EPISODES="$EPISODES" UPDATES="$UPDATES" SEEDS="$SEEDS" JOBS="$JOBS" ANONYMIZE_MANIFEST="$ANONYMIZE_MANIFEST" bash scripts/run_paper_cpu_cluster.sh

echo "==> E2: N-scaling diagnostic benchmark"
EPISODES="${SCALING_EPISODES:-300}" \
UPDATES="$UPDATES" \
SEEDS="${SCALING_SEEDS:-777 778 779}" \
JOBS="$JOBS" \
PRETRAIN_SAMPLES="$PRETRAIN_SAMPLES" \
PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS" \
bash scripts/run_scaling.sh

.venv/bin/python -m symqnet.analysis.complexity \
  --out-csv runs/scaling/complexity.csv \
  --out-tex runs/scaling/complexity_table.tex

echo "==> E3: native noisy evaluation"
CONFIG=configs/noisy_native.json \
RUN_ROOT=runs/noisy_native \
EPISODES="${NOISE_EPISODES:-300}" \
UPDATES="$UPDATES" \
SEEDS="${NOISE_SEEDS:-777 778 779}" \
JOBS="$JOBS" \
COMPACT_TABLE_SHOTS=128 \
WITH_CRLB=0 \
ANONYMIZE_MANIFEST="$ANONYMIZE_MANIFEST" \
bash scripts/run_paper_cpu_cluster.sh

echo "==> E4: wider-prior transfer evaluation"
ood_checkpoints=()
for seed in $SEEDS; do
  ood_checkpoints+=(--checkpoint "runs/main_result/symqnet_seed_$seed/best_agent.pt")
done
.venv/bin/python -m symqnet.cross_eval \
  --config configs/ood_wide.json \
  "${ood_checkpoints[@]}" \
  --agent-name symqnet \
  --episodes "${TRANSFER_EPISODES:-300}" \
  --out runs/ood_wide/shot_budget.csv \
  --episodes-out runs/ood_wide/episodes.csv \
  --task-bank runs/ood_wide/task_bank.npz \
  --include-baselines
.venv/bin/python -m symqnet.analysis.paired_main \
  --episodes-csv runs/ood_wide/episodes.csv \
  --reference symqnet \
  --baselines bald_2step_fast fisher_greedy_fast fixed_optimized random \
  --out-csv runs/ood_wide/paired_main.csv \
  --out-tex runs/ood_wide/paired_main_table.tex

if [[ "$RUN_XXZ_TRANSFER" == "1" ]]; then
  echo "==> E4b: XXZ zero-shot transfer evaluation"
  EPISODES="${XXZ_TRANSFER_EPISODES:-300}" \
  SEEDS="${XXZ_TRANSFER_SEEDS:-777 778 779}" \
  bash scripts/run_xxz_transfer.sh
else
  echo "==> Skipping optional XXZ zero-shot transfer; set RUN_XXZ_TRANSFER=1 for appendix evidence"
fi

echo "==> E5: trimmed paper ablations"
CONFIG_ROOT=configs/ablations_paper \
RUN_ROOT=runs/ablations_paper \
UPDATES="$UPDATES" \
EPISODES="${ABLATION_EPISODES:-300}" \
SEEDS="${ABLATION_SEEDS:-777 778 779}" \
ABLATION_CONFIGS="full no_vae no_smc_feedback no_graph mlp_only" \
ABLATION_BASELINES="no_vae no_smc_feedback no_graph mlp_only" \
bash scripts/run_ablations.sh

echo "==> E6: reward-alignment ablation"
CONFIG=configs/reward_mse_delta.json \
RUN_ROOT=runs/reward_mse_delta \
EPISODES="${REWARD_EPISODES:-300}" \
UPDATES="$UPDATES" \
SEEDS="${REWARD_SEEDS:-777 778 779}" \
JOBS="$JOBS" \
COMPACT_TABLE_SHOTS=128 \
WITH_CRLB=0 \
ANONYMIZE_MANIFEST="$ANONYMIZE_MANIFEST" \
bash scripts/run_paper_cpu_cluster.sh

if [[ "$PAPER_READINESS" == "1" ]]; then
  extra_required=(
    runs/scaling/scaling_summary.csv
    runs/scaling/claim_gate.json
    runs/scaling/mps_validation.json
    runs/scaling/complexity_table.tex
    runs/noisy_native/paired_main.csv
    runs/ood_wide/paired_main.csv
    runs/ablations_paper/paired_ablations.csv
    runs/reward_mse_delta/paired_main.csv
  )
  if [[ "$RUN_XXZ_TRANSFER" == "1" ]]; then
    extra_required+=(runs/transfer_xxz/paired_main.csv)
  fi
  .venv/bin/python -m symqnet.analysis.paper_readiness \
    --run-root runs/main_result \
    --config configs/default.json \
    --extra-required-files "${extra_required[@]}" \
    --out runs/main_result/paper_readiness_report.json
fi
