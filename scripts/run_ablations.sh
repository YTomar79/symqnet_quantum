#!/usr/bin/env bash
set -euo pipefail

UPDATES="${UPDATES:-2500}"
EPISODES="${EPISODES:-200}"
SEEDS="${SEEDS:-777 778 779}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/ablations}"
RUN_ROOT="${RUN_ROOT:-runs/ablations}"
CSV_OUT="${CSV_OUT:-$RUN_ROOT/shot_budget.csv}"
EPISODES_OUT="${EPISODES_OUT:-$RUN_ROOT/episodes.csv}"
TASK_BANK="${TASK_BANK:-$RUN_ROOT/task_bank.npz}"
VALIDATION_TASK_BANK="${VALIDATION_TASK_BANK:-$RUN_ROOT/validation_task_bank.npz}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-32}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-25}"
ABLATION_DELTA_SHOT="${ABLATION_DELTA_SHOT:-128}"
ABLATION_BASELINES="${ABLATION_BASELINES:-no_vae no_graph no_transformer no_smc_feedback}"

if [[ -n "${ABLATION_CONFIGS:-}" ]]; then
  CONFIGS=()
  for name in $ABLATION_CONFIGS; do
    CONFIGS+=("$CONFIG_ROOT/$name.json:$name")
  done
else
  CONFIGS=(
    "$CONFIG_ROOT/full.json:full"
    "$CONFIG_ROOT/no_vae.json:no_vae"
    "$CONFIG_ROOT/no_graph.json:no_graph"
    "$CONFIG_ROOT/random_graph.json:random_graph"
    "$CONFIG_ROOT/no_transformer.json:no_transformer"
    "$CONFIG_ROOT/no_smc_feedback.json:no_smc_feedback"
    "$CONFIG_ROOT/belief_mean_only.json:belief_mean_only"
    "$CONFIG_ROOT/belief_cov_only.json:belief_cov_only"
    "$CONFIG_ROOT/history_5.json:history_5"
    "$CONFIG_ROOT/history_40.json:history_40"
    "$CONFIG_ROOT/star_graph.json:star_graph"
    "$CONFIG_ROOT/mlp_only.json:mlp_only"
  )
fi

rm -f "$CSV_OUT" "$EPISODES_OUT"

first_eval=1
for item in "${CONFIGS[@]}"; do
  config="${item%%:*}"
  name="${item##*:}"
  if [[ ! -f "$config" ]]; then
    continue
  fi
  for seed in $SEEDS; do
    seed_dir="$RUN_ROOT/$name/seed_$seed"
    echo "==> Training $name seed $seed"
    .venv/bin/python -m symqnet.train_ppo \
      --config "$config" \
      --updates "$UPDATES" \
      --seed "$seed" \
      --output-dir "$seed_dir" \
      --validation-task-bank "$VALIDATION_TASK_BANK" \
      --validation-episodes "$VALIDATION_EPISODES" \
      --validation-interval "$VALIDATION_INTERVAL"

    echo "==> Evaluating $name seed $seed"
    eval_cmd=(.venv/bin/python -m symqnet.eval
      --config "$config" \
      --episodes "$EPISODES" \
      --agent-checkpoint "$seed_dir/best_agent.pt" \
      --agent-name "$name" \
      --train-seed "$seed" \
      --skip-baselines \
      --out "$CSV_OUT" \
      --episodes-out "$EPISODES_OUT" \
      --task-bank "$TASK_BANK" \
      --with-crlb)
    if [[ "$first_eval" -eq 0 ]]; then
      eval_cmd+=(--append)
    fi
    "${eval_cmd[@]}"
    first_eval=0
  done
done

.venv/bin/python -m symqnet.plot_shot_budget \
  --csv "$CSV_OUT" \
  --out "${CSV_OUT%.csv}.svg" \
  --title "SymQNet ablations: shot budget vs final theta-MSE"

.venv/bin/python -m symqnet.analysis.paired_main \
  --episodes-csv "$EPISODES_OUT" \
  --reference full \
  --baselines $ABLATION_BASELINES \
  --out-csv "$RUN_ROOT/paired_ablations.csv" \
  --out-tex "$RUN_ROOT/paired_ablations_table.tex"

.venv/bin/python -m symqnet.analysis.paper_figures ablation-delta \
  --csv "$CSV_OUT" \
  --out "$RUN_ROOT/ablation_delta_${ABLATION_DELTA_SHOT}.svg" \
  --reference full \
  --shot "$ABLATION_DELTA_SHOT"

.venv/bin/python -m symqnet.analysis.ablation_table \
  --csv "$CSV_OUT" \
  --out "$RUN_ROOT/ablation_delta_${ABLATION_DELTA_SHOT}.tex" \
  --reference full \
  --shot "$ABLATION_DELTA_SHOT"
