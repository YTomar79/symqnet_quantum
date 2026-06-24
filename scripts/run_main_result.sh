#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.json}"
RUN_ROOT="${RUN_ROOT:-runs/main_result}"
UPDATES="${UPDATES:-2500}"
EPISODES="${EPISODES:-500}"
SEEDS="${SEEDS:-777 778 779 780 781}"
WITH_CRLB="${WITH_CRLB:-0}"
TASK_BANK="${TASK_BANK:-$RUN_ROOT/task_bank.npz}"
VALIDATION_TASK_BANK="${VALIDATION_TASK_BANK:-$RUN_ROOT/validation_task_bank.npz}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-32}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-25}"
N_QUBITS="${N_QUBITS:-5}"
M_EVO="${M_EVO:-5}"
MAIN_COMPARISONS="${MAIN_COMPARISONS:-bald_2step_fast fisher_greedy_fast fixed_optimized random}"
PAPER_MIN_EPISODES="${PAPER_MIN_EPISODES:-100}"
COMPACT_TABLE_SHOTS="${COMPACT_TABLE_SHOTS:-128 512}"
ALLOW_RANDOM_VAE="${ALLOW_RANDOM_VAE:-0}"

CSV_OUT="$RUN_ROOT/shot_budget.csv"
EPISODES_OUT="$RUN_ROOT/episodes.csv"
rm -f "$CSV_OUT" "$EPISODES_OUT"
mkdir -p "$RUN_ROOT"

crlb_flag=()
if [[ "$WITH_CRLB" == "1" ]]; then
  crlb_flag=(--with-crlb)
fi
vae_flag=()
if [[ "$ALLOW_RANDOM_VAE" == "1" ]]; then
  vae_flag=(--allow-random-vae)
fi

echo "==> Evaluating non-RL baselines"
.venv/bin/python -m symqnet.eval \
  --config "$CONFIG" \
  --episodes "$EPISODES" \
  --out "$CSV_OUT" \
  --episodes-out "$EPISODES_OUT" \
  --task-bank "$TASK_BANK" \
  "${crlb_flag[@]}"

for seed in $SEEDS; do
  echo "==> Training SymQNet seed $seed"
  .venv/bin/python -m symqnet.train_ppo \
    --config "$CONFIG" \
    --updates "$UPDATES" \
    --seed "$seed" \
    --output-dir "$RUN_ROOT/symqnet_seed_$seed" \
    --validation-task-bank "$VALIDATION_TASK_BANK" \
    --validation-episodes "$VALIDATION_EPISODES" \
    --validation-interval "$VALIDATION_INTERVAL" \
    ${vae_flag[@]+"${vae_flag[@]}"}

  echo "==> Evaluating SymQNet seed $seed"
  .venv/bin/python -m symqnet.eval \
    --config "$CONFIG" \
    --episodes "$EPISODES" \
    --agent-checkpoint "$RUN_ROOT/symqnet_seed_$seed/best_agent.pt" \
    --agent-name "symqnet" \
    --train-seed "$seed" \
    --skip-baselines \
    --append \
    --out "$CSV_OUT" \
    --episodes-out "$EPISODES_OUT" \
    --task-bank "$TASK_BANK" \
    "${crlb_flag[@]}"
done

.venv/bin/python -m symqnet.plot_shot_budget \
  --csv "$CSV_OUT" \
  --out "$RUN_ROOT/shot_budget.svg"

.venv/bin/python -m symqnet.analysis.tables \
  --csv "$CSV_OUT" \
  --out "$RUN_ROOT/wallclock_mse_crlb_table.tex"

.venv/bin/python -m symqnet.analysis.reward_scatter \
  --episodes-csv "$EPISODES_OUT" \
  --out "$RUN_ROOT/reward_vs_objective.svg"

.venv/bin/python -m symqnet.analysis.paired_main \
  --episodes-csv "$EPISODES_OUT" \
  --reference symqnet \
  --baselines $MAIN_COMPARISONS \
  --out-csv "$RUN_ROOT/paired_main.csv" \
  --out-tex "$RUN_ROOT/paired_main_table.tex" \
  --acceptance-tex "$RUN_ROOT/main_result_table.tex"

if [[ "$EPISODES" -ge "$PAPER_MIN_EPISODES" ]]; then
  .venv/bin/python -m symqnet.analysis.compact_table \
    --paired-csv "$RUN_ROOT/paired_main.csv" \
    --shots $COMPACT_TABLE_SHOTS \
    --baselines $MAIN_COMPARISONS \
    --min-paired-tasks "$PAPER_MIN_EPISODES" \
    --out-csv "$RUN_ROOT/compact_main_table.csv" \
    --out-tex "$RUN_ROOT/compact_main_table.tex"
else
  echo "==> Skipping camera-ready compact table for smoke-sized EPISODES=$EPISODES"
fi

.venv/bin/python -m symqnet.analysis.seed_stability \
  --episodes-csv "$EPISODES_OUT" \
  --method symqnet \
  --out-csv "$RUN_ROOT/seed_stability.csv" \
  --out-tex "$RUN_ROOT/seed_stability_table.tex"

.venv/bin/python -m symqnet.analysis.reward_objective \
  --episodes-csv "$EPISODES_OUT" \
  --methods symqnet \
  --out-csv "$RUN_ROOT/reward_objective.csv" \
  --out-tex "$RUN_ROOT/reward_objective_table.tex"

.venv/bin/python -m symqnet.analysis.baseline_params \
  --summary-csv "$CSV_OUT" \
  --config "$CONFIG" \
  --out-csv "$RUN_ROOT/baseline_params.csv" \
  --out-tex "$RUN_ROOT/baseline_params_table.tex"

.venv/bin/python -m symqnet.analysis.paper_figures latency \
  --csv "$CSV_OUT" \
  --out "$RUN_ROOT/latency.svg"

.venv/bin/python -m symqnet.analysis.paper_figures pareto \
  --csv "$CSV_OUT" \
  --out "$RUN_ROOT/mse_latency_pareto.svg"

.venv/bin/python -m symqnet.analysis.paper_figures action-heatmap \
  --episodes-csv "$EPISODES_OUT" \
  --out "$RUN_ROOT/action_heatmap_symqnet.svg" \
  --n-qubits "$N_QUBITS" \
  --m-evo "$M_EVO" \
  --method symqnet \
  --shot 128

.venv/bin/python -m symqnet.analysis.validate_results \
  --summary-csv "$CSV_OUT" \
  --episodes-csv "$EPISODES_OUT" \
  --require-agent-checkpoints \
  --out "$RUN_ROOT/validation_report.json"
