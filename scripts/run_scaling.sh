#!/usr/bin/env bash
set -euo pipefail

N_VALUES="${N_VALUES:-8 10 12}"
EPISODES="${EPISODES:-100}"
UPDATES="${UPDATES:-300}"
SEEDS="${SEEDS:-777 778 779}"
JOBS="${JOBS:-4}"
RUN_ROOT="${RUN_ROOT:-runs/scaling}"
PRETRAIN_SAMPLES="${PRETRAIN_SAMPLES:-15000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
DRY_RUN="${DRY_RUN:-0}"
WITH_CRLB="${WITH_CRLB:-0}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_roots=()
for n in $N_VALUES; do
  config="configs/scaling/n$n.json"
  out="$RUN_ROOT/n$n"
  vae="artifacts/vae_n${n}_l16.pt"
  run_roots+=("$out")
  if [[ "$DRY_RUN" == "1" || ! -f "$vae" ]]; then
    run_cmd .venv/bin/python -m symqnet.pretrain_vae \
      --config "$config" \
      --samples "$PRETRAIN_SAMPLES" \
      --epochs "$PRETRAIN_EPOCHS"
  fi
  dad_config="configs/scaling/dad_n$n.json"
  if [[ -f "$dad_config" ]]; then
    run_cmd .venv/bin/python -m symqnet.paper_cpu_cluster \
      --config "$config" \
      --run-root "$out" \
      --episodes "$EPISODES" \
      --updates "$UPDATES" \
      --seeds $SEEDS \
      --jobs "$JOBS" \
      --with-crlb "$WITH_CRLB" \
      --main-comparisons dad_transformer bald_2step_fast fisher_greedy_fast fixed_optimized random \
      --extra-agent "dad_transformer=$dad_config" \
      --compact-table-shots 128
  else
    run_cmd .venv/bin/python -m symqnet.paper_cpu_cluster \
      --config "$config" \
      --run-root "$out" \
      --episodes "$EPISODES" \
      --updates "$UPDATES" \
      --seeds $SEEDS \
      --jobs "$JOBS" \
      --with-crlb "$WITH_CRLB" \
      --main-comparisons bald_2step_fast fisher_greedy_fast fixed_optimized random \
      --compact-table-shots 128
  fi
done

run_cmd .venv/bin/python -m symqnet.analysis.merge_scaling \
  --run-roots "${run_roots[@]}" \
  --out "$RUN_ROOT/scaling_summary.csv"

run_cmd .venv/bin/python -m symqnet.analysis.paper_figures scaling \
  --csv "$RUN_ROOT/scaling_summary.csv" \
  --out "$RUN_ROOT/scaling_summary.svg"

run_cmd .venv/bin/python -m symqnet.analysis.mps_validation \
  --n-values ${MPS_VALIDATION_N_VALUES:-4 5 6 7} \
  --tasks "${MPS_VALIDATION_TASKS:-3}" \
  --bond-dim "${MPS_BOND_DIM:-32}" \
  --trotter-steps "${MPS_TROTTER_STEPS:-8}" \
  --out "$RUN_ROOT/mps_validation.json"

run_cmd .venv/bin/python -m symqnet.analysis.claim_gate \
  --scaling-csv "$RUN_ROOT/scaling_summary.csv" \
  --mps-validation-json "$RUN_ROOT/mps_validation.json" \
  --out "$RUN_ROOT/claim_gate.json"
