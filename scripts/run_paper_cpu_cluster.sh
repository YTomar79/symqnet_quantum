#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.json}"
RUN_ROOT="${RUN_ROOT:-runs/main_result}"
EPISODES="${EPISODES:-500}"
UPDATES="${UPDATES:-2500}"
SEEDS="${SEEDS:-777 778 779 780 781}"
JOBS="${JOBS:-4}"
WITH_CRLB="${WITH_CRLB:-0}"
MAIN_COMPARISONS="${MAIN_COMPARISONS:-dad_transformer bald_2step_fast fisher_greedy_fast fixed_optimized random}"
EXTRA_AGENT="${EXTRA_AGENT:-}"
if [[ -z "$EXTRA_AGENT" && "$CONFIG" == "configs/default.json" ]]; then
  EXTRA_AGENT="dad_transformer=configs/dad_transformer.json"
fi
COMPACT_TABLE_SHOTS="${COMPACT_TABLE_SHOTS:-128 512}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-32}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-25}"
PRETRAIN_SAMPLES="${PRETRAIN_SAMPLES:-15000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
ANONYMIZE_MANIFEST="${ANONYMIZE_MANIFEST:-0}"
anonymize_flag=()
if [[ "$ANONYMIZE_MANIFEST" == "1" ]]; then
  anonymize_flag=(--anonymize-manifest)
fi
extra_agent_flag=()
if [[ -n "$EXTRA_AGENT" ]]; then
  extra_agent_flag=(--extra-agent "$EXTRA_AGENT")
fi

if [[ "$CONFIG" == "configs/default.json" && ! -f artifacts/vae_n5_l16.pt ]]; then
  echo "==> Pretraining N=5 VAE required by configs/default.json"
  .venv/bin/python -m symqnet.pretrain_vae \
    --config "$CONFIG" \
    --samples "$PRETRAIN_SAMPLES" \
    --epochs "$PRETRAIN_EPOCHS"
fi

.venv/bin/python -m symqnet.paper_cpu_cluster \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT" \
  --episodes "$EPISODES" \
  --updates "$UPDATES" \
  --seeds $SEEDS \
  --jobs "$JOBS" \
  --with-crlb "$WITH_CRLB" \
  --main-comparisons $MAIN_COMPARISONS \
  --validation-episodes "$VALIDATION_EPISODES" \
  --validation-interval "$VALIDATION_INTERVAL" \
  --compact-table-shots $COMPACT_TABLE_SHOTS \
  "${extra_agent_flag[@]}" \
  "${anonymize_flag[@]}"
