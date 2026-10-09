#!/usr/bin/env bash
set -euo pipefail

# Use the same n=10 target split as the main DL3DV evaluation.
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
BASE_OUT="${BASE_OUT:-outputs-5-14/ablation_study-3}"
SUB_DIR="${SUB_DIR:-dl3dv_ctx2_tgt1_n10_w_dilation}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/region_metrics_dl3dv}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

python tools/evaluate_region_metrics.py \
  --index "$INDEX_PATH" \
  --baseline-root "$BASE_OUT/baseline/$SUB_DIR" \
  --generated-only-root "$BASE_OUT/generated-only/$SUB_DIR" \
  --gscompleter-root "$BASE_OUT/gscompleter/$SUB_DIR" \
  --mask-root "${MASK_ROOT:-$BASE_OUT/gscompleter/$SUB_DIR}" \
  --output-dir "$OUTPUT_DIR" \
  --region "${REGION:-completed}" \
  "${LIMIT_ARGS[@]}"
