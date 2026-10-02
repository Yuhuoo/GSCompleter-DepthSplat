#!/usr/bin/env bash
set -euo pipefail

# Use the exact n=10 index used by the original GSCompleter DL3DV experiment.
# Here n=10 is the candidate-frame/extrapolation window, not the number of
# context views. The index contains 2 context views and 1 target.
DATA_ROOT="${DATA_ROOT:-datasets/dl3dv}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dl3dv_pairs}"

if [[ ! -f "$INDEX_PATH" ]]; then
  echo "Missing the original DL3DV n=10 index: $INDEX_PATH" >&2
  echo "Set INDEX_PATH to the existing evaluation index before running." >&2
  exit 1
fi

python tools/export_dl3dv_pairs.py \
  --data-root "$DATA_ROOT" \
  --index-path "$INDEX_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --overwrite
