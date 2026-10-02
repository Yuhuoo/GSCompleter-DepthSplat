#!/usr/bin/env bash
set -euo pipefail

PAIRS_ROOT="${PAIRS_ROOT:-outputs/dl3dv_pairs}"
GS_OUTPUT="${GS_OUTPUT:-outputs-5-14/ablation_study-3/baseline/dl3dv_ctx2_tgt1_n10_w_dilation}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/dl3dv_baseline_target}"

python tools/collect_dl3dv_outputs.py \
  --pairs "$PAIRS_ROOT" \
  --gs-output "$GS_OUTPUT" \
  --baseline-root "$BASELINE_ROOT" \
  --overwrite
