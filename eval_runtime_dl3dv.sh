#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-datasets/dl3dv}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
CHECKPOINT="${CHECKPOINT:-pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/runtime/dl3dv_ctx2_tgt1_n10}"
LIMIT="${LIMIT:-}"
MODE="${MODE:-both}"

[[ -d "$DATA_ROOT" ]] || { echo "Missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
[[ -f "$INDEX_PATH" ]] || { echo "Missing INDEX_PATH: $INDEX_PATH" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing CHECKPOINT: $CHECKPOINT" >&2; exit 1; }

if [[ -n "$LIMIT" ]]; then
  [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be a positive integer" >&2; exit 1; }
  export RUNTIME_LIMIT="$LIMIT"
else
  unset RUNTIME_LIMIT || true
fi

echo "Runtime protocol: DL3DV ctx2-tgt1-n10"
echo "GPU: $GPU"
echo "Output root: $OUTPUT_DIR"
echo "Mode: $MODE"
if [[ -n "$LIMIT" ]]; then
  echo "Scene limit: $LIMIT"
else
  echo "Scene limit: full test index"
fi

run_case() {
  local case_name="$1"
  local refinement="$2"
  local gscompleter="$3"
  local stereo_anchor="$4"
  local depth_alignment="$5"
  local rc_register="$6"
  local mv_refinement="$7"
  local case_output="$OUTPUT_DIR/$case_name"

  echo
  echo "===== Runtime case: $case_name ====="
  CUDA_VISIBLE_DEVICES="$GPU" \
  GS_RUNTIME_NO_DEBUG_IO=1 \
  GS_RUNTIME_QUIET=1 \
  python -m src.infer +experiment=dl3dv \
    mode=test \
    "dataset.roots=[$DATA_ROOT]" \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=2 \
    "dataset.view_sampler.index_path=$INDEX_PATH" \
    model.encoder.num_scales=2 \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=8 \
    model.encoder.monodepth_vit_type=vitb \
    "checkpointing.pretrained_model=$CHECKPOINT" \
    "output_dir=$case_output" \
    test.compute_scores=false \
    test.save_image=false \
    test.save_gt_image=false \
    test.save_virtual_video=false \
    test.render_context=false \
    test.save_gaussian=false \
    test.save_colmap=false \
    "test.using_refinement=$refinement" \
    "test.using_gscompleter=$gscompleter" \
    "test.using_stereo_anchor=$stereo_anchor" \
    "test.using_depth_alignment=$depth_alignment" \
    "test.using_rc_register=$rc_register" \
    "test.using_mv_refinement=$mv_refinement" \
    test.using_target_align=false \
    hydra.output_subdir=null

  python tools/summarize_runtime.py --input "$case_output/runtime_summary.json"
}

case "$MODE" in
  ours)
    run_case gscompleter true true true true true true
    ;;
  baseline)
    run_case baseline false false false false false false
    ;;
  both)
    run_case baseline false false false false false false
    run_case gscompleter true true true true true true
    ;;
  *)
    echo "MODE must be one of: ours, baseline, both" >&2
    exit 1
    ;;
esac
