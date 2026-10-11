#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-datasets/dl3dv}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
CHECKPOINT="${CHECKPOINT:-pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sensitivity/dl3dv_ctx2_tgt1_n10}"
LIMIT="${LIMIT:-20}"

[[ -d "$DATA_ROOT" ]] || { echo "Missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
[[ -f "$INDEX_PATH" ]] || { echo "Missing INDEX_PATH: $INDEX_PATH" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing CHECKPOINT: $CHECKPOINT" >&2; exit 1; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be positive" >&2; exit 1; }

export RUNTIME_LIMIT="$LIMIT"
unset RUNTIME_WARMUP || true

run_case() {
  local case_name="$1"
  local stereo_anchor="$2"
  local angle_deg="$3"
  local hole_threshold="$4"
  local dilation_radius="$5"
  local depth_noise_std="$6"
  local case_output="$OUTPUT_DIR/$case_name"

  echo "===== Sensitivity case: $case_name ====="
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
    test.compute_scores=true \
    test.save_image=false \
    test.save_gt_image=false \
    test.save_virtual_video=false \
    test.render_context=false \
    test.save_gaussian=false \
    test.save_colmap=false \
    test.using_refinement=true \
    test.using_gscompleter=true \
    "test.using_stereo_anchor=$stereo_anchor" \
    "test.angle_deg=$angle_deg" \
    test.using_depth_alignment=true \
    test.using_rc_register=true \
    test.using_mv_refinement=true \
    test.using_target_align=false \
    "test.hole_mask_threshold=$hole_threshold" \
    "test.dilation_radius=$dilation_radius" \
    "test.depth_noise_std=$depth_noise_std" \
    hydra.output_subdir=null
}

# Shared default configuration.
run_case default true 45.0 0.5 3 0.0

# Anchor choices and maximum rotation thresholds.
run_case anchor_nearest false 45.0 0.5 3 0.0
run_case anchor_stereo_15 true 15.0 0.5 3 0.0
run_case anchor_stereo_30 true 30.0 0.5 3 0.0
run_case anchor_stereo_60 true 60.0 0.5 3 0.0

# Hole-mask threshold and insertion dilation.
run_case hole_threshold_030 true 45.0 0.3 3 0.0
run_case hole_threshold_070 true 45.0 0.7 3 0.0
run_case dilation_0 true 45.0 0.5 0 0.0
run_case dilation_1 true 45.0 0.5 1 0.0
run_case dilation_5 true 45.0 0.5 5 0.0

# Relative sparse-depth noise: 0%, 5%, and 10% standard deviation.
run_case depth_noise_005 true 45.0 0.5 3 0.05
run_case depth_noise_010 true 45.0 0.5 3 0.10

python tools/summarize_sensitivity.py --root "$OUTPUT_DIR"
