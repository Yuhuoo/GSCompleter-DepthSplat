#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-datasets/dl3dv}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
CHECKPOINT="${CHECKPOINT:-pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/failure_cases/dl3dv_ctx2_tgt1_n10}"
CASES="${CASES:-}"

[[ -n "$CASES" ]] || {
  echo "Set CASES='label:scene_id,label:scene_id,...'" >&2
  echo "Example: CASES='pe_field_failure:<scene>,weak_overlap:<scene>'" >&2
  exit 1
}
[[ -d "$DATA_ROOT" ]] || { echo "Missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
[[ -f "$INDEX_PATH" ]] || { echo "Missing INDEX_PATH: $INDEX_PATH" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing CHECKPOINT: $CHECKPOINT" >&2; exit 1; }

IFS=',' read -r -a CASE_LIST <<< "$CASES"
for case_spec in "${CASE_LIST[@]}"; do
  label="${case_spec%%:*}"
  scene="${case_spec#*:}"
  [[ -n "$label" && -n "$scene" && "$label" != "$case_spec" ]] || {
    echo "Invalid case specification: $case_spec" >&2
    exit 1
  }
  case_output="$OUTPUT_DIR/$label/$scene"
  echo "===== Failure case: $label ($scene) ====="

  CUDA_VISIBLE_DEVICES="$GPU" \
  GS_RUNTIME_NO_DEBUG_IO=0 \
  GS_RUNTIME_QUIET=0 \
  RUNTIME_LIMIT=1 \
  python -m src.infer +experiment=dl3dv \
    mode=test \
    "dataset.roots=[$DATA_ROOT]" \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=2 \
    "dataset.view_sampler.index_path=$INDEX_PATH" \
    "dataset.overfit_to_scene=$scene" \
    model.encoder.num_scales=2 \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=8 \
    model.encoder.monodepth_vit_type=vitb \
    "checkpointing.pretrained_model=$CHECKPOINT" \
    "output_dir=$case_output" \
    test.compute_scores=true \
    test.save_image=true \
    test.save_gt_image=true \
    test.save_virtual_video=false \
    test.render_context=false \
    test.save_gaussian=false \
    test.save_colmap=false \
    test.using_refinement=true \
    test.using_gscompleter=true \
    test.using_stereo_anchor=true \
    test.angle_deg=45.0 \
    test.using_depth_alignment=true \
    test.using_rc_register=true \
    test.using_mv_refinement=true \
    test.using_target_align=false \
    test.hole_mask_threshold=0.5 \
    test.dilation_radius=3 \
    test.depth_noise_std=0.0 \
    hydra.output_subdir=null
done
