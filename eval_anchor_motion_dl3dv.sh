#!/usr/bin/env bash
set -euo pipefail

# Run the two anchor choices on the frozen DL3DV n=10 split, then stratify by
# the target-to-nearest-context distance from measure_extrapolation_distance.py.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-datasets/dl3dv}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
CHECKPOINT="${CHECKPOINT:-pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth}"
DISTANCE_CSV="${DISTANCE_CSV:-outputs/extrapolation_pose_distance.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/anchor_motion/dl3dv_ctx2_tgt1_n10}"
RUN_INFER="${RUN_INFER:-1}"

[[ -f "$DISTANCE_CSV" ]] || { echo "Missing distance CSV: $DISTANCE_CSV" >&2; exit 1; }
[[ "$RUN_INFER" == 0 || "$RUN_INFER" == 1 ]] || { echo "RUN_INFER must be 0 or 1" >&2; exit 1; }

if [[ "$RUN_INFER" == 1 ]]; then
  [[ -d "$DATA_ROOT" ]] || { echo "Missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
  [[ -f "$INDEX_PATH" ]] || { echo "Missing INDEX_PATH: $INDEX_PATH" >&2; exit 1; }
  [[ -f "$CHECKPOINT" ]] || { echo "Missing CHECKPOINT: $CHECKPOINT" >&2; exit 1; }
  export RUNTIME_LIMIT=140
  unset RUNTIME_WARMUP || true

  run_case() {
    local case_name="$1"
    local stereo_anchor="$2"
    echo "===== Anchor case: $case_name ====="
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
      "output_dir=$OUTPUT_DIR/$case_name" \
      test.compute_scores=true \
      test.save_scene_metrics=true \
      test.save_image=false \
      test.save_gt_image=false \
      test.save_virtual_video=false \
      test.render_context=false \
      test.save_gaussian=false \
      test.save_colmap=false \
      test.using_refinement=true \
      test.using_gscompleter=true \
      "test.using_stereo_anchor=$stereo_anchor" \
      test.angle_deg=45.0 \
      test.using_depth_alignment=true \
      test.using_rc_register=true \
      test.using_mv_refinement=true \
      test.using_target_align=false \
      test.hole_mask_threshold=0.5 \
      test.dilation_radius=3 \
      test.depth_noise_std=0.0 \
      hydra.output_subdir=null
  }

  run_case default true
  run_case anchor_nearest false
fi

python tools/summarize_anchor_motion.py \
  --default "$OUTPUT_DIR/default/per_scene_metrics.csv" \
  --nearest "$OUTPUT_DIR/anchor_nearest/per_scene_metrics.csv" \
  --distance-csv "$DISTANCE_CSV" \
  --output-dir "$OUTPUT_DIR/stratified"
