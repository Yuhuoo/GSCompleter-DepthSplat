#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-datasets/dl3dv}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
CHECKPOINT="${CHECKPOINT:-pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/registration_robustness/dl3dv_ctx2_tgt1_n10}"
LIMIT="${LIMIT:-20}"
RESUME="${RESUME:-0}"

# The perturbation units are meters in the normalized scene coordinates for
# lateral/object shifts, and pixels for silhouette_shift.
LATERAL_OFFSET="${LATERAL_OFFSET:-0.05}"
OBJECT_SHIFT="${OBJECT_SHIFT:-0.05}"
SILHOUETTE_SHIFT="${SILHOUETTE_SHIFT:-12.0}"

# Hydra parses numeric command-line overrides strictly.  Normalize integer
# environment values such as SILHOUETTE_SHIFT=12 to a float literal because
# TestCfg.perturbation_strength is typed as float.
as_float_literal() {
  local value="$1"
  if [[ "$value" == *.* || "$value" == *e* || "$value" == *E* ]]; then
    printf '%s' "$value"
  else
    printf '%s.0' "$value"
  fi
}

[[ -d "$DATA_ROOT" ]] || { echo "Missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
[[ -f "$INDEX_PATH" ]] || { echo "Missing INDEX_PATH: $INDEX_PATH" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing CHECKPOINT: $CHECKPOINT" >&2; exit 1; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be positive" >&2; exit 1; }
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || { echo "RESUME must be 0 or 1" >&2; exit 1; }

export RUNTIME_LIMIT="$LIMIT"
unset RUNTIME_WARMUP || true

run_case() {
  local case_name="$1"
  local registration_mode="$2"
  local perturbation="$3"
  local strength="$4"
  local case_output="$OUTPUT_DIR/$case_name"
  local strength_literal
  strength_literal="$(as_float_literal "$strength")"

  if [[ "$RESUME" == "1" && -f "$case_output/metrics_summary.txt" ]]; then
    echo "===== Skipping completed case: $case_name ====="
    return
  fi

  echo "===== Registration robustness case: $case_name ====="
  mkdir -p "$case_output"
  printf 'registration_mode=%s\nregistration_perturbation=%s\nperturbation_strength=%s\n' \
    "$registration_mode" "$perturbation" "$strength" \
    > "$case_output/experiment_config.txt"
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
    test.using_stereo_anchor=true \
    test.angle_deg=45.0 \
    test.using_depth_alignment=true \
    test.using_rc_register=true \
    test.using_mv_refinement=true \
    test.using_target_align=false \
    "test.registration_mode=$registration_mode" \
    "test.registration_perturbation=$perturbation" \
    "test.perturbation_strength=$strength_literal" \
    test.hole_mask_threshold=0.5 \
    test.dilation_radius=3 \
    test.depth_noise_std=0.0 \
    hydra.output_subdir=null
}

# Clean comparison and three controlled perturbation families.
run_case clean_ray ray none 0.0
run_case clean_free free none 0.0
run_case lateral_ray ray lateral_offset "$LATERAL_OFFSET"
run_case lateral_free free lateral_offset "$LATERAL_OFFSET"
run_case silhouette_ray ray silhouette_shift "$SILHOUETTE_SHIFT"
run_case silhouette_free free silhouette_shift "$SILHOUETTE_SHIFT"
run_case object_ray ray object_shift "$OBJECT_SHIFT"
run_case object_free free object_shift "$OBJECT_SHIFT"

python tools/summarize_registration_robustness.py --root "$OUTPUT_DIR"
