#!/usr/bin/env bash
set -euo pipefail

# Runtime configuration. Values can be overridden at invocation time, e.g.
# GPU_ID=1 bash eval_three_way_re10k_acid.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/project/agentend2end/xieyuan-24039/ga/software/anaconda3}"
CONDA_ENV="${CONDA_ENV:-gscompleter}"
GPU_ID="${GPU_ID:-0}"
OUT_ROOT="${OUT_ROOT:-outputs-9-30/three_way_ablation}"

# Initialise Conda from the installation used by this project.
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    printf 'Conda initialization script not found: %s\n' "$CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python executable not found or not executable: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

RE10K_INDEX="assets/evaluation_index_extrapolation/re10k/re10k_ctx2_tgt1_n30.json"
ACID_INDEX="assets/evaluation_index_extrapolation/acid/acid_ctx2_tgt1_n30.json"
RE10K_CHECKPOINT="pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth"
ACID_CHECKPOINT="pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth"

required_files=(
    "$RE10K_INDEX"
    "$ACID_INDEX"
    "$RE10K_CHECKPOINT"
    "$ACID_CHECKPOINT"
)

missing=()
for path in "${required_files[@]}"; do
    [[ -f "$path" ]] || missing+=("$path")
done

if (( ${#missing[@]} > 0 )); then
    printf 'Missing evaluation files:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    printf 'Run this script from the GSCompleter-DepthSplat repository root after placing the splits and checkpoints at these paths.\n' >&2
    exit 1
fi

run_eval() {
    local dataset="$1"
    local variant="$2"
    local experiment index_path checkpoint vit_type output_dir
    local dataset_args=()
    local refinement=false gscompleter=false stereo_anchor=false
    local depth_alignment=false rc_register=false mv_refinement=false target_align=false

    case "$dataset" in
        re10k)
            experiment="+experiment=re10k"
            index_path="$RE10K_INDEX"
            checkpoint="$RE10K_CHECKPOINT"
            vit_type=vitb
            ;;
        acid)
            experiment="+experiment=re10k"
            index_path="$ACID_INDEX"
            checkpoint="$ACID_CHECKPOINT"
            vit_type=vitl
            dataset_args+=("dataset.roots=[datasets/acid]" "dataset.view_sampler.num_context_views=2")
            ;;
        *)
            printf 'Unknown dataset: %s\n' "$dataset" >&2
            return 2
            ;;
    esac

    case "$variant" in
        baseline)
            ;;
        generated-only)
            # Add generated Gaussians in the dilated hole region without GSCompleter refinement.
            refinement=true
            stereo_anchor=true
            ;;
        gscompleter)
            refinement=true
            gscompleter=true
            stereo_anchor=true
            depth_alignment=true
            rc_register=true
            mv_refinement=true
            ;;
        *)
            printf 'Unknown variant: %s\n' "$variant" >&2
            return 2
            ;;
    esac

    output_dir="$OUT_ROOT/$dataset/$variant"
    printf '\n[%s | %s | GPU %s] output: %s\n' "$dataset" "$variant" "$GPU_ID" "$output_dir"

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m src.infer \
        "$experiment" \
        mode=test \
        "${dataset_args[@]}" \
        dataset/view_sampler=evaluation \
        "dataset.view_sampler.index_path=$index_path" \
        dataset.test_chunk_interval=1 \
        model.encoder.num_scales=2 \
        model.encoder.upsample_factor=2 \
        model.encoder.lowest_feature_resolution=4 \
        "model.encoder.monodepth_vit_type=$vit_type" \
        "checkpointing.pretrained_model=$checkpoint" \
        "output_dir=$output_dir" \
        "test.using_refinement=$refinement" \
        "test.using_gscompleter=$gscompleter" \
        "test.using_stereo_anchor=$stereo_anchor" \
        "test.using_depth_alignment=$depth_alignment" \
        "test.using_rc_register=$rc_register" \
        "test.using_mv_refinement=$mv_refinement" \
        "test.using_target_align=$target_align" \
        test.save_image=false \
        test.save_gt_image=false \
        test.render_context=true \
        hydra/job_logging=stdout \
        hydra.output_subdir=null
}

for dataset in re10k acid; do
    for variant in baseline generated-only gscompleter; do
        run_eval "$dataset" "$variant"
    done
done

printf '\nAll six evaluations completed. Results are under: %s\n' "$OUT_ROOT"
