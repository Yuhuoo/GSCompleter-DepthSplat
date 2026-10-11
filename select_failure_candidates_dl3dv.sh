#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STATS_CSV="${STATS_CSV:-outputs/paired_stats_dl3dv/per_scene.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/failure_candidates/dl3dv_ctx2_tgt1_n10}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
TOP_K="${TOP_K:-5}"
MAX_CANDIDATES="${MAX_CANDIDATES:-12}"

[[ -f "$STATS_CSV" ]] || {
  echo "Missing per-scene metrics: $STATS_CSV" >&2
  echo "Run analyze_paired_scene_metrics.py first, or set STATS_CSV to its per_scene.csv." >&2
  exit 1
}
[[ -f "$INDEX_PATH" ]] || { echo "Missing evaluation index: $INDEX_PATH" >&2; exit 1; }

python tools/select_failure_candidates.py \
  --per-scene "$STATS_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --index "$INDEX_PATH" \
  --top-k "$TOP_K" \
  --max-candidates "$MAX_CANDIDATES"
