#!/usr/bin/env bash
# Evaluate a trained checkpoint on RFW, bin by ITA, and build the results table.
# Usage: MODEL_NAME=bupt_adaface_ir50 CKPT=/path/to.ckpt bash scripts/run_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ADAFACE_REPO=${ADAFACE_REPO:-/models/AdaFace}
RFW_ROOT=${RFW_ROOT:-/data/RFW/test}
OUTPUT_DIR=${OUTPUT_DIR:-/data/results}
ARCH=${ARCH:-ir_50}
MODEL_NAME=${MODEL_NAME:-bupt_adaface_ir50}
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.ckpt}

echo "==> RFW verification eval"
python src/evaluate_rfw.py \
  --checkpoint "$CKPT" \
  --adaface-repo "$ADAFACE_REPO" \
  --rfw-root "$RFW_ROOT" \
  --arch "$ARCH" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME"

echo "==> ITA-binned eval + plot"
python src/evaluate_ita.py \
  --rfw-root "$RFW_ROOT" \
  --scores-dir "$OUTPUT_DIR" \
  --models "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR"

echo "==> Results table (markdown + LaTeX, vs literature baselines)"
python src/results_table.py \
  --rfw-results "$OUTPUT_DIR/${MODEL_NAME}_rfw_results.json" \
  --output-dir "$OUTPUT_DIR"

echo "Done. See $OUTPUT_DIR/results_table.md and ita_vs_tar.png"
