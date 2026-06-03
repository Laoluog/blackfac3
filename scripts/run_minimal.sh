#!/usr/bin/env bash
# Minimal, honest, <$25 pipeline on a single A100. Produces the whole story:
#   baseline (no fairness) vs ITA-reweighted training, then ITA-weighted
#   distillation into a small edge student, evaluated on RFW with an
#   ITA-stratified breakdown + ONNX export.
#
# Steps (sequential, each resumable — skips work whose output already exists):
#   0. smoke test (20 steps) — prove the loop runs end to end (~2 min)
#   1. precompute ITA over the training images (cached, shared by all runs)
#   2. train IR-50 baseline           (reweight none)
#   3. train IR-50 + ITA reweight     (the headline model)
#   4. distill IR-50(ITA) -> ir_18    (ITA-weighted feature distillation)
#   5. eval all 3 on RFW + ITA-binned table + plot + results table
#   6. export the edge student to ONNX + CPU latency
#
# Usage:
#   bash scripts/prepare_data.sh           # once, writes data/prepared/paths.env
#   nohup bash scripts/run_minimal.sh > run.log 2>&1 &   # survives logout
#   tail -f run.log
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- config (override via env) -------------------------------------------
ADAFACE_REPO=${ADAFACE_REPO:-/models/AdaFace}
DATA_DIR=${DATA_DIR:-./data/prepared}
# shellcheck disable=SC1091
[ -f "$DATA_DIR/paths.env" ] && source "$DATA_DIR/paths.env"
IMAGES_ROOT=${IMAGES_ROOT:?run scripts/prepare_data.sh first (IMAGES_ROOT unset)}
RFW_ROOT=${RFW_ROOT:?run scripts/prepare_data.sh first (RFW_ROOT unset)}

OUT=${OUT:-./experiments}            # checkpoints + logs
RESULTS=${RESULTS:-./results}        # eval outputs, table, plot, onnx
ITA_CACHE=${ITA_CACHE:-$OUT/train_ita.json}
TEACHER_ARCH=${TEACHER_ARCH:-ir_50}
STUDENT_ARCH=${STUDENT_ARCH:-ir_18}
EPOCHS=${EPOCHS:-12}                 # matched-budget; bump to 26 for max quality
DISTILL_EPOCHS=${DISTILL_EPOCHS:-10}
BATCH=${BATCH:-512}                  # A100 80GB headroom
WORKERS=${WORKERS:-12}
LR_MILESTONES=${LR_MILESTONES:-6,9,11}
DISTILL_MILESTONES=${DISTILL_MILESTONES:-5,8,9}

mkdir -p "$OUT" "$RESULTS"
TRAIN_COMMON=(--data-root "$IMAGES_ROOT" --data-format imagefolder \
  --adaface-repo "$ADAFACE_REPO" --batch-size "$BATCH" --num-workers "$WORKERS" --amp)

step() { echo; echo "########## $* ##########"; date; }
have() { [ -f "$1" ]; }

# ---- 0. smoke test --------------------------------------------------------
step "0/6 smoke test (20 steps)"
python train.py "${TRAIN_COMMON[@]}" --arch "$TEACHER_ARCH" --head adaface \
  --reweight none --epochs 1 --max-steps 20 \
  --output-dir "$OUT" --prefix smoke
echo "smoke OK"

# ---- 1. precompute ITA ----------------------------------------------------
step "1/6 precompute ITA cache"
if have "$ITA_CACHE"; then echo "cache exists: $ITA_CACHE"; else
  python precompute_ita.py --data-root "$IMAGES_ROOT" --out "$ITA_CACHE"
fi

# ---- 2. baseline IR-50 (no reweight) -------------------------------------
BASE_CKPT="$OUT/bupt_${TEACHER_ARCH}_none_last.ckpt"
step "2/6 train baseline $TEACHER_ARCH (reweight none)"
if have "$BASE_CKPT"; then echo "exists: $BASE_CKPT"; else
  python train.py "${TRAIN_COMMON[@]}" --arch "$TEACHER_ARCH" --head adaface \
    --reweight none --epochs "$EPOCHS" --lr-milestones "$LR_MILESTONES" \
    --output-dir "$OUT" --prefix "bupt_${TEACHER_ARCH}_none"
fi

# ---- 3. ITA-reweighted IR-50 (headline) ----------------------------------
ITA_CKPT="$OUT/bupt_${TEACHER_ARCH}_ita_last.ckpt"
step "3/6 train $TEACHER_ARCH + ITA reweight (headline)"
if have "$ITA_CKPT"; then echo "exists: $ITA_CKPT"; else
  python train.py "${TRAIN_COMMON[@]}" --arch "$TEACHER_ARCH" --head adaface \
    --reweight ita --ita-json "$ITA_CACHE" --epochs "$EPOCHS" \
    --lr-milestones "$LR_MILESTONES" \
    --output-dir "$OUT" --prefix "bupt_${TEACHER_ARCH}_ita"
fi

# ---- 4. distill IR-50(ITA) -> small student ------------------------------
STU_CKPT="$OUT/distill_${STUDENT_ARCH}_ita_last.ckpt"
step "4/6 distill $TEACHER_ARCH(ITA) -> $STUDENT_ARCH (ITA-weighted)"
if have "$STU_CKPT"; then echo "exists: $STU_CKPT"; else
  python distill.py --data-root "$IMAGES_ROOT" --data-format imagefolder \
    --adaface-repo "$ADAFACE_REPO" --batch-size "$BATCH" --num-workers "$WORKERS" --amp \
    --teacher-arch "$TEACHER_ARCH" --teacher-ckpt "$ITA_CKPT" --student-arch "$STUDENT_ARCH" \
    --reweight ita --ita-json "$ITA_CACHE" --epochs "$DISTILL_EPOCHS" \
    --lr-milestones "$DISTILL_MILESTONES" \
    --output-dir "$OUT" --prefix "distill_${STUDENT_ARCH}_ita"
fi

# ---- 5. evaluate everything ----------------------------------------------
step "5/6 RFW eval + ITA breakdown + tables"
eval_one() {  # name  arch  ckpt
  local name=$1 arch=$2 ckpt=$3
  if have "$RESULTS/${name}_rfw_results.json"; then echo "eval exists: $name"; return; fi
  python evaluate_rfw.py --checkpoint "$ckpt" --adaface-repo "$ADAFACE_REPO" \
    --rfw-root "$RFW_ROOT" --arch "$arch" --output-dir "$RESULTS" --model-name "$name"
}
eval_one "baseline_${TEACHER_ARCH}" "$TEACHER_ARCH" "$BASE_CKPT"
eval_one "ita_${TEACHER_ARCH}"      "$TEACHER_ARCH" "$ITA_CKPT"
eval_one "distill_${STUDENT_ARCH}"  "$STUDENT_ARCH" "$STU_CKPT"

python evaluate_ita.py --rfw-root "$RFW_ROOT" --scores-dir "$RESULTS" \
  --models "baseline_${TEACHER_ARCH}" "ita_${TEACHER_ARCH}" "distill_${STUDENT_ARCH}" \
  --output-dir "$RESULTS"

python results_table.py --output-dir "$RESULTS" --rfw-results \
  "$RESULTS/baseline_${TEACHER_ARCH}_rfw_results.json" \
  "$RESULTS/ita_${TEACHER_ARCH}_rfw_results.json" \
  "$RESULTS/distill_${STUDENT_ARCH}_rfw_results.json"

# ---- 6. edge export -------------------------------------------------------
step "6/6 export edge student to ONNX + benchmark"
python export_edge.py --checkpoint "$STU_CKPT" --adaface-repo "$ADAFACE_REPO" \
  --arch "$STUDENT_ARCH" --out "$RESULTS/edge_${STUDENT_ARCH}.onnx" | tee "$RESULTS/edge_summary.txt"

step "DONE"
echo "Artifacts in: $RESULTS"
echo "  - results_table.md / .tex      (RFW vs literature)"
echo "  - ita_binned_results.json      (per skin-tone-bin TAR)"
echo "  - ita_vs_tar.png               (the money plot)"
echo "  - *_rfw_results.json           (per-model RFW metrics)"
echo "  - edge_${STUDENT_ARCH}.onnx + edge_summary.txt"
echo "Send these back (+ run.log) and I'll build the paper. Then DESTROY the pod."
