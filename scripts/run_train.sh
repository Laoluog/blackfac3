#!/usr/bin/env bash
# Train AdaFace on BUPT-Balancedface (mxrec). Edit the paths/vars then run.
# Usage: bash scripts/run_train.sh
set -euo pipefail

ADAFACE_REPO=${ADAFACE_REPO:-/models/AdaFace}
DATA_ROOT=${DATA_ROOT:-/data/BUPT-Balancedface/rec_for_mxnet}   # must hold train.rec + train.idx
LST=${LST:-/data/BUPT-Balancedface/train_balancedface.lst}
OUTPUT_DIR=${OUTPUT_DIR:-/data/experiments}
ARCH=${ARCH:-ir_50}            # ir_50 to iterate cheaply; ir_101 for the final model
HEAD=${HEAD:-adaface}          # adaface | arcface | cosface
REWEIGHT=${REWEIGHT:-none}     # none | race | ita
EPOCHS=${EPOCHS:-26}
BATCH=${BATCH:-256}
MAXSTEPS=${MAXSTEPS:-0}         # >0 = smoke test (stop each epoch after N batches)
PREFIX=${PREFIX:-bupt_${HEAD}_${ARCH}_${REWEIGHT}}

python train.py \
  --data-root "$DATA_ROOT" \
  --data-format mxrec \
  --lst "$LST" \
  --adaface-repo "$ADAFACE_REPO" \
  --arch "$ARCH" \
  --head "$HEAD" \
  --reweight "$REWEIGHT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --max-steps "$MAXSTEPS" \
  --amp \
  --num-workers 8 \
  --output-dir "$OUTPUT_DIR" \
  --prefix "$PREFIX"
