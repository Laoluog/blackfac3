#!/usr/bin/env bash
# One command to start your first run on a fresh DO GPU droplet.
#   - installs deps + AdaFace (if not already)
#   - checks your data is actually in place (fails loudly if not)
#   - runs a 20-step smoke test to catch crashes in ~1 min
#   - launches the real IR-101 balanced baseline in the background (survives logout)
#
# Usage:  bash scripts/first_run.sh
# Tail:   tail -f /data/experiments/train.log
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="$HOME/bf-venv"
ADAFACE_REPO=${ADAFACE_REPO:-/models/AdaFace}
DATA_ROOT=${DATA_ROOT:-/data/BUPT-Balancedface/rec_for_mxnet}
LST=${LST:-/data/BUPT-Balancedface/train_balancedface.lst}
OUTPUT_DIR=${OUTPUT_DIR:-/data/experiments}
export ADAFACE_REPO DATA_ROOT LST OUTPUT_DIR
export ARCH=${ARCH:-ir_101} HEAD=${HEAD:-adaface} REWEIGHT=${REWEIGHT:-none} EPOCHS=${EPOCHS:-26}

# 1. setup if needed
if [ ! -d "$VENV" ]; then
  echo "==> First time: installing environment (this takes a few minutes)..."
  bash scripts/setup_droplet.sh
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 2. preflight — fail loudly BEFORE spending GPU time
echo "==> Preflight checks"
fail=0
for f in "$DATA_ROOT/train.rec" "$DATA_ROOT/train.idx"; do
  if [ -f "$f" ]; then echo "  ok   $f"; else echo "  MISSING  $f"; fail=1; fi
done
[ -f "$LST" ] && echo "  ok   $LST" || echo "  note: $LST not found (fine; num_classes read from rec header)"
[ -f "$ADAFACE_REPO/net.py" ] && echo "  ok   $ADAFACE_REPO/net.py" || { echo "  MISSING  $ADAFACE_REPO/net.py"; fail=1; }
python -c "import torch; assert torch.cuda.is_available()" \
  && echo "  ok   CUDA visible" || { echo "  MISSING  CUDA / GPU"; fail=1; }
if [ "$fail" -ne 0 ]; then
  echo "Preflight failed — fix the MISSING items above, then re-run."; exit 1
fi
mkdir -p "$OUTPUT_DIR"

# 3. smoke test (20 steps, ~1 min) — proves the whole loop works end to end
echo "==> Smoke test (20 steps)"
EPOCHS=1 MAXSTEPS=20 PREFIX=smoke bash scripts/run_train.sh
echo "==> Smoke test PASSED"

# 4. real run, backgrounded so you can disconnect
echo "==> Launching real run: $ARCH / $HEAD / reweight=$REWEIGHT / $EPOCHS epochs"
nohup bash scripts/run_train.sh > "$OUTPUT_DIR/train.log" 2>&1 &
echo "    PID $! — logging to $OUTPUT_DIR/train.log"
echo
echo "Watch it:   tail -f $OUTPUT_DIR/train.log"
echo "When done:  ARCH=$ARCH MODEL_NAME=bupt_${HEAD}_${ARCH} \\"
echo "              CKPT=$OUTPUT_DIR/bupt_${HEAD}_${ARCH}_${REWEIGHT}_last.ckpt \\"
echo "              bash scripts/run_eval.sh"
echo "Then DESTROY the droplet to stop billing."
