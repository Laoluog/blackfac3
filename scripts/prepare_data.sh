#!/usr/bin/env bash
# Extract the downloaded archives into the layout run_minimal.sh expects, then
# validate and print the two paths you'll actually use (IMAGES_ROOT, RFW_ROOT).
#
# Inputs (env vars, with sane defaults):
#   RAW_DIR   directory holding your downloads (default: ./data)
#   DATA_DIR  where to extract everything       (default: ./data/prepared)
#
# Handles:
#   * RFW test:  test.tar.gz                 -> $DATA_DIR/RFW/test/{data,txts}
#   * BUPT imgs: the images archive(s)       -> $DATA_DIR/BUPT-images/<identity>/*.jpg
#                (auto-detects the identity-folder root; override w/ BUPT_IMAGES_ROOT)
#   * the .lst:  train_balancedface.lst      -> copied next to the images root
#
# Usage:  bash scripts/prepare_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RAW_DIR=${RAW_DIR:-./data}
DATA_DIR=${DATA_DIR:-./data/prepared}
mkdir -p "$DATA_DIR"

echo "==> RAW_DIR=$RAW_DIR   DATA_DIR=$DATA_DIR"

# ---------------------------------------------------------------------------
# 1. RFW test set  ->  $DATA_DIR/RFW/test  (must contain data/ and txts/)
# ---------------------------------------------------------------------------
RFW_ROOT="$DATA_DIR/RFW/test"
if [ -d "$RFW_ROOT/data" ] && [ -d "$RFW_ROOT/txts" ]; then
  echo "==> RFW already extracted at $RFW_ROOT"
elif [ -f "$RAW_DIR/test.tar.gz" ]; then
  echo "==> Extracting RFW test set..."
  mkdir -p "$DATA_DIR/RFW"
  tar xzf "$RAW_DIR/test.tar.gz" -C "$DATA_DIR/RFW"
else
  echo "  WARN: $RAW_DIR/test.tar.gz not found and $RFW_ROOT not present"
fi

# ---------------------------------------------------------------------------
# 2. BUPT training images  ->  identity-folder root (ImageFolder layout)
# ---------------------------------------------------------------------------
IMG_DEST="$DATA_DIR/BUPT-images"
mkdir -p "$IMG_DEST"

# Extract any not-yet-extracted image archives found in RAW_DIR. We match common
# BUPT image archive names; adjust the glob if yours differs.
shopt -s nullglob
for arc in "$RAW_DIR"/*[Ii]mage*.tar.gz "$RAW_DIR"/*[Bb]alanced*image*.tar.gz \
           "$RAW_DIR"/*[Ii]mage*.tar "$RAW_DIR"/*[Ii]mage*.zip; do
  echo "==> Extracting $arc ..."
  case "$arc" in
    *.tar.gz) tar xzf "$arc" -C "$IMG_DEST" ;;
    *.tar)    tar xf  "$arc" -C "$IMG_DEST" ;;
    *.zip)    unzip -q -o "$arc" -d "$IMG_DEST" ;;
  esac
done
shopt -u nullglob

# Auto-detect the identity-folder root: the directory with the most immediate
# subdirectories (BUPT has ~28k identity folders). Override with BUPT_IMAGES_ROOT.
if [ -n "${BUPT_IMAGES_ROOT:-}" ]; then
  IMAGES_ROOT="$BUPT_IMAGES_ROOT"
else
  IMAGES_ROOT=$(
    find "$IMG_DEST" -maxdepth 3 -type d -printf '%h\n' 2>/dev/null \
      | sort | uniq -c | sort -rn | head -1 | awk '{$1=""; sub(/^ /,""); print}'
  )
  [ -z "$IMAGES_ROOT" ] && IMAGES_ROOT="$IMG_DEST"
fi

# ---------------------------------------------------------------------------
# 3. label list next to the images
# ---------------------------------------------------------------------------
if [ -f "$RAW_DIR/train_balancedface.lst" ] && [ ! -f "$IMAGES_ROOT/../train_balancedface.lst" ]; then
  cp "$RAW_DIR/train_balancedface.lst" "$DATA_DIR/train_balancedface.lst" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 4. validate + report
# ---------------------------------------------------------------------------
echo
echo "================ resolved paths ================"
n_ids=$(find "$IMAGES_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
n_jpg=$(find "$IMAGES_ROOT" -maxdepth 2 -name '*.jpg' 2>/dev/null | head -5000 | wc -l | tr -d ' ')
echo "IMAGES_ROOT = $IMAGES_ROOT   (identity dirs: $n_ids, jpgs sampled: $n_jpg+)"
echo "RFW_ROOT    = $RFW_ROOT"
echo "================================================"
fail=0
if [ "$n_ids" -lt 100 ]; then
  echo "  WARN: only $n_ids identity dirs under IMAGES_ROOT — wrong root? set BUPT_IMAGES_ROOT and re-run."
  fail=1
fi
[ -d "$RFW_ROOT/data" ] && [ -d "$RFW_ROOT/txts" ] || { echo "  WARN: RFW_ROOT missing data/ or txts/"; fail=1; }

cat > "$DATA_DIR/paths.env" <<EOF
# sourced by run_minimal.sh
export IMAGES_ROOT="$IMAGES_ROOT"
export RFW_ROOT="$RFW_ROOT"
EOF
echo "==> wrote $DATA_DIR/paths.env  (run_minimal.sh reads this)"
[ "$fail" -eq 0 ] && echo "==> data ready." || echo "==> data NOT fully ready — fix WARNings above."
