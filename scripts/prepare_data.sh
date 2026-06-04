#!/usr/bin/env bash
# Extract the downloaded archives into the layout run_minimal.sh expects, then
# validate and print the two paths you'll actually use (IMAGES_ROOT, RFW_ROOT).
#
# Inputs (env vars, with sane defaults):
#   RAW_DIR               directory holding your downloads   (default: ./data)
#   DATA_DIR              where to extract everything         (default: ./data/prepared)
#   BUPT_IMAGES_ARCHIVE   the images .tar.gz to extract       (default: auto-detect)
#
# The BUPT images archive ships as race_per_7000/<Race>/<identity>/<imgs>.jpg —
# i.e. an extra *race* level. torchvision.ImageFolder would wrongly treat the 4
# races as the classes, so we flatten it: one symlink per identity directory
# into a single root, giving ~28k identity classes (the correct label space).
#
# Usage:  bash scripts/prepare_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RAW_DIR=${RAW_DIR:-./data}
DATA_DIR=${DATA_DIR:-./data/prepared}
mkdir -p "$DATA_DIR"
DATA_DIR=$(cd "$DATA_DIR" && pwd)   # absolute, so symlink targets resolve anywhere
RAW_DIR=$(cd "$RAW_DIR" && pwd)

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
  tar xf --no-same-owner "$RAW_DIR/test.tar.gz" -C "$DATA_DIR/RFW" \
    || echo "  WARN: tar reported non-fatal errors during RFW extract; continuing"
else
  echo "  WARN: $RAW_DIR/test.tar.gz not found and $RFW_ROOT not present"
fi

# ---------------------------------------------------------------------------
# 2. BUPT training images
#    extract  ->  $DATA_DIR/BUPT-raw  (race_per_7000/<Race>/<identity>/<imgs>)
#    flatten  ->  $DATA_DIR/BUPT-flat (identity symlinks = ImageFolder root)
# ---------------------------------------------------------------------------
IMG_RAW="$DATA_DIR/BUPT-raw"
IMAGES_ROOT="$DATA_DIR/BUPT-flat"
mkdir -p "$IMG_RAW" "$IMAGES_ROOT"

# Pick the images archive. NB the rec archive "Equalizedface (1).tar.gz" must NOT
# be extracted here — default detection prefers the bare-name images archive and
# explicitly skips any "(1)" rec archive.
if [ -z "${BUPT_IMAGES_ARCHIVE:-}" ]; then
  shopt -s nullglob
  for cand in "$RAW_DIR"/Equalizedface.tar.gz "$RAW_DIR"/equalizedface.tar.gz \
              "$RAW_DIR"/*[Bb]alanced*image*.tar.gz "$RAW_DIR"/*[Ii]mage*.tar.gz; do
    case "$cand" in *"(1)"*) continue;; esac
    [ -f "$cand" ] && { BUPT_IMAGES_ARCHIVE="$cand"; break; }
  done
  shopt -u nullglob
fi

if find "$IMG_RAW" -maxdepth 4 -iname '*.jpg' 2>/dev/null | head -1 | grep -q .; then
  echo "==> images already extracted under $IMG_RAW"
elif [ -n "${BUPT_IMAGES_ARCHIVE:-}" ] && [ -f "$BUPT_IMAGES_ARCHIVE" ]; then
  echo "==> Extracting images from $BUPT_IMAGES_ARCHIVE (large — be patient)..."
  # Use auto-detect (tar xf) and tolerate a truncated archive: extract every
  # complete file and continue rather than aborting on the final short read.
  tar xf --no-same-owner "$BUPT_IMAGES_ARCHIVE" -C "$IMG_RAW" \
    || echo "  WARN: tar reported errors (archive likely truncated); continuing with extracted files"
else
  echo "  WARN: no images archive found. Set BUPT_IMAGES_ARCHIVE=/path/to/archive.tar.gz"
fi

# Flatten: one symlink per identity dir (a dir that directly contains jpgs).
# Works for both race/identity/img (3-level) and identity/img (2-level) layouts.
echo "==> Flattening identity folders into $IMAGES_ROOT ..."
n_link=0; n_collide=0
while IFS= read -r idir; do
  name=$(basename "$idir")
  link="$IMAGES_ROOT/$name"
  if [ -e "$link" ]; then          # globally-unique MIDs make this rare
    name="$(basename "$(dirname "$idir")")__$name"
    link="$IMAGES_ROOT/$name"
    n_collide=$((n_collide + 1))
  fi
  ln -sfn "$idir" "$link"
  n_link=$((n_link + 1))
done < <(find "$IMG_RAW" -type f -iname '*.jpg' -printf '%h\n' 2>/dev/null | sort -u)
echo "    linked $n_link identity dirs ($n_collide collisions disambiguated)"

# ---------------------------------------------------------------------------
# 3. label list (only used by the mxrec path; harmless to stage here)
# ---------------------------------------------------------------------------
if [ -f "$RAW_DIR/train_balancedface.lst" ] && [ ! -f "$DATA_DIR/train_balancedface.lst" ]; then
  cp "$RAW_DIR/train_balancedface.lst" "$DATA_DIR/train_balancedface.lst" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 4. validate + report  (-L so symlinked identity dirs are followed)
# ---------------------------------------------------------------------------
echo
echo "================ resolved paths ================"
n_ids=$(find -L "$IMAGES_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
n_jpg=$(find -L "$IMAGES_ROOT" -maxdepth 2 -iname '*.jpg' 2>/dev/null | head -5000 | wc -l | tr -d ' ')
echo "IMAGES_ROOT = $IMAGES_ROOT   (identity dirs: $n_ids, jpgs sampled: $n_jpg+)"
echo "RFW_ROOT    = $RFW_ROOT"
echo "================================================"
fail=0
if [ "$n_ids" -lt 1000 ]; then
  echo "  WARN: only $n_ids identity dirs — extraction/flatten may have failed."
  echo "        (expected ~28000). Check BUPT_IMAGES_ARCHIVE and re-run."
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
