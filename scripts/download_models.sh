#!/usr/bin/env bash
#
# Download the ONNX graphs and the small feature assets into app/src/main/assets,
# then convert the .npz files to the app's raw little-endian .bin format.
#
# Requires: curl, python3 + numpy (pip install numpy).
#
# Usage:  bash scripts/download_models.sh
#
set -euo pipefail

HF="https://huggingface.co/Bbkblo/DhVaani-0.5-ONNX/resolve/main"
ASSETS="app/src/main/assets"
mkdir -p "$ASSETS"

# download <file> <min_bytes>
download() {
  local file="$1"
  local minbytes="$2"
  local target="$ASSETS/$file"
  for attempt in 1 2 3 4 5; do
    echo "==> Downloading $file (attempt $attempt)"
    if curl -L --fail -C - -o "$target" "$HF/$file" 2>/dev/null; then
      local size
      size=$(stat -c%s "$target" 2>/dev/null || echo 0)
      if [ "$size" -ge "$minbytes" ]; then
        echo "    OK  $file ($size bytes)"
        return 0
      fi
      echo "    too small ($size < $minbytes), retrying"
      rm -f "$target"
    else
      echo "    curl failed, retrying"
    fi
    sleep 2
  done
  echo "ERROR: could not download $file" >&2
  return 1
}

# --- large ONNX graphs (choose int8 if available, else fp32) -------------------
download "text_encoder_int8.onnx"   5_000_000
download "fm_decoder_int8.onnx"   110_000_000
download "vocoder_backbone.onnx"   30_000_000

# --- small feature / vocab assets ---------------------------------------------
download "mel_fb.npz"       200_000
download "vocos_head.npz"   1_000_000
download "tokens.txt"         1_000

# --- convert npz -> bin --------------------------------------------------------
echo "==> Converting .npz feature files to .bin"
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "numpy missing; attempting install ..."
  if ! python3 -m pip install --quiet numpy 2>/dev/null; then
    echo "numpy is required. Install:  python3 -m pip install numpy" >&2
    exit 1
  fi
fi
python3 scripts/make_assets.py --in "$ASSETS" --out "$ASSETS"

echo
echo "Done. Assets are ready in $ASSETS."
echo "Build with:  ./gradlew assembleDebug"
echo "APK:  app/build/outputs/apk/debug/app-debug.apk"
