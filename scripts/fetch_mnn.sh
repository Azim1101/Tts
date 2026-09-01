#!/usr/bin/env bash
# Downloads the official prebuilt MNN Android libraries + headers into
# dhvaani/src/main/cpp/mnn/. Run once from the repository root.
set -euo pipefail

MNN_VERSION="${MNN_VERSION:-3.6.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/dhvaani/src/main/cpp/mnn"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Fetching MNN $MNN_VERSION -> $DEST"
mkdir -p "$DEST/include" "$DEST/libs"

# ---- prebuilt .so (official release asset) ---------------------------------
ZIP="mnn_${MNN_VERSION}_android_armv7_armv8_cpu_opencl_vulkan.zip"
URL="https://github.com/alibaba/MNN/releases/download/${MNN_VERSION}/${ZIP}"
echo "--> Downloading prebuilts: $URL"
curl -fL --progress-bar -o "$TMP/$ZIP" "$URL"
unzip -q -o "$TMP/$ZIP" -d "$TMP/libs"
SRC="$(find "$TMP/libs" -type d -name 'arm64-v8a' | head -1)"
SRC_PARENT="$(dirname "$SRC")"

for abi in arm64-v8a armeabi-v7a; do
    [ -d "$SRC_PARENT/$abi" ] || continue
    mkdir -p "$DEST/libs/$abi"
    # Required: core + Express (these models contain subgraphs) + libc++_shared
    for so in libMNN.so libMNN_Express.so libc++_shared.so; do
        if [ -f "$SRC_PARENT/$abi/$so" ]; then
            cp -f "$SRC_PARENT/$abi/$so" "$DEST/libs/$abi/"
        fi
    done
    echo "    $abi libs: $(ls "$DEST/libs/$abi" | tr '\n' ' ')"
done

# ---- headers (from the matching source tag) --------------------------------
echo "--> Downloading MNN headers ($MNN_VERSION)"
curl -fL --progress-bar -o "$TMP/src.tar.gz" \
    "https://codeload.github.com/alibaba/MNN/tar.gz/refs/tags/${MNN_VERSION}"
tar xzf "$TMP/src.tar.gz" -C "$TMP" "MNN-${MNN_VERSION}/include"
rm -rf "$DEST/include"
mv "$TMP/MNN-${MNN_VERSION}/include" "$DEST/include"

echo "==> MNN setup complete"
echo "    headers: $DEST/include/MNN"
echo "    libs   : $DEST/libs/<abi>"
