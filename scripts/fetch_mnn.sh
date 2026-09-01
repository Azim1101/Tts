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

for abi in arm64-v8a armeabi-v7a; do
    SRC_DIR="$(find "$TMP/libs" -type d -name "$abi" | head -n 1)"
    if [ -n "$SRC_DIR" ] && [ -d "$SRC_DIR" ]; then
        mkdir -p "$DEST/libs/$abi"
        shopt -s nullglob
        for so in "$SRC_DIR"/*.so; do
            cp -f "$so" "$DEST/libs/$abi/"
        done
        shopt -u nullglob
        echo "    $abi libs: $(ls "$DEST/libs/$abi" | tr '\n' ' ')"
    fi
done

# If libc++_shared.so is missing from arm64-v8a, search elsewhere in the archive
if [ ! -f "$DEST/libs/arm64-v8a/libc++_shared.so" ]; then
    CXX="$(find "$TMP/libs" -type f -name 'libc++_shared.so' -path '*arm64*' | head -n 1)"
    if [ -n "$CXX" ] && [ -f "$CXX" ]; then
        cp -f "$CXX" "$DEST/libs/arm64-v8a/libc++_shared.so"
        echo "    arm64-v8a: added libc++_shared.so"
    fi
fi

# ---- headers (from the matching source tag) --------------------------------
echo "--> Downloading MNN headers ($MNN_VERSION)"
curl -fL --progress-bar -o "$TMP/src.tar.gz" \
    "https://codeload.github.com/alibaba/MNN/tar.gz/refs/tags/${MNN_VERSION}"
tar xzf "$TMP/src.tar.gz" -C "$TMP"

INC_DIR="$(find "$TMP" -type d -path '*/MNN*/include' | head -n 1)"
if [ -z "$INC_DIR" ]; then
    INC_DIR="$(find "$TMP" -type d -name 'include' | head -n 1)"
fi

if [ -z "$INC_DIR" ]; then
    echo "ERROR: could not find include directory in MNN source archive" >&2
    exit 1
fi

rm -rf "$DEST/include"
cp -r "$INC_DIR" "$DEST/include"

echo "==> MNN setup complete"
echo "    headers: $DEST/include/MNN"
echo "    libs   : $DEST/libs/<abi>"
