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
mkdir -p "$DEST/libs"

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
            # Do not copy libc++_shared.so: CMake with ANDROID_STL=c++_shared automatically
            # provides libc++_shared.so from NDK toolchain, preventing duplicate symbol collisions.
            if [ "$(basename "$so")" != "libc++_shared.so" ]; then
                cp -f "$so" "$DEST/libs/$abi/"
            fi
        done
        shopt -u nullglob
        echo "    $abi libs: $(ls "$DEST/libs/$abi" | tr '\n' ' ')"
    fi
done

# ---- headers (from the matching source tag) --------------------------------
echo "--> Downloading MNN headers ($MNN_VERSION)"
curl -fL --progress-bar -o "$TMP/src.tar.gz" \
    "https://codeload.github.com/alibaba/MNN/tar.gz/refs/tags/${MNN_VERSION}"
tar xzf "$TMP/src.tar.gz" -C "$TMP" "MNN-${MNN_VERSION}/include"

rm -rf "$DEST/include"
SRC_INC="$(find "$TMP" -type d -name "include" | head -n 1)"
mv "$SRC_INC" "$DEST/include"

if [ ! -f "$DEST/include/MNN/expr/Module.hpp" ]; then
    echo "ERROR: $DEST/include/MNN/expr/Module.hpp not found!" >&2
    exit 1
fi

echo "==> MNN setup complete"
echo "    headers: $DEST/include/MNN"
echo "    libs   : $DEST/libs/<abi>"
