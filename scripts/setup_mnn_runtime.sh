#!/usr/bin/env bash
#
# Fetch the official MNN Android runtime (native libs) and place it where this
# MNN-only APK can use it:
#
#   app/src/main/jniLibs/arm64-v8a/libMNN.so
#   app/src/main/jniLibs/arm64-v8a/libMNN_CL.so        (optional GPU)
#   app/src/main/jniLibs/arm64-v8a/libMNN_Vulkan.so    (optional GPU)
#   app/src/main/jniLibs/arm64-v8a/libMNN_GL.so        (optional GPU)
#   app/src/main/jniLibs/arm64-v8a/libmnncore.so       (JNI wrapper)
#   app/src/main/jniLibs/arm64-v8a/libc++_shared.so
#
# This is REQUIRED for the MNN-only build. If the package cannot be fetched the
# script exits non-zero so the CI/build never produces a broken APK.
#
# The MNN Java bindings live in this repo under
# `app/src/main/java/com/taobao/android/mnn/` — they call the native functions
# exported by `libmnncore.so`, so no MNN.jar is needed.
#
# Usage:  bash scripts/setup_mnn_runtime.sh
# Env:    MNN_VERSION (default 3.6.1), MNN_ANDROID_URL (custom zip URL)
set -eo pipefail

VERSION="${MNN_VERSION:-3.6.1}"
URL="${MNN_ANDROID_URL:-https://github.com/alibaba/MNN/releases/download/${VERSION}/mnn_${VERSION}_android_armv7_armv8_cpu_opencl_vulkan.zip}"
TMP="$(mktemp -d)"
JNI_DIR="app/src/main/jniLibs/arm64-v8a"

mkdir -p "$JNI_DIR"

echo "==> Downloading MNN Android runtime ${VERSION}"
if ! curl -L --retry 5 --retry-delay 2 --retry-all-errors \
      -o "$TMP/mnn_android.zip" "$URL"; then
  echo "ERROR: MNN runtime download failed from $URL" >&2
  rm -rf "$TMP"
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "ERROR: 'unzip' is required to extract the MNN runtime." >&2
  rm -rf "$TMP"
  exit 1
fi

( cd "$TMP" && unzip -q mnn_android.zip )

echo "==> Installing arm64-v8a native libs"
# The MNN Android package ships one subdirectory per ABI:
#   .../arm64-v8a/     (64-bit ARM)
#   .../armeabi-v7a/   (32-bit ARM)
# This APK builds ONLY for arm64-v8a (see abiFilters in app/build.gradle.kts),
# so we copy ONLY the arm64 binaries. We must NOT fall back to "everything
# named *.so": the two ABIs share basenames (libMNN.so, libmnncore.so,
# libc++_shared.so), and since "arm64-v8a" sorts before "armeabi-v7a" the old
# logic copied the arm64 libs first and then OVERWROTE them with the 32-bit
# armeabi-v7a ones. The APK then shipped 32-bit .so files that fail to load on
# arm64 devices — surfacing as the "MNN_RUNTIME_MISSING" launch error even on
# real phones.
ARM_DIR="$(find "$TMP" -type d -name 'arm64-v8a' | head -n 1)"
if [ -z "$ARM_DIR" ]; then
  echo "ERROR: no arm64-v8a directory found inside the MNN package" >&2
  rm -rf "$TMP"
  exit 1
fi

shopt -s nullglob
installed=0
for f in "$ARM_DIR"/*.so; do
  cp "$f" "$JNI_DIR/"
  echo "   installed: $(basename "$f")"
  installed=$((installed + 1))
done
shopt -u nullglob
if [ "$installed" -eq 0 ]; then
  echo "ERROR: no *.so files found in $ARM_DIR" >&2
  rm -rf "$TMP"
  exit 1
fi

echo "==> Verifying required MNN libraries"
if [ ! -s "$JNI_DIR/libMNN.so" ]; then
  echo "ERROR: $JNI_DIR/libMNN.so missing" >&2
  rm -rf "$TMP"
  exit 1
fi
if [ ! -s "$JNI_DIR/libmnncore.so" ]; then
  echo "ERROR: $JNI_DIR/libmnncore.so missing (MNN JNI wrapper)" >&2
  rm -rf "$TMP"
  exit 1
fi
# libMNN.so / libmnncore.so are built against the NDK shared libc++; without
# libc++_shared.so in the same dir the APK fails to load libMNN.so at runtime
# (UnsatisfiedLinkError "dlopen failed ... not found") — surfaced as the
# "com.taobao.android.mnn.MNNNetNative" launch error. Fail the build here so
# CI never ships a broken APK.
if [ ! -s "$JNI_DIR/libc++_shared.so" ]; then
  # Try the arm64 NDK libc++ if it lives elsewhere in the package.
  CXX="$(find "$TMP" -type f -name 'libc++_shared.so' -path '*arm64*' | head -n 1)"
  if [ -n "$CXX" ] && [ -s "$CXX" ]; then
    cp "$CXX" "$JNI_DIR/libc++_shared.so"
    echo "   installed: libc++_shared.so (from $CXX)"
  fi
fi
if [ ! -s "$JNI_DIR/libc++_shared.so" ]; then
  echo "ERROR: $JNI_DIR/libc++_shared.so missing - MNN cannot load without it" >&2
  echo "      Provide it via the NDK: find \$ANDROID_NDK -name libc++_shared.so" >&2
  rm -rf "$TMP"
  exit 1
fi

echo "==> MNN runtime ready in $JNI_DIR"
echo "    Build with:  ./gradlew assembleDebug"
rm -rf "$TMP"
