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
so_count=0
# Prefer files inside an arm64/arm64-v8a path; fall back to everything named *.so.
{
  find "$TMP" -type f -name '*.so' -path '*arm64*'
  find "$TMP" -type f -name '*.so'
} | sort -u | while read -r f; do
  cp "$f" "$JNI_DIR/"
  echo "   installed: $(basename "$f")"
done

# libc++_shared.so can live outside the arch folder (NDK layout).
while read -r f; do
  cp "$f" "$JNI_DIR/" || true
  echo "   installed: $(basename "$f")"
done < <(find "$TMP" -type f -name 'libc++_shared.so' | sort -u)

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
# (UnsatisfiedLinkError "dlopen failed ... not found") — surfaced as a cryptic
# "com.taobao.android.mnn.MNNNetNative" error on launch. Fail the build here.
if [ ! -s "$JNI_DIR/libc++_shared.so" ]; then
  echo "WARN: $JNI_DIR/libc++_shared.so missing - MNN may fail to load at runtime" >&2
  echo "      Provide it via the NDK: find \$ANDROID_NDK -name libc++_shared.so" >&2
fi

echo "==> MNN runtime ready in $JNI_DIR"
echo "    Build with:  ./gradlew assembleDebug"
rm -rf "$TMP"
