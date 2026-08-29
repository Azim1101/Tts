#!/usr/bin/env bash
#
# Download the ONNX graphs and small feature assets into app/src/main/assets,
# then convert the .npz files to the app's raw little-endian .bin format.
#
# Uses the `huggingface_hub` python client (handles the LFS redirect, streaming,
# resume and retries) with a plain `curl` fallback for environments without it.
# Every file is verified by a minimum-byte-size check so truncated downloads
# (the ~125 MB fm_decoder is the usual suspect) are retried.
#
# Requires: python3 (+ numpy). The script auto-installs numpy & huggingface_hub.
#
# Usage:  bash scripts/download_models.sh
set -euo pipefail

ASSETS="app/src/main/assets"
mkdir -p "$ASSETS"

# Make sure python deps are available (best-effort in CI where pip usually works).
for pkg in numpy huggingface_hub; do
  if ! python3 -c "import $pkg" 2>/dev/null; then
    echo "==> Installing $pkg ..."
    python3 -m pip install --quiet "$pkg" 2>/dev/null || \
      python3 -m pip install --quiet --user "$pkg" 2>/dev/null || \
      { echo "ERROR: cannot install $pkg. Install: python3 -m pip install $pkg" >&2; exit 1; }
  fi
done

echo "==> Downloading model assets into $ASSETS"
python3 - "$ASSETS" <<'PY'
import os, sys, time

# heavy downloader first, then the light feature/vocab files
FILES = [
    ("text_encoder_int8.onnx", 5_000_000),
    ("fm_decoder_int8.onnx",   110_000_000),
    ("vocoder_backbone.onnx",   30_000_000),
    ("mel_fb.npz",              200_000),
    ("vocos_head.npz",          1_000_000),
    ("tokens.txt",              1_000),
]

REPO = "Bbkblo/DhVaani-0.5-ONNX"
out_dir = sys.argv[1]

def get_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    print(">> using huggingface_hub", flush=True)
    # download the whole (small) set matching our files, retrying missing ones
    names = [f for f, _ in FILES]
    for attempt in range(1, 5):
        try:
            snapshot_download(repo_id=REPO, local_dir=out_dir, allow_patterns=names)
            break
        except Exception as e:
            print(f"   snapshot download attempt {attempt} failed: {e}", flush=True)
            time.sleep(3)
except ImportError:
    snapshot_download = None
    print(">> huggingface_hub unavailable; using curl fallback", flush=True)

if snapshot_download is None:
    from subprocess import run

def has_snapshot(name):
    # snapshot_download with local_dir writes files directly into out_dir
    return get_size(os.path.join(out_dir, name))

def ensure(name, minbytes):
    if has_snapshot(name) >= minbytes:
        print(f"   OK  {name}  ({get_size(os.path.join(out_dir, name))} bytes)", flush=True)
        return True
    for attempt in range(1, 7):
        print(f"   download {name} (attempt {attempt})", flush=True)
        if snapshot_download is not None:
            try:
                hf_hub_download(repo_id=REPO, filename=name, local_dir=out_dir,
                                force_download=True)
            except Exception as e:
                print(f"      error: {e}", flush=True)
        else:
            r = run(["curl", "-sSL", "--retry", "5", "--retry-delay", "2",
                     "--retry-all-errors", "--connect-timeout", "60",
                     "--max-time", "1800",
                     "-A", "DhVaani-build/0.5",
                     "-o", os.path.join(out_dir, name),
                     f"https://huggingface.co/{REPO}/resolve/main/{name}"])
            if r.returncode != 0:
                print("      curl failed", flush=True)
                continue
        size = get_size(os.path.join(out_dir, name))
        if size >= minbytes:
            print(f"   OK  {name}  ({size} bytes)", flush=True)
            return True
        print(f"      too small ({size} < {minbytes}); retrying", flush=True)
        time.sleep(3)
    print(f"   FAILED: {name}", flush=True)
    return False

ok = all(ensure(name, minbytes) for name, minbytes in FILES)
if not ok:
    raise SystemExit("One or more model assets failed to download.")
print(">> all assets downloaded", flush=True)
PY

echo "==> Converting .npz feature files to .bin"
python3 scripts/make_assets.py --in "$ASSETS" --out "$ASSETS"

echo
echo "Done. Assets are ready in $ASSETS."
echo "Build with:  ./gradlew assembleDebug"
echo "APK:  app/build/outputs/apk/debug/app-debug.apk"
