#!/usr/bin/env bash
#
# Download the MNN model graphs + small feature assets into app/src/main/assets,
# then convert the .npz files to the app's raw little-endian .bin format.
#
# This is the *bundled model* build path (optional / developer / prebuilt APK).
# For the small APK + "user downloads the model" flow, the app itself downloads
# the selected model from Hugging Face on first launch, so CI can skip this
# script entirely (the default app flow is first-launch download).
#
# Usage:  bash scripts/download_models.sh
# Requires: python3 (+ numpy). The script auto-installs numpy & huggingface_hub.
set -euo pipefail

ASSETS="app/src/main/assets"
mkdir -p "$ASSETS"

REPO="Bbkblo/DhVaani-0.5-MNN"
FILES=(text_encoder_int8.mnn fm_decoder_int8.mnn vocoder_backbone.mnn mel_fb.npz vocos_head.npz tokens.txt model.json)
MIN=(5_000_000 110_000_000 30_000_000 200_000 1_000_000 1_000 100)

# Make sure python deps are available (best-effort in CI where pip usually works).
for pkg in numpy huggingface_hub; do
  if ! python3 -c "import $pkg" 2>/dev/null; then
    echo "==> Installing $pkg ..."
    python3 -m pip install --quiet "$pkg" 2>/dev/null || \
      python3 -m pip install --quiet --user "$pkg" 2>/dev/null || \
      { echo "ERROR: cannot install $pkg. Install: python3 -m pip install $pkg" >&2; exit 1; }
  fi
done

echo "==> Downloading MNN model assets from $REPO into $ASSETS"
REPO="$REPO" python3 - "$ASSETS" "${FILES[*]}" "${MIN[*]}" <<'PY'
import os, sys, time

REPO = os.environ["REPO"]
out_dir = sys.argv[1]
names = sys.argv[2].split()
mins = [int(x) for x in sys.argv[3].split()]
FILES = list(zip(names, mins))

def get_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    print(">> using huggingface_hub", flush=True)
except ImportError:
    snapshot_download = None
    hf_hub_download = None
    print(">> huggingface_hub unavailable; using curl fallback", flush=True)

if snapshot_download is None:
    from subprocess import run

for attempt in range(1, 5):
    try:
        if snapshot_download is not None:
            snapshot_download(repo_id=REPO, local_dir=out_dir, allow_patterns=names)
        break
    except Exception as e:
        print(f"   snapshot download attempt {attempt} failed: {e}", flush=True)
        time.sleep(3)

def ensure(name, minbytes):
    if get_size(os.path.join(out_dir, name)) >= minbytes:
        print(f"   OK  {name}  ({get_size(os.path.join(out_dir, name))} bytes)", flush=True)
        return True
    for attempt in range(1, 7):
        print(f"   download {name} (attempt {attempt})", flush=True)
        if snapshot_download is not None and hf_hub_download is not None:
            try:
                hf_hub_download(repo_id=REPO, filename=name, local_dir=out_dir, force_download=True)
            except Exception as e:
                print(f"      error: {e}", flush=True)
        else:
            r = run(["curl", "-sSL", "--retry", "5", "--retry-delay", "2",
                     "--retry-all-errors", "--connect-timeout", "60",
                     "--max-time", "1800",
                     "-A", "DhVaani-build/0.8",
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
echo "(For a small APK, skip this script — the app downloads the MNN model on first launch.)"
