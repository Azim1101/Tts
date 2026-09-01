#!/usr/bin/env bash
#
# Download the DhVaani-0.5 MNN models and assets into app/src/main/assets/dhvaani.
#
# Source: https://huggingface.co/Bbkblo/DhVaani-0.5-MNN
#
# Usage:  bash scripts/download_models.sh
set -euo pipefail

ASSETS="${1:-app/src/main/assets/dhvaani}"
mkdir -p "$ASSETS"

echo "==> Downloading DhVaani MNN model assets into $ASSETS"
python3 - "$ASSETS" <<'PY'
import os, sys, time
from urllib.request import urlopen, Request

FILES = [
    ("text_encoder_int8.mnn", 5_000_000),
    ("fm_decoder_int8.mnn",   110_000_000),
    ("vocoder_backbone.mnn",   12_000_000),
    ("mel_fb.bin",             200_000),
    ("vocos_head.bin",         2_000_000),
    ("tokens.txt",             1_000),
    ("model.json",             500),
]

REPO = "Bbkblo/DhVaani-0.5-MNN"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"
out_dir = sys.argv[1]

def get_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    print(">> using huggingface_hub", flush=True)
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
    print(">> huggingface_hub unavailable; using streaming curl / urllib fallback", flush=True)

def has_file(name, minbytes):
    return get_size(os.path.join(out_dir, name)) >= minbytes

def ensure_file(name, minbytes):
    dest = os.path.join(out_dir, name)
    if has_file(name, minbytes):
        print(f"   OK  {name}  ({get_size(dest)} bytes)", flush=True)
        return True
    url = f"{BASE_URL}/{name}"
    part = dest + ".part"
    for attempt in range(1, 7):
        print(f"   download {name} (attempt {attempt})", flush=True)
        try:
            if os.path.exists(part):
                os.remove(part)
            req = Request(url, headers={"User-Agent": "DhVaani-build/1.0"})
            with urlopen(req, timeout=180) as response, open(part, "wb") as out_f:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
            if get_size(part) >= minbytes:
                os.replace(part, dest)
                print(f"   OK  {name}  ({get_size(dest)} bytes)", flush=True)
                return True
        except Exception as e:
            print(f"      error: {e}", flush=True)
        time.sleep(3)
    print(f"   FAILED: {name}", flush=True)
    return False

ok = all(ensure_file(name, minbytes) for name, minbytes in FILES)
if not ok:
    raise SystemExit("One or more model assets failed to download.")
print(">> all MNN assets ready in " + out_dir, flush=True)
PY

echo
echo "Done. Assets are ready in $ASSETS."
echo "Build with:  ./gradlew assembleDebug"
echo "APK:  app/build/outputs/apk/debug/app-debug.apk"
